"""FP8 NT GEMM for 1D2D block scales: ``scale_a`` per 1x128, ``scale_b`` per 128x128.

A persistent grid of one producer and two consumer warp-groups: the producer
fills a shared-memory ring through TMA, including one M-contiguous ``scale_a``
column per stage; each consumer folds every K-step's partial under its scales.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.gemm.call_spec import GemmCall
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import device_calibration, get_sm_count

__all__ = ["GemmFp81D2DFwdKernel"]

_FP8_1D2D_HELPER_PATH = csrc_path("fp8_1d2d_helper.h")
_TMA_BFLOAT16 = 9
_TMA_INTERLEAVE_NONE = 0
_TMA_SWIZZLE_32B = 1
_TMA_SWIZZLE_64B = 2
_TMA_SWIZZLE_128B = 3
_TMA_L2_128B = 2
_TMA_OOB_NONE = 0
_BLOCK_K = 128
# The driver reserves this much of every block's shared memory
# (cudaDevAttrReservedSharedMemoryPerBlock).
_SMEM_RESERVED_PER_BLOCK = 1024


def _dynamic_smem_limit(device_index: Optional[int]) -> int:
    """Dynamic shared memory one block may claim on the target device, in bytes."""
    index = torch.cuda.current_device() if device_index is None else device_index
    props = torch.cuda.get_device_properties(index)
    return int(props.shared_memory_per_block_optin) - _SMEM_RESERVED_PER_BLOCK


def _dynamic_smem_bytes(
    *,
    block_m: int,
    block_n: int,
    num_stages: int,
    scale_k: int,
    scale_b_count: int,
    shared_epilogue: bool,
    stage_scale_b_per_k: bool,
) -> int:
    """Dynamic shared memory ``main`` allocates for this schedule, in bytes.

    Barriers and the staged B-scale buffer are not dynamic, so neither is counted.
    """
    total = num_stages * (block_m + block_n) * _BLOCK_K  # a_shared + b_shared, one byte per element
    total += num_stages * block_m * 4  # scale_a_shared
    if shared_epilogue:
        total += block_m * block_n * 2  # shared_c, bfloat16
    if not stage_scale_b_per_k:
        total += scale_b_count * scale_k * 4  # one_scale_b
    return total


_LEGAL_BLOCK_M = (64, 128, 256)
_LEGAL_BLOCK_N = (16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192)


def _validate_schedule(
    m: int,
    n: int,
    k: int,
    *,
    block_m: int,
    block_n: int,
    num_stages: int,
    num_sms: int,
    num_multicast: int,
    multicast_on_a: bool,
    group_size_m: int,
    group_unroll: int,
    sm_count: int,
    smem_limit: int,
    shared_epilogue: bool,
) -> None:
    """Answer a configuration before anything is compiled.

    Raises:
        ValueError: Naming the parameter that cannot be built for this shape.
    """
    if block_m not in _LEGAL_BLOCK_M:
        raise ValueError(f"block_m must be one of {_LEGAL_BLOCK_M}, got {block_m}")
    if block_n not in _LEGAL_BLOCK_N:
        raise ValueError(f"block_n must be one of {_LEGAL_BLOCK_N}, got {block_n}")
    if block_m == 256 and block_n > 128:
        raise ValueError("block_m=256 requires block_n<=128")
    if group_size_m < 1:
        raise ValueError(f"group_size_m must be positive, got {group_size_m}")
    if group_unroll < 1:
        raise ValueError(f"group_unroll must be positive, got {group_unroll}")
    if num_stages < 1:
        raise ValueError(f"num_stages must be positive, got {num_stages}")
    if num_sms < 1 or num_sms > sm_count:
        raise ValueError(f"num_sms must be in [1, {sm_count}], got {num_sms}")
    if num_multicast not in (1, 2):
        raise ValueError(f"num_multicast must be 1 or 2, got {num_multicast}")
    if num_sms % num_multicast:
        raise ValueError(f"num_sms={num_sms} must be divisible by {num_multicast=}")
    if num_multicast == 1 and multicast_on_a:
        raise ValueError("multicast_on_a requires num_multicast=2")
    if num_multicast == 2 and group_size_m % 2:
        raise ValueError(f"multicast pairs tiles within a group, so {group_size_m=} must be even")

    num_pid_m = -(-m // block_m)
    num_pid_n = -(-n // block_n)
    if num_multicast == 2:
        multicast_tiles = num_pid_n if multicast_on_a else num_pid_m
        if multicast_tiles % 2:
            axis = "N" if multicast_on_a else "M"
            raise ValueError(f"multicast on {axis} requires an even tile count")

    scale_k = -(-k // _BLOCK_K)
    worker_count = num_sms // num_multicast
    max_waves = -(-(num_pid_m * num_pid_n // num_multicast) // worker_count)
    smem_bytes = _dynamic_smem_bytes(
        block_m=block_m,
        block_n=block_n,
        num_stages=num_stages,
        scale_k=scale_k,
        scale_b_count=1 if _BLOCK_K % block_n == 0 else 2,
        shared_epilogue=shared_epilogue,
        stage_scale_b_per_k=max_waves > 1 and scale_k < 16,
    )
    if smem_bytes > smem_limit:
        raise ValueError(
            f"this schedule needs {smem_bytes} bytes of shared memory, over the "
            f"{smem_limit} a block may claim; lower num_stages={num_stages}, "
            f"block_m={block_m} or block_n={block_n}"
        )


_FP8_1D2D_CONFIGS: dict[str, dict[tuple[int, int, int], dict[str, int | bool]]] = {
    "h200": {
        (128, 2112, 7168): {
            "block_m": 64,
            "block_n": 32,
            "num_stages": 14,
            "num_sms": 132,
            "num_multicast": 1,
            "multicast_on_a": False,
            "group_size_m": 16,
            "group_unroll": 1,
        },
        (128, 7168, 2048): {
            "block_m": 128,
            "block_n": 64,
            "num_stages": 8,
            "num_sms": 112,
            "num_multicast": 1,
            "multicast_on_a": False,
            "group_size_m": 16,
            "group_unroll": 1,
        },
        (4096, 2112, 7168): {
            "block_m": 128,
            "block_n": 192,
            "num_stages": 4,
            "num_sms": 118,
            "num_multicast": 2,
            "multicast_on_a": False,
            "group_size_m": 2,
            "group_unroll": 1,
        },
        (4096, 4096, 7168): {
            "block_m": 256,
            "block_n": 128,
            "num_stages": 3,
            "num_sms": 128,
            "num_multicast": 2,
            "multicast_on_a": True,
            "group_size_m": 16,
            "group_unroll": 1,
        },
        (4096, 7168, 2048): {
            "block_m": 256,
            "block_n": 128,
            "num_stages": 3,
            "num_sms": 128,
            "num_multicast": 2,
            "multicast_on_a": True,
            "group_size_m": 8,
            "group_unroll": 3,
        },
        (4096, 7168, 16384): {
            "block_m": 256,
            "block_n": 128,
            "num_stages": 3,
            "num_sms": 128,
            "num_multicast": 2,
            "multicast_on_a": True,
            "group_size_m": 16,
            "group_unroll": 1,
        },
        (4096, 24576, 1536): {
            "block_m": 128,
            "block_n": 192,
            "num_stages": 4,
            "num_sms": 132,
            "num_multicast": 1,
            "multicast_on_a": False,
            "group_size_m": 32,
            "group_unroll": 1,
        },
    },
}


@functools.lru_cache(maxsize=32)
def _gemm_fp8_1d2d_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    *,
    sm_count: int,
    smem_limit: int,
    shared_epilogue: bool = False,
) -> Callable:
    """Build the persistent 1D2D GEMM; the returned factory takes the tile config."""
    half_m = 64
    block_k = _BLOCK_K
    accum_dtype = "float"
    scale_k = (k + block_k - 1) // block_k

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=[
            "-O3",
            "--use_fast_math",
            "-DENABLE_BF16",
            "-include",
            _FP8_1D2D_HELPER_PATH,
        ],
    )
    def kernel_func(
        block_m: int = 128,
        block_n: int = 128,
        num_stages: int = 3,
        num_sms: int = sm_count,
        num_multicast: int = 1,
        multicast_on_a: bool = False,
        group_size_m: int = 16,
        group_unroll: int = 1,
    ) -> Callable:
        _validate_schedule(
            m,
            n,
            k,
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
            num_sms=num_sms,
            num_multicast=num_multicast,
            multicast_on_a=multicast_on_a,
            group_size_m=group_size_m,
            group_unroll=group_unroll,
            sm_count=sm_count,
            smem_limit=smem_limit,
            shared_epilogue=shared_epilogue,
        )
        scale_b_count = 1 if _BLOCK_K % block_n == 0 else 2
        wgmma_helper = f"tl::fp8_gemm_wgmma_64x128_by_128x{block_n}_lo"
        promote_helper = (
            f"tl::fp8_gemm_1d2d_promote_64x{block_n}"
            if scale_b_count == 1
            else f"tl::fp8_gemm_1d2d_promote_two_b_scales_64x{block_n}"
        )
        store_helper = (
            f"tl::fp8_gemm_raw_acc_stsm_bf16_swizzled_bm{block_m}_64x{block_n}"
            if shared_epilogue
            else f"tl::fp8_gemm_raw_acc_store_global_64x{block_n}_v2"
        )
        epilogue_swizzle_bytes = next(w for w in (128, 64, 32) if (block_n * 2) % w == 0)
        epilogue_block_n = epilogue_swizzle_bytes // 2
        epilogue_store_count = block_n // epilogue_block_n
        epilogue_swizzle = {
            32: _TMA_SWIZZLE_32B,
            64: _TMA_SWIZZLE_64B,
            128: _TMA_SWIZZLE_128B,
        }[epilogue_swizzle_bytes]
        fragment_regs = (half_m * block_n) // 128
        wave_block_m = 64 if block_m == 64 else 128
        m_waves = block_m // wave_block_m
        math_threads = 128 if block_m == 64 else 256
        total_threads = math_threads + 128
        math_warps = math_threads // 32
        num_pid_m = -(-m // block_m)
        num_pid_n = -(-n // block_n)
        total_tiles = num_pid_m * num_pid_n
        worker_count = num_sms // num_multicast
        total_tasks = total_tiles // num_multicast
        max_waves = -(-total_tasks // worker_count)
        k_unroll_group = next(g for g in (8, 4, 1) if scale_k % g == 0)
        stage_1d2d_b_per_k = max_waves > 1 and scale_k < 16
        static_ring = scale_k % num_stages == 0 or (max_waves == 1 and num_multicast == 1)
        rings_per_task = scale_k // num_stages
        if static_ring and scale_k % num_stages == 0 and k_unroll_group < num_stages <= 16:
            k_unroll_group = num_stages
        k_unroll_factor = k_unroll_group * group_unroll
        a_tma_annotations = {"cluster_mask": 3} if num_multicast == 2 and multicast_on_a else None
        b_tma_annotations = (
            {"cluster_mask": 3} if num_multicast == 2 and not multicast_on_a else None
        )
        # TileLang cannot bound the shuffled guard, so the election scope is named here.
        tma_plain_ann = {"leader_scope_threads": 32}
        tma_a_ann = {**(a_tma_annotations or {}), **tma_plain_ann}
        tma_b_ann = {**(b_tma_annotations or {}), **tma_plain_ann}

        @T.macro
        def decode(task_id, cluster_rank, mt, nt):
            if num_multicast == 2:
                primary = num_pid_n if multicast_on_a else num_pid_m
                secondary = num_pid_m if multicast_on_a else num_pid_n
                block_id = task_id * 2 + cluster_rank
                tiles_per_group = T.int32(group_size_m * secondary)
                first = (block_id // tiles_per_group) * T.int32(group_size_m)
                in_group = block_id % tiles_per_group
                group_len = T.min(T.int32(group_size_m), T.int32(primary) - first)
                if multicast_on_a:
                    mt[0] = in_group // group_len
                    nt[0] = first + in_group % group_len
                else:
                    mt[0] = first + in_group % group_len
                    nt[0] = in_group // group_len
            else:
                tiles_per_group = T.int32(group_size_m * num_pid_n)
                group_id = task_id // tiles_per_group
                first_m = group_id * T.int32(group_size_m)
                group_m = T.min(T.int32(group_size_m), T.int32(num_pid_m) - first_m)
                mt[0] = first_m + (task_id % tiles_per_group) % group_m
                nt[0] = (task_id % tiles_per_group) // group_m

        def launch():
            if num_multicast == 2:
                return T.ClusterKernel(num_sms, cluster_dims=2, threads=total_threads)
            return T.Kernel(num_sms, threads=total_threads)

        @T.prim_func
        def main(
            a: T.Tensor((m, k), dtype),
            b: T.Tensor((n, k), dtype),
            scale_a: T.Tensor((scale_k, m), "float32"),
            scale_b: T.Tensor(((n + 127) // 128, scale_k), "float32"),
            c: T.Tensor((m, n), out_dtype),
        ) -> None:
            with launch() as (pid,):
                a_shared = T.alloc_shared((num_stages, block_m, block_k), dtype)
                b_shared = T.alloc_shared((num_stages, block_n, block_k), dtype)
                partial = T.alloc_local((fragment_regs,), accum_dtype)
                final = T.alloc_local((m_waves, fragment_regs), accum_dtype)
                if shared_epilogue:
                    shared_c = T.alloc_shared((block_m * block_n,), out_dtype)
                scale_a_shared = T.alloc_shared((num_stages, 1, block_m), accum_dtype)
                if stage_1d2d_b_per_k:
                    # scope="shared": a shared.dyn store here makes TileLang fence it against the
                    # TMA writes with a barrier one warp cannot complete.
                    one_scale_b = T.alloc_shared(
                        (num_stages, scale_b_count), accum_dtype, scope="shared"
                    )
                else:
                    one_scale_b = T.alloc_shared((scale_b_count, scale_k), accum_dtype)
                T.annotate_layout(
                    {
                        a_shared: tilelang.layout.make_swizzled_layout(a_shared),
                        b_shared: tilelang.layout.make_swizzled_layout(b_shared),
                    }
                )

                full = T.alloc_barrier([1] * num_stages)
                if num_multicast == 2:
                    empty = T.alloc_cluster_barrier([num_multicast * math_warps] * num_stages)
                else:
                    empty = T.alloc_barrier([math_warps] * num_stages)
                producer_index = T.alloc_var("int32", init=0)
                consumer_index = T.alloc_var("int32", init=0)
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")
                scales = T.alloc_local((4,), accum_dtype)
                tx = T.get_thread_binding()
                cluster_rank = T.block_rank_in_cluster() if num_multicast == 2 else T.int32(0)
                if shared_epilogue:
                    output_desc = T.create_tma_descriptor(
                        _TMA_BFLOAT16,
                        2,
                        c.data,
                        n,
                        m,
                        1,
                        n * 2,
                        epilogue_block_n,
                        block_m,
                        1,
                        1,
                        _TMA_INTERLEAVE_NONE,
                        epilogue_swizzle,
                        _TMA_L2_128B,
                        _TMA_OOB_NONE,
                    )
                # Lanes 4i..4i+3 of warp w hold accumulator rows w*16 + i and w*16 + i + 8.
                acc_row0 = ((tx // 32) % 4) * 16 + (tx % 32) // 4

                if tx >= math_threads:
                    T.dec_max_nreg(40)
                    producer_tx = tx - math_threads
                    # Shuffled so the value stays warp-uniform for NVCC's uniform datapath.
                    producer_warp = T.tvm_warp_shuffle(
                        T.uint32(0xFFFFFFFF), producer_tx // 32, 0, 32, 32
                    )
                    for wave in T.serial(max_waves):
                        task_id = T.int32(worker_count) * wave + pid // num_multicast
                        if task_id < total_tasks:
                            decode(task_id, cluster_rank, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            if producer_warp == 0:
                                for kk in T.unroll(
                                    scale_k, unroll_factor=k_unroll_group if static_ring else 1
                                ):
                                    if static_ring:
                                        slot = kk % num_stages
                                        phase = (wave * rings_per_task + kk // num_stages) & 1
                                    else:
                                        slot = producer_index % num_stages
                                        phase = (producer_index // num_stages) & 1
                                    T.barrier_wait(empty[slot], phase ^ 1)
                                    T.tma_copy(
                                        a[
                                            m_start : m_start + block_m,
                                            kk * block_k : (kk + 1) * block_k,
                                        ],
                                        a_shared[slot, :, :],
                                        barrier=full[slot],
                                        annotations=tma_a_ann,
                                    )
                                    T.tma_copy(
                                        scale_a[
                                            kk : kk + 1,
                                            m_start : m_start + block_m,
                                        ],
                                        scale_a_shared[slot, :, :],
                                        barrier=full[slot],
                                        annotations=tma_a_ann,
                                    )
                                    T.tma_copy(
                                        b[
                                            n_start : n_start + block_n,
                                            kk * block_k : (kk + 1) * block_k,
                                        ],
                                        b_shared[slot, :, :],
                                        barrier=full[slot],
                                        annotations=tma_b_ann,
                                    )
                                    if stage_1d2d_b_per_k and producer_tx == 0:
                                        for scale_b_row in T.unroll(scale_b_count):
                                            scale_row = T.min(
                                                n_start // 128 + scale_b_row,
                                                (n + 127) // 128 - 1,
                                            )
                                            one_scale_b[slot, scale_b_row] = scale_b[scale_row, kk]
                                    if producer_tx == 0:
                                        T.barrier_arrive(full[slot])
                                    if not static_ring:
                                        producer_index = producer_index + 1
                    if num_multicast == 2 and producer_warp == 0:
                        tasks_done = T.max(
                            T.int32(0),
                            (T.int32(total_tasks) - pid // num_multicast + worker_count - 1)
                            // worker_count,
                        )
                        for drain_slot in T.unroll(num_stages):
                            if static_ring:
                                slot = drain_slot
                                phase = (tasks_done * rings_per_task) & 1
                            else:
                                slot = producer_index % num_stages
                                phase = (producer_index // num_stages) & 1
                            T.barrier_wait(empty[slot], phase ^ 1)
                            if not static_ring:
                                producer_index = producer_index + 1

                else:
                    T.inc_max_nreg(232)
                    math_wg_idx = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), tx // 128, 0, 32, 32)
                    math_wg_offset = math_wg_idx * half_m
                    a_desc_lo = T.call_extern(
                        "uint32",
                        "tl::fp8_gemm_wgmma_desc_lo",
                        T.address_of(a_shared[0, math_wg_offset, 0]),
                    )
                    b_desc_lo = T.call_extern(
                        "uint32",
                        "tl::fp8_gemm_wgmma_desc_lo",
                        T.address_of(b_shared[0, 0, 0]),
                    )
                    if num_multicast == 2:
                        empty_remote = T.call_extern(
                            "uint32",
                            "tl::fp8_smem_mapa",
                            T.call_extern("uint32", "tl::fp8_smem_u32", T.address_of(empty[0])),
                            T.uint32(tx % 2),
                        )
                    scale_a_word = T.call_extern(
                        "uint32",
                        "tl::fp8_smem_u32",
                        T.address_of(scale_a_shared[0, 0, math_wg_offset + acc_row0]),
                    )
                    for wave in T.serial(max_waves):
                        task_id = T.int32(worker_count) * wave + pid // num_multicast
                        if task_id < total_tasks:
                            decode(task_id, cluster_rank, mt, nt)
                            m_start = mt[0] * block_m
                            n_start = nt[0] * block_n
                            if not stage_1d2d_b_per_k:
                                T.sync_threads(barrier_id=10, arrive_count=math_threads)
                                for scale_iter in T.serial(
                                    -(-(scale_b_count * scale_k) // math_threads)
                                ):
                                    scale_linear = tx + scale_iter * math_threads
                                    if scale_linear < scale_b_count * scale_k:
                                        scale_b_row = scale_linear // scale_k
                                        scale_col = scale_linear % scale_k
                                        scale_row = T.min(
                                            n_start // 128 + scale_b_row,
                                            (n + 127) // 128 - 1,
                                        )
                                        one_scale_b[scale_b_row, scale_col] = scale_b[
                                            scale_row, scale_col
                                        ]
                                T.sync_threads(barrier_id=10, arrive_count=math_threads)
                            T.clear(final)
                            if scale_b_count == 2:
                                first_scale_iters = T.min(block_n, 128 - n_start % 128) // 8
                            for kk in T.unroll(scale_k, unroll_factor=k_unroll_factor):
                                if static_ring:
                                    slot = kk % num_stages
                                    phase = (wave * rings_per_task + kk // num_stages) & 1
                                else:
                                    slot = consumer_index % num_stages
                                    phase = (consumer_index // num_stages) & 1
                                T.barrier_wait(full[slot], phase)
                                if stage_1d2d_b_per_k:
                                    scales[2] = one_scale_b[slot, 0]
                                    if scale_b_count == 2:
                                        scales[3] = one_scale_b[slot, 1]
                                else:
                                    scales[2] = one_scale_b[0, kk]
                                    if scale_b_count == 2:
                                        scales[3] = one_scale_b[1, kk]
                                for m_wave in T.unroll(m_waves):
                                    scale_a_offset = (slot * block_m + m_wave * wave_block_m) * 4
                                    scales[0] = T.call_extern(
                                        "float32",
                                        "tl::fp8_lds_f32",
                                        scale_a_word + T.uint32(scale_a_offset),
                                    )
                                    scales[1] = T.call_extern(
                                        "float32",
                                        "tl::fp8_lds_f32",
                                        scale_a_word + T.uint32(scale_a_offset + 8 * 4),
                                    )
                                    T.call_extern(
                                        "handle",
                                        wgmma_helper,
                                        partial.data,
                                        a_desc_lo
                                        + T.uint32(
                                            (slot * block_m + m_wave * wave_block_m)
                                            * (block_k // 16)
                                        ),
                                        b_desc_lo + T.uint32(slot * block_n * (block_k // 16)),
                                    )
                                    T.wait_wgmma(0)
                                    if m_wave == m_waves - 1:
                                        if num_multicast == 2:
                                            if tx % 32 < num_multicast:
                                                T.call_extern(
                                                    "handle",
                                                    "tl::fp8_mbar_arrive_cluster",
                                                    empty_remote + T.uint32(slot * 8),
                                                )
                                        elif tx % 32 == 0:
                                            T.barrier_arrive(empty[slot])
                                    promote_args = (
                                        partial.data,
                                        T.address_of(final[m_wave, 0]),
                                        scales[0],
                                        scales[1],
                                        scales[2],
                                    )
                                    if scale_b_count == 1:
                                        T.call_extern("handle", promote_helper, *promote_args)
                                    elif block_n == 192:
                                        if first_scale_iters == 8:
                                            T.call_extern(
                                                "handle",
                                                f"{promote_helper}_split8",
                                                *promote_args,
                                                scales[3],
                                            )
                                        else:
                                            T.call_extern(
                                                "handle",
                                                f"{promote_helper}_split16",
                                                *promote_args,
                                                scales[3],
                                            )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            promote_helper,
                                            *promote_args,
                                            scales[3],
                                            first_scale_iters,
                                        )
                                if not static_ring:
                                    consumer_index = consumer_index + 1
                            if shared_epilogue:
                                if max_waves > 1:
                                    if tx < epilogue_store_count:
                                        T.tma_store_wait(0)
                                    T.sync_threads(barrier_id=13, arrive_count=math_threads)
                                for m_wave in T.unroll(m_waves):
                                    T.call_extern(
                                        "handle",
                                        store_helper,
                                        T.address_of(final[m_wave, 0]),
                                        T.address_of(shared_c[0]),
                                        m_wave * wave_block_m + math_wg_offset,
                                    )
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=14, arrive_count=math_threads)
                                if tx < epilogue_store_count:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_tma_store_2d_issue",
                                        output_desc,
                                        T.address_of(shared_c[tx * block_m * epilogue_block_n]),
                                        n_start + tx * epilogue_block_n,
                                        m_start,
                                    )
                                    T.tma_store_arrive()
                            else:
                                for m_wave in T.unroll(m_waves):
                                    T.call_extern(
                                        "handle",
                                        store_helper,
                                        T.address_of(final[m_wave, 0]),
                                        c.data,
                                        m_start + m_wave * wave_block_m + math_wg_offset,
                                        n_start,
                                        m,
                                        n,
                                    )
                    if shared_epilogue and tx < epilogue_store_count:
                        T.tma_store_wait(0)

        return main

    return kernel_func


class GemmFp81D2DFwdKernel(Kernel):
    """FP8 NT GEMM for 1D2D scales, bfloat16 output, no bias.

    ``scale_a`` is logically ``[M, ceil(K/128)]`` with stride ``(1, M)``;
    ``scale_b`` is row-major ``[ceil(N/128), ceil(K/128)]``.

    Args:
        m: Rows of ``a``; at least 128.
        n: Rows of ``b``, columns of the output.
        k: Contraction dim.
        dtype: Operand dtype; ``torch.float8_e4m3fn``.
        out_dtype: Output dtype; ``torch.bfloat16``.
        config: Kernel config override; unset keys take their default.
        tune: Whether to autotune over :attr:`autotune_configs`.
        device_index: The device the kernel is built for.
        shared_epilogue: Whether to stage the tile through shared memory and store
            it with TMA. ``None`` takes the calibrated choice for this shape.
    """

    @staticmethod
    def _uses_shared_epilogue(n: int) -> bool:
        """Whether the TMA epilogue can address this ``n``; a narrower one is stored packed."""
        return n % 8 == 0

    @staticmethod
    def _shape_refusal(m: int, n: int, k: int) -> Optional[str]:
        """Why this kernel cannot address these shapes, or ``None`` when it can.

        TMA addresses the contiguous dimension in 16-byte units -- ``k`` for the fp8
        operands, ``m`` for ``scale_a.T`` -- and the packed global store writes ``c``
        two BF16 columns at a time.
        """
        if m < 128:
            return f"m={m} is below one 128-row tile"
        offenders = [
            f"{name}={value} is not a multiple of {unit} ({what})"
            for name, value, unit, what in (
                ("k", k, 16, "a and b are read K-major through TMA, 16 fp8 per 16 bytes"),
                ("m", m, 4, "scale_a.T is read M-major through TMA, 4 fp32 per 16 bytes"),
                ("n", n, 2, "the epilogue writes c two BF16 columns at a time"),
            )
            if value % unit
        ]
        if not offenders:
            return None
        return "; ".join(offenders)

    supported_archs = [90]

    @classmethod
    def applies(cls, call: GemmCall) -> bool:
        return (
            call.block_scale_grid == "1d2d"
            and call.scale_a_stride == (1, call.m)
            and call.dtype == torch.float8_e4m3fn
            and call.out_dtype == torch.bfloat16
            and not call.has_bias
            and cls._shape_refusal(call.m, call.n, call.k) is None
        )

    @classmethod
    def entry_for(cls, call: GemmCall) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (call.m, call.n, call.k, call.dtype, call.out_dtype, index)
        return identity, lambda: cls(
            call.m,
            call.n,
            call.k,
            call.dtype,
            call.out_dtype,
            tune=call.tune,
            device_index=index,
        )

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
        shared_epilogue: Optional[bool] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if dtype != torch.float8_e4m3fn:
            raise NotImplementedError(f"{type(self).__name__} takes float8_e4m3fn, got {dtype}")
        if out_dtype != torch.bfloat16:
            raise NotImplementedError(f"{type(self).__name__} writes bfloat16, got {out_dtype}")

        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.sm_count = get_sm_count(self.device_index)
        calibration = device_calibration(self.device_index)
        self._calibrated = _FP8_1D2D_CONFIGS.get(calibration, {}).get((m, n, k))
        self.shared_epilogue = (
            self._uses_shared_epilogue(n) if shared_epilogue is None else bool(shared_epilogue)
        )

        refusal = self._shape_refusal(m, n, k)
        if refusal is not None:
            raise ValueError(f"{type(self).__name__} cannot serve m={m} n={n} k={k}: {refusal}")

        self.smem_limit = _dynamic_smem_limit(self.device_index)
        self.kernel = _gemm_fp8_1d2d_kernel(
            m,
            n,
            k,
            self.dtype_str,
            self.out_dtype_str,
            sm_count=self.sm_count,
            smem_limit=self.smem_limit,
            shared_epilogue=self.shared_epilogue,
        )
        self.init_config(config, tune)

    @property
    def out_dtype_str(self) -> str:
        return self.dtype_to_str(self.out_dtype)

    @property
    def default_config(self) -> dict:
        if self._calibrated is not None:
            return dict(self._calibrated)
        m_tiles = (self.m + 127) // 128
        target_n = self.n * m_tiles / self.sm_count
        if target_n <= 24:
            block_n = 16
        elif target_n <= 48:
            block_n = 32
        elif target_n <= 96:
            block_n = 64
        else:
            block_n = 128
        return {
            "block_m": 128,
            "block_n": block_n,
            "num_stages": 3,
            "num_sms": self.sm_count,
            "num_multicast": 1,
            "multicast_on_a": False,
            "group_size_m": 16,
            "group_unroll": 1,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        """The legal SM90 1D2D configuration space."""
        block_m_options = [64, 128] if self.m < 256 else [64, 128, 256]
        block_n_options = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192]
        default = self.default_config
        group_size_m = int(default["group_size_m"])
        group_unroll = int(default["group_unroll"])
        scale_k = -(-self.k // _BLOCK_K)

        configs = []
        for block_m in block_m_options:
            for block_n in block_n_options:
                if block_m > 128 and block_n > 128:
                    continue
                num_blocks = -(-self.m // block_m) * -(-self.n // block_n)
                num_waves = -(-num_blocks // self.sm_count)
                min_sms = -(-num_blocks // num_waves)
                multicast_options = [(1, False)]
                if self.m >= 512 and self.sm_count % 2 == 0 and group_size_m % 2 == 0:
                    if -(-self.m // block_m) % 2 == 0:
                        multicast_options.append((2, False))
                    if -(-self.n // block_n) % 2 == 0:
                        multicast_options.append((2, True))
                for num_stages in range(1, 13):
                    smem_bytes = _dynamic_smem_bytes(
                        block_m=block_m,
                        block_n=block_n,
                        num_stages=num_stages,
                        scale_k=scale_k,
                        scale_b_count=1 if 128 % block_n == 0 else 2,
                        shared_epilogue=self.shared_epilogue,
                        stage_scale_b_per_k=False,
                    )
                    if smem_bytes > self.smem_limit:
                        continue
                    for num_multicast, multicast_on_a in multicast_options:
                        aligned_min_sms = -(-min_sms // num_multicast) * num_multicast
                        for num_sms in sorted({aligned_min_sms, self.sm_count}):
                            configs.append(
                                {
                                    "block_m": block_m,
                                    "block_n": block_n,
                                    "num_stages": num_stages,
                                    "num_sms": num_sms,
                                    "num_multicast": num_multicast,
                                    "multicast_on_a": multicast_on_a,
                                    "group_size_m": group_size_m,
                                    "group_unroll": group_unroll,
                                }
                            )
        return configs

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError(f"{type(self).__name__} has no bias epilogue")
        return self.kernel(**self.config)(a, b, scale_a.T, scale_b)
