#pragma once

#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/common.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/instruction/wgmma.h>

#include <cuda.h>
#include <cutlass/float8.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>

namespace tl {

__device__ __forceinline__ float fp8_scale_mul(float lhs, float rhs) {
  float out;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(lhs), "f"(rhs));
  return out;
}

// Broadcast from lane 0 so NVCC keeps this and every offset added to it uniform.
__device__ __forceinline__ uint32_t fp8_gemm_wgmma_desc_lo(fp8_e4_t* smem) {
  GmmaDescriptor desc;
  initialize_wgmma_descriptor<1, 1, 64>(desc, smem);
  return __shfl_sync(0xffffffff, desc.reg32_[0], 0);
}

template <int BlockN>
__device__ __forceinline__ void fp8_gemm_wgmma_64x128_by_128xN_lo(
    float* accumulator, uint32_t a_lo, uint32_t b_lo) {
  constexpr uint64_t kDescHi = uint64_t(0x40000040u) << 32;
  warpgroup_fence_operand(accumulator, BlockN / 2);
  warpgroup_arrive();
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    wgmma_ss<DataType::kFloat8_e4m3, DataType::kFloat8_e4m3,
             DataType::kFloat32, 64, BlockN, 32, false, false, 1, 1>(
        kDescHi | uint64_t(a_lo + ki * 2), kDescHi | uint64_t(b_lo + ki * 2),
        reinterpret_cast<uint32_t*>(accumulator), 0 < ki ? 1 : 0);
  }
  warpgroup_commit_batch();
  warpgroup_fence_operand(accumulator, BlockN / 2);
}

// The caller commits the bulk group and waits for it, so the wait can be deferred.
TL_DEVICE void fp8_tma_store_2d_issue(const CUtensorMap& descriptor,
                                      void const* smem_ptr, int x, int y) {
  uint64_t desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t src = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
      "[%0, {%2, %3}], [%1];" : : "l"(desc), "r"(src), "r"(x), "r"(y) : "memory");
}

// The scales arrive by value: the stage is released, so nothing here may read it.
template <int BlockN>
__device__ __forceinline__ void fp8_gemm_1d2d_promote(
    float* partial, float* final_accum, float scale_a_row0, float scale_a_row1,
    float scale_b) {
  float const scale0 = scale_a_row0 * scale_b;
  float const scale1 = scale_a_row1 * scale_b;
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    final_accum[i * 4 + 0] += scale0 * partial[i * 4 + 0];
    final_accum[i * 4 + 1] += scale0 * partial[i * 4 + 1];
    final_accum[i * 4 + 2] += scale1 * partial[i * 4 + 2];
    final_accum[i * 4 + 3] += scale1 * partial[i * 4 + 3];
  }
}

template <int BlockN>
__device__ __forceinline__ void fp8_gemm_1d2d_promote_two_b_scales(
    float* partial, float* final_accum, float scale_a_row0, float scale_a_row1,
    float scale_b_0, float scale_b_1, int first_scale_iters) {
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    float const scale_b = i < first_scale_iters ? scale_b_0 : scale_b_1;
    float const scale0 = scale_a_row0 * scale_b;
    float const scale1 = scale_a_row1 * scale_b;
    final_accum[i * 4 + 0] += scale0 * partial[i * 4 + 0];
    final_accum[i * 4 + 1] += scale0 * partial[i * 4 + 1];
    final_accum[i * 4 + 2] += scale1 * partial[i * 4 + 2];
    final_accum[i * 4 + 3] += scale1 * partial[i * 4 + 3];
  }
}

template <int BlockN, int FirstScaleIters>
__device__ __forceinline__ void fp8_gemm_1d2d_promote_two_b_scales_split(
    float* partial, float* final_accum, float scale_a_row0, float scale_a_row1,
    float scale_b_0, float scale_b_1) {
  float const scale_0_0 = fp8_scale_mul(scale_a_row0, scale_b_0);
  float const scale_1_0 = fp8_scale_mul(scale_a_row1, scale_b_0);
#pragma unroll
  for (int i = 0; i < FirstScaleIters; ++i) {
    final_accum[i * 4 + 0] += scale_0_0 * partial[i * 4 + 0];
    final_accum[i * 4 + 1] += scale_0_0 * partial[i * 4 + 1];
    final_accum[i * 4 + 2] += scale_1_0 * partial[i * 4 + 2];
    final_accum[i * 4 + 3] += scale_1_0 * partial[i * 4 + 3];
  }
  float const scale_0_1 = fp8_scale_mul(scale_a_row0, scale_b_1);
  float const scale_1_1 = fp8_scale_mul(scale_a_row1, scale_b_1);
#pragma unroll
  for (int i = FirstScaleIters; i < BlockN / 8; ++i) {
    final_accum[i * 4 + 0] += scale_0_1 * partial[i * 4 + 0];
    final_accum[i * 4 + 1] += scale_0_1 * partial[i * 4 + 1];
    final_accum[i * 4 + 2] += scale_1_1 * partial[i * 4 + 2];
    final_accum[i * 4 + 3] += scale_1_1 * partial[i * 4 + 3];
  }
}

template <int BlockM, int BlockN>
__device__ __forceinline__ void fp8_gemm_raw_acc_stsm_bf16_swizzled(
    float* accumulator, bfloat16_t* output_smem, int m_offset) {
  constexpr int kElemBytes = sizeof(bfloat16_t);
  constexpr int kTileBytes = BlockN * kElemBytes;
  constexpr int kSwizzleBytes =
      kTileBytes % 128 == 0 ? 128 : (kTileBytes % 64 == 0 ? 64 : 32);
  constexpr int kTmaBlockN = kSwizzleBytes / kElemBytes;
  constexpr int kBankGroups = kSwizzleBytes / 16;
  constexpr int kWgmmaMPerWarp = 16;
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const warp_in_group = (static_cast<int>(threadIdx.x) >> 5) & 3;
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    int const atom_offset = i / (kTmaBlockN / 8);
    int const in_atom_offset = i % (kTmaBlockN / 8);
    int const bank_group_index = in_atom_offset + lane * kBankGroups;
    int const row = kBankGroups == 8 ? in_atom_offset / 8 + lane
                                      : bank_group_index / 8;
    int col = kBankGroups == 8 ? in_atom_offset : bank_group_index % 8;
    col ^= row % kBankGroups;
    auto* dst = reinterpret_cast<cute::uint128_t*>(
        reinterpret_cast<uint8_t*>(output_smem) +
        warp_in_group * (kWgmmaMPerWarp * kSwizzleBytes) +
        m_offset * kSwizzleBytes + atom_offset * BlockM * kSwizzleBytes +
        row * 128 + col * 16);
    nv_bfloat162 v0 = __float22bfloat162_rn(
        {accumulator[i * 4 + 0], accumulator[i * 4 + 1]});
    nv_bfloat162 v1 = __float22bfloat162_rn(
        {accumulator[i * 4 + 2], accumulator[i * 4 + 3]});
    cute::SM90_U32x2_STSM_N::copy(
        *reinterpret_cast<uint32_t*>(&v0), *reinterpret_cast<uint32_t*>(&v1),
        *dst);
  }
}

template <int BlockN>
__device__ __forceinline__ void fp8_gemm_raw_acc_store_global_vec2(
    float* accumulator, bfloat16_t const* output, int m_start, int n_start,
    int shape_m, int shape_n) {
  bfloat16_t* out = const_cast<bfloat16_t*>(output);
  int const lane = static_cast<int>(threadIdx.x) & 31;
  int const warp_in_group = (static_cast<int>(threadIdx.x) >> 5) & 3;
  int const row0 = m_start + warp_in_group * 16 + lane / 4;
  int const row1 = row0 + 8;
#pragma unroll
  for (int i = 0; i < BlockN / 8; ++i) {
    int const col = n_start + i * 8 + (lane % 4) * 2;
    if (col + 1 < shape_n) {
      if (row0 < shape_m) {
        uint32_t packed;
        bfloat16_t* values = reinterpret_cast<bfloat16_t*>(&packed);
        values[0] = static_cast<bfloat16_t>(accumulator[i * 4 + 0]);
        values[1] = static_cast<bfloat16_t>(accumulator[i * 4 + 1]);
        *reinterpret_cast<uint32_t*>(out + row0 * shape_n + col) = packed;
      }
      if (row1 < shape_m) {
        uint32_t packed;
        bfloat16_t* values = reinterpret_cast<bfloat16_t*>(&packed);
        values[0] = static_cast<bfloat16_t>(accumulator[i * 4 + 2]);
        values[1] = static_cast<bfloat16_t>(accumulator[i * 4 + 3]);
        *reinterpret_cast<uint32_t*>(out + row1 * shape_n + col) = packed;
      }
    }
  }
}

// The u32 conversion is opaque to NVCC, which keeps the word in a register.
__device__ __forceinline__ uint32_t fp8_smem_u32(void const* ptr) {
  uint32_t addr;
  asm("{\n\t.reg .u64 t;\n\tcvta.to.shared.u64 t, %1;\n\tcvt.u32.u64 %0, t;\n\t}"
      : "=r"(addr) : "l"(ptr));
  return addr;
}

__device__ __forceinline__ uint32_t fp8_smem_mapa(uint32_t addr, uint32_t cta) {
  uint32_t out;
  asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(cta));
  return out;
}

// ``addr`` must already be a cluster-window address (``fp8_smem_mapa``).
__device__ __forceinline__ void fp8_mbar_arrive_cluster(uint32_t addr) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
               : : "r"(addr) : "memory");
}

__device__ __forceinline__ float fp8_lds_f32(uint32_t addr) {
  float out;
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(out) : "r"(addr) : "memory");
  return out;
}

// Each list is what the builder can name; a width no configuration selects is absent.

#define TL_DEFINE_FP8_GEMM_1D2D_HELPERS(N)                                      \
__device__ __forceinline__ void fp8_gemm_wgmma_64x128_by_128x##N##_lo(           \
    float* acc, uint32_t a_lo, uint32_t b_lo) {                                  \
  fp8_gemm_wgmma_64x128_by_128xN_lo<N>(acc, a_lo, b_lo);                         \
}                                                                                \
__device__ __forceinline__ void fp8_gemm_raw_acc_stsm_bf16_swizzled_bm64_64x##N( \
    float* acc, bfloat16_t* out, int m_offset) {                                  \
  fp8_gemm_raw_acc_stsm_bf16_swizzled<64, N>(acc, out, m_offset);                 \
}                                                                                 \
__device__ __forceinline__ void fp8_gemm_raw_acc_stsm_bf16_swizzled_bm128_64x##N(\
    float* acc, bfloat16_t* out, int m_offset) {                                  \
  fp8_gemm_raw_acc_stsm_bf16_swizzled<128, N>(acc, out, m_offset);                \
}                                                                                 \
__device__ __forceinline__ void fp8_gemm_raw_acc_store_global_64x##N##_v2(       \
    float* acc, bfloat16_t const* out, int ms, int ns, int m, int n) {           \
  fp8_gemm_raw_acc_store_global_vec2<N>(acc, out, ms, ns, m, n);                 \
}

#define TL_DEFINE_FP8_GEMM_1D2D_BM256_HELPER(N)                                  \
__device__ __forceinline__ void fp8_gemm_raw_acc_stsm_bf16_swizzled_bm256_64x##N(\
    float* acc, bfloat16_t* out, int m_offset) {                                  \
  fp8_gemm_raw_acc_stsm_bf16_swizzled<256, N>(acc, out, m_offset);                \
}

#define TL_DEFINE_FP8_GEMM_1D2D_UNIFORM_SFB_HELPER(N)                            \
__device__ __forceinline__ void fp8_gemm_1d2d_promote_64x##N(                    \
    float* p, float* f, float sa0, float sa1, float sb) {                         \
  fp8_gemm_1d2d_promote<N>(p, f, sa0, sa1, sb);                                  \
}

#define TL_DEFINE_FP8_GEMM_1D2D_SPLIT_SFB_HELPER(N)                              \
__device__ __forceinline__ void fp8_gemm_1d2d_promote_two_b_scales_64x##N(       \
    float* p, float* f, float sa0, float sa1, float sb0, float sb1, int split) {  \
  fp8_gemm_1d2d_promote_two_b_scales<N>(p, f, sa0, sa1, sb0, sb1, split);        \
}

#define TL_DEFINE_FP8_GEMM_1D2D_SPLIT_HELPER(N, SPLIT)                           \
__device__ __forceinline__ void                                                   \
fp8_gemm_1d2d_promote_two_b_scales_64x##N##_split##SPLIT(                        \
    float* p, float* f, float sa0, float sa1, float sb0, float sb1) {             \
  fp8_gemm_1d2d_promote_two_b_scales_split<N, SPLIT>(p, f, sa0, sa1, sb0, sb1);  \
}

TL_DEFINE_FP8_GEMM_1D2D_HELPERS(16)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(32)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(48)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(64)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(80)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(96)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(112)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(128)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(144)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(160)
TL_DEFINE_FP8_GEMM_1D2D_HELPERS(192)

TL_DEFINE_FP8_GEMM_1D2D_BM256_HELPER(16)
TL_DEFINE_FP8_GEMM_1D2D_BM256_HELPER(32)
TL_DEFINE_FP8_GEMM_1D2D_BM256_HELPER(48)
TL_DEFINE_FP8_GEMM_1D2D_BM256_HELPER(64)
TL_DEFINE_FP8_GEMM_1D2D_BM256_HELPER(80)
TL_DEFINE_FP8_GEMM_1D2D_BM256_HELPER(96)
TL_DEFINE_FP8_GEMM_1D2D_BM256_HELPER(112)
TL_DEFINE_FP8_GEMM_1D2D_BM256_HELPER(128)

TL_DEFINE_FP8_GEMM_1D2D_UNIFORM_SFB_HELPER(16)
TL_DEFINE_FP8_GEMM_1D2D_UNIFORM_SFB_HELPER(32)
TL_DEFINE_FP8_GEMM_1D2D_UNIFORM_SFB_HELPER(64)
TL_DEFINE_FP8_GEMM_1D2D_UNIFORM_SFB_HELPER(128)

TL_DEFINE_FP8_GEMM_1D2D_SPLIT_SFB_HELPER(48)
TL_DEFINE_FP8_GEMM_1D2D_SPLIT_SFB_HELPER(80)
TL_DEFINE_FP8_GEMM_1D2D_SPLIT_SFB_HELPER(96)
TL_DEFINE_FP8_GEMM_1D2D_SPLIT_SFB_HELPER(112)
TL_DEFINE_FP8_GEMM_1D2D_SPLIT_SFB_HELPER(144)
TL_DEFINE_FP8_GEMM_1D2D_SPLIT_SFB_HELPER(160)
TL_DEFINE_FP8_GEMM_1D2D_SPLIT_HELPER(192, 8)
TL_DEFINE_FP8_GEMM_1D2D_SPLIT_HELPER(192, 16)

#undef TL_DEFINE_FP8_GEMM_1D2D_HELPERS
#undef TL_DEFINE_FP8_GEMM_1D2D_BM256_HELPER
#undef TL_DEFINE_FP8_GEMM_1D2D_UNIFORM_SFB_HELPER
#undef TL_DEFINE_FP8_GEMM_1D2D_SPLIT_SFB_HELPER
#undef TL_DEFINE_FP8_GEMM_1D2D_SPLIT_HELPER

}  // namespace tl
