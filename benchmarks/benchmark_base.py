import logging
import subprocess
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import pytest
import torch
from torch.autograd.profiler import DeviceType

from tileops.manifest import load_workloads

# Workload dict keys reserved by the benchmark harness. Everything else on
# a workload entry (e.g. ``dim``, ``keepdim``, ``correction``) is treated
# as an op-call parameter.
#
# The current harness is explicitly scoped to **single-input ops whose
# sole tensor input is named ``x``**. Multi-input ops (e.g. attention
# families that declare ``q_shape`` / ``kv_shape``) are not supported:
# :func:`workloads_to_params` will raise ``KeyError`` if ``x_shape`` is
# absent. Extending to signature-aware tensor binding is tracked as a
# follow-up and must also update ``docs/design/manifest.md``.
_WORKLOAD_META_KEYS: frozenset[str] = frozenset(
    {"x_shape", "dtypes", "label"}
)

# ---------------------------------------------------------------------------
# Benchmark capability protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ShapeDtypeWorkload(Protocol):
    """Structural type for workloads that carry shape and dtype metadata.

    Any object with ``shape`` and ``dtype`` satisfies this protocol.
    Used by helpers that only need tensor metadata, not input generation
    capability.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype


@runtime_checkable
class InputGeneratingWorkload(Protocol):
    """Structural type for workloads that can generate benchmark inputs."""

    def gen_inputs(self) -> tuple[Any, ...]: ...


@runtime_checkable
class BenchmarkWorkload(ShapeDtypeWorkload, InputGeneratingWorkload, Protocol):
    """Full benchmark workload: shape/dtype metadata + input generation.

    This is the standard contract for benchmark workloads that need both
    roofline metadata extraction and input tensor generation.
    Workloads satisfy this protocol when they define ``shape`` and ``dtype``
    metadata in addition to implementing ``gen_inputs()``.
    """

    ...


# Backward-compatible alias
RooflineWorkload = ShapeDtypeWorkload

W = TypeVar("W")


_logger = logging.getLogger("tileops.bench")

# Thread-local storage for conftest hook to pick up per-test bench results.
# A single test function may call record() multiple times (tileops + baseline).
_bench_results = threading.local()


@dataclass
class _PendingMeasurement:
    fn: Callable
    args: tuple[Any, ...]
    n_warmup: int
    n_repeat: int
    n_trials: int
    result: dict
    build_result: Callable[[float], dict]
    enable_grad: bool = False
    fallback_latency_ms: Optional[float] = None


class _CaseProfilerSession:
    """Execute measurements immediately inside one case-wide Kineto session."""

    def __init__(self) -> None:
        self.measurements: list[_PendingMeasurement] = []
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()
        self.canary = torch.zeros(1024, device="cuda")
        self.canary.add_(1)
        torch.cuda.synchronize()
        self._run_canary(_CANARY_START_REGION)

    def add(self, measurement: _PendingMeasurement) -> dict:
        """Run one request now; its CUPTI result is parsed when the case ends."""
        measurement_index = len(self.measurements)
        self.measurements.append(measurement)
        try:
            run, arg_pool = _prepare_measurement_runner(measurement)
            cache = _get_l2_flush_cache()
            event_trial_means = []
            for trial_index in range(measurement.n_trials):
                region = f"{_KERNEL_REGION}:{measurement_index}:{trial_index}"
                starts = [
                    torch.cuda.Event(enable_timing=True)
                    for _ in range(measurement.n_repeat)
                ]
                ends = [
                    torch.cuda.Event(enable_timing=True)
                    for _ in range(measurement.n_repeat)
                ]
                for repeat_index in range(measurement.n_repeat):
                    cache.zero_()
                    torch.cuda.synchronize()
                    starts[repeat_index].record()
                    with torch.profiler.record_function(region):
                        run(repeat_index)
                        torch.cuda.synchronize()
                    ends[repeat_index].record()
                torch.cuda.synchronize()
                times = [
                    start.elapsed_time(end)
                    for start, end in zip(starts, ends, strict=True)
                ]
                event_trial_means.append(sum(times) / len(times))
            if arg_pool is not None:
                del arg_pool
        except BaseException:
            # The synchronous profile contract requires the caller to observe
            # setup/kernel exceptions at this call site.  Do not leave a
            # half-recorded request for teardown to process and mask that
            # original exception (notably pytest.skip for unsupported shapes).
            self.measurements.pop()
            raise
        event_trial_means.sort()
        measurement.fallback_latency_ms = event_trial_means[len(event_trial_means) // 2]
        return measurement.result

    def _run_canary(self, region: str) -> None:
        for _ in range(_CANARY_REPEATS):
            with torch.profiler.record_function(region):
                self.canary.add_(1)
                torch.cuda.synchronize()

    def finish(self) -> dict[str, Any]:
        """Close the trace, validate canaries, and fill registered results."""
        self._run_canary(_CANARY_END_REGION)
        profiler_error = None
        try:
            self.profiler.__exit__(None, None, None)
            kineto_results = self.profiler.profiler.kineto_results
        except RuntimeError as error:
            kineto_results = None
            profiler_error = f"{type(error).__name__}: {error}"
        return _parse_case_measurements(
            self.measurements, kineto_results, profiler_error
        )


def _begin_case_profiler_session() -> None:
    if getattr(_bench_results, "case_profiler", None) is not None:
        raise RuntimeError("benchmark case profiler session is already active")
    _bench_results.case_profiler = _CaseProfilerSession()


def _active_case_profiler_session() -> Optional[_CaseProfilerSession]:
    return getattr(_bench_results, "case_profiler", None)


def _finish_case_profiler_session() -> Optional[dict[str, Any]]:
    session = _active_case_profiler_session()
    if session is None:
        return None
    try:
        return session.finish()
    finally:
        _bench_results.case_profiler = None


# Name of the CPU ``record_function`` scope wrapping the timed call.  The L2
# flush is synchronized before this scope opens, and the measured work is
# synchronized before it closes.  Consequently its CPU timestamps form a
# reliable device-work window even when Kineto's projected GPU annotation is
# incomplete (notably for kernels launched by autograd worker threads).
_KERNEL_REGION = "tileops_bench_kernel"
_CANARY_START_REGION = "tileops_bench_canary_start"
_CANARY_END_REGION = "tileops_bench_canary_end"
_CANARY_REPEATS = 3


def _sum_kernel_time_us(kineto_results):
    """Sum device time of the kernels the timed call launched.

    Sums only kernels inside a :data:`_KERNEL_REGION` annotation window, so the
    L2-flush fill is excluded and the kernel under test is counted regardless of
    its name. A call launching several kernels contributes all of them.

    Iterates the C++ Kineto events directly to bypass ``key_averages()``, which
    is ~16x slower (~130ms of Python parsing/tree-building) for large traces.

    Returns:
        ``(total_us, n_regions)``: summed kernel time in microseconds and the
        number of CPU annotation windows. The caller checks ``n_regions ==
        n_repeat`` to confirm every timed iteration was recorded.
    """
    total_us, count = _region_kernel_times_us(
        kineto_results, {_KERNEL_REGION}
    )[_KERNEL_REGION]
    return total_us, count


# ---------------------------------------------------------------------------
# L2 cache flush buffer (sized to actual L2, allocated lazily)
# ---------------------------------------------------------------------------

_l2_flush_cache: Optional[torch.Tensor] = None


def _get_l2_flush_cache() -> torch.Tensor:
    global _l2_flush_cache
    if _l2_flush_cache is None:
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
        if l2_bytes <= 0:
            l2_bytes = int(256e6)  # fallback
        _l2_flush_cache = torch.empty(l2_bytes // 4, dtype=torch.int, device="cuda")
    return _l2_flush_cache


# ---------------------------------------------------------------------------
# NVIDIA SOL-ExecBench–style benchmark
# ---------------------------------------------------------------------------

def bench_kernel(
    fn: Callable,
    args: tuple[Any, ...] = (),
    n_warmup: int = 10,
    n_repeat: int = 50,
    n_trials: int = 3,
) -> float:
    """Benchmark a GPU kernel with pure kernel timing via CUPTI.

    Protocol (adapted from NVIDIA SOL-ExecBench, arxiv.org/abs/2603.19173):
      1. Lock GPU clocks externally (nvidia-smi).
      2. Run *n_warmup* un-timed iterations with L2 flush.
      3. For each of *n_trials* trials, profile *n_repeat* iterations
         under CUPTI to get pure kernel execution time (no launch overhead).
         L2 is flushed before every iteration.  Input tensors are cloned
         each iteration so the kernel always sees fresh addresses.
      4. Report the median trial mean (robust to outlier trials).

    Uses CUPTI via torch.profiler for accurate kernel-only timing, with
    direct Kineto C++ event iteration to avoid Python parsing overhead.
    Falls back to CUDA events if CUPTI is unavailable.

    Args:
        fn: Callable to benchmark.  If *args* is provided, called as
            ``fn(*cloned_args)``; otherwise called as ``fn()``.
        args: Tensor arguments to clone each iteration.  Non-tensor
            values are passed through unchanged.
        n_warmup: Warmup iterations (default 10).
        n_repeat: Timed iterations per trial (default 50).
        n_trials: Independent trials (default 3).

    Returns:
        Kernel latency in **milliseconds**.
    """
    if not isinstance(args, tuple):
        raise TypeError(
            f"bench_kernel expects a tuple of args, got {type(args).__name__}. "
            "Check that gen_inputs() returns a tuple."
        )

    from tilelang.profiler.bench import suppress_stdout_stderr

    cache = _get_l2_flush_cache()
    has_args = len(args) > 0

    # Pre-clone a small pool of input tensors so the kernel sees different
    # addresses across iterations.  Skip cloning if total tensor memory
    # exceeds 1 GB to avoid OOM on large workloads.
    _N_CLONES = 3
    _MAX_CLONE_BYTES = 1 << 30  # 1 GB
    if has_args:
        tensor_mask = tuple(isinstance(a, torch.Tensor) for a in args)
        total_bytes = sum(a.nelement() * a.element_size()
                          for a, m in zip(args, tensor_mask, strict=True) if m)
        if total_bytes * _N_CLONES <= _MAX_CLONE_BYTES:
            arg_pool = [
                tuple(a.clone() if m else a for a, m in zip(args, tensor_mask, strict=True))
                for _ in range(_N_CLONES)
            ]
            def _run(i):
                return fn(*arg_pool[i % _N_CLONES])
        else:
            arg_pool = None
            def _run(i):
                return fn(*args)
    else:
        arg_pool = None
        def _run(i):
            return fn()

    # Warmup (no profiling)
    for i in range(n_warmup):
        cache.zero_()
        _run(i % n_repeat)
    torch.cuda.synchronize()

    # Timed trials with CUPTI.  Each trial opens its own torch.profiler context
    # around exactly n_repeat iterations and reads the trace after the context
    # closes; summed device kernel time / n_repeat is the mean per-call kernel
    # time.  We deliberately do NOT use torch.profiler.schedule: that mechanism
    # is for sampling a window out of a long step()-driven loop, and forcing
    # n_repeat calls into a single "step" let queued, un-synchronized launches
    # leak across the warmup/active boundary.  A plain per-trial context records
    # exactly the calls we want — no schedule, no on_trace_ready callback.
    #
    # Only the timed call is wrapped in record_function(_KERNEL_REGION).  The
    # parser attributes device events purely by timestamp interval, and Kineto's
    # projection of the annotation window onto the device timeline is not
    # guaranteed to exclude a flush merely enqueued before the window (issue
    # #1723: under cold TileLang cache + autotune the flush event was observed
    # inside the window, adding one flush duration per measured repeat).  We
    # therefore synchronize after cache.zero_() so the flush completes before
    # the window opens; L2 is still cold for the measured call, and the extra
    # sync only adds host-side latency, never device time.  The post-call sync
    # keeps the measured kernels recorded before the next flush.
    trial_means: list[float] = []
    try:
        with suppress_stdout_stderr():
            for _ in range(n_trials):
                with torch.profiler.profile(
                    # CPU activity is required for Kineto to project the
                    # annotation onto the device timeline (CUDA-only emits no
                    # window); it adds only host-side overhead, not kernel time.
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                ) as profiler:
                    for i in range(n_repeat):
                        cache.zero_()
                        # Drain the flush so its device event ends before the
                        # timed window opens; without this, projection quirks
                        # can place it inside the window (see comment above).
                        torch.cuda.synchronize()
                        with torch.profiler.record_function(_KERNEL_REGION):
                            _run(i)
                            # Keep the CPU annotation open until all measured
                            # device work completes.  GPU annotation projection
                            # is incomplete for autograd worker-thread launches.
                            torch.cuda.synchronize()
                total_us, n_regions = _sum_kernel_time_us(profiler.profiler.kineto_results)
                # Scope failed to project on some iteration → trace untrustworthy,
                # fall back to CUDA events.
                if n_regions != n_repeat:
                    raise RuntimeError
                trial_means.append((total_us / n_repeat) * 1e-3)
    except RuntimeError:
        trial_means = []

    # Fallback to CUDA events if CUPTI failed
    if not trial_means:
        for _ in range(n_trials):
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
            for i in range(n_repeat):
                cache.zero_()
                start_events[i].record()
                _run(i)
                end_events[i].record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events, strict=True)]
            trial_means.append(sum(times) / len(times))

    # Free the arg pool and release cached GPU memory to prevent
    # accumulation across hundreds of benchmark calls.
    if arg_pool is not None:
        del arg_pool
    torch.cuda.empty_cache()

    trial_means.sort()
    return trial_means[len(trial_means) // 2]


def _prepare_measurement_runner(measurement: _PendingMeasurement):
    """Warm a deferred measurement and return its indexed runner and arg pool."""
    cache = _get_l2_flush_cache()
    args = measurement.args
    has_args = bool(args)
    arg_pool = None
    if has_args:
        tensor_mask = tuple(isinstance(a, torch.Tensor) for a in args)
        total_bytes = sum(
            a.nelement() * a.element_size()
            for a, is_tensor in zip(args, tensor_mask, strict=True)
            if is_tensor
        )
        if total_bytes * 3 <= 1 << 30:
            arg_pool = [
                tuple(
                    a.clone() if is_tensor else a
                    for a, is_tensor in zip(args, tensor_mask, strict=True)
                )
                for _ in range(3)
            ]

    def run(index: int):
        call_args = arg_pool[index % len(arg_pool)] if arg_pool else args
        grad_context = torch.enable_grad() if measurement.enable_grad else torch.no_grad()
        with grad_context:
            return measurement.fn(*call_args)

    for i in range(measurement.n_warmup):
        cache.zero_()
        run(i)
    torch.cuda.synchronize()
    return run, arg_pool


def _region_kernel_times_us(kineto_results, region_names: set[str]):
    """Return CUDA kernel time inside each named CPU annotation window.

    GPU user annotations are projections made by Kineto.  They can cover only
    the launches performed by the thread that opened ``record_function``; an
    autograd worker may therefore launch valid kernels outside that projected
    GPU range.  CPU annotations do not have that limitation.  Callers keep the
    scope open through ``torch.cuda.synchronize()``, making its timestamps a
    complete boundary for work launched on every CUDA stream/thread.
    """
    stats = _region_kernel_stats(kineto_results, region_names)
    return {name: (total_us, windows) for name, (total_us, windows, _) in stats.items()}


def _region_kernel_stats(kineto_results, region_names: set[str]):
    """Return kernel time, CPU-window count, and kernel count by region."""
    import bisect

    windows: dict[str, list[tuple[int, int]]] = {name: [] for name in region_names}
    kernels: list[tuple[int, int]] = []
    for event in kineto_results.events():
        if event.device_type() == DeviceType.CPU and event.is_user_annotation():
            if event.name() in windows:
                windows[event.name()].append((event.start_ns(), event.end_ns()))
            continue
        if event.device_type() == DeviceType.CUDA and not event.is_user_annotation():
            kernels.append((event.start_ns(), event.duration_ns()))

    output = {}
    for name, named_windows in windows.items():
        named_windows.sort()
        starts = [window[0] for window in named_windows]
        ends = [window[1] for window in named_windows]
        total_us = 0.0
        kernel_count = 0
        for start_ns, duration_ns in kernels:
            index = bisect.bisect_right(starts, start_ns) - 1
            if index >= 0 and start_ns < ends[index]:
                total_us += duration_ns / 1000.0
                kernel_count += 1
        output[name] = (total_us, len(named_windows), kernel_count)
    return output


def _gpu_annotation_counts(kineto_results, region_names: set[str]) -> dict[str, int]:
    counts = {name: 0 for name in region_names}
    for event in kineto_results.events():
        if (
            event.device_type() == DeviceType.CUDA
            and event.is_user_annotation()
            and event.name() in counts
        ):
            counts[event.name()] += 1
    return counts


def _parse_case_measurements(
    measurements: list[_PendingMeasurement],
    kineto_results: Any,
    profiler_error: Optional[str],
) -> dict[str, Any]:
    """Parse a closed case trace and fill results without rerunning kernels."""
    region_names = {
        f"{_KERNEL_REGION}:{measurement_index}:{trial_index}"
        for measurement_index, measurement in enumerate(measurements)
        for trial_index in range(measurement.n_trials)
    }
    canary_names = {_CANARY_START_REGION, _CANARY_END_REGION}
    all_region_names = region_names | canary_names
    region_results = (
        _region_kernel_stats(kineto_results, all_region_names)
        if kineto_results is not None
        else None
    )
    gpu_annotation_counts = (
        _gpu_annotation_counts(kineto_results, canary_names)
        if kineto_results is not None
        else {name: 0 for name in canary_names}
    )

    health: dict[str, Any] = {
        "healthy": region_results is not None,
        "reason": "ok" if region_results is not None else "kineto_runtime_error",
        "expected": _CANARY_REPEATS,
    }
    for position, name in (
        ("start", _CANARY_START_REGION),
        ("end", _CANARY_END_REGION),
    ):
        _, cpu_count, kernel_count = (
            region_results[name] if region_results is not None else (0.0, 0, 0)
        )
        gpu_count = gpu_annotation_counts[name]
        health[f"{position}_cpu_count"] = cpu_count
        health[f"{position}_kernel_count"] = kernel_count
        health[f"{position}_gpu_annotation_count"] = gpu_count
        for kind, count in (
            ("cpu", cpu_count),
            ("kernel", kernel_count),
            ("gpu_annotation", gpu_count),
        ):
            if health["healthy"] and count != _CANARY_REPEATS:
                health["healthy"] = False
                health["reason"] = f"{position}_{kind}_canary_mismatch"

    for measurement_index, measurement in enumerate(measurements):
        trial_means = []
        region_counts: list[int] = []
        fallback_reason = None
        if region_results is not None:
            region_counts_match = True
            for trial_index in range(measurement.n_trials):
                region = f"{_KERNEL_REGION}:{measurement_index}:{trial_index}"
                total_us, count, _ = region_results[region]
                region_counts.append(count)
                if count != measurement.n_repeat:
                    region_counts_match = False
                    if count < measurement.n_repeat:
                        fallback_reason = "missing_cpu_annotations"
                    else:
                        fallback_reason = "duplicate_cpu_annotations"
                else:
                    trial_means.append(total_us / measurement.n_repeat * 1e-3)
            if not region_counts_match:
                trial_means = []
        else:
            fallback_reason = "kineto_runtime_error"
        if trial_means:
            trial_means.sort()
            latency = trial_means[len(trial_means) // 2]
            timing_source = "cupti"
        else:
            latency = measurement.fallback_latency_ms
            if latency is None:
                raise RuntimeError("measurement completed without CUDA event timing")
            timing_source = "cuda_event"
        built_result = measurement.build_result(latency)
        built_result["timing_source"] = timing_source
        built_result["expected_region_count"] = measurement.n_repeat
        built_result["observed_region_counts"] = "/".join(map(str, region_counts))
        if fallback_reason is not None:
            built_result["fallback_reason"] = fallback_reason
        if profiler_error is not None:
            built_result["kineto_error"] = profiler_error
        measurement.result.update(built_result)

    torch.cuda.empty_cache()
    return health


def _get_env_metadata() -> list[str]:
    """Collect GPU model, driver version, CUDA version, and torch version."""
    lines = []
    lines.append(f"- **Torch version**: {torch.__version__}")
    lines.append(f"- **CUDA version (torch)**: {torch.version.cuda or 'N/A'}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        lines.append(f"- **GPU model**: {gpu_name}")
    else:
        lines.append("- **GPU model**: N/A (no CUDA device)")

    # Try to get NVIDIA driver version and clocks from nvidia-smi.
    gpu_query_fields = [
        "driver_version",
        "clocks.current.sm",
        "clocks.current.memory",
        "clocks.applications.graphics",
        "clocks.applications.memory",
    ]
    gpu_query_values = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(gpu_query_fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpu_query_values = [
                part.strip() for part in result.stdout.splitlines()[0].split(",")
            ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    driver = gpu_query_values[0] if len(gpu_query_values) == len(gpu_query_fields) else "N/A"
    lines.append(f"- **Driver version**: {driver}")

    if len(gpu_query_values) == len(gpu_query_fields):
        sm_clock, mem_clock, app_sm_clock, app_mem_clock = gpu_query_values[1:]
        lines.append(
            "- **GPU clocks**: "
            f"SM current {sm_clock} MHz, memory current {mem_clock} MHz, "
            f"application SM {app_sm_clock} MHz, "
            f"application memory {app_mem_clock} MHz"
        )

    return lines


class BenchmarkBase(Generic[W], ABC):
    """Abstract base class for op benchmarking.

    Generic over workload type so subclasses can declare the exact
    capability they need.  ``WorkloadBase`` remains the typical in-repo
    implementation, but the public contract is the type parameter.

    Subclass must implement calculate_flops() and calculate_memory().
    """

    def __init__(self, workload: W):
        self.workload = workload

    @abstractmethod
    def calculate_flops(self) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def calculate_memory(self) -> Optional[float]:
        raise NotImplementedError

    def profile(self,
                functor: Any,
                *inputs: Any,
                n_warmup: int = 10,
                n_repeat: int = 50,
                n_trials: int = 3) -> dict:
        """Profile a callable and return structured results.

        Uses the NVIDIA SOL-ExecBench protocol: CUPTI kernel timing,
        10 warmup, 50 repeats × 3 trials, L2 flush sized to actual
        cache, input tensors cloned each iteration.
        """
        session = _active_case_profiler_session()
        if session is not None:
            return session.add(_PendingMeasurement(
                fn=functor,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
                result={},
                build_result=self._build_result,
            ))
        with torch.no_grad():
            latency = bench_kernel(
                functor,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
        return self._build_result(latency)

    def profile_autograd(self, functor: Any) -> dict:
        """Profile a callable that requires autograd (e.g. fwd+bwd).

        Same as profile() but without torch.no_grad(), so the callable
        can build autograd graphs and call .backward() internally.
        The functor must be a zero-arg closure that captures its inputs.
        """
        session = _active_case_profiler_session()
        if session is not None:
            return session.add(_PendingMeasurement(
                fn=functor,
                args=(),
                n_warmup=10,
                n_repeat=50,
                n_trials=3,
                result={},
                build_result=self._build_result,
                enable_grad=True,
            ))
        latency = bench_kernel(functor)
        return self._build_result(latency)

    def _build_result(self, latency: float) -> dict:
        result = {"latency_ms": latency}
        flops = self.calculate_flops()
        if flops is not None:
            result["tflops"] = flops / latency * 1e-9
        memory = self.calculate_memory()
        if memory is not None:
            result["bandwidth_tbs"] = memory / latency * 1e-9
        return result


# ---------------------------------------------------------------------------
# Manifest-driven benchmark helpers
# ---------------------------------------------------------------------------


def _workload_extra_params(w: dict) -> dict[str, Any]:
    """Return op-specific params attached to a manifest workload entry.

    A workload entry may carry optional op-call parameter values beyond
    ``x_shape`` / ``dtypes`` / ``label`` (e.g. ``dim``, ``keepdim``,
    ``correction``). These are forwarded to the op constructor by benchmark
    files that opt into ``include_extra=True``.

    Only the reserved meta keys (``x_shape``, ``dtypes``, ``label``) and
    dunder-style metadata keys are stripped; everything else — including
    any other ``*_shape`` keys — is surfaced as an op param. This matches
    the single-input ``x_shape``-only harness contract documented in
    :data:`_WORKLOAD_META_KEYS`; multi-input ops with ``q_shape`` /
    ``kv_shape`` are out of scope and would need a dedicated harness.
    """
    return {
        k: v
        for k, v in w.items()
        if k not in _WORKLOAD_META_KEYS and not k.startswith("__")
    }


def workloads_to_params(op_name: str, include_extra: bool = False) -> list:
    """Convert manifest workload dicts for *op_name* to pytest params.

    By default (``include_extra=False``) each entry becomes
    ``pytest.param(shape, dtype, id=...)`` — compatible with existing bench
    files that use ``@pytest.mark.parametrize("shape, dtype", ...)``.

    With ``include_extra=True`` each entry becomes
    ``pytest.param(shape, dtype, extra_params, id=...)`` where
    ``extra_params`` is a dict of op-call params declared on the workload
    entry (e.g. ``{"dim": 0, "keepdim": False}``). Use this when the
    benchmark needs to drive op calls from manifest-declared workload params.
    """
    workloads = load_workloads(op_name)
    params = []
    for w in workloads:
        if "x_shape" not in w:
            raise KeyError(
                f"workloads_to_params({op_name!r}) only supports single-input "
                "ops whose tensor input is named 'x' (workload must declare "
                "'x_shape'); multi-input ops with q_shape/kv_shape/... are "
                "out of scope for this harness."
            )
        shape = tuple(w["x_shape"])
        label = w.get("label", "x".join(str(s) for s in shape))
        extra = _workload_extra_params(w) if include_extra else {}
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            # Copy ``extra`` per parametrization so accidental mutation in
            # one test case cannot leak into later parametrized cases that
            # share the same workload entry.
            param_args = (
                (shape, dtype, dict(extra))
                if include_extra
                else (shape, dtype)
            )
            params.append(pytest.param(*param_args, id=f"{label}-{dtype_str}"))
    return params


class ManifestBenchmark(BenchmarkBase[ShapeDtypeWorkload]):
    """Generic benchmark that reads FLOP/memory counts from an Op instance.

    Accepts an op name, an instantiated Op, and any workload satisfying
    :class:`ShapeDtypeWorkload`.  The op must implement ``eval_roofline()``.
    Dynamic-shape ops may bind roofline variables during ``forward()``, so
    this helper calls ``op.eval_roofline()`` only while building a result
    after profiling has executed the op.

    Usage::

        op = SumFwdOp(dtype=dtype, dim=0)
        bm = ManifestBenchmark("SumFwdOp", op, workload)
        result = bm.profile(op, *inputs)
    """

    def __init__(
        self,
        op_name: str,
        op: Any,
        workload: ShapeDtypeWorkload,
    ):
        super().__init__(workload)
        self._op_name = op_name
        self._op = op
        self._roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            flops, mem_bytes = self._op.eval_roofline()
            self._roofline_cache = (float(flops), float(mem_bytes))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


def _extract_op_config(op: object) -> Optional[dict]:
    """Return the kernel config for an Op instance, or None if unavailable.

    Handles the three Op patterns currently used in tileops:

      1. **Eager-init** (e.g. ``GemmOp``): ``op.kernel`` is a Kernel
         instance set in ``__init__``.
      2. **Lazy with dummy kernel** (e.g. ``FFTC2COp``): ``op.kernel`` is a
         default Kernel and ``op._kernel_cache`` may hold others.
      3. **Pure lazy cache** (e.g. ``_SoftmaxBaseOp`` and the spec-conformant
         reduction ops): ``op._kernel_cache`` is the only source; ``op.kernel``
         is unset.

    A direct ``op.config`` attribute (legacy / explicit override) takes
    precedence over kernel introspection.
    """
    op_config = getattr(op, "config", None)
    if op_config:
        return op_config

    kernel = getattr(op, "kernel", None)
    op_config = getattr(kernel, "config", None) if kernel is not None else None
    if op_config:
        return op_config

    # Pure lazy-cache pattern: pick any cached kernel's config. All cached
    # kernels for a given op share dtype/op_kind, so taking the first is
    # sufficient for the benchmark report (which records one entry per call).
    cache = getattr(op, "_kernel_cache", None)
    if cache:
        try:
            first_kernel = next(iter(cache.values()))
        except StopIteration:
            first_kernel = None
        if first_kernel is not None:
            op_config = getattr(first_kernel, "config", None)
            if op_config:
                return op_config

    return None


class BenchmarkReport:
    """Collects benchmark results and dumps a markdown report.

    All methods are static — use as BenchmarkReport.record(...).
    Call clear() at session start, dump() at session end.
    """
    _records: dict = {}

    @staticmethod
    def record(op_or_name, params: dict, result: dict, tag: str = "tileops") -> None:
        """Record a benchmark result.

        Args:
            op_or_name: Op instance or benchmark group name string.
                If an Op instance, class name and module are extracted automatically.
            params: Parameter dict (typically from locals())
            result: Dict with latency_ms, tflops, bandwidth_tbs
            tag: Label to distinguish implementations (e.g. "tileops", "FA3", "fla")
        """
        if isinstance(op_or_name, str):
            name = op_or_name
            op_module = None
            op_config = None
        else:
            name = op_or_name.__class__.__name__
            op_module = op_or_name.__class__.__module__
            op_config = _extract_op_config(op_or_name)

        # Filter params to only include serializable benchmark parameters.
        # Tuples of primitives (e.g. ``shape=(4096, 4096)``) are preserved
        # verbatim so the profile log carries the original input geometry
        # rather than a flattened element count.
        def _is_serializable(v: Any) -> bool:
            if isinstance(v, (int, float, bool, str, torch.dtype)):
                return True
            if isinstance(v, tuple):
                return all(_is_serializable(x) for x in v)
            return False

        filtered_params = {
            k: v for k, v in params.items()
            if k not in ("test", "bm", "op", "inputs", "result", "result_bl",
                         "baseline_fn", "tune")
            and not k.startswith("_")
            and _is_serializable(v)
        }
        record_entry = {
            "params": filtered_params,
            "result": result,
            "tag": tag,
        }
        if op_config:
            record_entry["config"] = op_config
        BenchmarkReport._records.setdefault(name, []).append(record_entry)

        # Accumulate in thread-local for conftest hook.
        if not hasattr(_bench_results, "entries"):
            _bench_results.entries = []
        # Keep the result mapping by reference: case-scoped profiling fills it
        # after the test body has registered every implementation.
        entry = {"tag": tag, "op": name, "result": result}
        if op_module:
            entry["op_module"] = op_module
        _bench_results.entries.append(entry)

        if result:
            _logger.info("op=%s module=%s tag=%s latency_ms=%.4f tflops=%.2f",
                         name, op_module or "N/A", tag,
                         result.get("latency_ms", 0),
                         result.get("tflops", 0))

    @staticmethod
    def dump(path: str) -> None:
        """Write all collected results to a markdown-formatted log file."""
        if not BenchmarkReport._records:
            return

        lines = [
            "# TileOPs Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Environment",
            "",
        ]
        lines.extend(_get_env_metadata())
        lines.append("")

        default_result_keys = ["latency_ms", "tflops", "bandwidth_tbs"]

        for name, entries in BenchmarkReport._records.items():
            if not entries:
                continue

            lines.append(f"## {name}")
            lines.append("")

            # Group by tag
            tag_entries = {}
            for entry in entries:
                tag_entries.setdefault(entry["tag"], []).append(entry)
            result_keys = list(default_result_keys)
            for entry in entries:
                for key in entry["result"]:
                    if key not in result_keys:
                        result_keys.append(key)

            for tag, tag_group in tag_entries.items():
                lines.append(f"### {tag}")
                lines.append("")

                param_keys = list(tag_group[0]["params"].keys())
                has_config = any("config" in e for e in tag_group)
                header_parts = param_keys + result_keys
                if has_config:
                    header_parts.append("config")
                lines.append("| " + " | ".join(header_parts) + " |")
                lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")

                for entry in tag_group:
                    row = [str(entry["params"].get(k, "")) for k in param_keys]
                    for rk in result_keys:
                        val = entry["result"].get(rk)
                        if isinstance(val, (int, float)):
                            row.append(f"{val:.4f}")
                        else:
                            row.append(str(val) if val is not None else "N/A")
                    if has_config:
                        cfg = entry.get("config")
                        row.append(str(cfg) if cfg else "")
                    lines.append("| " + " | ".join(row) + " |")

                lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        print(f"Benchmark report saved to {path}")

    @staticmethod
    def clear() -> None:
        """Clear all collected records."""
        BenchmarkReport._records.clear()
