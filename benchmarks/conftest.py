import gc
import json
import os

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    _begin_case_profiler_session,
    _bench_results,
    _finish_case_profiler_session,
)

# Skip NSA benchmarks until the underlying op failures are resolved.
collect_ignore_glob = [
    "ops/attention/bench_deepseek_nsa*.py",
]

def _normalized_benchmark_nodeid(item: pytest.Item) -> str:
    nodeid = item.nodeid
    if nodeid.startswith("benchmarks/"):
        return nodeid
    if nodeid.startswith("ops/"):
        return f"benchmarks/{nodeid}"
    return nodeid


def _is_fp8_e4m3_benchmark(item: pytest.Item) -> bool:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False
    return callspec.params.get("dtype") == torch.float8_e4m3fn


def _release_cuda_cache_after_case() -> None:
    """Drop per-case Python references and cached CUDA blocks between benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1235)


def pytest_sessionstart(session):
    BenchmarkReport.clear()


def pytest_sessionfinish(session, exitstatus):
    BenchmarkReport.dump(os.environ.get("TILEOPS_BENCH_PROFILE_PATH", "profile_run.log"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    fp8_e4m3_skip = pytest.mark.skip(
        reason=(
            "Skipped under tilelang 0.1.9: fp8 e4m3 benchmark fails due to "
            "lowering regression; re-enable when fp8 e4m3 benchmarks run "
            "cleanly against current tilelang."
        )
    )

    for item in items:
        nodeid = _normalized_benchmark_nodeid(item)
        path = nodeid.split("::", 1)[0]

        if (
            path == "benchmarks/ops/bench_elementwise_fp8.py"
            and _is_fp8_e4m3_benchmark(item)
        ):
            item.add_marker(fp8_e4m3_skip)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """After bench test execution, attach perf data to the item as properties."""
    _bench_results.entries = []
    _begin_case_profiler_session()
    try:
        yield
        health = _finish_case_profiler_session()
        if health is not None:
            health |= {
                "nodeid": _normalized_benchmark_nodeid(item),
                "worker_id": os.environ.get("TILEOPS_BENCH_WORKER_ID", ""),
                "process_case_index": int(
                    os.environ.get("TILEOPS_BENCH_PROCESS_CASE_INDEX", "0")
                ),
            }
            health_path = os.environ.get("TILEOPS_BENCH_HEALTH_PATH")
            if health_path:
                with open(health_path, "w") as stream:
                    json.dump(health, stream)
            for key, value in health.items():
                item.user_properties.append((f"kineto_health_{key}", value))
            if not health["healthy"]:
                pytest.fail(
                    f"Kineto health check failed: {health['reason']} "
                    f"(start cpu/kernel/gpu="
                    f"{health['start_cpu_count']}/{health['start_kernel_count']}/"
                    f"{health['start_gpu_annotation_count']}, end="
                    f"{health['end_cpu_count']}/{health['end_kernel_count']}/"
                    f"{health['end_gpu_annotation_count']})"
                )
        entries = getattr(_bench_results, "entries", [])
        if not entries:
            return

        entries = [
            {key: value for key, value in entry.items() if key != "result"}
            | entry["result"]
            for entry in entries
        ]

        # Separate tileops entry (tag starts with "tileops") from baselines.
        tileops_entry = None
        baseline_entries = []
        for e in entries:
            if e["tag"].startswith("tileops"):
                if tileops_entry is None:
                    tileops_entry = e
            else:
                baseline_entries.append(e)

        if tileops_entry:
            item.user_properties.append(("op", tileops_entry["op"]))
            if "op_module" in tileops_entry:
                item.user_properties.append(("op_module", tileops_entry["op_module"]))
            tag = tileops_entry["tag"]
            if tag != "tileops" and tag.startswith("tileops_"):
                item.user_properties.append(("tileops_variant", tag[len("tileops_"):]))
            item.user_properties.append(("tileops_latency_ms",
                                         f"{tileops_entry.get('latency_ms', 0):.4f}"))
            item.user_properties.append((
                "tileops_timing_source",
                tileops_entry.get("timing_source", "unknown"),
            ))
            for key in (
                "fallback_reason", "expected_region_count",
                "observed_region_counts", "kineto_error",
            ):
                if key in tileops_entry:
                    item.user_properties.append((f"tileops_{key}", tileops_entry[key]))
            tflops = tileops_entry.get("tflops")
            if tflops is not None:
                item.user_properties.append(("tileops_tflops", f"{tflops:.2f}"))
            bw = tileops_entry.get("bandwidth_tbs")
            if bw is not None:
                item.user_properties.append(("tileops_bandwidth_tbs", f"{bw:.2f}"))

        # Write all baselines into JUnit XML properties.
        # The first baseline uses the legacy unprefixed names (baseline_tag, etc.)
        # for backward compatibility.  Additional baselines use "{tag}_latency_ms",
        # "{tag}_tflops", "{tag}_ratio" so the report can display multiple columns.
        for idx, be in enumerate(baseline_entries):
            tag = be["tag"]
            bl_latency = be.get("latency_ms", 0)
            bl_tflops = be.get("tflops")

            if idx == 0:
                # Legacy unprefixed keys — consumed by existing nightly_report.py
                item.user_properties.append(("baseline_tag", tag))
                item.user_properties.append(("baseline_latency_ms", f"{bl_latency:.4f}"))
                item.user_properties.append((
                    "baseline_timing_source",
                    be.get("timing_source", "unknown"),
                ))
                if bl_tflops is not None:
                    item.user_properties.append(("baseline_tflops", f"{bl_tflops:.2f}"))
                if tileops_entry:
                    tl = tileops_entry.get("latency_ms", 0)
                    if tl > 0 and bl_latency > 0:
                        item.user_properties.append(("baseline_ratio",
                                                     f"{bl_latency / tl:.4f}"))

            # Tag-prefixed keys — always written for every baseline
            item.user_properties.append((f"{tag}_latency_ms", f"{bl_latency:.4f}"))
            item.user_properties.append((
                f"{tag}_timing_source",
                be.get("timing_source", "unknown"),
            ))
            for key in (
                "fallback_reason", "expected_region_count",
                "observed_region_counts", "kineto_error",
            ):
                if key in be:
                    item.user_properties.append((f"{tag}_{key}", be[key]))
            if bl_tflops is not None:
                item.user_properties.append((f"{tag}_tflops", f"{bl_tflops:.2f}"))
            if tileops_entry:
                tl = tileops_entry.get("latency_ms", 0)
                if tl > 0 and bl_latency > 0:
                    item.user_properties.append((f"{tag}_ratio", f"{bl_latency / tl:.4f}"))
    finally:
        # Also clears a partially registered session when the test body fails.
        _bench_results.case_profiler = None
        _bench_results.entries = []
        _release_cuda_cache_after_case()
