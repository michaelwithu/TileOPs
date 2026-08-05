#!/usr/bin/env python3
"""Run benchmark cases in recyclable, Kineto-health-checked workers.

Each worker reuses its Python and CUDA process across cases. Every case still
gets a separate Kineto context with start/end canaries. If a canary reports
missing CPU scopes, kernels, or projected GPU annotations, the supervisor
discards that case, recycles the worker, and retries it in a fresh process.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def collect_nodeids(target: str, pytest_args: list[str]) -> list[str]:
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", target, *pytest_args]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith((" ", "="))
    ]


def merge_junit_xml(inputs: list[Path], output: Path) -> None:
    root = ET.Element("testsuites", {"name": "pytest tests"})
    totals: dict[str, float] = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "time": 0.0,
    }
    for path in inputs:
        parsed = ET.parse(path).getroot()
        suites = [parsed] if parsed.tag == "testsuite" else list(parsed.findall("testsuite"))
        for suite in suites:
            root.append(suite)
            for key in totals:
                totals[key] += float(suite.attrib.get(key, 0))
    for key, value in totals.items():
        root.set(key, f"{value:.6f}" if key == "time" else str(int(value)))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def merge_profile_logs(inputs: list[Path], output: Path) -> None:
    sections = []
    for path in inputs:
        if path.exists():
            sections.append(path.read_text())
    if sections:
        output.write_text("\n\n".join(sections))


def append_case_audit(xml_path: Path, audit_path: Path, nodeid: str) -> None:
    """Persist per-implementation latency/source rows as soon as a case ends."""
    write_header = not audit_path.exists()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.parse(xml_path).getroot()
    with audit_path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "nodeid", "testcase", "outcome", "implementation",
            "latency_ms", "timing_source", "fallback_reason",
            "expected_region_count", "observed_region_counts", "kineto_error",
            "worker_id", "process_case_index", "health_reason",
            "canary_expected", "start_cpu_count", "start_kernel_count",
            "start_gpu_annotation_count", "end_cpu_count", "end_kernel_count",
            "end_gpu_annotation_count",
        ])
        if write_header:
            writer.writeheader()
        for testcase in root.iter("testcase"):
            properties = {
                prop.attrib["name"]: prop.attrib.get("value", "")
                for prop in testcase.iter("property")
            }
            outcome = (
                "failed" if testcase.find("failure") is not None else
                "error" if testcase.find("error") is not None else
                "skipped" if testcase.find("skipped") is not None else
                "passed"
            )
            source_keys = [
                key for key in properties
                if key.endswith("_timing_source") and key != "baseline_timing_source"
            ]
            if not source_keys:
                writer.writerow({
                    "nodeid": nodeid,
                    "testcase": testcase.attrib.get("name", ""),
                    "outcome": outcome,
                    "implementation": "",
                    "latency_ms": "",
                    "timing_source": "",
                    "fallback_reason": "",
                    "expected_region_count": "",
                    "observed_region_counts": "",
                    "kineto_error": "",
                    "worker_id": properties.get("kineto_health_worker_id", ""),
                    "process_case_index": properties.get(
                        "kineto_health_process_case_index", ""
                    ),
                    "health_reason": properties.get("kineto_health_reason", ""),
                    "canary_expected": properties.get("kineto_health_expected", ""),
                    "start_cpu_count": properties.get(
                        "kineto_health_start_cpu_count", ""
                    ),
                    "start_kernel_count": properties.get(
                        "kineto_health_start_kernel_count", ""
                    ),
                    "start_gpu_annotation_count": properties.get(
                        "kineto_health_start_gpu_annotation_count", ""
                    ),
                    "end_cpu_count": properties.get("kineto_health_end_cpu_count", ""),
                    "end_kernel_count": properties.get(
                        "kineto_health_end_kernel_count", ""
                    ),
                    "end_gpu_annotation_count": properties.get(
                        "kineto_health_end_gpu_annotation_count", ""
                    ),
                })
            for source_key in source_keys:
                implementation = source_key.removesuffix("_timing_source")
                writer.writerow({
                    "nodeid": nodeid,
                    "testcase": testcase.attrib.get("name", ""),
                    "outcome": outcome,
                    "implementation": implementation,
                    "latency_ms": properties.get(f"{implementation}_latency_ms", ""),
                    "timing_source": properties[source_key],
                    "fallback_reason": properties.get(
                        f"{implementation}_fallback_reason", ""
                    ),
                    "expected_region_count": properties.get(
                        f"{implementation}_expected_region_count", ""
                    ),
                    "observed_region_counts": properties.get(
                        f"{implementation}_observed_region_counts", ""
                    ),
                    "kineto_error": properties.get(
                        f"{implementation}_kineto_error", ""
                    ),
                    "worker_id": properties.get("kineto_health_worker_id", ""),
                    "process_case_index": properties.get(
                        "kineto_health_process_case_index", ""
                    ),
                    "health_reason": properties.get("kineto_health_reason", ""),
                    "canary_expected": properties.get("kineto_health_expected", ""),
                    "start_cpu_count": properties.get(
                        "kineto_health_start_cpu_count", ""
                    ),
                    "start_kernel_count": properties.get(
                        "kineto_health_start_kernel_count", ""
                    ),
                    "start_gpu_annotation_count": properties.get(
                        "kineto_health_start_gpu_annotation_count", ""
                    ),
                    "end_cpu_count": properties.get("kineto_health_end_cpu_count", ""),
                    "end_kernel_count": properties.get(
                        "kineto_health_end_kernel_count", ""
                    ),
                    "end_gpu_annotation_count": properties.get(
                        "kineto_health_end_gpu_annotation_count", ""
                    ),
                })


def run_worker(
    manifest_path: Path,
    state_path: Path,
    worker_dir: Path,
    worker_id: int,
    max_cases: int,
    max_seconds: float,
) -> int:
    """Run pytest repeatedly in one process until unhealthy or rotation limit."""
    import pytest

    manifest = json.loads(manifest_path.read_text())
    nodeids: list[str] = manifest["nodeids"]
    pytest_args: list[str] = manifest["pytest_args"]
    completed = []
    started = time.monotonic()
    state: dict = {"completed": completed, "worker_id": worker_id}

    for process_index, nodeid in enumerate(nodeids, start=1):
        case_id = manifest["start_index"] + process_index - 1
        junit_path = worker_dir / f"case_{case_id}.xml"
        profile_path = worker_dir / f"case_{case_id}.md"
        health_path = worker_dir / f"case_{case_id}.health.json"
        health_path.unlink(missing_ok=True)
        os.environ["TILEOPS_BENCH_PROFILE_PATH"] = str(profile_path)
        os.environ["TILEOPS_BENCH_HEALTH_PATH"] = str(health_path)
        os.environ["TILEOPS_BENCH_WORKER_ID"] = str(worker_id)
        os.environ["TILEOPS_BENCH_PROCESS_CASE_INDEX"] = str(process_index)

        print(
            f"[worker {worker_id} case {process_index}] {nodeid}",
            flush=True,
        )
        exit_code = int(pytest.main([
            "-q", nodeid, f"--junit-xml={junit_path}", *pytest_args,
        ]))
        health = json.loads(health_path.read_text()) if health_path.exists() else None
        case_result = {
            "nodeid": nodeid,
            "junit": str(junit_path),
            "profile": str(profile_path),
            "exit_code": exit_code,
            "health": health,
        }
        # A test skipped during setup never enters pytest_runtest_call and thus
        # legitimately has no health report.  Missing health is only a worker
        # failure when pytest itself also reports a non-zero exit status.
        missing_health_failure = health is None and exit_code != 0
        unhealthy = health is not None and not health.get("healthy", False)
        if missing_health_failure or unhealthy:
            state |= {
                "retry": case_result,
                "restart_reason": (
                    health.get("reason", "unknown")
                    if health is not None
                    else "missing_health_report"
                ),
                "worker_uptime_seconds": time.monotonic() - started,
            }
            state_path.write_text(json.dumps(state, indent=2))
            return 75

        completed.append(case_result)
        state_path.write_text(json.dumps(state, indent=2))
        reached_case_limit = max_cases > 0 and process_index >= max_cases
        reached_time_limit = max_seconds > 0 and time.monotonic() - started >= max_seconds
        if reached_case_limit or reached_time_limit:
            state |= {
                "rotated": True,
                "worker_uptime_seconds": time.monotonic() - started,
            }
            state_path.write_text(json.dumps(state, indent=2))
            return 0

    state["worker_uptime_seconds"] = time.monotonic() - started
    state_path.write_text(json.dumps(state, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", default="benchmarks/ops")
    parser.add_argument("--junit-xml", default="bench_results.xml")
    parser.add_argument("--profile-log", default="profile_run.log")
    parser.add_argument("--audit-csv", default="benchmark_audit.csv")
    parser.add_argument(
        "--max-worker-cases",
        type=int,
        default=0,
        help="proactively rotate after N cases; 0 keeps the worker until unhealthy",
    )
    parser.add_argument(
        "--max-worker-seconds",
        type=float,
        default=0,
        help="proactively rotate after N seconds; 0 keeps the worker until unhealthy",
    )
    parser.add_argument("--max-health-retries", type=int, default=2)
    parser.add_argument("--worker-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-state", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, default=0, help=argparse.SUPPRESS)
    args, pytest_args = parser.parse_known_args()
    pytest_args = pytest_args[1:] if pytest_args[:1] == ["--"] else pytest_args

    if args.worker_manifest:
        if args.worker_state is None or args.worker_dir is None:
            parser.error("worker mode requires --worker-state and --worker-dir")
        return run_worker(
            args.worker_manifest,
            args.worker_state,
            args.worker_dir,
            args.worker_id,
            args.max_worker_cases,
            args.max_worker_seconds,
        )

    nodeids = collect_nodeids(args.target, pytest_args)
    print(f"Collected {len(nodeids)} benchmark cases", flush=True)
    if not nodeids:
        return 5

    junit_inputs: list[Path] = []
    profile_inputs: list[Path] = []
    failed = 0
    retry_counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="tileops-benchmark-cases-") as temp:
        temp_path = Path(temp)
        cursor = 0
        worker_id = 0
        while cursor < len(nodeids):
            worker_id += 1
            manifest_path = temp_path / f"worker_{worker_id}.manifest.json"
            state_path = temp_path / f"worker_{worker_id}.state.json"
            manifest_path.write_text(json.dumps({
                "nodeids": nodeids[cursor:],
                "pytest_args": pytest_args,
                "start_index": cursor + 1,
            }))
            env = os.environ.copy()
            command = [
                sys.executable,
                __file__,
                f"--worker-manifest={manifest_path}",
                f"--worker-state={state_path}",
                f"--worker-dir={temp_path}",
                f"--worker-id={worker_id}",
                f"--max-worker-cases={args.max_worker_cases}",
                f"--max-worker-seconds={args.max_worker_seconds}",
            ]
            result = subprocess.run(command, env=env)
            if not state_path.exists():
                print(f"Worker {worker_id} exited without state", file=sys.stderr)
                return result.returncode or 1
            state = json.loads(state_path.read_text())
            for case in state["completed"]:
                junit_path = Path(case["junit"])
                profile_path = Path(case["profile"])
                if junit_path.exists():
                    junit_inputs.append(junit_path)
                    append_case_audit(junit_path, Path(args.audit_csv), case["nodeid"])
                if profile_path.exists():
                    profile_inputs.append(profile_path)
                failed += case["exit_code"] != 0
                cursor += 1

            retry = state.get("retry")
            if retry is not None:
                nodeid = retry["nodeid"]
                retry_counts[nodeid] = retry_counts.get(nodeid, 0) + 1
                print(
                    f"Worker {worker_id} unhealthy after {len(state['completed'])} "
                    f"completed cases: {state['restart_reason']}; retrying {nodeid}",
                    flush=True,
                )
                if retry_counts[nodeid] > args.max_health_retries:
                    print(
                        f"Kineto health failed {retry_counts[nodeid]} times for {nodeid}",
                        file=sys.stderr,
                    )
                    junit_path = Path(retry["junit"])
                    profile_path = Path(retry["profile"])
                    if junit_path.exists():
                        junit_inputs.append(junit_path)
                        append_case_audit(junit_path, Path(args.audit_csv), nodeid)
                    if profile_path.exists():
                        profile_inputs.append(profile_path)
                    failed += 1
                    cursor += 1

        merge_junit_xml(junit_inputs, Path(args.junit_xml))
        merge_profile_logs(profile_inputs, Path(args.profile_log))

    print(
        f"Completed {len(nodeids)} cases in {worker_id} workers; failed={failed}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
