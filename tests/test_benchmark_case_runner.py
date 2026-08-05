"""Tests for per-case benchmark subprocess artifact aggregation."""

import json
import os
import xml.etree.ElementTree as ET

import pytest

from scripts.run_benchmark_cases import (
    append_case_audit,
    merge_junit_xml,
    merge_profile_logs,
    run_worker,
)

pytestmark = pytest.mark.full


def _write_suite(path, *, tests, failures, elapsed):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", {
        "name": path.stem,
        "tests": str(tests),
        "failures": str(failures),
        "errors": "0",
        "skipped": "0",
        "time": str(elapsed),
    })
    ET.SubElement(suite, "testcase", {"name": path.stem})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_merge_junit_xml_sums_child_suites(tmp_path):
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    output = tmp_path / "combined.xml"
    _write_suite(first, tests=1, failures=0, elapsed=1.25)
    _write_suite(second, tests=1, failures=1, elapsed=2.5)

    merge_junit_xml([first, second], output)

    root = ET.parse(output).getroot()
    assert root.attrib["tests"] == "2"
    assert root.attrib["failures"] == "1"
    assert float(root.attrib["time"]) == 3.75
    assert len(root.findall("testsuite")) == 2


def test_merge_profile_logs_keeps_case_order(tmp_path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    output = tmp_path / "combined.md"
    first.write_text("case one")
    second.write_text("case two")

    merge_profile_logs([first, second], output)

    assert output.read_text() == "case one\n\ncase two"


def test_append_case_audit_records_timing_source(tmp_path):
    xml = tmp_path / "case.xml"
    audit = tmp_path / "audit.csv"
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    case = ET.SubElement(suite, "testcase", {"name": "gemm"})
    properties = ET.SubElement(case, "properties")
    ET.SubElement(properties, "property", {"name": "tileops_latency_ms", "value": "0.0035"})
    ET.SubElement(properties, "property", {"name": "tileops_timing_source", "value": "cupti"})
    ET.ElementTree(root).write(xml)

    append_case_audit(xml, audit, "bench.py::gemm")

    assert audit.read_text().splitlines() == [
        "nodeid,testcase,outcome,implementation,latency_ms,timing_source,"
        "fallback_reason,expected_region_count,observed_region_counts,kineto_error,"
        "worker_id,process_case_index,health_reason,canary_expected,start_cpu_count,"
        "start_kernel_count,start_gpu_annotation_count,end_cpu_count,end_kernel_count,"
        "end_gpu_annotation_count",
        "bench.py::gemm,gemm,passed,tileops,0.0035,cupti,,,,,,,,,,,,,,",
    ]


def test_worker_recycles_and_retries_unhealthy_case(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    state = tmp_path / "state.json"
    manifest.write_text(json.dumps({
        "nodeids": ["bench.py::first", "bench.py::second"],
        "pytest_args": [],
        "start_index": 1,
    }))

    def fake_pytest_main(_args):
        process_index = int(os.environ["TILEOPS_BENCH_PROCESS_CASE_INDEX"])
        health_path = os.environ["TILEOPS_BENCH_HEALTH_PATH"]
        healthy = process_index == 1
        with open(health_path, "w") as stream:
            json.dump({"healthy": healthy, "reason": "ok" if healthy else "missing"}, stream)
        return 0 if healthy else 1

    monkeypatch.setattr(pytest, "main", fake_pytest_main)

    assert run_worker(manifest, state, tmp_path, 7, 100, 900) == 75
    result = json.loads(state.read_text())
    assert [case["nodeid"] for case in result["completed"]] == ["bench.py::first"]
    assert result["retry"]["nodeid"] == "bench.py::second"
    assert result["restart_reason"] == "missing"


def test_worker_rotates_at_case_limit(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    state = tmp_path / "state.json"
    manifest.write_text(json.dumps({
        "nodeids": ["bench.py::first", "bench.py::second"],
        "pytest_args": [],
        "start_index": 1,
    }))

    def fake_pytest_main(_args):
        with open(os.environ["TILEOPS_BENCH_HEALTH_PATH"], "w") as stream:
            json.dump({"healthy": True, "reason": "ok"}, stream)
        return 0

    monkeypatch.setattr(pytest, "main", fake_pytest_main)

    assert run_worker(manifest, state, tmp_path, 3, 1, 900) == 0
    result = json.loads(state.read_text())
    assert len(result["completed"]) == 1
    assert result["rotated"] is True


def test_worker_has_no_fixed_case_rotation_by_default(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    state = tmp_path / "state.json"
    nodeids = [f"bench.py::case_{index}" for index in range(150)]
    manifest.write_text(json.dumps({
        "nodeids": nodeids,
        "pytest_args": [],
        "start_index": 1,
    }))

    def fake_pytest_main(_args):
        with open(os.environ["TILEOPS_BENCH_HEALTH_PATH"], "w") as stream:
            json.dump({"healthy": True, "reason": "ok"}, stream)
        return 0

    monkeypatch.setattr(pytest, "main", fake_pytest_main)

    assert run_worker(manifest, state, tmp_path, 4, 0, 0) == 0
    result = json.loads(state.read_text())
    assert [case["nodeid"] for case in result["completed"]] == nodeids
    assert "rotated" not in result


def test_worker_restarts_when_case_writes_no_health_report(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    state = tmp_path / "state.json"
    manifest.write_text(json.dumps({
        "nodeids": ["bench.py::case"],
        "pytest_args": [],
        "start_index": 1,
    }))
    monkeypatch.setattr(pytest, "main", lambda _args: 1)

    assert run_worker(manifest, state, tmp_path, 5, 0, 0) == 75
    result = json.loads(state.read_text())
    assert result["completed"] == []
    assert result["retry"]["nodeid"] == "bench.py::case"
    assert result["restart_reason"] == "missing_health_report"


def test_worker_accepts_skipped_case_without_health_report(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    state = tmp_path / "state.json"
    manifest.write_text(json.dumps({
        "nodeids": ["bench.py::skipped"],
        "pytest_args": [],
        "start_index": 1,
    }))
    monkeypatch.setattr(pytest, "main", lambda _args: 0)

    assert run_worker(manifest, state, tmp_path, 6, 0, 0) == 0
    result = json.loads(state.read_text())
    assert [case["nodeid"] for case in result["completed"]] == ["bench.py::skipped"]
    assert result["completed"][0]["health"] is None
    assert "retry" not in result
