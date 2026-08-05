"""Unit tests for benchmark Kineto region attribution and execution semantics."""

from contextlib import nullcontext

import pytest
from torch.autograd.profiler import DeviceType

import benchmarks.benchmark_base as benchmark_base
from benchmarks.benchmark_base import _region_kernel_times_us

pytestmark = pytest.mark.full


class _Event:
    def __init__(self, name, device, start, end, *, annotation=False):
        self._name = name
        self._device = device
        self._start = start
        self._end = end
        self._annotation = annotation

    def name(self):
        return self._name

    def device_type(self):
        return self._device

    def start_ns(self):
        return self._start

    def end_ns(self):
        return self._end

    def duration_ns(self):
        return self._end - self._start

    def is_user_annotation(self):
        return self._annotation


class _Results:
    def __init__(self, events):
        self._events = events

    def events(self):
        return self._events


def _cpu_region(name, start, end):
    return _Event(name, DeviceType.CPU, start, end, annotation=True)


def _gpu_region(name, start, end):
    return _Event(name, DeviceType.CUDA, start, end, annotation=True)


def _kernel(start, end):
    return _Event("kernel", DeviceType.CUDA, start, end)


def test_region_sums_all_kernels_in_cpu_scope_and_excludes_flush():
    results = _Results([
        _kernel(0, 10),  # L2 flush before the timed scope.
        _cpu_region("trial", 20, 100),
        _kernel(30, 40),
        _kernel(50, 75),
        _kernel(110, 120),
    ])

    assert _region_kernel_times_us(results, {"trial"}) == {
        "trial": (0.035, 1),
    }


def test_region_uses_cpu_scope_when_gpu_projection_is_incomplete():
    results = _Results([
        _cpu_region("backward", 100, 300),
        _gpu_region("backward", 120, 160),
        _kernel(125, 140),  # Main-thread launch covered by GPU projection.
        _kernel(220, 250),  # Autograd-worker launch outside that projection.
    ])

    assert _region_kernel_times_us(results, {"backward"}) == {
        "backward": (0.045, 1),
    }


def test_region_counts_repeated_cpu_scopes_and_ignores_gpu_duplicates():
    results = _Results([
        _cpu_region("trial", 100, 150),
        _cpu_region("trial", 200, 250),
        _gpu_region("trial", 110, 140),
        _gpu_region("trial", 210, 220),
        _gpu_region("trial", 225, 240),
        _kernel(120, 130),
        _kernel(215, 225),
        _kernel(230, 245),
    ])

    assert _region_kernel_times_us(results, {"trial"}) == {
        "trial": (0.035, 2),
    }


def test_region_keeps_trial_names_isolated():
    results = _Results([
        _cpu_region("trial:0", 10, 50),
        _cpu_region("trial:1", 60, 100),
        _kernel(20, 30),
        _kernel(70, 90),
    ])

    assert _region_kernel_times_us(results, {"trial:0", "trial:1"}) == {
        "trial:0": (0.01, 1),
        "trial:1": (0.02, 1),
    }


class _FakeCudaEvent:
    def record(self):
        pass

    def elapsed_time(self, _other):
        return 1.0


class _FakeCache:
    def zero_(self):
        pass


def _measurement(fn, *, n_repeat=2):
    return benchmark_base._PendingMeasurement(
        fn=fn,
        args=(),
        n_warmup=0,
        n_repeat=n_repeat,
        n_trials=1,
        result={},
        build_result=lambda latency: {"latency_ms": latency},
    )


def _fake_immediate_session(monkeypatch, fn):
    session = benchmark_base._CaseProfilerSession.__new__(
        benchmark_base._CaseProfilerSession
    )
    session.measurements = []
    monkeypatch.setattr(
        benchmark_base,
        "_prepare_measurement_runner",
        lambda _measurement: (lambda _index: fn(), None),
    )
    monkeypatch.setattr(benchmark_base, "_get_l2_flush_cache", _FakeCache)
    monkeypatch.setattr(benchmark_base.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        benchmark_base.torch.cuda,
        "Event",
        lambda **_kwargs: _FakeCudaEvent(),
    )
    monkeypatch.setattr(
        benchmark_base.torch.profiler,
        "record_function",
        lambda _name: nullcontext(),
    )
    return session


def test_case_profile_executes_before_add_returns(monkeypatch):
    calls = []
    session = _fake_immediate_session(monkeypatch, lambda: calls.append("called"))
    measurement = _measurement(lambda: None)

    result = session.add(measurement)

    assert result is measurement.result
    assert calls == ["called", "called"]
    assert measurement.fallback_latency_ms == 1.0
    assert session.measurements == [measurement]


def test_case_profile_propagates_exception_without_pending_measurement(monkeypatch):
    error = RuntimeError("unsupported shape")

    def fail():
        raise error

    session = _fake_immediate_session(monkeypatch, fail)

    with pytest.raises(RuntimeError, match="unsupported shape"):
        session.add(_measurement(lambda: None, n_repeat=1))

    assert session.measurements == []
