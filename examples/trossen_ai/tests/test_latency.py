"""Offline tests for inference-latency accounting."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latency import LatencyTracker  # noqa: E402


def test_nothing_to_report_before_any_call():
    assert LatencyTracker().summary() is None
    assert LatencyTracker().percentiles() == {}


def test_the_first_call_is_held_aside_as_warmup():
    """A first call carrying JAX compilation would otherwise sit in the window
    and drag every percentile with it for the rest of the session."""
    tracker = LatencyTracker()
    tracker.record(30.0)  # 30 s compile
    assert tracker.warmup_ms == pytest.approx(30_000.0)
    assert tracker.samples == 0
    assert tracker.summary() is None

    for _ in range(10):
        tracker.record(0.2)
    assert tracker.percentiles()["p50_ms"] == pytest.approx(200.0)


def test_percentiles_track_the_round_trip():
    tracker = LatencyTracker()
    tracker.record(0.1)  # warmup, discarded
    for value in [0.1] * 95 + [1.0] * 5:
        tracker.record(value)
    stats = tracker.percentiles()
    assert stats["p50_ms"] == pytest.approx(100.0)
    assert stats["p95_ms"] > 100.0
    assert stats["max_ms"] == pytest.approx(1000.0)


def test_server_time_and_overhead_are_split():
    """The whole point: a slow model and a slow client look identical in a
    single round-trip number."""
    tracker = LatencyTracker()
    tracker.record(0.3, 250.0)  # warmup
    for _ in range(20):
        tracker.record(0.3, 250.0)
    stats = tracker.percentiles()
    assert stats["server_ms"] == pytest.approx(250.0)
    assert stats["overhead_ms"] == pytest.approx(50.0)


def test_overhead_is_absent_when_the_server_reports_no_timing():
    """Backends other than openpi's own server may send none; the round-trip
    numbers must still be reported."""
    tracker = LatencyTracker()
    tracker.record(0.2)
    for _ in range(5):
        tracker.record(0.2)
    stats = tracker.percentiles()
    assert "server_ms" not in stats
    assert "overhead_ms" not in stats
    assert "server" not in tracker.summary()


def test_the_window_is_bounded():
    tracker = LatencyTracker(window=10)
    tracker.record(0.1)
    for _ in range(100):
        tracker.record(0.5)
    assert tracker.samples == 10
    assert tracker.calls == 101


def test_the_window_forgets_an_old_slow_patch():
    tracker = LatencyTracker(window=10)
    tracker.record(0.1)
    for _ in range(10):
        tracker.record(5.0)
    for _ in range(10):
        tracker.record(0.1)
    assert tracker.percentiles()["max_ms"] == pytest.approx(100.0)


def test_reset_clears_everything():
    tracker = LatencyTracker()
    for _ in range(5):
        tracker.record(0.2, 100.0)
    tracker.reset()
    assert tracker.samples == 0
    assert tracker.calls == 0
    assert tracker.warmup_ms is None
    assert tracker.summary() is None


def test_summary_reads_as_one_line():
    tracker = LatencyTracker()
    tracker.record(0.4, 300.0)
    for _ in range(5):
        tracker.record(0.4, 300.0)
    summary = tracker.summary()
    assert summary.startswith("infer p50 400 p95 400 ms")
    assert "server 300" in summary
    assert "overhead 100" in summary
