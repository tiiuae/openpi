"""Offline tests for inference-latency accounting."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latency import (  # noqa: E402
    InferenceTiming,
    LatencyTracker,
    SchedulingLagProbe,
    server_timing,
)


def _timing(pack=5.0, wire=200.0, unpack=2.0, server=180.0) -> InferenceTiming:
    return InferenceTiming(pack_ms=pack, wire_ms=wire, unpack_ms=unpack, server_ms=server)


# -- the split -----------------------------------------------------------------


def test_round_trip_is_the_sum_of_the_phases():
    """It must equal what a naive timer around infer() would have shown, or the
    split is not a decomposition of the same thing."""
    timing = _timing(pack=5.0, wire=200.0, unpack=2.0)
    assert timing.round_trip_ms == pytest.approx(207.0)


def test_network_is_wire_time_the_server_did_not_account_for():
    assert _timing(wire=200.0, server=180.0).network_ms == pytest.approx(20.0)


def test_network_is_unknown_when_the_server_reports_nothing():
    """starVLA and the REST-backed policies may send no timing block."""
    assert InferenceTiming(1.0, 100.0, 1.0, server_ms=None).network_ms is None


def test_network_never_goes_negative():
    """Two machines' clocks measuring overlapping spans can disagree by a hair."""
    assert _timing(wire=100.0, server=100.4).network_ms == 0.0


def test_client_time_is_the_gil_holding_part():
    assert _timing(pack=5.0, unpack=2.0).client_ms == pytest.approx(7.0)


# -- the tracker ---------------------------------------------------------------


def test_nothing_to_report_before_any_call():
    assert LatencyTracker().summary() is None
    assert LatencyTracker().percentiles() == {}


def test_the_first_call_is_held_aside_as_warmup():
    """A first call carrying JAX compilation would otherwise sit in the window
    and drag every percentile with it for the rest of the session."""
    tracker = LatencyTracker()
    tracker.record(_timing(wire=30_000.0))
    assert tracker.warmup.wire_ms == pytest.approx(30_000.0)
    assert tracker.samples == 0
    # Shown, so a long first call is explained rather than looking like a hang,
    # but labelled as excluded from the statistics.
    assert "warmup" in tracker.summary()
    assert "30007 ms" in tracker.summary()

    for _ in range(10):
        tracker.record(_timing())
    assert tracker.percentiles()["p50_ms"] == pytest.approx(207.0)


def test_each_phase_is_reported_separately():
    """The whole point: a slow model, a slow link and a starved client are three
    different problems that a single round-trip number cannot tell apart."""
    tracker = LatencyTracker()
    tracker.record(_timing())
    for _ in range(20):
        tracker.record(_timing(pack=8.0, wire=300.0, unpack=2.0, server=250.0))
    stats = tracker.percentiles()
    assert stats["pack_ms"] == pytest.approx(8.0)
    assert stats["wire_ms"] == pytest.approx(300.0)
    assert stats["unpack_ms"] == pytest.approx(2.0)
    assert stats["server_ms"] == pytest.approx(250.0)
    assert stats["network_ms"] == pytest.approx(50.0)


def test_percentiles_track_the_round_trip():
    tracker = LatencyTracker()
    tracker.record(_timing())
    for wire in [100.0] * 95 + [1000.0] * 5:
        tracker.record(_timing(pack=0.0, wire=wire, unpack=0.0, server=None))
    stats = tracker.percentiles()
    assert stats["p50_ms"] == pytest.approx(100.0)
    assert stats["p95_ms"] > 100.0
    assert stats["max_ms"] == pytest.approx(1000.0)


def test_server_and_network_are_absent_when_unreported():
    tracker = LatencyTracker()
    tracker.record(_timing(server=None))
    for _ in range(5):
        tracker.record(_timing(server=None))
    stats = tracker.percentiles()
    assert "server_ms" not in stats
    assert "network_ms" not in stats
    assert "wire_ms" in stats  # still measurable without the server's help


def test_the_window_is_bounded():
    tracker = LatencyTracker(window=10)
    for _ in range(101):
        tracker.record(_timing())
    assert tracker.samples == 10
    assert tracker.calls == 101


def test_the_window_forgets_an_old_slow_patch():
    tracker = LatencyTracker(window=10)
    tracker.record(_timing())
    for _ in range(10):
        tracker.record(_timing(wire=5000.0))
    for _ in range(10):
        tracker.record(_timing(pack=0.0, wire=100.0, unpack=0.0))
    assert tracker.percentiles()["max_ms"] == pytest.approx(100.0)


def test_reset_clears_everything():
    tracker = LatencyTracker()
    for _ in range(5):
        tracker.record(_timing())
    tracker.reset()
    assert tracker.samples == 0
    assert tracker.calls == 0
    assert tracker.warmup is None
    assert tracker.summary() is None


def test_summary_names_each_contributor():
    tracker = LatencyTracker()
    tracker.record(_timing())
    for _ in range(5):
        tracker.record(_timing(pack=5.0, wire=200.0, unpack=2.0, server=180.0))
    summary = tracker.summary()
    assert summary.startswith("last 207 ms (server 180 · net 20 · pack 7)")
    assert "p50 207 · p95 207 · max 207 ms" in summary
    assert summary.endswith("6 calls")


def test_summary_leads_with_the_latest_call_not_the_median():
    """A 200-sample median barely moves; while testing, the operator needs to
    see the call that just happened, spike included."""
    tracker = LatencyTracker()
    tracker.record(_timing())
    for _ in range(50):
        tracker.record(_timing(pack=0.0, wire=100.0, unpack=0.0, server=90.0))
    tracker.record(_timing(pack=0.0, wire=900.0, unpack=0.0, server=880.0))
    summary = tracker.summary()
    assert summary.startswith("last 900 ms (server 880 · net 20 · pack 0)")
    assert "p50 100" in summary
    assert "max 900 ms" in summary


def test_summary_omits_server_and_net_when_the_backend_reports_neither():
    tracker = LatencyTracker()
    for _ in range(3):
        tracker.record(_timing(server=None))
    summary = tracker.summary()
    assert "server" not in summary
    assert "net" not in summary
    assert "pack 7" in summary


def test_summary_includes_scheduling_lag_once_measured():
    tracker = LatencyTracker()
    tracker.record(_timing())
    for _ in range(5):
        tracker.record(_timing())
    tracker.lag.record(12.0)
    assert "gil lag 12 ms" in tracker.summary()


# -- scheduling lag ------------------------------------------------------------


def test_lag_is_unknown_before_it_has_measured_anything():
    assert SchedulingLagProbe().p95_ms() is None


def test_lag_reports_a_percentile_of_its_samples():
    probe = SchedulingLagProbe()
    for value in [1.0] * 95 + [50.0] * 5:
        probe.record(value)
    assert probe.p95_ms() > 1.0


def test_the_probe_measures_a_quiet_interpreter_as_near_zero():
    probe = SchedulingLagProbe(interval=0.01)
    probe.start()
    time.sleep(0.2)
    probe.stop()
    assert probe.samples > 0
    assert probe.p95_ms() < 50.0  # generous: CI machines are noisy


def test_the_probe_can_be_stopped_and_restarted():
    probe = SchedulingLagProbe(interval=0.01)
    probe.start()
    probe.start()  # idempotent
    time.sleep(0.05)
    probe.stop()
    probe.stop()  # idempotent


# -- server timing block -------------------------------------------------------


def test_server_timing_is_returned_when_present():
    assert server_timing({"actions": [], "server_timing": {"infer_ms": 12.5}}) == {"infer_ms": 12.5}


@pytest.mark.parametrize("reply", [{"actions": []}, {"server_timing": "nope"}, None, "string"])
def test_missing_server_timing_is_not_an_error(reply):
    assert server_timing(reply) == {}


# -- terminal display ----------------------------------------------------------


def _render(latency: str | None, *, paused: bool) -> str:
    rich_console = pytest.importorskip("rich.console")
    import terminal_ui  # noqa: PLC0415 - needs rich, which the bare test env may lack

    listener = terminal_ui.PinnedPromptListener.__new__(terminal_ui.PinnedPromptListener)
    terminal_ui.BasePromptListener.__init__(listener, "pick up the cup")
    listener._buffer = ""
    listener.set_latency(latency)
    listener.set_paused(paused)
    console = rich_console.Console(width=200, force_terminal=False, color_system=None, record=True)
    console.print(listener._render())
    return console.export_text()


def test_the_terminal_shows_latency_on_its_own_line():
    text = _render("last 212 ms (server 180 · net 25 · pack 7)", paused=False)
    lines = [line for line in text.splitlines() if "inference" in line]
    assert lines == ["⏱ inference last 212 ms (server 180 · net 25 · pack 7)"]


def test_the_terminal_keeps_showing_latency_while_paused():
    """It used to vanish on pause, which is exactly when a stall needs reading."""
    text = _render("last 950 ms (server 930 · net 20 · pack 0)", paused=True)
    assert "last 950 ms" in text
    assert "paused" in text


def test_the_terminal_says_it_is_waiting_before_the_first_result():
    assert "waiting for the first result" in _render(None, paused=False)
