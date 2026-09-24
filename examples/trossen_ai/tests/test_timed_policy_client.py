"""Offline tests for the timing client.

The timed client reimplements ``infer()`` so it can get between packing, the
wire and unpacking. That is a copy of upstream's five lines, so the risk is
drift: these tests pin it to the same wire behaviour, driven by a fake socket.
No server, no network.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpi_client import msgpack_numpy  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

from latency import LatencyTracker  # noqa: E402
from timed_policy_client import TimedWebsocketClientPolicy  # noqa: E402

DIM = 14


class FakeSocket:
    """Records what was sent and replies with a queued frame."""

    def __init__(self, reply, wire_delay: float = 0.0):
        self.reply = reply
        self.wire_delay = wire_delay
        self.sent: list[bytes] = []

    def send(self, data):
        self.sent.append(data)

    def recv(self):
        if self.wire_delay:
            time.sleep(self.wire_delay)
        return self.reply


def _client(reply_obj=None, *, raw_reply=None, wire_delay=0.0, tracker=None):
    """A TimedWebsocketClientPolicy wired to a fake socket, skipping __init__
    so no connection is attempted."""
    client = TimedWebsocketClientPolicy.__new__(TimedWebsocketClientPolicy)
    client._packer = msgpack_numpy.Packer()
    frame = raw_reply if raw_reply is not None else msgpack_numpy.packb(reply_obj)
    client._ws = FakeSocket(frame, wire_delay)
    client.latency = tracker if tracker is not None else LatencyTracker()
    return client


def _observation():
    return {
        "state": np.zeros(DIM),
        "images": {"cam_high": np.zeros((3, 224, 224), dtype=np.uint8)},
        "prompt": "pick up the cup",
    }


def test_the_reply_is_returned_unchanged():
    expected = {"actions": np.ones((25, DIM)), "server_timing": {"infer_ms": 42.0}}
    client = _client(expected)
    reply = client.infer(_observation())
    np.testing.assert_array_equal(reply["actions"], expected["actions"])


def test_the_observation_goes_out_in_the_upstream_wire_format():
    """Pinned against the parent class so the copied infer() cannot drift."""
    observation = _observation()
    timed = _client({"actions": np.zeros((1, DIM))})
    timed.infer(observation)

    upstream = websocket_client_policy.WebsocketClientPolicy.__new__(
        websocket_client_policy.WebsocketClientPolicy
    )
    upstream._packer = msgpack_numpy.Packer()
    upstream._ws = FakeSocket(msgpack_numpy.packb({"actions": np.zeros((1, DIM))}))
    upstream.infer(observation)

    assert timed._ws.sent == upstream._ws.sent


def test_a_string_frame_is_still_raised_as_a_server_error():
    client = _client(raw_reply="Traceback: boom")
    with pytest.raises(RuntimeError, match="Error in inference server"):
        client.infer(_observation())


def test_a_failed_call_records_no_timing():
    """A call that raised never completed, so it must not enter the stats."""
    tracker = LatencyTracker()
    client = _client(raw_reply="Traceback: boom", tracker=tracker)
    with pytest.raises(RuntimeError):
        client.infer(_observation())
    assert tracker.calls == 0


def test_each_phase_is_recorded():
    tracker = LatencyTracker()
    client = _client({"actions": np.zeros((25, DIM)), "server_timing": {"infer_ms": 5.0}}, tracker=tracker)
    client.infer(_observation())
    timing = tracker.warmup
    assert timing is not None
    assert timing.pack_ms > 0.0
    assert timing.unpack_ms > 0.0
    assert timing.server_ms == pytest.approx(5.0)


def test_wire_time_reflects_time_spent_waiting_for_the_server():
    tracker = LatencyTracker()
    client = _client({"actions": np.zeros((1, DIM))}, wire_delay=0.05, tracker=tracker)
    client.infer(_observation())
    timing = tracker.warmup
    assert timing.wire_ms >= 45.0
    # Packing a small observation is nothing like the wire wait.
    assert timing.pack_ms < timing.wire_ms


def test_server_time_is_none_when_the_backend_reports_none():
    tracker = LatencyTracker()
    client = _client({"actions": np.zeros((1, DIM))}, tracker=tracker)
    client.infer(_observation())
    assert tracker.warmup.server_ms is None


def test_repeated_calls_accumulate_after_the_warmup():
    tracker = LatencyTracker()
    client = _client({"actions": np.zeros((1, DIM))}, tracker=tracker)
    for _ in range(4):
        client.infer(_observation())
    assert tracker.calls == 4
    assert tracker.samples == 3
    assert tracker.summary() is not None
