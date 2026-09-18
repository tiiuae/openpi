"""A policy client that times each phase of an inference call.

``WebsocketClientPolicy.infer()`` packs, sends, receives and unpacks in one
method, so timing it from outside can only ever produce a single number that
mixes client CPU work, the network and the model. This subclass runs the same
three steps with a timer between them and hands the split to a
``LatencyTracker``.

The body of ``infer`` deliberately mirrors the parent's, line for line, rather
than calling it: the whole point is to get between the steps. It is five lines,
and if upstream changes the wire protocol this must change with it —
``test_timed_policy_client.py`` asserts the two stay equivalent on the wire.

Callers use it exactly like the normal client, so nothing downstream needs to
know that timing is happening.
"""

from __future__ import annotations

import time
from typing import Any

from latency import InferenceTiming
from latency import LatencyTracker
from latency import server_timing
from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


class TimedWebsocketClientPolicy(websocket_client_policy.WebsocketClientPolicy):
    """``WebsocketClientPolicy`` that records where each call's time went."""

    def __init__(self, *args: Any, latency: LatencyTracker | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.latency = latency if latency is not None else LatencyTracker()

    def infer(self, obs: dict) -> dict:
        # Phase 1: serialize. Roughly 440 KB of images, on this thread, holding
        # the GIL. Normally well under a millisecond, so a large number here
        # means this process is starved rather than the server being slow.
        started = time.perf_counter()
        data = self._packer.pack(obs)
        packed_at = time.perf_counter()

        # Phase 2: the wire. Send, wait for the server, receive. The GIL is
        # released across the socket calls, so this is the only phase that can
        # overlap with the control loop — and the only one containing the model.
        self._ws.send(data)
        response = self._ws.recv()
        received_at = time.perf_counter()

        if isinstance(response, str):
            # We expect bytes; a string frame is the server reporting an error.
            raise RuntimeError(f"Error in inference server:\n{response}")

        # Phase 3: deserialize the reply. Small, on this thread, GIL again.
        reply = msgpack_numpy.unpackb(response)
        done_at = time.perf_counter()

        self.latency.record(
            InferenceTiming(
                pack_ms=(packed_at - started) * 1e3,
                wire_ms=(received_at - packed_at) * 1e3,
                unpack_ms=(done_at - received_at) * 1e3,
                server_ms=server_timing(reply).get("infer_ms"),
            )
        )
        return reply
