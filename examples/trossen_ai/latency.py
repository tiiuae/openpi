"""Inference timing that says which part was slow.

Timing a policy call by wrapping ``client.infer()`` in ``perf_counter()`` gives
one number for six different things, which is why that number looks erratic:

1. msgpack packing of the observation — three 224x224 RGB frames is ~450 KB
   per request, and it is packed on the calling thread;
2. the socket send and the network hop;
3. any time the request waits on a busy server;
4. the model itself;
5. the response hop, and unpacking it;
6. **time the calling thread spent waiting for the GIL.** In async mode the
   worker competes with a 25 Hz control loop that is doing camera reads,
   ``cv2.resize`` and numpy work. The worker cannot return from ``recv()``
   until it holds the GIL again, so control-loop work lands inside the
   measurement even though it has nothing to do with inference.

On top of that, the client keeps a request in flight at all times (the loop
submits every control step, latest-wins), so the server is measured at 100%
duty cycle, never at idle — and the very first call includes JAX compilation,
which can be tens of seconds.

openpi's websocket server already reports what the model actually took, in
``reply["server_timing"]["infer_ms"]``; the client used to throw it away. Keep
both numbers and the difference is everything on the client's side of the wire:

    round trip = server infer + overhead(pack + network + unpack + GIL wait)

A high *server* number is a model or GPU problem. A high *overhead* number on a
LAN is this process — most often GIL contention with the control loop. That is
the split that tells you which one you have.

Pure numpy, no transport: see ``tests/test_latency.py``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Rolling window used for the percentiles. At a typical 3-5 inferences a second,
# 200 samples is roughly the last minute of operation.
DEFAULT_WINDOW = 200


def server_timing(reply: object) -> dict:
    """The server's own timing block, or an empty dict when it sends none.

    openpi's websocket server reports ``infer_ms`` (the model alone) and
    ``prev_total_ms`` (the previous request's whole handler, including sending
    the response). Other backends may report nothing; absent timing is normal,
    never an error.
    """
    if isinstance(reply, Mapping):
        timing = reply.get("server_timing")
        if isinstance(timing, Mapping):
            return dict(timing)
    return {}


class LatencyTracker:
    """Rolling inference timings, split into server time and client overhead.

    Thread-safe enough for this use: samples are appended by the inference
    thread and read by the control thread, and ``deque.append`` with a bounded
    ``maxlen`` is atomic under the GIL. A summary can miss the sample that
    arrives while it is being computed, which does not matter for a status line.
    """

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        self._round_trip_ms: deque[float] = deque(maxlen=window)
        self._server_ms: deque[float] = deque(maxlen=window)
        self._warmup_ms: float | None = None
        self.calls = 0

    def reset(self) -> None:
        self._round_trip_ms.clear()
        self._server_ms.clear()
        self._warmup_ms = None
        self.calls = 0

    def record(self, round_trip_s: float, server_infer_ms: float | None = None) -> None:
        """Add one completed inference.

        The first call is held aside rather than averaged in: it carries JAX
        compilation or model warmup and is not representative of anything the
        control loop will see again.
        """
        self.calls += 1
        round_trip_ms = round_trip_s * 1e3
        if self.calls == 1:
            self._warmup_ms = round_trip_ms
            logger.info(
                "First inference took %.0f ms — model warmup/compilation, excluded from the rolling stats",
                round_trip_ms,
            )
            return
        self._round_trip_ms.append(round_trip_ms)
        if server_infer_ms is not None:
            self._server_ms.append(float(server_infer_ms))
        logger.debug(
            "inference round trip %.1f ms (server %s)",
            round_trip_ms,
            f"{server_infer_ms:.1f} ms" if server_infer_ms is not None else "not reported",
        )

    @property
    def warmup_ms(self) -> float | None:
        """The first call's round trip, which usually includes compilation."""
        return self._warmup_ms

    @property
    def samples(self) -> int:
        return len(self._round_trip_ms)

    def percentiles(self) -> dict[str, float]:
        """p50/p95/max round trip, mean server time and mean overhead, in ms."""
        if not self._round_trip_ms:
            return {}
        round_trip = np.fromiter(self._round_trip_ms, dtype=float)
        stats = {
            "p50_ms": float(np.percentile(round_trip, 50)),
            "p95_ms": float(np.percentile(round_trip, 95)),
            "max_ms": float(round_trip.max()),
        }
        if self._server_ms:
            server = np.fromiter(self._server_ms, dtype=float)
            stats["server_ms"] = float(server.mean())
            # Compared over the same window rather than per-sample: the two
            # deques can differ by the one sample still being appended.
            stats["overhead_ms"] = float(round_trip[-len(server) :].mean() - server.mean())
        return stats

    def summary(self) -> str | None:
        """One line for the status line, or None before there is anything to say."""
        stats = self.percentiles()
        if not stats:
            return None
        text = f"infer p50 {stats['p50_ms']:.0f} p95 {stats['p95_ms']:.0f} ms"
        if "server_ms" in stats:
            text += f" (server {stats['server_ms']:.0f}, overhead {stats['overhead_ms']:.0f})"
        return text
