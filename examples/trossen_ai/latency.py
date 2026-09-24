"""Inference timing that says which part was slow.

Timing a policy call by wrapping ``client.infer()`` in ``perf_counter()`` gives
one number covering six different things, which is why that number looks
erratic. This module measures them apart.

``WebsocketClientPolicy.infer()`` does three things in sequence, and they have
completely different characters:

======  ==========================  =========================================
phase   what happens                why it can be slow
======  ==========================  =========================================
pack    msgpack the observation     ~440 KB of images, on the calling thread,
                                    holding the GIL
wire    send, wait, receive         the network, the server queue, and the
                                    model — the GIL is released here
unpack  msgpack the reply           small, on the calling thread, GIL again
======  ==========================  =========================================

The server reports what the model alone took, in
``reply["server_timing"]["infer_ms"]``, so the wire phase splits further:

    wire = server infer + network

Read the result like this:

* high **server** — a model or GPU problem, nothing the client can fix;
* high **network** — the link, or requests queueing on a busy server;
* high **pack/unpack** — this process is CPU-starved, which on a 25 Hz control
  loop usually means GIL contention with the control thread.

A note on that last row, since it is the popular suspect: packing an
observation is close to a memcpy for numpy arrays, and measures well under a
millisecond for the three-camera payload. So a large *pack* number is a strong
signal and a small one rules the client out, which is exactly what makes the
split worth having — it names the contributor instead of assuming one.

``SchedulingLagProbe`` measures interpreter starvation directly rather than
leaving it to be inferred: a thread that sleeps for a known interval and
reports how late it actually woke. It is a property of this process, not of the
policy server, so it separates "the server is slow" from "we were not running".

Two more things distort a naive average. The control loop submits an
observation every control step (latest-wins), so a request is in flight at all
times and the server is only ever measured at 100% duty cycle. And the first
call carries model warmup or JAX compilation, which can be tens of seconds and
would then sit in the average for the rest of the session; it is recorded
separately.

Pure numpy and stdlib, no transport: see ``tests/test_latency.py``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import logging
import threading
import time

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


@dataclass(frozen=True)
class InferenceTiming:
    """One inference, split by where the time went. Milliseconds."""

    pack_ms: float
    wire_ms: float
    unpack_ms: float
    server_ms: float | None = None

    @property
    def round_trip_ms(self) -> float:
        """What a naive ``perf_counter()`` around ``infer()`` would have shown."""
        return self.pack_ms + self.wire_ms + self.unpack_ms

    @property
    def client_ms(self) -> float:
        """Time spent on this thread, holding the GIL, serializing."""
        return self.pack_ms + self.unpack_ms

    @property
    def network_ms(self) -> float | None:
        """Wire time the server did not account for: transit and queueing.

        None when the backend reports no timing. Clamped at zero because the
        two clocks are different machines' and can disagree by a hair.
        """
        if self.server_ms is None:
            return None
        return max(self.wire_ms - self.server_ms, 0.0)


class SchedulingLagProbe:
    """How late a thread wakes from a short sleep, as a proxy for GIL contention.

    A background thread sleeps for ``interval`` and records how much longer than
    that it actually took to get going again. On an idle interpreter the lag is
    well under a millisecond. When the 25 Hz control loop is holding the GIL for
    long stretches (camera reads, ``cv2.resize``, numpy on big arrays) every
    other thread wakes late by that much, including the one waiting on the
    policy server's reply.

    This is what turns "inference looks slow" into "inference looks slow because
    this process is starved", without having to guess.
    """

    def __init__(self, interval: float = 0.05, window: int = DEFAULT_WINDOW) -> None:
        self._interval = interval
        self._lags_ms: deque[float] = deque(maxlen=window)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="scheduling-lag-probe")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            before = time.perf_counter()
            self._stop.wait(self._interval)
            overshoot = (time.perf_counter() - before) - self._interval
            self._lags_ms.append(max(overshoot, 0.0) * 1e3)

    def record(self, lag_ms: float) -> None:
        """Add a sample directly. For tests, and for callers with their own timer."""
        self._lags_ms.append(lag_ms)

    @property
    def samples(self) -> int:
        return len(self._lags_ms)

    def p95_ms(self) -> float | None:
        if not self._lags_ms:
            return None
        return float(np.percentile(np.fromiter(self._lags_ms, dtype=float), 95))


class LatencyTracker:
    """Rolling inference timings, split by phase.

    Thread-safe enough for this use: samples are appended by the inference
    thread and read by the control thread, and ``deque.append`` with a bounded
    ``maxlen`` is atomic under the GIL. A summary can miss the sample that
    arrives while it is being computed, which does not matter for a status line.
    """

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        self._window = window
        self._round_trip_ms: deque[float] = deque(maxlen=window)
        self._pack_ms: deque[float] = deque(maxlen=window)
        self._wire_ms: deque[float] = deque(maxlen=window)
        self._unpack_ms: deque[float] = deque(maxlen=window)
        self._server_ms: deque[float] = deque(maxlen=window)
        self._network_ms: deque[float] = deque(maxlen=window)
        self._warmup: InferenceTiming | None = None
        self.calls = 0
        self.lag = SchedulingLagProbe(window=window)

    def reset(self) -> None:
        for samples in self._all_series().values():
            samples.clear()
        self._warmup = None
        self.calls = 0

    def _all_series(self) -> dict[str, deque]:
        return {
            "round_trip": self._round_trip_ms,
            "pack": self._pack_ms,
            "wire": self._wire_ms,
            "unpack": self._unpack_ms,
            "server": self._server_ms,
            "network": self._network_ms,
        }

    def record(self, timing: InferenceTiming) -> None:
        """Add one completed inference.

        The first call is held aside rather than averaged in: it carries model
        warmup or JAX compilation and is not representative of anything the
        control loop will see again.
        """
        self.calls += 1
        if self.calls == 1:
            self._warmup = timing
            logger.info(
                "First inference took %.0f ms (pack %.0f, wire %.0f, unpack %.0f) — model warmup, "
                "excluded from the rolling stats",
                timing.round_trip_ms,
                timing.pack_ms,
                timing.wire_ms,
                timing.unpack_ms,
            )
            return

        self._round_trip_ms.append(timing.round_trip_ms)
        self._pack_ms.append(timing.pack_ms)
        self._wire_ms.append(timing.wire_ms)
        self._unpack_ms.append(timing.unpack_ms)
        if timing.server_ms is not None:
            self._server_ms.append(timing.server_ms)
        if timing.network_ms is not None:
            self._network_ms.append(timing.network_ms)

        logger.debug(
            "inference %.1f ms = pack %.1f + wire %.1f + unpack %.1f (server %s)",
            timing.round_trip_ms,
            timing.pack_ms,
            timing.wire_ms,
            timing.unpack_ms,
            f"{timing.server_ms:.1f}" if timing.server_ms is not None else "not reported",
        )

    @property
    def warmup(self) -> InferenceTiming | None:
        """The first call, which usually includes compilation."""
        return self._warmup

    @property
    def samples(self) -> int:
        return len(self._round_trip_ms)

    def percentiles(self) -> dict[str, float]:
        """Round-trip p50/p95/max plus the mean of each phase, in milliseconds."""
        if not self._round_trip_ms:
            return {}
        round_trip = np.fromiter(self._round_trip_ms, dtype=float)
        stats = {
            "p50_ms": float(np.percentile(round_trip, 50)),
            "p95_ms": float(np.percentile(round_trip, 95)),
            "max_ms": float(round_trip.max()),
        }
        for name, samples in self._all_series().items():
            if name != "round_trip" and samples:
                stats[f"{name}_ms"] = float(np.fromiter(samples, dtype=float).mean())
        lag = self.lag.p95_ms()
        if lag is not None:
            stats["sched_lag_ms"] = lag
        return stats

    def summary(self) -> str | None:
        """One line for the status line, or None before there is anything to say."""
        stats = self.percentiles()
        if not stats:
            return None

        text = f"infer p50 {stats['p50_ms']:.0f} p95 {stats['p95_ms']:.0f} ms"
        parts = []
        if "server_ms" in stats:
            parts.append(f"server {stats['server_ms']:.0f}")
        if "network_ms" in stats:
            parts.append(f"net {stats['network_ms']:.0f}")
        client_ms = stats.get("pack_ms", 0.0) + stats.get("unpack_ms", 0.0)
        if client_ms:
            parts.append(f"pack {client_ms:.0f}")
        if parts:
            text += " (" + ", ".join(parts) + ")"
        if "sched_lag_ms" in stats:
            text += f" · gil lag {stats['sched_lag_ms']:.0f} ms"
        return text
