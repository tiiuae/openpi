"""Rolling telemetry metrics fed from the event stream.

All windows are bounded deques; snapshot() returns a JSON-ready dict the server
emits at ~2 Hz.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class Metrics:
    def __init__(self, window: int = 200) -> None:
        self._rtt = deque(maxlen=window)
        self._action_ts = deque(maxlen=window)
        self._actions = deque(maxlen=window)
        self._drops = 0

    def add_rtt(self, rtt_ms: float) -> None:
        self._rtt.append(float(rtt_ms))

    def add_action(self, ts: float, action) -> None:
        self._action_ts.append(float(ts))
        self._actions.append(np.asarray(action, dtype=float).flatten())

    def add_drop(self) -> None:
        self._drops += 1

    def _loop_hz(self) -> float | None:
        if len(self._action_ts) < 2:
            return None
        span = self._action_ts[-1] - self._action_ts[0]
        if span <= 0:
            return None
        return (len(self._action_ts) - 1) / span

    def _jitter(self) -> float | None:
        if len(self._actions) < 2:
            return None
        deltas = [float(np.linalg.norm(self._actions[i] - self._actions[i - 1]))
                  for i in range(1, len(self._actions))]
        return float(np.std(deltas))

    def snapshot(self) -> dict:
        rtt = list(self._rtt)
        return {
            "rtt_last": rtt[-1] if rtt else None,
            "rtt_p50": float(np.percentile(rtt, 50)) if rtt else None,
            "rtt_p95": float(np.percentile(rtt, 95)) if rtt else None,
            "loop_hz": self._loop_hz(),
            "jitter": self._jitter(),
            "drops": self._drops,
        }
