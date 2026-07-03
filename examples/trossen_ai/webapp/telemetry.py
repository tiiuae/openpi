"""Telemetry seam between the control loop and the web layer.

The bridge/worker call a TelemetrySink. NullSink is the default everywhere, so
the CLI is unaffected. QueueSink serializes events into a thread-safe queue that
the WebSocket drains and forwards to the browser as JSON.
"""
from __future__ import annotations

import base64
import queue
from typing import Protocol

import numpy as np


class TelemetrySink(Protocol):
    def on_log(self, level: str, msg: str, ts: float) -> None: ...
    def on_action(self, step: int, action: np.ndarray, ts: float) -> None: ...
    def on_inference(self, rtt_ms: float, ts: float) -> None: ...
    def on_chunk(self, query_step: int, chunk: np.ndarray, ts: float) -> None: ...
    def on_ee_chunk(self, query_step: int, ee_chunk: np.ndarray, ts: float) -> None: ...
    def on_overlap(self, step: int, count: int) -> None: ...
    def on_weights(self, step: int, weights: np.ndarray, ts: float) -> None: ...
    def on_images(self, images: dict[str, bytes], ts: float) -> None: ...
    def on_status(self, kind: str, payload: dict) -> None: ...


class NullSink:
    """No-op sink. Default for all CLI paths — zero behavior change."""

    def on_log(self, level: str, msg: str, ts: float) -> None: ...
    def on_action(self, step: int, action: np.ndarray, ts: float) -> None: ...
    def on_inference(self, rtt_ms: float, ts: float) -> None: ...
    def on_chunk(self, query_step: int, chunk: np.ndarray, ts: float) -> None: ...
    def on_ee_chunk(self, query_step: int, ee_chunk: np.ndarray, ts: float) -> None: ...
    def on_overlap(self, step: int, count: int) -> None: ...
    def on_weights(self, step: int, weights: np.ndarray, ts: float) -> None: ...
    def on_images(self, images: dict[str, bytes], ts: float) -> None: ...
    def on_status(self, kind: str, payload: dict) -> None: ...


class QueueSink:
    """Serialize telemetry into a queue of JSON-ready dicts.

    Tracks the most recent chunk so each action event can carry the raw (newest)
    prediction for the same step alongside the blended output, for the
    raw-vs-smoothed chart. Image events are throttled by wall-clock interval.
    """

    def __init__(self, q: "queue.Queue", image_min_interval_s: float = 0.1) -> None:
        self._q = q
        self._image_min_interval_s = image_min_interval_s
        self._last_chunk: tuple[int, np.ndarray] | None = None
        self._last_ee_chunk: tuple[int, np.ndarray] | None = None
        self._last_image_ts = float("-inf")

    def _put(self, evt: dict) -> None:
        """Enqueue an event, dropping the oldest if the queue is full.

        A bounded queue + drop-oldest keeps the browser view *current* under a
        fast control loop: if the producer outruns the WebSocket, stale events
        are discarded instead of building a multi-minute backlog that would make
        the chart lag far behind the robot and take ages to drain after Stop.
        """
        try:
            self._q.put_nowait(evt)
        except queue.Full:
            try:
                self._q.get_nowait()  # evict oldest, then retry
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(evt)
            except queue.Full:
                pass

    def on_log(self, level: str, msg: str, ts: float) -> None:
        self._put({"type": "log", "level": level, "msg": msg, "ts": ts})

    def on_action(self, step: int, action: np.ndarray, ts: float) -> None:
        raw = None
        if self._last_chunk is not None:
            qs, chunk = self._last_chunk
            offset = step - qs
            if 0 <= offset < len(chunk):
                raw = np.asarray(chunk[offset]).flatten().tolist()
        self._put({
            "type": "action",
            "step": step,
            "action": np.asarray(action).flatten().tolist(),
            "raw": raw,
            "ts": ts,
        })
        if self._last_ee_chunk is not None:
            _qs, _ee = self._last_ee_chunk
            _off = step - _qs
            if 0 <= _off < len(_ee):
                self._put({"type": "ee", "step": step,
                           "ee": np.asarray(_ee[_off]).flatten().tolist(), "ts": ts})

    def on_inference(self, rtt_ms: float, ts: float) -> None:
        self._put({"type": "inference", "rtt_ms": rtt_ms, "ts": ts})

    def on_chunk(self, query_step: int, chunk: np.ndarray, ts: float) -> None:
        arr = np.asarray(chunk)
        self._last_chunk = (query_step, arr)
        self._put({"type": "chunk", "query_step": query_step, "len": int(len(arr)), "ts": ts})

    def on_ee_chunk(self, query_step: int, ee_chunk: np.ndarray, ts: float) -> None:
        self._last_ee_chunk = (query_step, np.asarray(ee_chunk))

    def on_overlap(self, step: int, count: int) -> None:
        self._put({"type": "overlap", "step": step, "count": count})

    def on_weights(self, step: int, weights: np.ndarray, ts: float) -> None:
        self._put({
            "type": "weights",
            "step": step,
            "weights": np.asarray(weights).flatten().tolist(),
            "ts": ts,
        })

    def on_images(self, images: dict[str, bytes], ts: float) -> None:
        if ts - self._last_image_ts < self._image_min_interval_s:
            return
        self._last_image_ts = ts
        encoded = {k: base64.b64encode(v).decode("ascii") for k, v in images.items()}
        self._put({"type": "images", "images": encoded, "ts": ts})

    def on_status(self, kind: str, payload: dict) -> None:
        self._put({"type": "status", "kind": kind, "payload": payload})
