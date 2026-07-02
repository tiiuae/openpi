# webapp/recording.py
"""Disk recording sinks for Live experiment logging.

TeeSink fans telemetry to several sinks (browser QueueSink + disk RecordingSink).
RecordingSink writes one run directory: manifest.json, events.jsonl, summary.json.
Both are defensive: a failure in one sink never propagates into the control loop.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TeeSink:
    """Forward every TelemetrySink callback to each inner sink.

    A raise from one inner sink is caught and logged so one bad sink cannot
    starve the others (e.g. a disk-full RecordingSink must not kill the browser
    QueueSink). On close(), inner sinks that define close() are closed.
    """

    def __init__(self, sinks) -> None:
        self._sinks = list(sinks)

    def _fan(self, method: str, *args) -> None:
        for s in self._sinks:
            try:
                getattr(s, method)(*args)
            except Exception:  # noqa: BLE001 — never break the loop over one sink
                logger.exception("TeeSink: inner sink %s.%s failed", type(s).__name__, method)

    def on_log(self, level, msg, ts): self._fan("on_log", level, msg, ts)
    def on_action(self, step, action, ts): self._fan("on_action", step, action, ts)
    def on_inference(self, rtt_ms, ts): self._fan("on_inference", rtt_ms, ts)
    def on_chunk(self, query_step, chunk, ts): self._fan("on_chunk", query_step, chunk, ts)
    def on_overlap(self, step, count): self._fan("on_overlap", step, count)
    def on_weights(self, step, weights, ts): self._fan("on_weights", step, weights, ts)
    def on_images(self, images, ts): self._fan("on_images", images, ts)
    def on_status(self, kind, payload): self._fan("on_status", kind, payload)

    def close(self) -> None:
        for s in self._sinks:
            close = getattr(s, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.exception("TeeSink: inner sink %s.close failed", type(s).__name__)
