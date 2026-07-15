"""Forward Python logging records into a TelemetrySink.on_log."""
from __future__ import annotations

import logging
import time

from webapp.telemetry import TelemetrySink


class SinkLogHandler(logging.Handler):
    def __init__(self, sink: TelemetrySink) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink.on_log(record.levelname, record.getMessage(), time.time())
        except Exception:  # noqa: BLE001 — telemetry must never break the loop
            pass
