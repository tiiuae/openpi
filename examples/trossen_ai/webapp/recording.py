# webapp/recording.py
"""Disk recording sinks for Live experiment logging.

TeeSink fans telemetry to several sinks (browser QueueSink + disk RecordingSink).
RecordingSink writes one run directory: manifest.json, events.jsonl, summary.json.
Both are defensive: a failure in one sink never propagates into the control loop.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import numpy as np

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
    def on_ee_chunk(self, query_step, ee_chunk, ts): self._fan("on_ee_chunk", query_step, ee_chunk, ts)
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


class RecordingSink:
    """Record one live run to disk as manifest.json + events.jsonl (+ summary).

    Defensive by construction: any filesystem error disables recording (the sink
    becomes a no-op) and is logged once, so a broken disk never crashes the run.
    Images are intentionally not recorded (see spec). Summary math lives in
    Task 3's close().
    """

    def __init__(self, run_dir, *, run_id: str, config: dict, kind: str = "live") -> None:
        self._run_id = run_id
        self._dir = Path(run_dir)
        self._config = dict(config or {})
        self._kind = kind
        self._disabled = False
        self._fh = None
        self._last_chunk = None  # (query_step, np.ndarray) for raw lookup
        self._last_ee_chunk = None  # (query_step, np.ndarray) for ee lookup
        self._model = None
        # ---- summary accumulators (used in Task 3) ----
        self._rtts = []
        self._overlaps = []
        self._deltas = []
        self._action_ts = []
        self._steps = 0
        self._last_status = None
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._dir / "events.jsonl", "a", encoding="utf-8")
            self._write_manifest()
        except Exception:  # noqa: BLE001
            logger.exception("RecordingSink: init failed for %s; recording disabled", self._dir)
            self._disabled = True

    # ---- internals ----
    def _write_manifest(self) -> None:
        manifest = {
            "run_id": self._run_id,
            "started_at": time.time(),
            "kind": self._kind,
            "config": self._config,
            "model_name": self._config.get("model_name") or None,
            "model": self._model,
        }
        (self._dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _put(self, evt: dict) -> None:
        if self._disabled or self._fh is None:
            return
        try:
            self._fh.write(json.dumps(evt) + "\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001
            logger.exception("RecordingSink: write failed; disabling recording")
            self._disabled = True

    # ---- TelemetrySink ----
    def on_log(self, level, msg, ts):
        self._put({"type": "log", "level": level, "msg": msg, "ts": ts})

    def on_action(self, step, action, ts):
        act = np.asarray(action).flatten().tolist()
        raw = None
        if self._last_chunk is not None:
            qs, chunk = self._last_chunk
            off = step - qs
            if 0 <= off < len(chunk):
                raw = np.asarray(chunk[off]).flatten().tolist()
        if raw is not None:
            self._deltas.append(float(np.linalg.norm(np.array(act) - np.array(raw))))
        self._steps += 1
        self._action_ts.append(float(ts))
        if self._last_ee_chunk is not None:
            _qs, _ee = self._last_ee_chunk
            _off = step - _qs
            if 0 <= _off < len(_ee):
                self._put({"type": "ee", "step": step,
                           "ee": np.asarray(_ee[_off]).flatten().tolist(), "ts": ts})
        self._put({"type": "action", "step": step, "action": act, "raw": raw, "ts": ts})

    def on_inference(self, rtt_ms, ts):
        self._rtts.append(float(rtt_ms))
        self._put({"type": "inference", "rtt_ms": rtt_ms, "ts": ts})

    def on_chunk(self, query_step, chunk, ts):
        arr = np.asarray(chunk)
        self._last_chunk = (query_step, arr)
        self._put({"type": "chunk", "query_step": query_step, "len": int(len(arr)), "ts": ts})

    def on_ee_chunk(self, query_step, ee_chunk, ts):
        self._last_ee_chunk = (query_step, np.asarray(ee_chunk))

    def on_overlap(self, step, count):
        self._overlaps.append(int(count))
        self._put({"type": "overlap", "step": step, "count": count})

    def on_weights(self, step, weights, ts):
        self._put({"type": "weights", "step": step,
                   "weights": np.asarray(weights).flatten().tolist(), "ts": ts})

    def on_images(self, images, ts):
        pass  # images intentionally not recorded

    def on_status(self, kind, payload):
        self._last_status = (kind, dict(payload or {}))
        if kind == "model":
            self._model = payload
            if not self._disabled:
                try:
                    self._write_manifest()
                except Exception:  # noqa: BLE001
                    logger.exception("RecordingSink: manifest rewrite failed")
        self._put({"type": "status", "kind": kind, "payload": payload})

    def ingest_event(self, event: dict) -> None:
        """Record one already-serialized QueueSink event from Laptop B.

        Remote gateway telemetry already contains the raw action paired with each
        blended action, so ingest it directly instead of trying to reconstruct a
        full prediction chunk on Machine A.  Camera previews remain intentionally
        excluded, matching ``on_images``.
        """

        event_type = event.get("type")
        if event_type in {"gateway", "gateway_state", "metrics", "images"}:
            return
        if event_type == "action":
            action = np.asarray(event.get("action", []), dtype=float).flatten()
            raw_value = event.get("raw")
            if raw_value is not None:
                raw = np.asarray(raw_value, dtype=float).flatten()
                if raw.shape == action.shape:
                    self._deltas.append(float(np.linalg.norm(action - raw)))
            self._steps += 1
            try:
                self._action_ts.append(float(event["ts"]))
            except (KeyError, TypeError, ValueError):
                pass
        elif event_type == "inference":
            try:
                self._rtts.append(float(event["rtt_ms"]))
            except (KeyError, TypeError, ValueError):
                pass
        elif event_type == "overlap":
            try:
                self._overlaps.append(int(event["count"]))
            except (KeyError, TypeError, ValueError):
                pass
        elif event_type == "status":
            kind = str(event.get("kind", "unknown"))
            payload = event.get("payload")
            payload = dict(payload) if isinstance(payload, dict) else {}
            self._last_status = (kind, payload)
            if kind == "model":
                self._model = payload
                if not self._disabled:
                    try:
                        self._write_manifest()
                    except Exception:  # noqa: BLE001
                        logger.exception("RecordingSink: manifest rewrite failed")
        self._put(dict(event))

    def _end_reason(self) -> str:
        if self._last_status is None:
            return "unknown"
        kind, payload = self._last_status
        if kind in ("error", "connect_failed"):
            return "error"
        reason = str(payload.get("reason", "")).lower()
        if "estop" in reason:
            return "estop"
        if "cancel" in reason:
            return "stopped"
        if "finish" in reason:
            return "completed"
        return "unknown"

    def _pct(self, vals, p):
        return float(np.percentile(vals, p)) if vals else None

    def _loop_hz(self):
        if len(self._action_ts) < 2:
            return None
        span = self._action_ts[-1] - self._action_ts[0]
        return (len(self._action_ts) - 1) / span if span > 0 else None

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._fh = None
        if self._disabled:
            return
        span = (self._action_ts[-1] - self._action_ts[0]) if len(self._action_ts) >= 2 else 0.0
        summary = {
            "run_id": self._run_id,
            "ended_at": time.time(),
            "duration_s": round(span, 6),
            "steps": self._steps,
            "end_reason": self._end_reason(),
            "rtt_ms": {"mean": (sum(self._rtts) / len(self._rtts)) if self._rtts else None,
                       "p50": self._pct(self._rtts, 50),
                       "p95": self._pct(self._rtts, 95)},
            "loop_hz": self._loop_hz(),
            "overlap_mean": (sum(self._overlaps) / len(self._overlaps)) if self._overlaps else None,
            "smoothing_delta_mean": (sum(self._deltas) / len(self._deltas)) if self._deltas else None,
        }
        try:
            (self._dir / "summary.json").write_text(json.dumps(summary, indent=2))
        except Exception:  # noqa: BLE001
            logger.exception("RecordingSink: summary write failed")
