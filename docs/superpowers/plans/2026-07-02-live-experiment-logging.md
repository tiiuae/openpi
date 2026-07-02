# Live Experiment Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record each Live (autonomous) webapp run to disk (config, model name, action/smoothing/metrics timeseries, outcome rating) and add a compare view to diff runs as evaluation experiments.

**Architecture:** Recording attaches at the existing `TelemetrySink` seam. A new `TeeSink` fans every telemetry event to the existing `QueueSink` (browser) and a new `RecordingSink` (disk). Each live run writes a `runs/<run_id>/` directory (`manifest.json`, `events.jsonl`, `summary.json`, `rating.json`). A `RunStore` + REST endpoints read them back; a new `runs.html` compare view diffs configs and overlays metric charts.

**Tech Stack:** Python 3.13, FastAPI, pytest (`webapp/tests/`), numpy; vanilla ES-module JS + Chart.js (`webapp/static/js/`).

**Spec:** `docs/superpowers/specs/2026-07-02-live-experiment-logging-design.md`

**How to run tests:** from `examples/trossen_ai/`:
`python -m pytest webapp/tests/<file> -v`
**How to check JS:** from `examples/trossen_ai/webapp/static/js/`: `node --check <file>.js`

---

## File Structure

**Create:**
- `webapp/recording.py` — `TeeSink` + `RecordingSink` (disk recorder).
- `webapp/run_store.py` — `RunStore`: list/read runs, write rating, from a runs dir.
- `webapp/tests/test_recording.py` — TeeSink + RecordingSink unit tests.
- `webapp/tests/test_run_store.py` — RunStore unit tests.
- `webapp/static/runs.html` — compare view page.
- `webapp/static/js/runs.js` — compare view logic (list, config diff, overlay charts).
- `webapp/static/js/runlog.js` — end-of-run quick-rating modal.

**Modify:**
- `webapp/server.py` — `create_app(runs_dir=...)`, build TeeSink for live runs, `run_started` status, finalize on session end, run-store REST endpoints, `/runs` page route.
- `webapp/session.py` — call `sink.close()` in `_run`'s `finally` if present.
- `webapp/runners.py` — `LiveRunner.run` emits `on_status("model", metadata)` after bridge build.
- `webapp/static/js/config.js` — add `model_name` Policy field.
- `webapp/static/js/live.js` — import + wire `runlog.js` (remember run_id, show rating on end).
- `webapp/static/index.html` — load compare-runs link.
- `webapp/static/theme.css` — styles for the rating modal + compare view.

---

## Phase 1 — Backend recording

### Task 1: TeeSink

**Files:**
- Create: `webapp/recording.py`
- Test: `webapp/tests/test_recording.py`

- [ ] **Step 1: Write the failing test**

```python
# webapp/tests/test_recording.py
import numpy as np

from webapp.recording import TeeSink


class _Spy:
    def __init__(self):
        self.calls = []
    def on_log(self, level, msg, ts): self.calls.append(("log", level, msg, ts))
    def on_action(self, step, action, ts): self.calls.append(("action", step, ts))
    def on_inference(self, rtt_ms, ts): self.calls.append(("inference", rtt_ms, ts))
    def on_chunk(self, query_step, chunk, ts): self.calls.append(("chunk", query_step, ts))
    def on_overlap(self, step, count): self.calls.append(("overlap", step, count))
    def on_weights(self, step, weights, ts): self.calls.append(("weights", step, ts))
    def on_images(self, images, ts): self.calls.append(("images", ts))
    def on_status(self, kind, payload): self.calls.append(("status", kind))


class _Boom(_Spy):
    def on_overlap(self, step, count):
        raise RuntimeError("inner sink failed")


def test_tee_fans_out_to_all_sinks():
    a, b = _Spy(), _Spy()
    tee = TeeSink([a, b])
    tee.on_action(3, np.zeros(14), 1.0)
    tee.on_overlap(3, 2)
    tee.on_status("started", {})
    assert ("action", 3, 1.0) in a.calls and ("action", 3, 1.0) in b.calls
    assert ("overlap", 3, 2) in a.calls and ("overlap", 3, 2) in b.calls
    assert ("status", "started") in a.calls and ("status", "started") in b.calls


def test_tee_one_raising_sink_does_not_starve_others():
    boom, ok = _Boom(), _Spy()
    tee = TeeSink([boom, ok])
    tee.on_overlap(1, 1)  # boom raises internally; must not propagate
    assert ("overlap", 1, 1) in ok.calls


def test_tee_close_calls_close_on_inner_sinks_that_have_it():
    closed = []

    class _Closable(_Spy):
        def close(self): closed.append(True)

    tee = TeeSink([_Closable(), _Spy()])  # second has no close()
    tee.close()  # must not raise despite one sink lacking close()
    assert closed == [True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp/tests/test_recording.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp.recording'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp/tests/test_recording.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add webapp/recording.py webapp/tests/test_recording.py
git commit -m "feat(trossen_ai): TeeSink fans telemetry to multiple sinks"
```

---

### Task 2: RecordingSink — manifest + events.jsonl

**Files:**
- Modify: `webapp/recording.py`
- Test: `webapp/tests/test_recording.py`

- [ ] **Step 1: Write the failing test**

```python
# add to webapp/tests/test_recording.py
import json
from pathlib import Path

from webapp.recording import RecordingSink


def _read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def test_recording_writes_manifest_on_construction(tmp_path):
    run_dir = tmp_path / "2026-07-02-181500"
    RecordingSink(run_dir, run_id="2026-07-02-181500",
                  config={"model_name": "pi0-v3", "control_freq": 25})
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == "2026-07-02-181500"
    assert manifest["kind"] == "live"
    assert manifest["config"]["control_freq"] == 25
    assert manifest["model_name"] == "pi0-v3"     # copied from config
    assert manifest["model"] is None              # no server metadata yet
    assert "started_at" in manifest


def test_recording_appends_events_to_jsonl(tmp_path):
    import numpy as np
    run_dir = tmp_path / "r"
    sink = RecordingSink(run_dir, run_id="r", config={})
    sink.on_action(0, np.array([1.0, 2.0]), 1.0)
    sink.on_overlap(0, 3)
    sink.on_weights(0, np.array([0.6, 0.4]), 1.0)
    rows = _read_jsonl(run_dir / "events.jsonl")
    assert rows[0] == {"type": "action", "step": 0, "action": [1.0, 2.0], "raw": None, "ts": 1.0}
    assert {"type": "overlap", "step": 0, "count": 3} in rows
    assert {"type": "weights", "step": 0, "weights": [0.6, 0.4], "ts": 1.0} in rows


def test_recording_does_not_record_images(tmp_path):
    run_dir = tmp_path / "r"
    sink = RecordingSink(run_dir, run_id="r", config={})
    sink.on_images({"cam": b"xxx"}, 1.0)
    rows = _read_jsonl(run_dir / "events.jsonl")
    assert all(r["type"] != "images" for r in rows)


def test_recording_action_raw_from_latest_chunk(tmp_path):
    import numpy as np
    run_dir = tmp_path / "r"
    sink = RecordingSink(run_dir, run_id="r", config={})
    sink.on_chunk(5, np.array([[10.0], [11.0], [12.0]]), 1.0)
    sink.on_action(6, np.array([99.0]), 1.1)
    rows = _read_jsonl(run_dir / "events.jsonl")
    action = next(r for r in rows if r["type"] == "action")
    assert action["raw"] == [11.0]  # chunk[6-5]


def test_recording_folds_model_metadata_into_manifest(tmp_path):
    run_dir = tmp_path / "r"
    sink = RecordingSink(run_dir, run_id="r", config={"model_name": "user-typed"})
    sink.on_status("model", {"policy": "pi0", "ckpt": "step_40000"})
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["model"] == {"policy": "pi0", "ckpt": "step_40000"}
    assert manifest["model_name"] == "user-typed"  # user field stays authoritative


def test_recording_disk_error_degrades_to_noop(tmp_path):
    # Point the run dir at a path whose parent is a file -> mkdir fails.
    bad_parent = tmp_path / "afile"
    bad_parent.write_text("x")
    sink = RecordingSink(bad_parent / "run", run_id="run", config={})  # must not raise
    sink.on_overlap(0, 1)  # must not raise
    sink.close()           # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp/tests/test_recording.py -v -k recording`
Expected: FAIL — `ImportError: cannot import name 'RecordingSink'`

- [ ] **Step 3: Write minimal implementation**

Add to `webapp/recording.py`:

```python
import json
import time
from pathlib import Path

import numpy as np


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
        self._put({"type": "action", "step": step, "action": act, "raw": raw, "ts": ts})

    def on_inference(self, rtt_ms, ts):
        self._rtts.append(float(rtt_ms))
        self._put({"type": "inference", "rtt_ms": rtt_ms, "ts": ts})

    def on_chunk(self, query_step, chunk, ts):
        arr = np.asarray(chunk)
        self._last_chunk = (query_step, arr)
        self._put({"type": "chunk", "query_step": query_step, "len": int(len(arr)), "ts": ts})

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp/tests/test_recording.py -v -k recording`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webapp/recording.py webapp/tests/test_recording.py
git commit -m "feat(trossen_ai): RecordingSink writes manifest + events.jsonl"
```

---

### Task 3: RecordingSink — summary on close()

**Files:**
- Modify: `webapp/recording.py`
- Test: `webapp/tests/test_recording.py`

- [ ] **Step 1: Write the failing test**

```python
# add to webapp/tests/test_recording.py
def test_recording_close_writes_summary(tmp_path):
    import numpy as np
    run_dir = tmp_path / "r"
    sink = RecordingSink(run_dir, run_id="r", config={})
    sink.on_chunk(0, np.array([[0.0], [0.0]]), 1.0)
    sink.on_action(0, np.array([0.0]), 1.0)   # raw=0 -> delta 0
    sink.on_action(1, np.array([1.0]), 1.5)   # raw=0 -> delta 1
    sink.on_inference(40.0, 1.0)
    sink.on_inference(60.0, 1.5)
    sink.on_overlap(0, 2)
    sink.on_overlap(1, 4)
    sink.on_status("stopped", {"reason": "finished"})
    sink.close()
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["steps"] == 2
    assert s["end_reason"] == "completed"
    assert s["overlap_mean"] == 3.0
    assert abs(s["smoothing_delta_mean"] - 0.5) < 1e-9
    assert abs(s["rtt_ms"]["mean"] - 50.0) < 1e-9
    assert s["duration_s"] == 0.5


def test_recording_end_reason_mapping(tmp_path):
    cases = {
        ("stopped", "finished"): "completed",
        ("stopped", "estop"): "estop",
        ("stopped", "cancelled"): "stopped",
        ("error", None): "error",
        ("connect_failed", None): "error",
    }
    for (kind, reason), expected in cases.items():
        run_dir = tmp_path / f"{kind}-{reason}"
        sink = RecordingSink(run_dir, run_id="r", config={})
        payload = {"reason": reason} if reason else {"message": "x"}
        sink.on_status(kind, payload)
        sink.close()
        s = json.loads((run_dir / "summary.json").read_text())
        assert s["end_reason"] == expected, (kind, reason)


def test_recording_close_without_status_is_unknown(tmp_path):
    run_dir = tmp_path / "r"
    sink = RecordingSink(run_dir, run_id="r", config={})
    sink.close()
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["end_reason"] == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp/tests/test_recording.py -v -k "close or end_reason"`
Expected: FAIL — `AttributeError: 'RecordingSink' object has no attribute 'close'`

- [ ] **Step 3: Write minimal implementation**

Add these methods to `RecordingSink` in `webapp/recording.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp/tests/test_recording.py -v`
Expected: PASS (all recording tests)

- [ ] **Step 5: Commit**

```bash
git add webapp/recording.py webapp/tests/test_recording.py
git commit -m "feat(trossen_ai): RecordingSink summary + end-reason on close"
```

---

### Task 4: RunStore

**Files:**
- Create: `webapp/run_store.py`
- Test: `webapp/tests/test_run_store.py`

- [ ] **Step 1: Write the failing test**

```python
# webapp/tests/test_run_store.py
import json
from pathlib import Path

from webapp.run_store import RunStore


def _make_run(base: Path, run_id: str, *, model_name="m", end_reason="completed",
              extra_events=None):
    d = base / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "started_at": 1.0, "kind": "live",
        "config": {"model_name": model_name, "smoothing": True}, "model_name": model_name,
        "model": None}))
    (d / "summary.json").write_text(json.dumps({
        "run_id": run_id, "steps": 10, "end_reason": end_reason,
        "rtt_ms": {"mean": 40.0}, "overlap_mean": 3.0, "smoothing_delta_mean": 0.1}))
    events = extra_events or [{"type": "overlap", "step": 0, "count": 2}]
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    return d


def test_list_returns_summaries_newest_first(tmp_path):
    _make_run(tmp_path, "2026-07-02-100000", model_name="old")
    _make_run(tmp_path, "2026-07-02-120000", model_name="new")
    store = RunStore(tmp_path)
    runs = store.list()
    assert [r["run_id"] for r in runs] == ["2026-07-02-120000", "2026-07-02-100000"]
    assert runs[0]["model_name"] == "new"
    assert runs[0]["steps"] == 10
    assert runs[0]["end_reason"] == "completed"


def test_list_tolerates_run_without_summary(tmp_path):
    d = tmp_path / "half"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"run_id": "half", "started_at": 1.0,
                                                 "config": {}, "model_name": None}))
    store = RunStore(tmp_path)
    runs = store.list()
    assert runs[0]["run_id"] == "half"
    assert runs[0]["end_reason"] == "unknown"


def test_get_returns_manifest_summary_rating(tmp_path):
    _make_run(tmp_path, "r1")
    store = RunStore(tmp_path)
    got = store.get("r1")
    assert got["manifest"]["run_id"] == "r1"
    assert got["summary"]["steps"] == 10
    assert got["rating"] is None
    assert "events" not in got  # events only when requested


def test_get_with_events(tmp_path):
    _make_run(tmp_path, "r1", extra_events=[
        {"type": "overlap", "step": 0, "count": 2},
        {"type": "action", "step": 0, "action": [1.0], "raw": [0.0], "ts": 1.0}])
    store = RunStore(tmp_path)
    got = store.get("r1", with_events=True)
    assert len(got["events"]) == 2


def test_set_rating_writes_file(tmp_path):
    _make_run(tmp_path, "r1")
    store = RunStore(tmp_path)
    store.set_rating("r1", {"success": "partial", "score": 3, "note": "ok"})
    saved = json.loads((tmp_path / "r1" / "rating.json").read_text())
    assert saved["success"] == "partial"
    assert saved["score"] == 3
    assert "rated_at" in saved
    assert store.get("r1")["rating"]["note"] == "ok"


def test_get_missing_run_raises_keyerror(tmp_path):
    store = RunStore(tmp_path)
    try:
        store.get("nope")
    except KeyError:
        return
    assert False, "expected KeyError"


def test_run_id_traversal_rejected(tmp_path):
    store = RunStore(tmp_path)
    for bad in ("../secret", "a/b", ".."):
        try:
            store.get(bad)
        except (KeyError, ValueError):
            continue
        assert False, f"expected rejection for {bad!r}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp/tests/test_run_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp.run_store'`

- [ ] **Step 3: Write minimal implementation**

```python
# webapp/run_store.py
"""Read/write access to recorded Live runs under a runs directory."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class RunStore:
    def __init__(self, runs_dir) -> None:
        self._dir = Path(runs_dir)

    def _run_path(self, run_id: str) -> Path:
        # Reject path traversal / nested ids; run dirs are flat single segments.
        if not _RUN_ID_RE.match(run_id or ""):
            raise ValueError(f"invalid run id: {run_id!r}")
        return self._dir / run_id

    @staticmethod
    def _read_json(path: Path):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def list(self) -> list[dict]:
        if not self._dir.is_dir():
            return []
        rows = []
        for d in self._dir.iterdir():
            if not d.is_dir():
                continue
            manifest = self._read_json(d / "manifest.json")
            if manifest is None:
                continue
            summary = self._read_json(d / "summary.json") or {}
            rating = self._read_json(d / "rating.json")
            rows.append({
                "run_id": manifest.get("run_id", d.name),
                "started_at": manifest.get("started_at"),
                "model_name": manifest.get("model_name"),
                "config": manifest.get("config", {}),
                "steps": summary.get("steps"),
                "end_reason": summary.get("end_reason", "unknown"),
                "rtt_mean": (summary.get("rtt_ms") or {}).get("mean"),
                "overlap_mean": summary.get("overlap_mean"),
                "smoothing_delta_mean": summary.get("smoothing_delta_mean"),
                "rating": rating,
            })
        rows.sort(key=lambda r: r["run_id"], reverse=True)
        return rows

    def get(self, run_id: str, with_events: bool = False) -> dict:
        d = self._run_path(run_id)
        manifest = self._read_json(d / "manifest.json")
        if manifest is None:
            raise KeyError(run_id)
        out = {
            "manifest": manifest,
            "summary": self._read_json(d / "summary.json"),
            "rating": self._read_json(d / "rating.json"),
        }
        if with_events:
            out["events"] = self._read_events(d / "events.jsonl")
        return out

    @staticmethod
    def _read_events(path: Path) -> list[dict]:
        try:
            text = path.read_text()
        except OSError:
            return []
        events = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a crash mid-write
        return events

    def set_rating(self, run_id: str, rating: dict) -> None:
        d = self._run_path(run_id)
        if not (d / "manifest.json").exists():
            raise KeyError(run_id)
        payload = dict(rating or {})
        payload["rated_at"] = time.time()
        (d / "rating.json").write_text(json.dumps(payload, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp/tests/test_run_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webapp/run_store.py webapp/tests/test_run_store.py
git commit -m "feat(trossen_ai): RunStore reads runs + writes ratings"
```

---

## Phase 2 — Wiring

### Task 5: Finalize sink on session end

**Files:**
- Modify: `webapp/session.py:78-80` (the `finally` block in `_run`)
- Test: `webapp/tests/test_session.py`

- [ ] **Step 1: Write the failing test**

```python
# add to webapp/tests/test_session.py
from webapp.session import SessionManager


def test_session_calls_sink_close_on_finish():
    closed = []

    class _Sink:
        def on_status(self, kind, payload): pass
        def close(self): closed.append(True)

    class _Runner:
        def __init__(self, kind, config, sink): pass
        def run(self): pass
        def stop(self): pass
        def estop(self): pass

    mgr = SessionManager(lambda k, c, s: _Runner(k, c, s))
    sink = _Sink()
    mgr.start("live", {}, sink)
    mgr._thread.join(timeout=5.0)
    assert closed == [True]


def test_session_close_called_even_when_runner_raises():
    closed = []

    class _Sink:
        def on_status(self, kind, payload): pass
        def close(self): closed.append(True)

    class _Boom:
        def __init__(self, kind, config, sink): pass
        def run(self): raise RuntimeError("crash")
        def stop(self): pass
        def estop(self): pass

    mgr = SessionManager(lambda k, c, s: _Boom(k, c, s))
    mgr.start("live", {}, _Sink())
    mgr._thread.join(timeout=5.0)
    assert closed == [True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp/tests/test_session.py -v -k close`
Expected: FAIL — `AssertionError: assert [] == [True]`

- [ ] **Step 3: Write minimal implementation**

In `webapp/session.py`, change the `finally` block in `_run` (currently lines 78-80):

```python
        finally:
            # Flush any recording sink so summary.json is written even on crash
            # / estop. Sinks without close() (NullSink, QueueSink) are skipped.
            close = getattr(sink, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.exception("Session %s: sink close failed", kind)
            root.removeHandler(handler)
            root.setLevel(prev_level)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp/tests/test_session.py -v`
Expected: PASS (all session tests, no regressions)

- [ ] **Step 5: Commit**

```bash
git add webapp/session.py webapp/tests/test_session.py
git commit -m "feat(trossen_ai): finalize recording sink on session end"
```

---

### Task 6: Build TeeSink + RecordingSink for live runs in server

**Files:**
- Modify: `webapp/server.py` — `create_app` signature (line 31-36), `start()` (lines 188-195)
- Test: `webapp/tests/test_server.py`

- [ ] **Step 1: Write the failing test**

```python
# add to webapp/tests/test_server.py
import json as _json


def test_live_run_records_to_runs_dir(tmp_path):
    class FakeRunner:
        def __init__(self, kind, config, sink):
            self._sink = sink
        def run(self):
            import numpy as np
            self._sink.on_status("started", {})
            self._sink.on_action(0, np.array([1.0]), 1.0)
            self._sink.on_overlap(0, 2)
            self._sink.on_status("stopped", {"reason": "finished"})
        def stop(self): pass
        def estop(self): pass

    app = create_app(runs_dir=tmp_path, runner_factory=lambda k, c, s: FakeRunner(k, c, s))
    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.send_json({"action": "start_live", "config": {"model_name": "pi0-test"}})
        run_id = None
        for _ in range(40):
            evt = ws.receive_json()
            if evt.get("type") == "status" and evt.get("kind") == "run_started":
                run_id = evt["payload"]["run_id"]
            if evt.get("type") == "status" and evt.get("kind") == "stopped":
                break
        assert run_id is not None
    # RecordingSink wrote a run dir with manifest + events (+ summary via close()).
    run_dir = tmp_path / run_id
    manifest = _json.loads((run_dir / "manifest.json").read_text())
    assert manifest["model_name"] == "pi0-test"
    assert (run_dir / "events.jsonl").read_text().strip() != ""


def test_non_live_run_does_not_record(tmp_path):
    class FakeRunner:
        def __init__(self, kind, config, sink): self._sink = sink
        def run(self): self._sink.on_status("stopped", {"reason": "finished"})
        def stop(self): pass
        def estop(self): pass

    app = create_app(runs_dir=tmp_path, runner_factory=lambda k, c, s: FakeRunner(k, c, s))
    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.send_json({"action": "start_replay", "config": {"dataset_dir": "/x"}})
        for _ in range(20):
            evt = ws.receive_json()
            if evt.get("type") == "status" and evt.get("kind") == "stopped":
                break
    assert list(tmp_path.iterdir()) == []  # nothing recorded for replay
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp/tests/test_server.py -v -k record`
Expected: FAIL — `TypeError: create_app() got an unexpected keyword argument 'runs_dir'`

- [ ] **Step 3: Write minimal implementation**

In `webapp/server.py`:

(a) Add imports near the top (after existing telemetry import):

```python
from webapp.recording import RecordingSink, TeeSink
from webapp.run_store import RunStore
```

(b) Extend `create_app` signature + defaults (lines 31-36):

```python
def create_app(presets_dir: str | Path | None = None, runner_factory=None,
               feedback_dir: str | Path | None = None,
               runs_dir: str | Path | None = None) -> FastAPI:
    if presets_dir is None:
        presets_dir = Path(__file__).parent / "presets"
    if feedback_dir is None:
        feedback_dir = Path(__file__).parent / "feedback"
    if runs_dir is None:
        runs_dir = Path(__file__).parent / "runs"
```

(c) After `store = ConfigStore(presets_dir)` (line 55) add:

```python
    runs_dir = Path(runs_dir)
    run_store = RunStore(runs_dir)
```

(d) Replace the `start()` helper inside `telemetry_ws` (lines 188-195) with a version that tees a RecordingSink for live runs and announces the run id:

```python
        import datetime as _dt

        def _new_run_id() -> str:
            stamp = _dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            run_id, n = stamp, 1
            while (runs_dir / run_id).exists():
                run_id = f"{stamp}-{n}"
                n += 1
            return run_id

        def start(kind: str, config: dict):
            try:
                sink = QueueSink(q)
                if kind == "live":
                    run_id = _new_run_id()
                    recorder = RecordingSink(runs_dir / run_id, run_id=run_id, config=config)
                    sink = TeeSink([sink, recorder])
                    # Tell the browser which run this is (for the rating prompt).
                    q.put({"type": "status", "kind": "run_started",
                           "payload": {"run_id": run_id}})
                session.start(kind, config, sink)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cannot start %s session: %s", kind, exc)
                q.put({"type": "status", "kind": "error", "payload": {"message": str(exc)}})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp/tests/test_server.py -v`
Expected: PASS (all server tests)

- [ ] **Step 5: Commit**

```bash
git add webapp/server.py webapp/tests/test_server.py
git commit -m "feat(trossen_ai): record live runs via TeeSink + RecordingSink"
```

---

### Task 7: Run store REST endpoints + /runs page route

**Files:**
- Modify: `webapp/server.py` — add routes near the other `/api/*` routes and a `/runs` page route near `/replay` (line 83-85)
- Test: `webapp/tests/test_server.py`

- [ ] **Step 1: Write the failing test**

```python
# add to webapp/tests/test_server.py
def test_runs_api_list_get_and_rating(tmp_path):
    # Seed one run dir directly.
    d = tmp_path / "2026-07-02-090000"
    d.mkdir()
    (d / "manifest.json").write_text(_json.dumps({
        "run_id": "2026-07-02-090000", "started_at": 1.0, "kind": "live",
        "config": {"model_name": "m"}, "model_name": "m", "model": None}))
    (d / "summary.json").write_text(_json.dumps({"steps": 5, "end_reason": "completed",
                                                 "rtt_ms": {"mean": 40.0}}))
    (d / "events.jsonl").write_text(_json.dumps({"type": "overlap", "step": 0, "count": 2}))

    client = TestClient(create_app(runs_dir=tmp_path))
    lst = client.get("/api/runs").json()
    assert lst[0]["run_id"] == "2026-07-02-090000"
    assert lst[0]["model_name"] == "m"

    detail = client.get("/api/runs/2026-07-02-090000").json()
    assert detail["summary"]["steps"] == 5
    assert "events" not in detail

    with_events = client.get("/api/runs/2026-07-02-090000", params={"events": 1}).json()
    assert len(with_events["events"]) == 1

    r = client.patch("/api/runs/2026-07-02-090000/rating",
                     json={"success": "yes", "score": 5, "note": "clean"})
    assert r.status_code == 200
    assert _json.loads((d / "rating.json").read_text())["success"] == "yes"


def test_runs_api_missing_run_404(tmp_path):
    client = TestClient(create_app(runs_dir=tmp_path))
    assert client.get("/api/runs/does-not-exist").status_code == 404


def test_runs_page_served():
    client = TestClient(create_app())
    r = client.get("/runs")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp/tests/test_server.py -v -k "runs_api or runs_page"`
Expected: FAIL — 404 on `/api/runs` and `/runs`

- [ ] **Step 3: Write minimal implementation**

In `webapp/server.py`:

(a) Add a page route next to `/replay` (after line 85):

```python
    @app.get("/runs")
    def runs_page():
        return FileResponse(STATIC_DIR / "runs.html")
```

(b) Add REST routes near the other `/api/*` handlers (e.g. after the feedback route ~line 114). Import `HTTPException` at top: `from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException`.

```python
    @app.get("/api/runs")
    def api_list_runs():
        return run_store.list()

    @app.get("/api/runs/{run_id}")
    def api_get_run(run_id: str, events: int = 0):
        try:
            return run_store.get(run_id, with_events=bool(events))
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="run not found")

    @app.patch("/api/runs/{run_id}/rating")
    def api_rate_run(run_id: str, rating: dict):
        try:
            run_store.set_rating(run_id, rating)
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="run not found")
        return {"ok": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp/tests/test_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webapp/server.py webapp/tests/test_server.py
git commit -m "feat(trossen_ai): run-store REST endpoints + /runs page route"
```

---

### Task 8: Emit policy model metadata from LiveRunner

**Files:**
- Modify: `webapp/runners.py:93-95` (after `self._bridge = self._make_bridge(cfg)`)
- Test: `webapp/tests/test_runners_lifecycle.py`

**IMPORTANT:** first confirm the live runner class name in `webapp/runners.py`
(grep for `class .*Runner` and `def run`; the class whose `run()` calls
`_make_bridge`). The plan below assumes `LiveRunner` — substitute the real name
in both the test and the edit if it differs.

- [ ] **Step 1: Write the failing test**

```python
# add to webapp/tests/test_runners_lifecycle.py
def test_live_runner_emits_model_metadata(monkeypatch):
    import runners

    events = []

    class _Sink:
        def on_status(self, kind, payload): events.append((kind, payload))
        def on_log(self, *a): pass

    class _Client:
        def get_server_metadata(self): return {"policy": "pi0", "ckpt": "step_40000"}

    class _Bridge:
        def __init__(self): self.policy_client = _Client()
        def run_episode(self, task_prompt=""): pass
        def cleanup(self): pass

    LiveRunner = runners.LiveRunner  # substitute real class name if different
    r = LiveRunner("live", {"policy_host": "h", "policy_port": 1,
                            "connect_timeout": 0.1}, _Sink())
    monkeypatch.setattr(runners, "wait_for_policy_server", lambda *a, **k: None)
    monkeypatch.setattr(r, "_make_bridge", lambda cfg: _Bridge())
    r.run()
    assert ("model", {"policy": "pi0", "ckpt": "step_40000"}) in events
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp/tests/test_runners_lifecycle.py -v -k model_metadata`
Expected: FAIL — `("model", ...)` not in events

- [ ] **Step 3: Write minimal implementation**

In `webapp/runners.py`, right after `self._bridge = self._make_bridge(cfg)` (line 93), before the `try:` around `run_episode`:

```python
        self._bridge = self._make_bridge(cfg)
        # Capture the policy server's identity for the run log. Best-effort: the
        # server may not expose a name, so never let this block the run.
        try:
            meta = self._bridge.policy_client.get_server_metadata()
            self._sink.on_status("model", meta)
        except Exception:  # noqa: BLE001
            logger.debug("policy server metadata unavailable", exc_info=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp/tests/test_runners_lifecycle.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webapp/runners.py webapp/tests/test_runners_lifecycle.py
git commit -m "feat(trossen_ai): emit policy model metadata into run log"
```

---

## Phase 3 — Frontend

> Frontend has no JS unit harness in this repo. Verify each JS task with
> `node --check <file>` for syntax, then the manual browser steps given.

### Task 9: Model name config field

**Files:**
- Modify: `webapp/static/js/config.js` — Policy group (lines 16-23)

- [ ] **Step 1: Add the field**

In `webapp/static/js/config.js`, in the `Policy` group's `fields` array, add as the first field (before `adapter`):

```javascript
    { key:"model_name", label:"Model name / tag", type:"text", def:"",
      help:"Name this policy checkpoint (e.g. pi0-trossen-joint-v3) so recorded runs can be compared by model. Saved into each run's log." },
```

- [ ] **Step 2: Syntax check**

Run: `node --check webapp/static/js/config.js`
Expected: no output (exit 0)

- [ ] **Step 3: Manual verify**

Start the webapp, open `/`, confirm a "Model name / tag" text input appears in the Policy card and its value is included when you Start a live run (it lands in `runs/<id>/manifest.json` as `model_name`).

- [ ] **Step 4: Commit**

```bash
git add webapp/static/js/config.js
git commit -m "feat(trossen_ai): add model name config field for run logging"
```

---

### Task 10: End-of-run quick-rating modal

**Files:**
- Create: `webapp/static/js/runlog.js`
- Modify: `webapp/static/js/live.js` (import + wire), `webapp/static/theme.css` (modal styles)

Note: `live.js` is already loaded as an ES module by `index.html`; `runlog.js`
is imported by `live.js`, so no new `<script>` tag is needed.

- [ ] **Step 1: Create the module**

```javascript
// webapp/static/js/runlog.js
// End-of-run quick rating. Dedicated to eval logging — NOT the feedback card
// (feedback is for webapp bug reports). Remembers the active run id from the
// run_started status and, when the session ends, prompts for success/score/note
// and PATCHes it to /api/runs/{id}/rating.
let activeRunId = null;

function modal() {
  let m = document.getElementById("runlog-modal");
  if (m) return m;
  m = document.createElement("div");
  m.id = "runlog-modal";
  m.className = "runlog-backdrop";
  m.innerHTML = `
    <div class="runlog-card">
      <h3>Rate this run</h3>
      <div class="runlog-row">
        <label>Result</label>
        <select id="runlog-success">
          <option value="yes">Success</option>
          <option value="partial" selected>Partial</option>
          <option value="no">Failure</option>
        </select>
      </div>
      <div class="runlog-row">
        <label>Score (1–5)</label>
        <input id="runlog-score" type="number" min="1" max="5" step="1">
      </div>
      <div class="runlog-row">
        <label>Note</label>
        <input id="runlog-note" type="text" placeholder="one-line note (optional)">
      </div>
      <div class="runlog-actions">
        <button class="btn-outline" id="runlog-skip">Skip</button>
        <button class="btn-primary" id="runlog-save">Save rating</button>
      </div>
    </div>`;
  document.body.appendChild(m);
  m.querySelector("#runlog-skip").addEventListener("click", () => { m.style.display = "none"; });
  m.querySelector("#runlog-save").addEventListener("click", async () => {
    if (!activeRunId) { m.style.display = "none"; return; }
    const body = {
      success: m.querySelector("#runlog-success").value,
      score: Number(m.querySelector("#runlog-score").value) || null,
      note: m.querySelector("#runlog-note").value.trim(),
    };
    try {
      await fetch(`/api/runs/${activeRunId}/rating`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (e) { console.error("rating save failed", e); }
    m.style.display = "none";
  });
  return m;
}

// Call from a status handler. run_started -> remember id; stopped/error -> prompt.
export function onRunStatus(evt) {
  if (evt.kind === "run_started") {
    activeRunId = evt.payload?.run_id || null;
    return;
  }
  if ((evt.kind === "stopped" || evt.kind === "error") && activeRunId) {
    const m = modal();
    m.querySelector("#runlog-score").value = "";
    m.querySelector("#runlog-note").value = "";
    m.style.display = "flex";
  }
}
```

- [ ] **Step 2: Wire it into live.js**

In `webapp/static/js/live.js`, add the import at the top:

```javascript
import { onRunStatus } from "./runlog.js";
```

Then add a status listener inside the `DOMContentLoaded` handler (next to the other `onMessage` calls):

```javascript
  onMessage("status", (e) => onRunStatus(e));
```

- [ ] **Step 3: Add modal styles**

Append to `webapp/static/theme.css`:

```css
/* ── End-of-run rating modal ── */
.runlog-backdrop { display:none; position:fixed; inset:0; z-index:50;
  background:rgba(1,4,9,.6); align-items:center; justify-content:center; }
.runlog-card { background:var(--panel,#161b22); border:1px solid var(--border);
  border-radius:10px; padding:18px 20px; width:min(380px,92vw); }
.runlog-card h3 { margin:0 0 12px; font-size:14px; color:var(--text); }
.runlog-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.runlog-row label { flex:0 0 90px; font-size:12px; color:var(--muted); }
.runlog-row select, .runlog-row input { flex:1; }
.runlog-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:6px; }
```

- [ ] **Step 4: Syntax check**

Run: `node --check webapp/static/js/runlog.js && node --check webapp/static/js/live.js`
Expected: no output (exit 0)

- [ ] **Step 5: Manual verify**

Start a dry-run live session, let it finish (or click Stop). The rating modal appears; saving PATCHes `/api/runs/<id>/rating` and writes `runs/<id>/rating.json`. Skip closes without writing.

- [ ] **Step 6: Commit**

```bash
git add webapp/static/js/runlog.js webapp/static/js/live.js webapp/static/theme.css
git commit -m "feat(trossen_ai): end-of-run quick-rating modal"
```

---

### Task 11: Compare view page

**Files:**
- Create: `webapp/static/runs.html`, `webapp/static/js/runs.js`
- Modify: `webapp/static/index.html` (add a "Compare runs" link), `webapp/static/theme.css` (diff table styles)
- Test: `webapp/tests/test_server.py` (assets reachable)

- [ ] **Step 1: Create runs.html**

Mirror `replay.html`'s head (same `theme.css` + Chart.js include). First
`grep -n "chart" webapp/static/replay.html` to find the exact Chart.js `<script>`
tag and copy it verbatim into the placeholder below.

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Compare Runs — Trossen</title>
  <link rel="stylesheet" href="/static/theme.css">
  <!-- Copy the exact Chart.js <script> tag used by replay.html here -->
</head>
<body>
  <header class="topbar"><h1>Compare Runs</h1><a href="/" class="btn-sm">← Live</a></header>
  <main class="runs-main">
    <section class="card">
      <h2>Runs <span class="muted">(pick 2+ to compare)</span></h2>
      <div id="runs-list" class="runs-list">Loading…</div>
    </section>
    <section class="card">
      <h2>Config diff</h2>
      <div id="runs-diff" class="runs-diff-wrap"><em class="muted">Select runs above.</em></div>
    </section>
    <section class="card">
      <h2>Smoothing Δ vs step</h2>
      <div class="chart-cj-container"><canvas id="runs-chart-delta"></canvas></div>
    </section>
    <section class="card">
      <h2>Overlap depth vs step</h2>
      <div class="chart-cj-container"><canvas id="runs-chart-overlap"></canvas></div>
    </section>
  </main>
  <script type="module" src="/static/js/runs.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create runs.js**

```javascript
// webapp/static/js/runs.js
import { makeSeriesChart } from "./charts.js";

const $ = (id) => document.getElementById(id);
const selected = new Set();
let deltaChart = null, overlapChart = null;

async function loadList() {
  const runs = await (await fetch("/api/runs")).json();
  const box = $("runs-list");
  if (!runs.length) { box.innerHTML = '<em class="muted">No runs recorded yet.</em>'; return; }
  box.innerHTML = runs.map(r => `
    <label class="run-item">
      <input type="checkbox" value="${r.run_id}">
      <span class="run-id">${r.run_id}</span>
      <span class="run-model">${r.model_name || "—"}</span>
      <span class="run-metric">${r.steps ?? "—"} steps</span>
      <span class="run-metric">${r.end_reason}</span>
      <span class="run-metric">${r.rating ? r.rating.success : "unrated"}</span>
    </label>`).join("");
  box.querySelectorAll("input[type=checkbox]").forEach(cb =>
    cb.addEventListener("change", () => {
      cb.checked ? selected.add(cb.value) : selected.delete(cb.value);
      refresh();
    }));
}

async function fetchRun(id) {
  return (await fetch(`/api/runs/${id}?events=1`)).json();
}

function configDiff(runs) {
  // runs: [{id, manifest}]. Build union of config keys; highlight rows that differ.
  const keys = new Set();
  runs.forEach(r => Object.keys(r.manifest.config || {}).forEach(k => keys.add(k)));
  const rows = [...keys].sort().map(k => {
    const vals = runs.map(r => String((r.manifest.config || {})[k] ?? ""));
    const differ = new Set(vals).size > 1;
    return `<tr class="${differ ? "diff" : ""}"><td>${k}</td>${vals.map(v => `<td>${v}</td>`).join("")}</tr>`;
  }).join("");
  return `<table class="runs-diff"><thead><tr><th>key</th>${
    runs.map(r => `<th>${r.id}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
}

// From an events array, series of {x:step, y:value}. delta from action events
// (‖action − raw‖), overlap from overlap events.
function deltaSeries(events) {
  const pts = [];
  for (const e of events) {
    if (e.type !== "action" || !e.raw || !e.action) continue;
    let s = 0; for (let i = 0; i < e.action.length; i++) { const d = e.action[i] - e.raw[i]; s += d * d; }
    pts.push({ x: e.step, y: Math.sqrt(s) });
  }
  return pts;
}
function overlapSeries(events) {
  return events.filter(e => e.type === "overlap").map(e => ({ x: e.step, y: e.count }));
}

async function refresh() {
  const ids = [...selected];
  if (ids.length < 2) {
    $("runs-diff").innerHTML = '<em class="muted">Select 2+ runs.</em>';
    return;
  }
  const runs = await Promise.all(ids.map(async id => ({ id, ...(await fetchRun(id)) })));
  $("runs-diff").innerHTML = configDiff(runs);

  const deltaData = runs.map(r => ({ label: r.id, points: deltaSeries(r.events) }));
  const overlapData = runs.map(r => ({ label: r.id, points: overlapSeries(r.events) }));
  if (deltaChart) deltaChart.destroy();
  if (overlapChart) overlapChart.destroy();
  deltaChart = makeSeriesChart($("runs-chart-delta"), deltaData);
  overlapChart = makeSeriesChart($("runs-chart-overlap"), overlapData);
}

loadList();
```

- [ ] **Step 3: Add the link on the live page**

In `webapp/static/index.html`, grep for `href="/replay"` to find the nav area and
add a matching link beside it:

```html
<a href="/runs" class="btn-sm">Compare runs</a>
```

- [ ] **Step 4: Add diff table + list styles**

Append to `webapp/static/theme.css`:

```css
/* ── Compare runs view ── */
.runs-main { max-width:1100px; margin:0 auto; padding:16px; display:grid; gap:14px; }
.runs-list { display:grid; gap:4px; }
.run-item { display:grid; grid-template-columns:24px 1.6fr 1.4fr repeat(3,1fr);
  align-items:center; gap:8px; font-size:12px; padding:4px 6px; border-radius:6px; }
.run-item:hover { background:var(--bg); }
.run-id { font-family:monospace; color:var(--text); }
.run-model { color:var(--muted); }
.run-metric { color:var(--muted); text-align:right; }
.runs-diff-wrap { overflow-x:auto; }
table.runs-diff { border-collapse:collapse; font-size:12px; width:100%; }
table.runs-diff th, table.runs-diff td { border:1px solid var(--border);
  padding:4px 8px; text-align:left; color:var(--text); }
table.runs-diff tr.diff td { background:rgba(255,184,108,.12); }
table.runs-diff tr.diff td:first-child { font-weight:700; }
```

- [ ] **Step 5: Syntax check + asset test**

Run: `node --check webapp/static/js/runs.js`
Expected: no output (exit 0)

Add a server test that the assets are reachable:

```python
# add to webapp/tests/test_server.py
def test_runs_static_assets_served():
    client = TestClient(create_app())
    assert client.get("/static/js/runs.js").status_code == 200
    assert client.get("/static/runs.html").status_code == 200
```

Run: `python -m pytest webapp/tests/test_server.py -v -k runs`
Expected: PASS

- [ ] **Step 6: Manual verify**

Record 2 live dry-run sessions with different configs (e.g. smoothing on vs off).
Open `/runs`, tick both: the config diff table highlights the differing rows and
the two Smoothing-Δ / overlap charts overlay both runs.

- [ ] **Step 7: Commit**

```bash
git add webapp/static/runs.html webapp/static/js/runs.js webapp/static/index.html webapp/static/theme.css webapp/tests/test_server.py
git commit -m "feat(trossen_ai): compare-runs view (config diff + overlay charts)"
```

---

## Final verification

- [ ] Run the whole webapp suite: from `examples/trossen_ai/`, `python -m pytest webapp/tests/ -v`. Expected: all pass, no regressions.
- [ ] Syntax-check all touched JS: `node --check` on `config.js runlog.js live.js runs.js`.
- [ ] Add recorded runs to `.gitignore` (they are data, not source):

```bash
echo "examples/trossen_ai/webapp/runs/" >> .gitignore
git add .gitignore && git commit -m "chore(trossen_ai): ignore recorded run logs"
```

- [ ] Confirm CLI path unaffected: NullSink still needs no `close()` (session uses `getattr`), and non-live webapp kinds record nothing (Task 6 test).

## Notes on decomposition

Phases are independently valuable: Phase 1 (recording lib) is pure + fully unit-tested; Phase 2 makes live runs actually write to disk; Phase 3 adds the human-facing rating + compare UI. If splitting across sessions, Phase 1+2 is a shippable "runs are logged" milestone before the compare view.
