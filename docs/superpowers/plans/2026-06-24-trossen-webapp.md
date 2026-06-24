# Trossen Control Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI + vanilla-JS/Chart.js web app (in its own folder) that configures and launches live control or dataset-replay sessions on the Trossen stack, and streams logs, action charts, camera images, RTT/latency/jitter metrics, and ensemble-smoothing visualizations in real time — by *observing* the existing control loop, not reimplementing it.

**Architecture:** A `TelemetrySink` protocol is the seam: the existing bridge/worker emit events to a sink that defaults to a no-op (`NullSink`), so CLI behavior is unchanged. The web app injects a `QueueSink` whose events a WebSocket drains to the browser. A `SessionManager` runs one bridge/replay session in a daemon thread (runner-factory DI so it is testable without hardware). Pure-logic modules (`telemetry`, `metrics`, `config_store`, `session`) are unit-tested off-hardware; the FastAPI server and static frontend wire them together.

**Tech Stack:** Python 3.10 (`lerobot` conda env), FastAPI, uvicorn, `websockets`, numpy, pytest; frontend = HTML + vanilla JS + Chart.js (CDN), no build step.

**Spec:** [`../specs/2026-06-24-trossen-webapp-design.md`](../specs/2026-06-24-trossen-webapp-design.md)

**Conventions for every task below:**
- `PYBIN=/home/edgeai/miniconda3/envs/lerobot/bin/python`
- Working directory: `examples/trossen_ai`.
- `tests/conftest.py` already puts `examples/trossen_ai/` on `sys.path`, so `webapp` imports as a package (`from webapp.telemetry import ...`).
- The robot runtime env (the one with `lerobot_robot_trossen`) is **not** the `lerobot` test env. Tasks 1, 3, 4, 5 run fully under `$PYBIN`. Task 2's instrumentation is verified by the existing suite under `$PYBIN`. Task 7 (server) is import-/route-tested under `$PYBIN` with FastAPI installed; real hardware (Task 6 runners) runs only on the robot env.

---

## Task 0: Dependencies + package skeleton

**Files:**
- Create: `examples/trossen_ai/webapp/__init__.py`
- Create: `examples/trossen_ai/webapp/tests/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Install web deps into the test env**

Run:
```bash
/home/edgeai/miniconda3/envs/lerobot/bin/python -m pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" "httpx>=0.27"
```
(`httpx` is for FastAPI's `TestClient`. On the robot runtime env, install the same three before serving.)

- [ ] **Step 2: Create the package files**

`examples/trossen_ai/webapp/__init__.py`:
```python
"""Browser UI + telemetry layer over the Trossen control stack (observe-only)."""
```
`examples/trossen_ai/webapp/tests/__init__.py`:
```python
```
(empty file)

- [ ] **Step 3: Gitignore saved presets**

Add to `.gitignore`:
```
# Web app saved config presets.
examples/trossen_ai/webapp/presets/
```

- [ ] **Step 4: Verify the package imports**

Run:
```bash
$PYBIN -c "import webapp; print('ok')"
```
Expected: `ok`.

- [ ] **Step 5: Commit**
```bash
git add examples/trossen_ai/webapp/__init__.py examples/trossen_ai/webapp/tests/__init__.py .gitignore
git commit -m "feat(webapp): package skeleton + web deps"
```

---

## Task 1: `telemetry.py` — sink protocol, events, QueueSink

**Files:**
- Create: `examples/trossen_ai/webapp/telemetry.py`
- Test: `examples/trossen_ai/webapp/tests/test_telemetry.py`

- [ ] **Step 1: Write the failing tests**

Create `examples/trossen_ai/webapp/tests/test_telemetry.py`:
```python
import queue

import numpy as np

from webapp.telemetry import NullSink, QueueSink


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_null_sink_accepts_all_calls():
    s = NullSink()
    s.on_log("INFO", "hi", 1.0)
    s.on_action(0, np.zeros(14), 1.0)
    s.on_inference(12.3, 1.0)
    s.on_chunk(0, np.ones((5, 14)), 1.0)
    s.on_overlap(0, 3)
    s.on_images({"cam": b"x"}, 1.0)
    s.on_status("started", {})  # no exception = pass


def test_queue_sink_emits_typed_events():
    q = queue.Queue()
    s = QueueSink(q)
    s.on_log("WARNING", "careful", 2.0)
    s.on_overlap(7, 4)
    events = _drain(q)
    assert {"type": "log", "level": "WARNING", "msg": "careful", "ts": 2.0} in events
    assert {"type": "overlap", "step": 7, "count": 4} in events


def test_queue_sink_action_includes_raw_from_latest_chunk():
    q = queue.Queue()
    s = QueueSink(q)
    # chunk queried at step 5 covers steps 5..7
    s.on_chunk(5, np.array([[10.0], [11.0], [12.0]]), 1.0)
    s.on_action(6, np.array([99.0]), 1.1)  # blended output for step 6
    events = _drain(q)
    action_evt = next(e for e in events if e["type"] == "action")
    assert action_evt["step"] == 6
    assert action_evt["action"] == [99.0]
    assert action_evt["raw"] == [11.0]  # chunk[6 - 5]


def test_queue_sink_action_raw_none_when_out_of_range():
    q = queue.Queue()
    s = QueueSink(q)
    s.on_chunk(5, np.array([[10.0]]), 1.0)  # covers only step 5
    s.on_action(9, np.array([1.0]), 1.1)
    action_evt = next(e for e in _drain(q) if e["type"] == "action")
    assert action_evt["raw"] is None


def test_queue_sink_throttles_images():
    q = queue.Queue()
    s = QueueSink(q, image_min_interval_s=10.0)
    s.on_images({"cam": b"abc"}, ts=100.0)
    s.on_images({"cam": b"def"}, ts=100.5)  # within interval -> dropped
    s.on_images({"cam": b"ghi"}, ts=200.0)  # after interval -> sent
    imgs = [e for e in _drain(q) if e["type"] == "images"]
    assert len(imgs) == 2
```

- [ ] **Step 2: Run to verify it fails**
```bash
$PYBIN -m pytest webapp/tests/test_telemetry.py -q
```
Expected: `ModuleNotFoundError: No module named 'webapp.telemetry'`.

- [ ] **Step 3: Implement `telemetry.py`**

Create `examples/trossen_ai/webapp/telemetry.py`:
```python
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
    def on_overlap(self, step: int, count: int) -> None: ...
    def on_images(self, images: dict[str, bytes], ts: float) -> None: ...
    def on_status(self, kind: str, payload: dict) -> None: ...


class NullSink:
    """No-op sink. Default for all CLI paths — zero behavior change."""

    def on_log(self, level: str, msg: str, ts: float) -> None: ...
    def on_action(self, step: int, action: np.ndarray, ts: float) -> None: ...
    def on_inference(self, rtt_ms: float, ts: float) -> None: ...
    def on_chunk(self, query_step: int, chunk: np.ndarray, ts: float) -> None: ...
    def on_overlap(self, step: int, count: int) -> None: ...
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
        self._last_image_ts = float("-inf")

    def on_log(self, level: str, msg: str, ts: float) -> None:
        self._q.put({"type": "log", "level": level, "msg": msg, "ts": ts})

    def on_action(self, step: int, action: np.ndarray, ts: float) -> None:
        raw = None
        if self._last_chunk is not None:
            qs, chunk = self._last_chunk
            offset = step - qs
            if 0 <= offset < len(chunk):
                raw = np.asarray(chunk[offset]).flatten().tolist()
        self._q.put({
            "type": "action",
            "step": step,
            "action": np.asarray(action).flatten().tolist(),
            "raw": raw,
            "ts": ts,
        })

    def on_inference(self, rtt_ms: float, ts: float) -> None:
        self._q.put({"type": "inference", "rtt_ms": rtt_ms, "ts": ts})

    def on_chunk(self, query_step: int, chunk: np.ndarray, ts: float) -> None:
        arr = np.asarray(chunk)
        self._last_chunk = (query_step, arr)
        self._q.put({"type": "chunk", "query_step": query_step, "len": int(len(arr)), "ts": ts})

    def on_overlap(self, step: int, count: int) -> None:
        self._q.put({"type": "overlap", "step": step, "count": count})

    def on_images(self, images: dict[str, bytes], ts: float) -> None:
        if ts - self._last_image_ts < self._image_min_interval_s:
            return
        self._last_image_ts = ts
        encoded = {k: base64.b64encode(v).decode("ascii") for k, v in images.items()}
        self._q.put({"type": "images", "images": encoded, "ts": ts})

    def on_status(self, kind: str, payload: dict) -> None:
        self._q.put({"type": "status", "kind": kind, "payload": payload})
```

- [ ] **Step 4: Run to verify it passes**
```bash
$PYBIN -m pytest webapp/tests/test_telemetry.py -q
```
Expected: `5 passed`.

- [ ] **Step 5: Commit**
```bash
git add examples/trossen_ai/webapp/telemetry.py examples/trossen_ai/webapp/tests/test_telemetry.py
git commit -m "feat(webapp): telemetry sink protocol + QueueSink"
```

---

## Task 2: Instrument bridge / worker / ensembles (default null-sink)

Adds the emit points. With `NullSink()` defaults, the existing 33 tests must stay green.

**Files:**
- Modify: `examples/trossen_ai/ensemble/base.py` (add `buffer_size` to ABC)
- Modify: `examples/trossen_ai/ensemble/exponential.py`, `examples/trossen_ai/ensemble/cogact.py` (implement `buffer_size`)
- Test: `examples/trossen_ai/webapp/tests/test_buffer_size.py`
- Create: `examples/trossen_ai/webapp/log_bridge.py`
- Modify: `examples/trossen_ai/async_worker.py` (sink param + emit)
- Modify: `examples/trossen_ai/trossen_bridge.py` (sink param + emit + log handler)

- [ ] **Step 1: Write the failing test for `buffer_size`**

Create `examples/trossen_ai/webapp/tests/test_buffer_size.py`:
```python
import numpy as np

from ensemble import CogACTEnsemble, ExponentialEnsemble


def test_exp_buffer_size_counts_buffered_entries():
    e = ExponentialEnsemble(decay=1.0)
    assert e.buffer_size() == 0
    e.add_chunk(0, np.ones((5, 2)))  # 5 future steps buffered
    assert e.buffer_size() == 5


def test_cogact_buffer_size_counts_chunks():
    e = CogACTEnsemble(max_buffer_size=25, mode="cogact")
    assert e.buffer_size() == 0
    e.add_chunk(0, np.ones((10, 4)))
    e.add_chunk(1, np.ones((10, 4)))
    assert e.buffer_size() == 2
```

- [ ] **Step 2: Run to verify it fails**
```bash
$PYBIN -m pytest webapp/tests/test_buffer_size.py -q
```
Expected: FAIL — `AttributeError: 'ExponentialEnsemble' object has no attribute 'buffer_size'`.

- [ ] **Step 3: Add `buffer_size` to the ABC**

In `ensemble/base.py`, add after the `reset` abstractmethod:
```python
    @abstractmethod
    def buffer_size(self) -> int:
        """Number of buffered items (for telemetry / memory-bound checks)."""
```

- [ ] **Step 4: Implement `buffer_size` in both ensembles**

In `ensemble/exponential.py`, add to the class, immediately after `reset` (before the `@register_ensemble` builder):
```python
    def buffer_size(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._buffer.values())
```
In `ensemble/cogact.py`, add to the class, immediately after `reset` (before the builder):
```python
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)
```

- [ ] **Step 5: Run to verify buffer_size passes + suite still green**
```bash
$PYBIN -m pytest webapp/tests/test_buffer_size.py tests/ -q
```
Expected: all pass (the previous 33 + 2 new).

- [ ] **Step 6: Create the log handler module**

Create `examples/trossen_ai/webapp/log_bridge.py`:
```python
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
```

- [ ] **Step 7: Add sink to `AsyncPolicyWorker` and emit RTT + chunk**

In `async_worker.py`:

a) Add to the imports at the top of the file:
```python
from webapp.telemetry import NullSink, TelemetrySink
```
(`time` and `numpy as np` are already imported.)

b) Replace the `__init__` signature + body start:
```python
    def __init__(self, policy_client, ensemble: ActionEnsemble, action_dim: int,
                 sink: TelemetrySink = NullSink()) -> None:
        self._client = policy_client
        self._ensemble = ensemble
        self._action_dim = action_dim
        self._sink = sink
```
(keep the remaining `self._pending`, `self._lock`, `self._first_result`, `self._running`, `self._thread` lines unchanged).

c) In `_loop`, replace the inference section inside the `try:` with:
```python
            obs, query_step = item
            try:
                t0 = time.perf_counter()
                response = self._client.infer(obs)
                rtt_ms = (time.perf_counter() - t0) * 1e3
                self._sink.on_inference(rtt_ms, time.time())
                chunk = np.asarray(response["actions"])[:, : self._action_dim]
                self._ensemble.add_chunk(query_step, chunk)
                self._sink.on_chunk(query_step, chunk, time.time())
                if first:
                    self._first_result.set()
                    first = False
            except Exception:
                logger.exception("AsyncPolicyWorker: inference error")
```

- [ ] **Step 8: Add sink + emits + log handler to the bridge**

In `trossen_bridge.py`:

a) Add to the local imports (after `from robot_control import build_stationary_robot`):
```python
from webapp.log_bridge import SinkLogHandler
from webapp.telemetry import NullSink, TelemetrySink
```

b) Add `sink` to `__init__` signature (after `adapter`):
```python
        adapter: "ActionSpaceAdapter | None" = None,
        sink: TelemetrySink = NullSink(),
```
Store it right after `self.adapter = ...`:
```python
        self.sink = sink
```
And pass it to the worker — replace the `self._policy_worker = (...)` assignment with:
```python
        self._policy_worker = (
            AsyncPolicyWorker(self.policy_client, self.ensemble, self.action_dim, sink=self.sink)
            if async_inference else None
        )
```

c) In `run_episode`, just after `fallback = HoldLastAction()`, add:
```python
        log_handler = SinkLogHandler(self.sink)
        logging.getLogger().addHandler(log_handler)
```
and inside the existing `finally:` block (first line of it), add:
```python
            logging.getLogger().removeHandler(log_handler)
```

d) **Sync branch** — replace:
```python
                        response = self.policy_client.infer(observation)
                        current_joints14 = extract_joints(obs_raw)
                        self.current_action_chunk = self.adapter.decode_chunk(response["actions"], current_joints14)
                        if self.ensemble is not None:
                            self.ensemble.add_chunk(self.episode_step, self.current_action_chunk)
```
with:
```python
                        _t0 = time.perf_counter()
                        response = self.policy_client.infer(observation)
                        self.sink.on_inference((time.perf_counter() - _t0) * 1e3, time.time())
                        current_joints14 = extract_joints(obs_raw)
                        self.current_action_chunk = self.adapter.decode_chunk(response["actions"], current_joints14)
                        if self.ensemble is not None:
                            self.ensemble.add_chunk(self.episode_step, self.current_action_chunk)
                            self.sink.on_chunk(self.episode_step, self.current_action_chunk, time.time())
```

e) Emit executed action + overlap + buffer once per step. Immediately **after** the existing action-logger block (the `if self.action_logger is not None and self.ensemble is not None:` block that calls `self.action_logger.log(...)`), add:
```python
                if self.ensemble is not None:
                    self.sink.on_overlap(self.episode_step, self.ensemble.get_overlap_count(self.episode_step))
                    self.sink.on_status("buffer", {"size": self.ensemble.buffer_size()})
                self.sink.on_action(self.episode_step, np.asarray(a_t), time.time())
```

f) Emit camera images (throttled inside QueueSink). In `_build_observation`, immediately before `return {"state": state, "images": images, "prompt": task_prompt}`, add:
```python
        self.sink.on_images({cam: cv2.imencode(".jpg", observation_dict[cam])[1].tobytes() for cam in cameras}, time.time())
```

- [ ] **Step 9: Verify the full suite stays green + byte-compile**
```bash
$PYBIN -m pytest tests/ webapp/tests/ -q
$PYBIN -m py_compile trossen_bridge.py async_worker.py ensemble/base.py ensemble/exponential.py ensemble/cogact.py webapp/log_bridge.py
```
Expected: all tests pass (existing 33 + buffer_size 2 + telemetry 5); compile silent.

- [ ] **Step 10: Commit**
```bash
git add examples/trossen_ai/ensemble examples/trossen_ai/async_worker.py examples/trossen_ai/trossen_bridge.py examples/trossen_ai/webapp/log_bridge.py examples/trossen_ai/webapp/tests/test_buffer_size.py
git commit -m "feat(webapp): instrument bridge/worker/ensembles with telemetry sink (null-default)"
```

---

## Task 3: `metrics.py` — RTT/latency/jitter/loop-Hz accumulators

**Files:**
- Create: `examples/trossen_ai/webapp/metrics.py`
- Test: `examples/trossen_ai/webapp/tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `examples/trossen_ai/webapp/tests/test_metrics.py`:
```python
import numpy as np

from webapp.metrics import Metrics


def test_rtt_percentiles():
    m = Metrics(window=100)
    for v in range(1, 101):  # 1..100 ms
        m.add_rtt(float(v))
    snap = m.snapshot()
    assert snap["rtt_last"] == 100.0
    assert 49 <= snap["rtt_p50"] <= 52
    assert 94 <= snap["rtt_p95"] <= 96


def test_loop_hz_from_action_timestamps():
    m = Metrics(window=100)
    for i in range(11):  # 10 intervals of 0.02s -> 50 Hz
        m.add_action(ts=i * 0.02, action=np.zeros(14))
    assert abs(m.snapshot()["loop_hz"] - 50.0) < 1.0


def test_jitter_is_rolling_stddev_of_action_deltas():
    m = Metrics(window=100)
    for i in range(5):  # constant action -> zero jitter
        m.add_action(ts=i * 0.02, action=np.ones(14))
    assert m.snapshot()["jitter"] == 0.0
    m.add_action(ts=0.12, action=np.ones(14) * 5)  # introduce variation
    assert m.snapshot()["jitter"] > 0.0


def test_drops_counter():
    m = Metrics(window=10)
    assert m.snapshot()["drops"] == 0
    m.add_drop(); m.add_drop()
    assert m.snapshot()["drops"] == 2


def test_empty_snapshot_is_safe():
    snap = Metrics(window=10).snapshot()
    assert snap["rtt_last"] is None
    assert snap["loop_hz"] is None
    assert snap["jitter"] is None
    assert snap["drops"] == 0
```

- [ ] **Step 2: Run to verify it fails**
```bash
$PYBIN -m pytest webapp/tests/test_metrics.py -q
```
Expected: `ModuleNotFoundError: No module named 'webapp.metrics'`.

- [ ] **Step 3: Implement `metrics.py`**

Create `examples/trossen_ai/webapp/metrics.py`:
```python
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
```

- [ ] **Step 4: Run to verify it passes**
```bash
$PYBIN -m pytest webapp/tests/test_metrics.py -q
```
Expected: `5 passed`.

- [ ] **Step 5: Commit**
```bash
git add examples/trossen_ai/webapp/metrics.py examples/trossen_ai/webapp/tests/test_metrics.py
git commit -m "feat(webapp): rolling RTT/latency/jitter/loop-Hz metrics"
```

---

## Task 4: `config_store.py` — named config presets

**Files:**
- Create: `examples/trossen_ai/webapp/config_store.py`
- Test: `examples/trossen_ai/webapp/tests/test_config_store.py`

- [ ] **Step 1: Write the failing tests**

Create `examples/trossen_ai/webapp/tests/test_config_store.py`:
```python
import pytest

from webapp.config_store import ConfigStore


def test_save_then_load_round_trip(tmp_path):
    store = ConfigStore(tmp_path)
    cfg = {"policy_host": "192.168.1.9", "control_freq": 25, "ensemble_type": "cogact"}
    store.save("rig-a", cfg)
    assert store.load("rig-a") == cfg


def test_list_names_sorted(tmp_path):
    store = ConfigStore(tmp_path)
    store.save("b", {}); store.save("a", {})
    assert store.list_names() == ["a", "b"]


def test_delete_removes(tmp_path):
    store = ConfigStore(tmp_path)
    store.save("x", {"a": 1})
    store.delete("x")
    assert store.list_names() == []


def test_load_missing_raises(tmp_path):
    with pytest.raises(KeyError):
        ConfigStore(tmp_path).load("nope")


def test_name_is_sanitized(tmp_path):
    store = ConfigStore(tmp_path)
    with pytest.raises(ValueError):
        store.save("../escape", {})
```

- [ ] **Step 2: Run to verify it fails**
```bash
$PYBIN -m pytest webapp/tests/test_config_store.py -q
```
Expected: `ModuleNotFoundError: No module named 'webapp.config_store'`.

- [ ] **Step 3: Implement `config_store.py`**

Create `examples/trossen_ai/webapp/config_store.py`:
```python
"""Named config presets persisted as JSON files in a directory."""
from __future__ import annotations

import json
import re
from pathlib import Path

_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


class ConfigStore:
    def __init__(self, presets_dir: str | Path) -> None:
        self._dir = Path(presets_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        if not _SAFE.match(name):
            raise ValueError(f"Invalid preset name {name!r} (use letters, digits, . _ -)")
        return self._dir / f"{name}.json"

    def save(self, name: str, config: dict) -> None:
        self._path(name).write_text(json.dumps(config, indent=2))

    def load(self, name: str) -> dict:
        path = self._path(name)
        if not path.is_file():
            raise KeyError(name)
        return json.loads(path.read_text())

    def list_names(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def delete(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)
```

- [ ] **Step 4: Run to verify it passes**
```bash
$PYBIN -m pytest webapp/tests/test_config_store.py -q
```
Expected: `5 passed`.

- [ ] **Step 5: Commit**
```bash
git add examples/trossen_ai/webapp/config_store.py examples/trossen_ai/webapp/tests/test_config_store.py
git commit -m "feat(webapp): JSON config-preset store"
```

---

## Task 5: `session.py` — SessionManager with runner-factory DI

**Files:**
- Create: `examples/trossen_ai/webapp/session.py`
- Test: `examples/trossen_ai/webapp/tests/test_session.py`

- [ ] **Step 1: Write the failing tests**

Create `examples/trossen_ai/webapp/tests/test_session.py`:
```python
import time

import pytest

from webapp.session import SessionManager


class _NoopSink:
    def on_status(self, *a): ...


class FakeRunner:
    """Stands in for a bridge/replay run. run() blocks until stop()."""

    def __init__(self, kind, config, sink):
        self.kind = kind
        self.config = config
        self.sink = sink
        self._running = False
        self.estopped = False

    def run(self):
        self._running = True
        self.sink.on_status("started", {"kind": self.kind})
        while self._running:
            time.sleep(0.005)
        self.sink.on_status("stopped", {})

    def stop(self):
        self._running = False

    def estop(self):
        self.estopped = True
        self._running = False


def _factory(created):
    def make(kind, config, sink):
        r = FakeRunner(kind, config, sink)
        created.append(r)
        return r
    return make


def test_start_then_stop():
    created = []
    sm = SessionManager(_factory(created))
    sm.start("live", {"x": 1}, sink=_NoopSink())
    time.sleep(0.02)
    assert sm.is_running()
    sm.stop()
    assert not sm.is_running()
    assert created[0].kind == "live"


def test_single_session_guard():
    created = []
    sm = SessionManager(_factory(created))
    sm.start("live", {}, sink=_NoopSink())
    time.sleep(0.02)
    with pytest.raises(RuntimeError):
        sm.start("replay", {}, sink=_NoopSink())
    sm.stop()


def test_estop_calls_runner_estop():
    created = []
    sm = SessionManager(_factory(created))
    sm.start("live", {}, sink=_NoopSink())
    time.sleep(0.02)
    sm.estop()
    assert created[0].estopped
    assert not sm.is_running()
```

- [ ] **Step 2: Run to verify it fails**
```bash
$PYBIN -m pytest webapp/tests/test_session.py -q
```
Expected: `ModuleNotFoundError: No module named 'webapp.session'`.

- [ ] **Step 3: Implement `session.py`**

Create `examples/trossen_ai/webapp/session.py`:
```python
"""Run exactly one control/replay session in a daemon thread.

The runner factory is injected so the hardware-bound runners (bridge / replay)
can be swapped for fakes in tests. A Runner must expose run() (blocking),
stop(), and estop().
"""
from __future__ import annotations

import threading
from typing import Callable, Protocol


class Runner(Protocol):
    def run(self) -> None: ...
    def stop(self) -> None: ...
    def estop(self) -> None: ...


RunnerFactory = Callable[[str, dict, object], Runner]


class SessionManager:
    def __init__(self, runner_factory: RunnerFactory) -> None:
        self._factory = runner_factory
        self._runner: Runner | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, kind: str, config: dict, sink: object) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError("A session is already running")
            self._runner = self._factory(kind, config, sink)
            self._thread = threading.Thread(target=self._runner.run, daemon=True, name=f"session-{kind}")
            self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        runner, thread = self._runner, self._thread
        if runner is not None:
            runner.stop()
        if thread is not None:
            thread.join(timeout=timeout)
        self._runner, self._thread = None, None

    def estop(self, timeout: float = 15.0) -> None:
        runner, thread = self._runner, self._thread
        if runner is not None:
            runner.estop()
        if thread is not None:
            thread.join(timeout=timeout)
        self._runner, self._thread = None, None
```

- [ ] **Step 4: Run to verify it passes**
```bash
$PYBIN -m pytest webapp/tests/test_session.py -q
```
Expected: `3 passed`.

- [ ] **Step 5: Commit**
```bash
git add examples/trossen_ai/webapp/session.py examples/trossen_ai/webapp/tests/test_session.py
git commit -m "feat(webapp): SessionManager with runner-factory DI"
```

---

## Task 6: Real runners — bridge + replay adapters for SessionManager

Wraps the hardware paths behind the `Runner` interface. These are **not** unit-tested (need hardware); they are syntax-checked only.

**Files:**
- Create: `examples/trossen_ai/webapp/runners.py`

- [ ] **Step 1: Implement the runners**

Create `examples/trossen_ai/webapp/runners.py`:
```python
"""Hardware-bound runners that adapt the bridge and replay tool to the Runner
interface used by SessionManager. Syntax-checked in CI; exercised on the rig.

`config` is the dict produced by the web form. The constructors forward only the
keys they declare; unknown keys are ignored.
"""
from __future__ import annotations

import time

import numpy as np

from adapters import EEAdapter, JointAdapter
from dataset_replay import EpisodeReader
from external.joint_to_ee.ee_to_joints import EEToJointsConverter
from external.joint_to_ee.kinematics import make_kinematics
from robot_control import RobotController, build_stationary_robot
from trossen_bridge import TrossenOpenPIBridge


class LiveRunner:
    """Runs TrossenOpenPIBridge.run_episode with a telemetry sink."""

    def __init__(self, kind: str, config: dict, sink) -> None:
        adapter = JointAdapter()
        if config.get("adapter") == "ee":
            conv = EEToJointsConverter(
                make_kinematics(),
                orientation_weight=float(config.get("ik_orientation_weight", 0.01)),
                pos_tol_m=float(config.get("ik_pos_tol_m", 1e-3)),
            )
            adapter = EEAdapter(conv)
        self._bridge = TrossenOpenPIBridge(
            policy_server_host=config.get("policy_host", "localhost"),
            policy_server_port=int(config.get("policy_port", 8000)),
            control_frequency=int(config.get("control_freq", 25)),
            test_mode=config.get("mode", "test"),
            max_steps=int(config.get("max_steps", 1000)),
            rate_of_inference=int(config.get("rate_of_inference", 20)),
            ensemble_type=config.get("ensemble_type", "exp"),
            cogact_mode=config.get("cogact_mode", "cogact"),
            async_inference=bool(config.get("async_inference", False)),
            use_left_arm_only=bool(config.get("use_left_arm_only", False)),
            use_right_arm_only=bool(config.get("use_right_arm_only", False)),
            starvla=bool(config.get("starvla", False)),
            adapter=adapter,
            sink=sink,
        )
        self._prompt = config.get("task_prompt", "")

    def run(self) -> None:
        self._bridge.run_episode(task_prompt=self._prompt)

    def stop(self) -> None:
        self._bridge.is_running = False

    def estop(self) -> None:
        self._bridge.is_running = False
        try:
            self._bridge.move_to_sleep_position(duration=10.0)
        finally:
            self._bridge.cleanup()


class ReplayRunner:
    """Replays a dataset episode (EE -> IK -> robot) with a telemetry sink."""

    def __init__(self, kind: str, config: dict, sink) -> None:
        self._sink = sink
        self._stopped = False
        reader = EpisodeReader(config["dataset_dir"])
        self._episode = reader.read_episode(int(config.get("episode_index", 0)))
        self._control_freq = int(config.get("control_freq") or self._episode.fps)
        self._converter = EEToJointsConverter(
            make_kinematics(),
            orientation_weight=float(config.get("ik_orientation_weight", 0.01)),
            pos_tol_m=float(config.get("ik_pos_tol_m", 1e-3)),
        )
        robot = build_stationary_robot(with_cameras=False)
        self._controller = RobotController(robot, control_frequency=self._control_freq,
                                           test_mode=config.get("mode", "test"))

    def run(self) -> None:
        cur = self._controller.current_joints14()
        joints = self._converter.decode_chunk(self._episode.ee_chunk16, cur)
        self._controller.move_to_start_position(joints[0], duration=5.0)
        dt = 1.0 / self._control_freq
        for step, a_t in enumerate(joints[1:], start=1):
            if self._stopped:
                break
            t0 = time.perf_counter()
            ok = self._controller.execute_action(a_t)
            self._sink.on_action(step, np.asarray(a_t), time.time())
            if not ok:
                self._sink.on_status("limit_violation", {"step": step})
                break
            elapsed = time.perf_counter() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
        self._controller.disconnect()

    def stop(self) -> None:
        self._stopped = True

    def estop(self) -> None:
        self._stopped = True
        self._controller.move_to_sleep_position(duration=10.0)


def make_runner(kind: str, config: dict, sink):
    if kind == "live":
        return LiveRunner(kind, config, sink)
    if kind == "replay":
        return ReplayRunner(kind, config, sink)
    raise ValueError(f"Unknown session kind {kind!r}")
```

- [ ] **Step 2: Syntax-check (full import needs the robot env; do not block on it here)**

Run:
```bash
$PYBIN -c "import ast; ast.parse(open('webapp/runners.py').read()); print('syntax ok')"
```
Expected: `syntax ok`.

- [ ] **Step 3: Commit**
```bash
git add examples/trossen_ai/webapp/runners.py
git commit -m "feat(webapp): live + replay runners adapting hardware paths to Runner"
```

---

## Task 7: `server.py` — FastAPI REST + WebSocket

**Files:**
- Create: `examples/trossen_ai/webapp/server.py`
- Create: `examples/trossen_ai/webapp/static/index.html` (placeholder; full UI in Task 8)
- Test: `examples/trossen_ai/webapp/tests/test_server.py`

- [ ] **Step 1: Create the placeholder `index.html` so the static mount + `/` route resolve**

Create `examples/trossen_ai/webapp/static/index.html`:
```html
<!doctype html>
<html><head><meta charset="utf-8"><title>Trossen Control</title></head>
<body><h1>Trossen Control</h1></body></html>
```

- [ ] **Step 2: Write the failing tests**

Create `examples/trossen_ai/webapp/tests/test_server.py`:
```python
from fastapi.testclient import TestClient

from webapp.server import create_app


def test_index_served():
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_presets_round_trip(tmp_path):
    client = TestClient(create_app(presets_dir=tmp_path))
    assert client.get("/api/presets").json() == []
    client.post("/api/presets", json={"name": "rig-a", "config": {"control_freq": 25}})
    assert client.get("/api/presets").json() == ["rig-a"]
    assert client.get("/api/presets/rig-a").json()["control_freq"] == 25
    client.delete("/api/presets/rig-a")
    assert client.get("/api/presets").json() == []


def test_health_returns_json():
    client = TestClient(create_app())
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "session_running" in r.json()
```

- [ ] **Step 3: Run to verify it fails**
```bash
$PYBIN -m pytest webapp/tests/test_server.py -q
```
Expected: `ModuleNotFoundError: No module named 'webapp.server'`.

- [ ] **Step 4: Implement `server.py`**

Create `examples/trossen_ai/webapp/server.py`:
```python
"""FastAPI app: static UI, preset REST, health, and a telemetry WebSocket.

The hardware-bound runner factory is imported lazily inside create_app so the
REST/WebSocket layer is testable off-robot (where lerobot_robot_trossen is
absent) by injecting a fake factory.
"""
from __future__ import annotations

import asyncio
import queue
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.config_store import ConfigStore
from webapp.metrics import Metrics
from webapp.session import SessionManager
from webapp.telemetry import QueueSink

STATIC_DIR = Path(__file__).parent / "static"


def create_app(presets_dir: str | Path | None = None, runner_factory=None) -> FastAPI:
    if presets_dir is None:
        presets_dir = Path(__file__).parent / "presets"
    if runner_factory is None:
        # Defer the robot-only import until a session actually starts, so the
        # REST/WebSocket layer (and the tests) work off-robot where
        # lerobot_robot_trossen is absent.
        def runner_factory(kind, config, sink):
            from webapp.runners import make_runner
            return make_runner(kind, config, sink)

    app = FastAPI(title="Trossen Control")
    store = ConfigStore(presets_dir)
    session = SessionManager(runner_factory)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health():
        return {"session_running": session.is_running()}

    @app.get("/api/presets")
    def list_presets():
        return store.list_names()

    @app.get("/api/presets/{name}")
    def get_preset(name: str):
        return store.load(name)

    @app.post("/api/presets")
    def save_preset(body: dict):
        store.save(body["name"], body["config"])
        return {"ok": True}

    @app.delete("/api/presets/{name}")
    def delete_preset(name: str):
        store.delete(name)
        return {"ok": True}

    @app.get("/api/episodes")
    def episodes(dataset_dir: str):
        from dataset_replay import EpisodeReader  # lazy
        r = EpisodeReader(dataset_dir)
        return {"fps": r.fps, "total_episodes": r.total_episodes}

    @app.websocket("/ws/telemetry")
    async def telemetry_ws(ws: WebSocket):
        await ws.accept()
        q: queue.Queue = queue.Queue()
        metrics = Metrics()
        loop = asyncio.get_event_loop()

        async def pump():
            while True:
                try:
                    evt = await loop.run_in_executor(None, q.get, True, 0.5)
                except queue.Empty:
                    continue
                if evt["type"] == "inference":
                    metrics.add_rtt(evt["rtt_ms"])
                elif evt["type"] == "action":
                    metrics.add_action(evt["ts"], evt["action"])
                await ws.send_json(evt)

        async def metrics_tick():
            while True:
                await asyncio.sleep(0.5)
                await ws.send_json({"type": "metrics", **metrics.snapshot()})

        pump_task = asyncio.ensure_future(pump())
        metrics_task = asyncio.ensure_future(metrics_tick())
        try:
            while True:
                cmd = await ws.receive_json()
                action = cmd.get("action")
                if action == "start_live":
                    session.start("live", cmd["config"], QueueSink(q))
                elif action == "start_replay":
                    session.start("replay", cmd["config"], QueueSink(q))
                elif action == "stop":
                    await loop.run_in_executor(None, session.stop)
                elif action == "estop":
                    await loop.run_in_executor(None, session.estop)
        except WebSocketDisconnect:
            await loop.run_in_executor(None, session.stop)
        finally:
            pump_task.cancel()
            metrics_task.cancel()

    return app


app = create_app()  # for `uvicorn webapp.server:app`; safe off-robot — runners import is deferred to session start
```

- [ ] **Step 5: Run to verify it passes**
```bash
$PYBIN -m pytest webapp/tests/test_server.py -q
```
Expected: `3 passed`. The deferred `runner_factory` means the robot-only `webapp.runners` import happens only when a session starts; these tests never start one, so REST/health/preset routes work under `$PYBIN` without `lerobot_robot_trossen`.

- [ ] **Step 6: Commit**
```bash
git add examples/trossen_ai/webapp/server.py examples/trossen_ai/webapp/static/index.html examples/trossen_ai/webapp/tests/test_server.py
git commit -m "feat(webapp): FastAPI server — presets, health, telemetry WebSocket"
```

---

## Task 8: Frontend — full single-page UI

No unit tests (browser UI); verified by static-serve smoke + manual rig check.

**Files:**
- Modify: `examples/trossen_ai/webapp/static/index.html`
- Create: `examples/trossen_ai/webapp/static/app.js`
- Create: `examples/trossen_ai/webapp/static/styles.css`

- [ ] **Step 1: Replace `index.html` with the full page**

Overwrite `examples/trossen_ai/webapp/static/index.html`:
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trossen Control</title>
  <link rel="stylesheet" href="/static/styles.css">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
  <header>
    <h1>Trossen Control</h1>
    <div id="health">
      <span class="badge" id="badge-rtt">rtt</span>
      <span class="badge" id="badge-session">idle</span>
    </div>
  </header>

  <main>
    <section id="config-panel">
      <h2>Configuration</h2>
      <div class="row">
        <label>Preset <select id="preset-select"></select></label>
        <button id="preset-load">Load</button>
        <input id="preset-name" placeholder="preset name">
        <button id="preset-save">Save</button>
        <button id="preset-delete">Delete</button>
      </div>
      <form id="config-form">
        <label>Policy host <input name="policy_host" value="192.168.50.174"></label>
        <label>Policy port <input name="policy_port" type="number" value="8800"></label>
        <label>Control Hz <input name="control_freq" type="number" value="25"></label>
        <label>Max steps <input name="max_steps" type="number" value="1000"></label>
        <label>Adapter
          <select name="adapter"><option value="joint">joint</option><option value="ee">ee</option></select>
        </label>
        <label>Ensemble
          <select name="ensemble_type"><option>exp</option><option>cogact</option><option>none</option></select>
        </label>
        <label>CogACT mode
          <select name="cogact_mode"><option>cogact</option><option>latest</option><option>hybrid</option></select>
        </label>
        <label>Async <input name="async_inference" type="checkbox"></label>
        <label>Left arm only <input name="use_left_arm_only" type="checkbox"></label>
        <label>Right arm only <input name="use_right_arm_only" type="checkbox"></label>
        <label>StarVLA <input name="starvla" type="checkbox"></label>
        <label>Task prompt <input name="task_prompt" value="move the arm to the left"></label>
        <label>IK orient w <input name="ik_orientation_weight" type="number" step="0.01" value="0.01"></label>
        <label>IK pos tol (m) <input name="ik_pos_tol_m" type="number" step="0.001" value="0.001"></label>
        <fieldset>
          <legend>Replay</legend>
          <label>Dataset dir <input name="dataset_dir" value="../../converted_to_EE"></label>
          <label>Episode <input name="episode_index" type="number" value="0"></label>
        </fieldset>
      </form>
      <div class="row">
        <label>Mode
          <select id="mode-select"><option value="test">test (no movement)</option><option value="autonomous">autonomous</option></select>
        </label>
        <button id="btn-start">Start Live</button>
        <button id="btn-replay">Replay Episode</button>
        <button id="btn-stop">Stop</button>
        <button id="btn-estop" class="estop">E-STOP</button>
      </div>
    </section>

    <section id="log-panel">
      <h2>Logs</h2>
      <div id="log-box"></div>
    </section>

    <section id="charts-panel">
      <h2>Actions</h2>
      <label>Joint <input id="joint-index" type="number" value="0" min="0" max="13"></label>
      <canvas id="chart-actions"></canvas>
      <h2>Raw vs Smoothed (selected joint)</h2>
      <canvas id="chart-rawvs"></canvas>
      <h2>Overlap timeline</h2>
      <canvas id="chart-overlap"></canvas>
      <h2>Buffer size</h2>
      <canvas id="chart-buffer"></canvas>
      <h2>Metrics</h2>
      <div id="metrics-box"></div>
    </section>

    <section id="images-panel">
      <h2>Cameras</h2>
      <div id="images-box"></div>
    </section>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `styles.css`**

Create `examples/trossen_ai/webapp/static/styles.css`:
```css
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; background: #0f1115; color: #e6e6e6; }
header { display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: #171a21; }
h1 { font-size: 18px; margin: 0; }
h2 { font-size: 14px; margin: 12px 0 6px; color: #9fb3c8; }
main { display: grid; grid-template-columns: 320px 1fr 1fr; gap: 12px; padding: 12px; }
section { background: #171a21; border-radius: 8px; padding: 10px; overflow: auto; }
#config-panel { grid-row: span 2; }
label { display: block; font-size: 12px; margin: 4px 0; }
input, select { width: 100%; background: #0f1115; color: #e6e6e6; border: 1px solid #2a2f3a; border-radius: 4px; padding: 4px; }
input[type=checkbox] { width: auto; }
.row { display: flex; gap: 6px; align-items: end; flex-wrap: wrap; margin: 8px 0; }
button { background: #2a6df4; color: #fff; border: 0; border-radius: 4px; padding: 6px 10px; cursor: pointer; }
button.estop { background: #d33; font-weight: bold; }
.badge { padding: 3px 8px; border-radius: 10px; font-size: 11px; background: #444; margin-left: 4px; }
.badge.ok { background: #2a7; } .badge.bad { background: #c33; }
#log-box { height: 240px; overflow: auto; font-family: monospace; font-size: 11px; white-space: pre-wrap; }
.log-INFO { color: #cfe; } .log-WARNING { color: #fc6; } .log-ERROR { color: #f66; }
#images-box img { max-width: 100%; margin-bottom: 6px; border-radius: 4px; }
#metrics-box { font-family: monospace; font-size: 12px; white-space: pre-wrap; }
canvas { background: #0f1115; border-radius: 4px; margin-bottom: 8px; }
```

- [ ] **Step 3: Create `app.js`**

Create `examples/trossen_ai/webapp/static/app.js`:
```javascript
const $ = (id) => document.getElementById(id);
const MAXPTS = 300;

function mkChart(id, datasets) {
  return new Chart($(id), {
    type: "line",
    data: { datasets },
    options: { animation: false, responsive: true, parsing: false,
      scales: { x: { type: "linear", display: false }, y: { ticks: { color: "#9fb3c8" } } },
      plugins: { legend: { labels: { color: "#9fb3c8" } } } },
  });
}
const ds = (label, color) => ({ label, borderColor: color, data: [], pointRadius: 0, borderWidth: 1 });

const charts = {
  actions: mkChart("chart-actions", [ds("joint", "#4ad")]),
  rawvs: mkChart("chart-rawvs", [ds("raw", "#f80"), ds("smoothed", "#4ad")]),
  overlap: mkChart("chart-overlap", [ds("overlaps", "#7c7")]),
  buffer: mkChart("chart-buffer", [ds("buffer size", "#c7f")]),
};

function push(chart, dsIndex, x, y) {
  const d = chart.data.datasets[dsIndex].data;
  d.push({ x, y });
  if (d.length > MAXPTS) d.shift();
}
setInterval(() => Object.values(charts).forEach((c) => c.update("none")), 200);

function jointIdx() { return parseInt($("joint-index").value || "0", 10); }

// ---- WebSocket ----
let ws;
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws/telemetry`);
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => { setBadge("badge-session", "disconnected", "bad"); setTimeout(connect, 1000); };
}
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function logLine(level, msg) {
  const box = $("log-box");
  const div = document.createElement("div");
  div.className = "log-" + level;
  div.textContent = `[${level}] ${msg}`;
  box.appendChild(div);
  if (box.childElementCount > 500) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}
function setBadge(id, text, cls) { const b = $(id); b.textContent = text; b.className = "badge " + (cls || ""); }
const fmt = (v) => (v == null ? "—" : Number(v).toFixed(1));

function handle(evt) {
  switch (evt.type) {
    case "log": logLine(evt.level, evt.msg); break;
    case "action": {
      const j = jointIdx();
      push(charts.actions, 0, evt.step, evt.action[j]);
      if (evt.raw) push(charts.rawvs, 0, evt.step, evt.raw[j]);
      push(charts.rawvs, 1, evt.step, evt.action[j]);
      break;
    }
    case "overlap": push(charts.overlap, 0, evt.step, evt.count); break;
    case "status":
      if (evt.kind === "buffer") { push(charts.buffer, 0, Date.now() / 1000, evt.payload.size); }
      else {
        logLine("INFO", `status: ${evt.kind} ${JSON.stringify(evt.payload)}`);
        if (evt.kind === "started") setBadge("badge-session", "running", "ok");
        if (evt.kind === "stopped") setBadge("badge-session", "idle", "");
      }
      break;
    case "inference": setBadge("badge-rtt", `${evt.rtt_ms.toFixed(0)}ms`, evt.rtt_ms < 100 ? "ok" : "bad"); break;
    case "images": {
      const box = $("images-box"); box.innerHTML = "";
      for (const [name, b64] of Object.entries(evt.images)) {
        const img = new Image(); img.src = "data:image/jpeg;base64," + b64; img.title = name;
        box.appendChild(img);
      }
      break;
    }
    case "metrics":
      $("metrics-box").textContent =
        `RTT last/p50/p95: ${fmt(evt.rtt_last)}/${fmt(evt.rtt_p50)}/${fmt(evt.rtt_p95)} ms\n` +
        `Loop Hz: ${fmt(evt.loop_hz)}   Jitter: ${fmt(evt.jitter)}   Drops: ${evt.drops}`;
      break;
  }
}

// ---- config form ----
function readConfig() {
  const f = $("config-form"); const cfg = {};
  for (const el of f.elements) {
    if (!el.name) continue;
    cfg[el.name] = el.type === "checkbox" ? el.checked : el.value;
  }
  cfg.mode = $("mode-select").value;
  return cfg;
}
function applyConfig(cfg) {
  const f = $("config-form");
  for (const el of f.elements) {
    if (!el.name || !(el.name in cfg)) continue;
    if (el.type === "checkbox") el.checked = !!cfg[el.name]; else el.value = cfg[el.name];
  }
  if (cfg.mode) $("mode-select").value = cfg.mode;
}

// ---- controls ----
$("btn-start").onclick = () => {
  const cfg = readConfig();
  if (cfg.mode === "autonomous" && !confirm("Autonomous mode moves the REAL robot. Continue?")) return;
  send({ action: "start_live", config: cfg });
};
$("btn-replay").onclick = () => {
  const cfg = readConfig();
  if (cfg.mode === "autonomous" && !confirm("Replay will move the REAL robot. Continue?")) return;
  send({ action: "start_replay", config: cfg });
};
$("btn-stop").onclick = () => send({ action: "stop" });
$("btn-estop").onclick = () => send({ action: "estop" });

// ---- presets ----
async function refreshPresets() {
  const names = await (await fetch("/api/presets")).json();
  const sel = $("preset-select"); sel.innerHTML = "";
  names.forEach((n) => { const o = document.createElement("option"); o.value = o.textContent = n; sel.appendChild(o); });
}
$("preset-save").onclick = async () => {
  const name = $("preset-name").value.trim(); if (!name) return;
  await fetch("/api/presets", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, config: readConfig() }) });
  refreshPresets();
};
$("preset-load").onclick = async () => {
  const name = $("preset-select").value; if (!name) return;
  applyConfig(await (await fetch("/api/presets/" + name)).json());
};
$("preset-delete").onclick = async () => {
  const name = $("preset-select").value; if (!name) return;
  await fetch("/api/presets/" + name, { method: "DELETE" }); refreshPresets();
};

connect();
refreshPresets();
```

- [ ] **Step 4: Smoke the page load + static assets**

Run:
```bash
$PYBIN -c "
from fastapi.testclient import TestClient
from webapp.server import create_app
c = TestClient(create_app())
assert c.get('/').status_code == 200
assert c.get('/static/app.js').status_code == 200
assert c.get('/static/styles.css').status_code == 200
print('static ok')
"
```
Expected: `static ok`.

- [ ] **Step 5: Commit**
```bash
git add examples/trossen_ai/webapp/static/index.html examples/trossen_ai/webapp/static/app.js examples/trossen_ai/webapp/static/styles.css
git commit -m "feat(webapp): full single-page UI (config, logs, charts, images, metrics)"
```

---

## Task 9: README + final verification

**Files:**
- Create: `examples/trossen_ai/webapp/README.md`

- [ ] **Step 1: Write the README**

Create `examples/trossen_ai/webapp/README.md`:
```markdown
# Trossen Control Web App

Browser UI over the existing Trossen ↔ OpenPI control stack. Observe-only: it
imports the bridge / ensemble / replay code and streams telemetry; it does not
reimplement the control loop.

## Run (on the robot machine)

```bash
# one-time, in the robot runtime env (the one with lerobot_robot_trossen):
pip install "fastapi>=0.110" "uvicorn[standard]>=0.27"

cd examples/trossen_ai
python -m uvicorn webapp.server:app --host 0.0.0.0 --port 8000
# open http://<robot-host>:8000
```

## Safety

- **Test mode is the default** (no movement). Autonomous requires an explicit confirm.
- The **E-STOP** button stops the session and moves the arms to the sleep pose.

## What you get

- Config form (all CLI flags) with savable presets.
- Live log box, per-joint action charts, raw-vs-smoothed overlay, overlap
  timeline, buffer size, RTT/latency/jitter metrics, camera frames.
- Live control sessions **and** dataset-episode replay, same telemetry.

## Tests (off-hardware, lerobot env)

```bash
PYBIN=/home/edgeai/miniconda3/envs/lerobot/bin/python
$PYBIN -m pytest webapp/tests/ tests/ -q
```
```

- [ ] **Step 2: Run the full off-hardware suite**
```bash
$PYBIN -m pytest webapp/tests/ tests/ -q
```
Expected: all pass (existing 33 + telemetry 5 + buffer_size 2 + metrics 5 + config_store 5 + session 3 + server 3 = 56).

- [ ] **Step 3: Byte-compile every webapp module**
```bash
$PYBIN -m py_compile webapp/telemetry.py webapp/log_bridge.py webapp/metrics.py webapp/config_store.py webapp/session.py webapp/server.py
$PYBIN -c "import ast; ast.parse(open('webapp/runners.py').read()); print('runners syntax ok')"
```
Expected: no compile output; `runners syntax ok`.

- [ ] **Step 4: Commit**
```bash
git add examples/trossen_ai/webapp/README.md
git commit -m "docs(webapp): README + run/safety notes"
```

- [ ] **Step 5: Manual rig checklist (on the robot machine — not automated)**
  - In the robot env: `pip install fastapi "uvicorn[standard]"`, then `uvicorn webapp.server:app` starts and the page loads.
  - With policy server up: Start Live in **test** mode → logs stream, action/overlap/buffer charts move, RTT badge updates, camera frames show.
  - Save a preset, reload it.
  - Replay an episode in **test** mode → action chart tracks the decoded joints.
  - E-STOP → session stops, arms move to sleep.
  - Only then try **autonomous** mode with a clear workspace.

---

## Notes

- The only changes to existing code are the default-`NullSink` `sink` params on
  `TrossenOpenPIBridge` / `AsyncPolicyWorker`, the per-step emit calls, and
  `buffer_size()` on the ensembles. With the null sink the CLI is behaviorally
  unchanged — the existing 33 tests guard this.
- `webapp/runners.py` and `webapp/server.py`'s lazy import of it are the only
  pieces that need the robot env; everything else is tested under `$PYBIN`.
- Throttling: images are rate-limited in `QueueSink`; metrics are emitted at ~2 Hz
  by the server; the action chart keeps the last 300 points client-side.
```
