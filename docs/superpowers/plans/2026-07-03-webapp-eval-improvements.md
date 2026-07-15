# Web-App Eval Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix navigation drift, a chart-resize bug, and an incomplete Compare view in the Trossen eval web app; unblock async inference in end-effector (EE) mode; and document a control-loop performance pitfall.

**Architecture:** Frontend is a static vanilla-JS app (`examples/trossen_ai/webapp/static`) using Chart.js. Telemetry flows control-loop → `TelemetrySink` (NullSink / QueueSink / RecordingSink, fanned by `TeeSink`) → `events.jsonl` per run. The control loop (`trossen_bridge.py`) decodes policy output through an `ActionSpaceAdapter` (JointAdapter identity / EEAdapter IK). Async inference runs the same decode in a background worker.

**Tech Stack:** Python 3.11 + pytest (control loop / telemetry), vanilla ES modules + Chart.js (frontend), Typer (CLI).

**Conventions:**
- Run Python tests from `examples/trossen_ai/`: `cd examples/trossen_ai && python -m pytest <path> -v`.
- Frontend has no JS test runner — JS/CSS tasks use manual browser verification.
- Commit after each task.

---

## File Structure

**New files**
- `examples/trossen_ai/webapp/static/js/nav.js` — single source of truth for the top nav (links, order, separator, active-by-path). (Task 1)
- `examples/trossen_ai/docs/async_vs_inference_interval.md` — perf explainer. (Task 7)

**Modified**
- `webapp/static/index.html`, `replay.html`, `teleop.html`, `runs.html` — use `nav.js`. (Task 1)
- `webapp/static/theme.css` — `.nav-sep` style (Task 1); resize `min-width:0` fixes (Task 2).
- `webapp/static/js/runs.js` + `runs.html` — single-run + joints/metrics/EE panels. (Tasks 3, 5)
- `adapters.py` — `ee_chunk()` hook. (Task 4)
- `webapp/telemetry.py`, `webapp/recording.py` — `on_ee_chunk` callback + `ee` event. (Task 4)
- `trossen_bridge.py` — emit `on_ee_chunk`; adapter-aware worker; remove EE-async gate. (Tasks 4, 6)
- `async_worker.py` — adapter-aware decode. (Task 6)
- `cli.py` — remove EE-async gate, update help. (Task 6)
- `docs/README.md` — link new doc. (Task 7)

---

## Task 1: Shared nav (items 1, 2, 4-reorder)

Replace the three hand-written nav blocks with one renderer. Result on every page:
`Live · Compare │ Replay · Teleop` (Compare after Live; `│` separator before the rarely-used pair; Compare now present on all pages).

**Files:**
- Create: `webapp/static/js/nav.js`
- Modify: `webapp/static/index.html:14-16`, `replay.html:23-24`, `teleop.html:21-22`, `runs.html:13`
- Modify: `webapp/static/theme.css` (add `.nav-sep`)

- [ ] **Step 1: Create `nav.js`**

```javascript
// webapp/static/js/nav.js
// Single source of truth for the top navigation. Ordered links with a visual
// separator between the common pair (Live, Compare) and the rarely-used pair
// (Replay, Teleop). Active link is marked by pathname.
const LINKS = [
  { href: "/", label: "Live" },
  { href: "/runs", label: "Compare" },
  { sep: true },
  { href: "/replay", label: "Replay" },
  { href: "/teleop", label: "Teleop" },
];

export function renderNav(container) {
  const path = location.pathname;
  container.innerHTML =
    `<strong style="color:var(--text);margin-right:16px">Trossen Control</strong>` +
    LINKS.map(l => {
      if (l.sep) return `<span class="nav-sep">|</span>`;
      const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
      return `<a href="${l.href}"${active ? ' class="active"' : ""}>${l.label}</a>`;
    }).join("");
}

const el = document.querySelector(".app-nav");
if (el) renderNav(el);
```

- [ ] **Step 2: Point `index.html` at `nav.js`**

Replace the nav contents at `index.html:14-16` (the `<div class="app-nav">…</div>` block) with an empty shell + module import. Change:

```html
    <div class="app-nav">
      <strong style="color:var(--text);margin-right:16px">Trossen Control</strong>
      <a href="/" class="active">Live</a><a href="/replay">Replay</a><a href="/teleop">Teleop</a><a href="/runs">Compare</a>
    </div>
```

to:

```html
    <div class="app-nav"></div>
    <script type="module" src="/static/js/nav.js"></script>
```

- [ ] **Step 3: Same for `replay.html:23-24`**

Replace:

```html
    <div class="app-nav"><strong style="color:var(--text);margin-right:16px">Trossen Control</strong>
      <a href="/">Live</a><a href="/replay" class="active">Replay</a><a href="/teleop">Teleop</a></div>
```

with:

```html
    <div class="app-nav"></div>
    <script type="module" src="/static/js/nav.js"></script>
```

- [ ] **Step 4: Same for `teleop.html:21-22`**

Replace:

```html
    <div class="app-nav"><strong style="color:var(--text);margin-right:16px">Trossen Control</strong>
      <a href="/">Live</a><a href="/replay">Replay</a><a href="/teleop" class="active">Teleop</a></div>
```

with:

```html
    <div class="app-nav"></div>
    <script type="module" src="/static/js/nav.js"></script>
```

- [ ] **Step 5: Add nav to `runs.html:13`**

`runs.html` uses a `.topbar`, not `.app-header`. Replace line 13:

```html
  <header class="topbar"><h1>Compare Runs</h1><a href="/" class="btn-sm">← Live</a></header>
```

with:

```html
  <header class="app-header">
    <div class="app-nav"></div>
    <script type="module" src="/static/js/nav.js"></script>
  </header>
```

- [ ] **Step 6: Add `.nav-sep` style to `theme.css`**

Search for `.app-nav` in `theme.css` and add this rule right after it:

```css
.nav-sep { color: var(--border); margin: 0 10px; user-select: none; }
```

- [ ] **Step 7: Manual verify**

Start the webapp (`cd examples/trossen_ai && python -m webapp.server` or the documented launch command), open each of `/`, `/replay`, `/teleop`, `/runs`. Confirm:
- Every page shows `Live · Compare │ Replay · Teleop`.
- The correct link is bold/active per page.
- The `|` separator renders between Compare and Replay.

- [ ] **Step 8: Commit**

```bash
git add examples/trossen_ai/webapp/static/js/nav.js examples/trossen_ai/webapp/static/index.html examples/trossen_ai/webapp/static/replay.html examples/trossen_ai/webapp/static/teleop.html examples/trossen_ai/webapp/static/runs.html examples/trossen_ai/webapp/static/theme.css
git commit -m "feat(trossen_ai): shared nav.js with Live·Compare | Replay·Teleop grouping"
```

---

## Task 2: Chart resize fix (item 3)

Root cause: `1fr` grid track + flex chart containers have `min-width:auto`, so they never shrink below rendered content; Chart.js `responsive` grows the canvas but the track won't let it shrink back → oversized boxes clipped when the window narrows.

**Files:**
- Modify: `webapp/static/theme.css` (`.live-grid`, `.live-main`, `.chart-cj-container`; `.runs-main` if applicable)

- [ ] **Step 1: Bound the grid track and allow children to shrink**

In `theme.css`, change `.live-grid` (currently `grid-template-columns:320px 1fr`):

```css
.live-grid { display:grid; grid-template-columns:320px minmax(0,1fr); gap:14px; padding:16px; }
```

Add/adjust:

```css
.live-main { min-width: 0; }
.chart-cj-container { min-width: 0; }
```

If a `.runs-main` grid/flex rule exists with a `1fr` track, apply the same `minmax(0,1fr)` treatment; if `.runs-main` is a single-column flow, no change needed.

- [ ] **Step 2: Manual verify**

Reload `/`. Widen the browser window until charts grow, then narrow it back below the starting width. Confirm the cards/charts reflow smaller and stay fully inside the viewport (no horizontal clipping, no half-hidden boxes). Repeat on `/runs`.

- [ ] **Step 3: Commit**

```bash
git add examples/trossen_ai/webapp/static/theme.css
git commit -m "fix(trossen_ai): charts shrink on window resize (min-width:0 + minmax track)"
```

---

## Task 3: Compare — single run + joint/metrics panels (items 4-single, 5 joints+metrics)

Allow rendering with ≥1 run selected, and add joint-trace + metrics panels beside the existing smoothing charts. Pure `events[]→series[]` helpers keep it verifiable by inspection.

**Files:**
- Modify: `webapp/static/runs.html:14-31` (add panels + copy)
- Modify: `webapp/static/js/runs.js`
- Modify: `webapp/static/theme.css` (metrics rows)

- [ ] **Step 1: Add panels + fix copy in `runs.html`**

Change the runs-list heading copy at `runs.html:16`:

```html
      <h2>Runs <span class="muted">(pick 1+ to compare)</span></h2>
```

Insert a joint-picker + joint chart and a metrics section into `<main class="runs-main">`, after the existing "Config diff" card and before the "Smoothing Δ vs step" card:

```html
    <section class="card">
      <h2>Joint traces
        <select id="runs-joint-sel" class="btn-sm"></select>
      </h2>
      <div class="chart-cj-container"><canvas id="runs-chart-joint"></canvas></div>
    </section>
    <section class="card">
      <h2>Metrics</h2>
      <div id="runs-metrics" class="runs-metrics"><em class="muted">Select runs above.</em></div>
      <div class="chart-cj-container"><canvas id="runs-chart-rtt"></canvas></div>
    </section>
```

- [ ] **Step 2: Import joint names and add series helpers in `runs.js`**

Change the import at `runs.js:2` to also pull joint names:

```javascript
import { makeSeriesChart, JOINT_NAMES_14 } from "./charts.js";
```

Add these pure helpers after `overlapSeries` (`runs.js:58`):

```javascript
// action[jointIdx] vs step, from action events.
function jointSeries(events, jointIdx) {
  return events
    .filter(e => e.type === "action" && Array.isArray(e.action) && e.action.length > jointIdx)
    .map(e => ({ x: e.step, y: e.action[jointIdx] }));
}
// inference RTT vs event index (inference events carry no step).
function rttSeries(events) {
  let i = 0;
  return events.filter(e => e.type === "inference").map(e => ({ x: i++, y: e.rtt_ms }));
}
// summary tiles: mean RTT (ms), effective Hz (from action ts deltas), mean overlap.
function metricsSummary(events) {
  const rtts = events.filter(e => e.type === "inference").map(e => e.rtt_ms);
  const ots = events.filter(e => e.type === "overlap").map(e => e.count);
  const ts = events.filter(e => e.type === "action" && typeof e.ts === "number").map(e => e.ts);
  const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : null;
  let hz = null;
  if (ts.length > 1) { const span = ts[ts.length - 1] - ts[0]; if (span > 0) hz = (ts.length - 1) / span; }
  return { rtt: mean(rtts), hz, overlap: mean(ots) };
}
```

- [ ] **Step 3: Populate the joint selector once, on load**

Add at the end of `loadList()` (after the `querySelectorAll(...)` block, before its closing brace at `runs.js:26`):

```javascript
  const sel = $("runs-joint-sel");
  if (sel && !sel.options.length) {
    sel.innerHTML = JOINT_NAMES_14.map((n, i) => `<option value="${i}">${n}</option>`).join("");
    sel.addEventListener("change", refresh);
  }
```

- [ ] **Step 4: Single-run gate + render new panels in `refresh()`**

Replace the whole `refresh()` function (`runs.js:60-75`) with:

```javascript
let jointChart = null, rttChart = null;

async function refresh() {
  const ids = [...selected];
  if (ids.length < 1) {
    $("runs-diff").innerHTML = '<em class="muted">Select 1+ runs.</em>';
    $("runs-metrics").innerHTML = '<em class="muted">Select runs above.</em>';
    return;
  }
  const runs = await Promise.all(ids.map(async id => ({ id, ...(await fetchRun(id)) })));
  $("runs-diff").innerHTML = configDiff(runs);

  const jointIdx = Number($("runs-joint-sel").value || 0);
  const deltaData = runs.map(r => ({ label: r.id, points: deltaSeries(r.events) }));
  const overlapData = runs.map(r => ({ label: r.id, points: overlapSeries(r.events) }));
  const jointData = runs.map(r => ({ label: r.id, points: jointSeries(r.events, jointIdx) }));
  const rttData = runs.map(r => ({ label: r.id, points: rttSeries(r.events) }));

  if (deltaChart) deltaChart.destroy();
  if (overlapChart) overlapChart.destroy();
  if (jointChart) jointChart.destroy();
  if (rttChart) rttChart.destroy();
  deltaChart = makeSeriesChart($("runs-chart-delta"), deltaData);
  overlapChart = makeSeriesChart($("runs-chart-overlap"), overlapData);
  jointChart = makeSeriesChart($("runs-chart-joint"), jointData);
  rttChart = makeSeriesChart($("runs-chart-rtt"), rttData);

  $("runs-metrics").innerHTML = runs.map(r => {
    const m = metricsSummary(r.events);
    const f = (v, d) => v == null ? "—" : v.toFixed(d);
    return `<div class="runs-metric-row"><b>${r.id}</b>
      <span>RTT ${f(m.rtt, 1)} ms</span>
      <span>${f(m.hz, 1)} Hz</span>
      <span>overlap ${f(m.overlap, 2)}</span></div>`;
  }).join("");
}
```

- [ ] **Step 5: Add minimal styling for metrics rows in `theme.css`**

```css
.runs-metrics { display:flex; flex-direction:column; gap:6px; margin-bottom:10px; }
.runs-metric-row { display:flex; gap:14px; font:12px var(--font); color:var(--muted); }
.runs-metric-row b { color:var(--text); min-width:120px; }
```

- [ ] **Step 6: Manual verify**

Reload `/runs`. Select **one** run: confirm config diff, joint chart (default joint), RTT chart, and a metrics row all render (no "select 2+" block). Change the joint dropdown → joint chart updates. Select a second run → all charts overlay both runs and two metrics rows show. Smoothing-Δ and overlap charts still render.

- [ ] **Step 7: Commit**

```bash
git add examples/trossen_ai/webapp/static/js/runs.js examples/trossen_ai/webapp/static/runs.html examples/trossen_ai/webapp/static/theme.css
git commit -m "feat(trossen_ai): compare supports single run + joint/metrics panels"
```

---

## Task 4: EE telemetry (item 5 EE data)

Persist the model's per-step EE target so the Compare EE panel is real. Uniform across sync and async: whenever an EE chunk is added, emit it via `on_ee_chunk`; sinks join it to executed steps and write `{"type":"ee","step":…,"ee":[…],"ts":…}`.

**Files:**
- Modify: `adapters.py` (add `ee_chunk`)
- Modify: `webapp/telemetry.py` (Protocol, NullSink, QueueSink)
- Modify: `webapp/recording.py` (TeeSink, RecordingSink)
- Modify: `trossen_bridge.py:329` (emit in sync path)
- Test: `webapp/tests/test_ee_telemetry.py` (new)

- [ ] **Step 1: Write the failing test**

Create `examples/trossen_ai/webapp/tests/test_ee_telemetry.py`:

```python
import json
import numpy as np
from webapp.recording import RecordingSink


def test_recording_sink_writes_ee_event(tmp_path):
    sink = RecordingSink(tmp_path / "run1", run_id="run1", config={})
    # EE chunk of 3 steps, 16-D each, added for query_step=0
    ee_chunk = np.arange(3 * 16, dtype=float).reshape(3, 16)
    sink.on_ee_chunk(0, ee_chunk, ts=1.0)
    # executed step 1 -> should log ee row 1
    sink.on_action(1, np.zeros(14), ts=2.0)
    sink.close()

    lines = (tmp_path / "run1" / "events.jsonl").read_text().splitlines()
    ee_events = [json.loads(x) for x in lines if json.loads(x).get("type") == "ee"]
    assert len(ee_events) == 1
    assert ee_events[0]["step"] == 1
    assert ee_events[0]["ee"] == list(range(16, 32))
```

- [ ] **Step 2: Run it — expect fail**

```bash
cd examples/trossen_ai && python -m pytest webapp/tests/test_ee_telemetry.py -v
```
Expected: FAIL — `RecordingSink` has no `on_ee_chunk`.

- [ ] **Step 3: Add `ee_chunk` to adapters**

In `adapters.py`, add to `ActionSpaceAdapter` (after `decode_chunk`, ~line 32):

```python
    def ee_chunk(self, raw_chunk: np.ndarray):
        """EE targets (N, 16) for logging, or None if this space is not EE."""
        return None
```

Add to `EEAdapter` (after its `decode_chunk`, ~line 61):

```python
    def ee_chunk(self, raw_chunk: np.ndarray):
        return np.asarray(raw_chunk)[:, :16]
```

- [ ] **Step 4: Add `on_ee_chunk` to telemetry sinks**

In `webapp/telemetry.py`, add to the `TelemetrySink` Protocol (after `on_chunk`, line 20):

```python
    def on_ee_chunk(self, query_step: int, ee_chunk: np.ndarray, ts: float) -> None: ...
```

Add the same no-op line to `NullSink` (after its `on_chunk`, line 33):

```python
    def on_ee_chunk(self, query_step: int, ee_chunk: np.ndarray, ts: float) -> None: ...
```

In `QueueSink.__init__`, add a tracker (after `self._last_chunk` line 51):

```python
        self._last_ee_chunk: tuple[int, np.ndarray] | None = None
```

Add the callback (after `on_chunk`, ~line 98):

```python
    def on_ee_chunk(self, query_step: int, ee_chunk: np.ndarray, ts: float) -> None:
        self._last_ee_chunk = (query_step, np.asarray(ee_chunk))
```

Emit the `ee` event from `QueueSink.on_action` — add at the very end of that method (after the existing `self._put({...action...})` block, ~line 90):

```python
        if self._last_ee_chunk is not None:
            _qs, _ee = self._last_ee_chunk
            _off = step - _qs
            if 0 <= _off < len(_ee):
                self._put({"type": "ee", "step": step,
                           "ee": np.asarray(_ee[_off]).flatten().tolist(), "ts": ts})
```

- [ ] **Step 5: Add `on_ee_chunk` to recording sinks**

In `webapp/recording.py`, add to `TeeSink` (after the `on_chunk` fan, line 41):

```python
    def on_ee_chunk(self, query_step, ee_chunk, ts): self._fan("on_ee_chunk", query_step, ee_chunk, ts)
```

In `RecordingSink.__init__`, add a tracker (after `self._last_chunk = None`, line 74):

```python
        self._last_ee_chunk = None  # (query_step, np.ndarray) for EE lookup
```

Add the callback (near `on_chunk`, ~line 135):

```python
    def on_ee_chunk(self, query_step, ee_chunk, ts):
        self._last_ee_chunk = (query_step, np.asarray(ee_chunk))
```

Emit the `ee` event from `RecordingSink.on_action` — insert immediately before the final `self._put({"type": "action", …})` line (~line 129):

```python
        if self._last_ee_chunk is not None:
            _qs, _ee = self._last_ee_chunk
            _off = step - _qs
            if 0 <= _off < len(_ee):
                self._put({"type": "ee", "step": step,
                           "ee": np.asarray(_ee[_off]).flatten().tolist(), "ts": ts})
```

- [ ] **Step 6: Run the test — expect pass**

```bash
cd examples/trossen_ai && python -m pytest webapp/tests/test_ee_telemetry.py -v
```
Expected: PASS.

- [ ] **Step 7: Emit `on_ee_chunk` from the sync control loop**

In `trossen_bridge.py`, in the synchronous branch right after `self.sink.on_chunk(...)` (line 329), add:

```python
                        _ee = self.adapter.ee_chunk(response["actions"])
                        if _ee is not None:
                            self.sink.on_ee_chunk(self.episode_step, _ee, time.time())
```

(Async emission is added in Task 6, inside the worker.)

- [ ] **Step 8: Run the telemetry/recording suites**

```bash
cd examples/trossen_ai && python -m pytest webapp/tests/test_recording.py webapp/tests/test_telemetry.py webapp/tests/test_ee_telemetry.py -v
```
Expected: PASS (no regressions).

- [ ] **Step 9: Commit**

```bash
git add examples/trossen_ai/adapters.py examples/trossen_ai/webapp/telemetry.py examples/trossen_ai/webapp/recording.py examples/trossen_ai/trossen_bridge.py examples/trossen_ai/webapp/tests/test_ee_telemetry.py
git commit -m "feat(trossen_ai): log per-step EE target as ee events (sync path)"
```

---

## Task 5: Compare EE panel (item 5 EE render)

Render an EE-trace panel that appears only for selected runs that actually have `ee` events (joint-mode/legacy runs omit it, never error).

**Files:**
- Modify: `webapp/static/runs.html` (EE panel markup)
- Modify: `webapp/static/js/runs.js` (`eeSeries`, `hasEE`, conditional render + EE dim picker)

- [ ] **Step 1: Add EE panel markup to `runs.html`**

After the Metrics `<section>` added in Task 3, insert:

```html
    <section class="card" id="runs-ee-card" style="display:none">
      <h2>End-effector
        <select id="runs-ee-sel" class="btn-sm"></select>
      </h2>
      <div class="chart-cj-container"><canvas id="runs-chart-ee"></canvas></div>
    </section>
```

- [ ] **Step 2: Add EE helpers to `runs.js`**

After `jointSeries` (from Task 3), add:

```javascript
// 16-D EE target names: per arm [x,y,z,qw,qx,qy,qz,grip].
const EE_NAMES_16 = ["l_x","l_y","l_z","l_qw","l_qx","l_qy","l_qz","l_grip",
                     "r_x","r_y","r_z","r_qw","r_qx","r_qy","r_qz","r_grip"];
function hasEE(events) { return events.some(e => e.type === "ee"); }
function eeSeries(events, dim) {
  return events
    .filter(e => e.type === "ee" && Array.isArray(e.ee) && e.ee.length > dim)
    .map(e => ({ x: e.step, y: e.ee[dim] }));
}
```

- [ ] **Step 3: Populate the EE dim selector once, on load**

In `loadList()`, alongside the joint-selector population from Task 3, add:

```javascript
  const esel = $("runs-ee-sel");
  if (esel && !esel.options.length) {
    esel.innerHTML = EE_NAMES_16.map((n, i) => `<option value="${i}">${n}</option>`).join("");
    esel.addEventListener("change", refresh);
  }
```

- [ ] **Step 4: Conditionally render the EE chart in `refresh()`**

Add an `eeChart` handle to the module-level chart declarations (the `let jointChart = null, rttChart = null;` line from Task 3):

```javascript
let jointChart = null, rttChart = null, eeChart = null;
```

At the end of `refresh()` (after the metrics block), add:

```javascript
  const eeRuns = runs.filter(r => hasEE(r.events));
  const eeCard = $("runs-ee-card");
  if (eeChart) { eeChart.destroy(); eeChart = null; }
  if (eeRuns.length) {
    eeCard.style.display = "";
    const eeDim = Number($("runs-ee-sel").value || 0);
    const eeData = eeRuns.map(r => ({ label: r.id, points: eeSeries(r.events, eeDim) }));
    eeChart = makeSeriesChart($("runs-chart-ee"), eeData);
  } else {
    eeCard.style.display = "none";
  }
```

- [ ] **Step 5: Manual verify**

Record (or reuse) an EE-mode run via `live-ee`, plus a joint run. On `/runs`:
- Select the joint-only run → EE card is hidden.
- Select the EE run → EE card appears; changing the EE-dim dropdown updates the chart.
- Select both → EE card shows only the EE run's trace; all other panels show both.

- [ ] **Step 6: Commit**

```bash
git add examples/trossen_ai/webapp/static/js/runs.js examples/trossen_ai/webapp/static/runs.html
git commit -m "feat(trossen_ai): compare EE-trace panel (shown only for EE runs)"
```

---

## Task 6: Async + EE inference (item 6)

Make `AsyncPolicyWorker` decode through the adapter exactly like the sync path, so EE works async; remove the two gates; emit EE telemetry from the worker.

**Files:**
- Modify: `async_worker.py`
- Modify: `trossen_bridge.py:100` (worker ctor), `:274-275` (remove gate), `:299` (pass joints14)
- Modify: `cli.py:93` (help), `:107-108` (docstring), `:110-111` (remove gate), `:137` (forward flag)
- Test: `webapp/tests/test_async_worker.py` (new)

- [ ] **Step 1: Write the failing test**

Create `examples/trossen_ai/webapp/tests/test_async_worker.py`:

```python
import numpy as np
from async_worker import AsyncPolicyWorker


class FakeClient:
    def __init__(self): self.calls = 0
    def infer(self, obs):
        self.calls += 1
        # raw model chunk: 2 steps x 20 dims
        return {"actions": np.ones((2, 20), dtype=float)}


class DoublingAdapter:
    """Stand-in adapter: 'decodes' by taking first 14 dims and doubling them."""
    def decode_chunk(self, raw_chunk, current_joints14):
        return np.asarray(raw_chunk)[:, :14] * 2.0
    def ee_chunk(self, raw_chunk):
        return None


class FakeEnsemble:
    def __init__(self): self.added = []
    def add_chunk(self, query_step, chunk): self.added.append((query_step, np.asarray(chunk)))


def test_worker_decodes_via_adapter():
    ens = FakeEnsemble()
    w = AsyncPolicyWorker(FakeClient(), ens, DoublingAdapter())
    w.start()
    w.submit({"x": 1}, query_step=5, joints14=np.zeros(14))
    assert w.wait_for_first(timeout=5.0)
    w.stop()
    assert ens.added, "ensemble received no chunk"
    qs, chunk = ens.added[0]
    assert qs == 5
    assert chunk.shape == (2, 14)          # decoded to joint dim
    assert np.allclose(chunk, 2.0)         # adapter's decode applied (not raw slice)
```

- [ ] **Step 2: Run it — expect fail**

```bash
cd examples/trossen_ai && python -m pytest webapp/tests/test_async_worker.py -v
```
Expected: FAIL — `submit()` has no `joints14` param / ctor takes `action_dim`.

- [ ] **Step 3: Make the worker adapter-aware**

In `async_worker.py`, change the constructor to take `adapter` instead of `action_dim`:

```python
    def __init__(
        self,
        policy_client,
        ensemble: TemporalEnsemble,
        adapter,
        sink: TelemetrySink = NullSink(),  # noqa
    ) -> None:
        self._client = policy_client
        self._ensemble = ensemble
        self._adapter = adapter
        self._sink = sink
        self._pending: tuple | None = None  # (obs_dict, query_step, joints14)
        self._lock = threading.Lock()
        self._first_result = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None
```

Change `submit`:

```python
    def submit(self, obs: dict, query_step: int, joints14) -> None:
        """Submit a fresh observation + current joints. Non-blocking. Overwrites any pending stale obs."""
        with self._lock:
            self._pending = (obs, query_step, joints14)
```

In `_loop`, replace the unpack line `obs, query_step = item` and the inference/slice block (lines 98-109) with:

```python
            obs, query_step, joints14 = item
            try:
                t0 = time.perf_counter()
                response = self._client.infer(obs)
                rtt_ms = (time.perf_counter() - t0) * 1e3
                self._sink.on_inference(rtt_ms, time.time())
                chunk = self._adapter.decode_chunk(response["actions"], joints14)
                self._ensemble.add_chunk(query_step, chunk)
                self._sink.on_chunk(query_step, chunk, time.time())
                ee = self._adapter.ee_chunk(response["actions"])
                if ee is not None:
                    self._sink.on_ee_chunk(query_step, ee, time.time())
                if first:
                    self._first_result.set()
                    first = False
            except Exception:
                logger.exception("AsyncPolicyWorker: inference error")
```

Also update the class docstring usage example `submit(obs, episode_step)` → `submit(obs, episode_step, joints14)`.

- [ ] **Step 4: Run the test — expect pass**

```bash
cd examples/trossen_ai && python -m pytest webapp/tests/test_async_worker.py -v
```
Expected: PASS.

- [ ] **Step 5: Update `trossen_bridge.py` construction + gate + submit**

At `trossen_bridge.py:100`, change worker construction to pass the adapter:

```python
            AsyncPolicyWorker(self.policy_client, self.ensemble, self.adapter, sink=self.sink)
```

Delete the gate at lines 274-275 entirely:

```python
        if self.async_inference and not isinstance(self.adapter, JointAdapter):
            raise NotImplementedError("Async inference with EE decoding is not supported yet.")
```

(Leave the `JointAdapter` import as-is; it is still referenced elsewhere. Remove it only if the linter flags it unused.)

At line 299, pass current joints to submit:

```python
                    self._policy_worker.submit(obs, self.episode_step, extract_joints(obs_raw))
```

- [ ] **Step 6: Update `cli.py` — remove EE gate, forward flag, fix help/docstring**

At `cli.py:93`, change the help text:

```python
    async_inference: bool = typer.Option(False, help="Background-thread inference (joint and EE mode)"),
```

In the `live_ee` docstring (`cli.py:107-108`), replace "Synchronous inference only (async EE decoding is not yet supported)." with "Supports `--async-inference` (IK runs in the worker thread)."

Delete the gate at lines 110-111:

```python
    if async_inference:
        raise typer.BadParameter("--async-inference is not supported in EE mode yet.")
```

At `cli.py:137`, forward the flag (currently hard-coded `async_inference=False`):

```python
        async_inference=async_inference,
```

- [ ] **Step 7: Run the affected suites**

```bash
cd examples/trossen_ai && python -m pytest webapp/tests/test_async_worker.py tests/test_ee_to_joints.py webapp/tests/test_cli_teleop.py -v
```
Expected: PASS. If any existing test constructed `AsyncPolicyWorker(..., action_dim)` positionally, update that call to pass an adapter (see Step 1 for the pattern).

- [ ] **Step 8: Commit**

```bash
git add examples/trossen_ai/async_worker.py examples/trossen_ai/trossen_bridge.py examples/trossen_ai/cli.py examples/trossen_ai/webapp/tests/test_async_worker.py
git commit -m "feat(trossen_ai): async inference works in EE mode (adapter decodes in worker)"
```

---

## Task 7: Docs — async vs inference-interval (item 7)

**Files:**
- Create: `examples/trossen_ai/docs/async_vs_inference_interval.md`
- Modify: `examples/trossen_ai/docs/README.md`

- [ ] **Step 1: Write the doc**

Create `examples/trossen_ai/docs/async_vs_inference_interval.md`:

```markdown
# Async inference vs. inference interval — control-loop timing

## TL;DR

Disabling async inference **and** setting the inference interval to **1 step**
makes the control loop call the policy server on **every** step. Because that call
is synchronous, the loop blocks on network + GPU latency each step, so the effective
control rate collapses from the target frequency toward `1 / RTT`.

## Why

The synchronous control loop (`trossen_bridge.py`, `run_episode`) requests a new
action chunk when `current_action_chunk is None or action_chunk_idx >= rate_of_inference`.
With `rate_of_inference = 1` that predicate is true every step, so each iteration runs:

    observation = build_observation(...)
    response = policy_client.infer(observation)   # blocks on server RTT
    chunk = adapter.decode_chunk(...)             # + IK in EE mode

The loop period becomes roughly:

    period ≈ dt + RTT_infer (+ IK_time in EE mode)
    effective_rate ≈ 1 / period

At a 25 Hz target (`dt = 40 ms`) an inference RTT of 60 ms drops the loop to
~10 Hz — the arm updates slower and motion looks laggy or jerky.

With `rate_of_inference > 1`, the loop only pays the RTT once per N steps and
replays chunk rows in between, so it holds close to the target frequency.

## What async does differently

With async enabled (`--async-inference`), inference runs in a background thread
(`async_worker.py`). Every control step submits the latest observation (non-blocking)
and reads a blended action from the temporal ensemble. The loop never waits on the
server, so it holds the control frequency regardless of RTT; new predictions are
folded in as they arrive. In EE mode the IK decode also happens in the worker thread,
off the control loop.

## Guidance

- Prefer async for smooth motion when the policy server RTT is non-trivial.
- If you must run synchronously, keep `rate_of_inference > 1` (chunked replay).
- `inference interval = 1` + async off is the worst case for loop rate — use it only
  for debugging single-step behavior, not for real runs.
```

- [ ] **Step 2: Link it from `docs/README.md`**

Add a bullet under the docs index in `examples/trossen_ai/docs/README.md` (match the existing list format):

```markdown
- [Async vs. inference interval](async_vs_inference_interval.md) — why sync + interval=1 slows the control loop.
```

- [ ] **Step 3: Commit**

```bash
git add examples/trossen_ai/docs/async_vs_inference_interval.md examples/trossen_ai/docs/README.md
git commit -m "docs(trossen_ai): explain async-off + inference-interval=1 loop slowdown"
```

---

## Final verification

- [ ] **Full Python suite**

```bash
cd examples/trossen_ai && python -m pytest -q
```
Expected: PASS (no regressions from telemetry/worker changes).

- [ ] **Manual frontend pass**: nav on all 4 pages; resize shrink on `/` and `/runs`; Compare with 1 run / 2 runs / an EE run.
