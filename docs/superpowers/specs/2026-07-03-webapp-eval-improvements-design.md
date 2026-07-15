# Web-App Eval Improvements — Design

**Date:** 2026-07-03
**Branch:** ibrahim/feat/web-app-eval
**Scope:** `examples/trossen_ai/webapp` (frontend + telemetry) and `examples/trossen_ai` (bridge/worker/cli/docs)

## Goal

Fix a cluster of usability and correctness issues in the Trossen eval web app:
navigation drift, a chart resize bug, an incomplete Compare view, and a hard block
on async inference in end-effector (EE) mode — plus a doc explaining a control-loop
performance pitfall.

## Background (as-is)

- **Nav** is hand-copied into 3 HTML files (`index.html`, `replay.html`, `teleop.html`).
  It has drifted: `replay.html`/`teleop.html` omit the **Compare** link entirely.
  Order today: `Live · Replay · Teleop · Compare`.
- **Compare** = the `/runs` page (`runs.html` + `js/runs.js`). It forces "pick 2+",
  and renders only two charts: smoothing-Δ vs step and overlap-depth vs step.
- **Per-run data** lives in `events.jsonl` (read by `run_store.py`). Event types:
  `action` (`step`, `action` = decoded 14-D joints, `raw` = pre-smoothing joints),
  `overlap` (`step`, `count`), `inference` (`rtt_ms`), `chunk`, `log`, `weights`.
  **EE pose is not persisted anywhere** — `raw` is decoded joints, not the EE target.
- **Charts** (`js/charts.js`) use Chart.js with `responsive:true,
  maintainAspectRatio:false`. Containers `.chart-cj-container` have fixed
  `height:220px` but no width bound. Grid/flex ancestors default to
  `min-width:auto`.
- **Async inference** (`async_worker.py`) takes the raw model output and slices
  `[:, :action_dim]`, adding it directly to the ensemble — it **never calls
  `adapter.decode_chunk`**. The synchronous path *does* decode via the adapter
  (`trossen_bridge.py:326`). EE mode is therefore blocked by a hard
  `NotImplementedError` (`trossen_bridge.py:274`) and a CLI `BadParameter`
  (`cli.py:110`).

## Design

Five independent clusters. A–C are frontend; D is the control loop; E is docs.

### A. Navigation — shared `nav.js` (items 1, 2, 4-reorder)

Replace the three hand-written nav blocks with one renderer.

- New `webapp/static/js/nav.js` exports a single ordered nav definition and a
  render function that injects the links into `.app-nav` and marks the active
  link by `location.pathname`.
- Order + grouping: **`Live · Compare │ Replay · Teleop`**.
  - Compare moves to right after Live (item 4-reorder).
  - A `.nav-sep` divider (`│`) separates the common pair (Live, Compare) from the
    rarely-used pair (Replay, Teleop) (item 2).
  - Compare now appears on every page, fixing its absence on Replay/Teleop (item 1).
- Each HTML page keeps a minimal `<div class="app-nav">` shell (brand label) and
  imports `nav.js`; `runs.html`'s topbar gains the same nav for consistency.
- `.nav-sep` styling added to `theme.css`.

**Unit boundary:** `nav.js` owns "what links exist and which is active." Pages own
nothing about nav ordering. Changing nav = editing one file.

### B. Compare page (items 4-single, 5)

**B1. Allow a single run (item 4).** Remove the 2+ gate in `runs.js`; render whenever
**≥1** run is selected. Update copy "pick 2+" → "pick 1+" in `runs.html`.

**B2. Complete panels (item 5).** Compare gains, reusing `charts.js`:

| Panel | Source (per-run events) | Notes |
|-------|-------------------------|-------|
| Joint traces | `action.action` (14-D) | Per-arm overlay across runs, one line per (run × joint) or grouped by arm; reuse `makeSeriesChart`. |
| Metrics | `inference.rtt_ms`, `overlap.count`, action timestamps | Summary tiles (mean RTT, effective Hz, mean overlap) + a chart. |
| End-effector | **new `ee` event** (see B3) | Rendered **only for runs that have `ee` events**; joint-mode/legacy runs simply omit this panel row. |
| Smoothing (keep) | existing | Smoothing-Δ + overlap charts stay. |

**B3. EE telemetry (new).** To make the EE panel real, log the pre-IK EE target in
EE-mode runs:

- Add a `TelemetrySink.on_ee_target(step, ee_pose, ts)` hook (default no-op in
  `NullSink`/base, fanned in `recording.py`, written as `{"type":"ee","step":…,
  "ee":[…16], "ts":…}`).
- In `trossen_bridge.py`, when the adapter is an `EEAdapter`, emit the raw model
  EE pose for the executed step alongside `on_action`. (Joint mode emits nothing —
  keeps joint runs unchanged.)
- `run_store` needs no change (it passes events through); `runs.js` reads `ee` events.

**Unit boundary:** each panel is a pure function `events[] → series[]` feeding a
Chart; panels are independent and a missing event type degrades to an omitted panel,
never an error.

### C. Chart resize bug (item 3)

Root cause: the `1fr` grid track and flex chart containers have `min-width:auto`, so
they refuse to shrink below rendered content; Chart.js `responsive` grows the canvas
but the track never shrinks it back, leaving oversized boxes clipped by the viewport.

Fix (CSS only, `theme.css`):

- `.live-grid { grid-template-columns: 320px minmax(0, 1fr); }`
- `.live-main { min-width: 0; }`
- `.chart-cj-container { min-width: 0; }`
- Apply the same `minmax(0, 1fr)` / `min-width:0` treatment to `runs-main` if it
  uses an analogous track.

No JS change. Verify by growing then shrinking the window: boxes reflow down and stay
within the viewport.

### D. Async + EE inference (item 6)

Make `AsyncPolicyWorker` adapter-aware so it decodes exactly like the sync path.

- Constructor takes the `adapter` (and ensemble) instead of a bare `action_dim`.
- `submit(obs, query_step, joints14)` — the control loop snapshots current joints at
  submit time (needed for EE IK, harmless for joints).
- Worker `_loop`: `chunk = adapter.decode_chunk(response["actions"], joints14)` then
  `ensemble.add_chunk(query_step, chunk)`. IK now runs in the worker thread — off the
  control loop, which is the entire point of async.
- `trossen_bridge.py`: pass `self.adapter` to the worker; in the async branch capture
  `joints14 = extract_joints(obs_raw)` and pass it to `submit`; **remove** the
  `NotImplementedError` gate at line 274.
- `cli.py`: **remove** the `BadParameter` at line 110; update the `async_inference`
  help text ("Background-thread inference (works in joint and EE mode)").

**Risk/assumption:** the `EEToJointsConverter` (placo IK) is only ever called from the
single worker thread, so no new concurrency hazard. Latest-wins semantics are
unchanged; a dropped stale obs just skips one IK solve.

### E. Docs — async vs inference-interval (item 7)

New `examples/trossen_ai/docs/async_vs_inference_interval.md`:

- Explains that with async **off** and `inference_interval = 1` step, the control
  loop calls `policy_client.infer()` **every step** inside the loop body
  (`trossen_bridge.py:317-323`). The loop period becomes `dt + rtt`, so effective
  control rate collapses toward `1 / rtt` (network/GPU-bound) instead of the target
  control frequency.
- Contrasts with async, where inference runs in a background thread and the loop
  holds control-frequency, consuming blended actions from the ensemble.
- Practical guidance: if running sync, use `inference_interval > 1` (chunked) or
  enable async; note the interaction with smoothing/ensemble.
- Cross-link from `docs/README.md`.

## Testing

- **D (async+EE):** unit test `AsyncPolicyWorker` with a fake adapter + fake client:
  assert the worker calls `decode_chunk(raw, joints14)` and adds the *decoded* chunk
  to the ensemble; assert joint-mode still works (JointAdapter identity). Extend/adjust
  existing `test_ee_to_joints.py` / worker coverage.
- **B (compare):** the `events[] → series[]` panel functions are pure — unit test each
  (joints, metrics, ee, single-run rendering with 1 selected).
- **A (nav):** light DOM test that `nav.js` marks the active link by path and renders
  the separator; or manual verification across the 4 pages.
- **C (resize):** manual — grow then shrink the browser; confirm no clipped boxes.
- **E:** docs only.

## Out of scope

- Refactoring the telemetry/recording architecture beyond the new `on_ee_target` hook.
- Any change to the ensemble/smoothing math.
- Unrelated nav/visual redesign.

## File-change summary

- `webapp/static/js/nav.js` (new), `index.html`, `replay.html`, `teleop.html`,
  `runs.html`, `theme.css` — A, C.
- `webapp/static/js/runs.js`, `runs.html` — B1, B2.
- `webapp/telemetry.py`, `webapp/recording.py`, `trossen_bridge.py` — B3, D.
- `async_worker.py`, `cli.py` — D.
- `examples/trossen_ai/docs/async_vs_inference_interval.md` (new), `docs/README.md` — E.
