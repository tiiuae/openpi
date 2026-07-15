# Live Experiment Logging — Design

**Date:** 2026-07-02
**Branch:** ibrahim/feat/web-app-eval
**Status:** Approved, ready for implementation plan

## Purpose

Record each Live (autonomous) run of the Trossen webapp so runs can be
**compared as evaluation experiments** — e.g. smoothing on/off, temporal vs
CogACT, goal-time multiplier, joint vs EE, and across policy model versions.
This backs the existing comparison notes (`LIVE_VS_MAIN_JOINT_EVAL.md`,
`ACTION_SMOOTHING_COGACT.md`, `GOAL_TIME_MULTIPLIER_LIVE.md`) with real
captured data instead of hand-written observations.

## Decisions (locked)

| Question | Decision |
|----------|----------|
| Primary purpose | Compare eval runs |
| Images | **Not logged** in v1 (actions / smoothing / metrics / config only) |
| Run outcome | **Dedicated quick-rating prompt** on session end (success + score + note) |
| Feedback box | **Not** reused — feedback card stays webapp bug-reporting only |
| Policy model name | **User-typed config field** (server does not reliably send it); `get_server_metadata()` stored as bonus if non-empty |
| Scope | Logs + run list + **compare view** |

## Non-Goals (v1)

- No image / video capture.
- No retraining-dataset (LeRobot/HDF5) export.
- Replay runners (`kind=="replay"`), sleep, home, teleop are **not** logged —
  only `kind=="live"`.
- No auth / multi-user run ownership.

## Architecture

Recording attaches at the existing telemetry seam (`webapp/telemetry.py`,
`TelemetrySink` protocol). The control loop and browser path are untouched.

```
control loop / bridge
        │  (TelemetrySink callbacks)
        ▼
     TeeSink ──────────────┐
        │                  │
        ▼                  ▼
   QueueSink          RecordingSink
        │                  │
      WS → browser      run dir on disk
```

### Chosen approach

- **Recording mechanism: TeeSink.** A `TeeSink` fans every callback to N inner
  sinks. `server.py` builds `TeeSink(QueueSink(q), RecordingSink(run_dir))` for
  live runs. Rejected: subclassing QueueSink (couples browser+disk); browser-side
  recording (loses data on disconnect, no server metadata).
- **Format: JSONL events + JSON manifest/summary.** Append-safe (survives a
  crash mid-run), streamable, greppable, trivial to load for compare. Rejected:
  SQLite (heavier for v1), single-JSON-per-run (must buffer whole run in memory).

## Storage layout

Per-run directory, sibling to `webapp/feedback/`. Base dir configurable
(default `webapp/runs/`, injectable like `presets_dir` / `feedback_dir`).

```
runs/<run_id>/            run_id = YYYY-MM-DD-HHMMSS[-N]
  manifest.json           written at run start
  events.jsonl            appended per telemetry event
  summary.json            written on finalize
  rating.json             written by the quick-rating prompt (optional)
```

### `manifest.json`

```json
{
  "run_id": "2026-07-02-181500",
  "started_at": 1751476500.12,
  "kind": "live",
  "config": { "...": "full readConfig() payload, includes model_name" },
  "model_name": "pi0-trossen-joint-v3",
  "model": { "...": "full get_server_metadata() dict, or null" },
  "git": { "branch": "ibrahim/feat/web-app-eval", "commit": "e79e46c" }
}
```

`model_name` is the **user-typed value** from the new config field — the
authoritative name for comparison. `model` is the raw `get_server_metadata()`
dict stored verbatim as a bonus when non-empty (the server does not reliably
report a name), else null.

### `events.jsonl`

One JSON object per line, exactly the sink event dicts already defined in
`telemetry.py` (minus `images`): `action`, `overlap`, `weights`, `inference`,
`chunk`, `status`, `log`. No transformation — the browser and the file see the
same events.

### `summary.json`

```json
{
  "run_id": "2026-07-02-181500",
  "ended_at": 1751476560.44,
  "duration_s": 60.3,
  "steps": 742,
  "end_reason": "completed",
  "rtt_ms": { "mean": 41.2, "p50": 39.0, "p95": 72.5 },
  "loop_hz": 24.6,
  "jitter_ms": 3.1,
  "drops": 0,
  "smoothing_delta_mean": 0.084,
  "overlap_mean": 3.2
}
```

`end_reason ∈ {completed, stopped, estop, error, unknown}`. Metrics reuse the
same math as `webapp/metrics.py` where possible.

### `rating.json`

```json
{ "success": "partial", "score": 3, "note": "dropped object on second grasp", "rated_at": 1751476590.0 }
```

`success ∈ {yes, partial, no}`, `score` optional 1–5, `note` optional one-liner.

## Components

1. **`RecordingSink`** (new, `webapp/recording.py`) — implements
   `TelemetrySink`. Constructor creates the run dir + opens `events.jsonl` for
   append and writes `manifest.json`. Each callback appends a JSON line and
   updates running metric accumulators. `on_status("model", md)` folds metadata
   into the manifest (rewrites manifest.json). `close(end_reason)` writes
   `summary.json`. All disk errors are caught + logged; the sink degrades to
   no-op, never raising into the control loop.

2. **`TeeSink`** (new, in `webapp/telemetry.py` or `webapp/recording.py`) —
   holds a list of sinks; every `TelemetrySink` method forwards to each. A raise
   from one inner sink is caught + logged so one bad sink can't starve the other.

3. **Model name config field** (`webapp/static/js/config.js`) — add a
   `model_name` text field to the Policy group (e.g. label "Model name / tag",
   help "Name this policy checkpoint so runs can be compared by model"). Flows
   through `readConfig()` into `manifest.config.model_name`, which
   `RecordingSink` copies to the top-level `manifest.model_name`. This is the
   authoritative name. **Bonus:** `TrossenBridge` still emits
   `sink.on_status("model", policy_client.get_server_metadata())` once after
   connect; RecordingSink stores it verbatim under `manifest.model` when
   non-empty (NullSink ignores → CLI unaffected). Never blocks a run if metadata
   is empty/absent.

4. **Wiring** (`webapp/server.py`) — in `start(kind, config)`:
   - For `kind == "live"`: allocate `run_id`, build
     `sink = TeeSink(QueueSink(q), RecordingSink(runs_dir/run_id, config))`,
     emit a `status` event `{"kind":"run_started","payload":{"run_id":...}}` so
     the browser knows the id.
   - Other kinds: unchanged `QueueSink(q)`.
   - Finalize: `RecordingSink.close(end_reason)` must run on session end. Hook it
     in `SessionManager._run`'s `finally` (via an optional sink `close()` the
     manager calls if present) so a crash still flushes `summary.json` with
     `end_reason=error`.

5. **Quick-rating prompt** (new, `webapp/static/js/runlog.js`) — on a
   `run_started` status, remember the run id. On session end (`stopped` /
   `error` status), show a dedicated modal: success (yes/partial/no), optional
   score, optional note → `PATCH /api/runs/{id}/rating`. Distinct from the
   feedback card. Dismissible (rating is optional).

6. **Run store API** (`webapp/server.py` + `webapp/run_store.py`):
   - `GET /api/runs` → list of summaries (run_id, started_at, model_name,
     config summary, rating, key metrics) sorted newest-first.
   - `GET /api/runs/{id}` → manifest + summary + rating (+ events on request via
     `?events=1`, streamed/paged to avoid loading a huge file at once).
   - `PATCH /api/runs/{id}/rating` → write `rating.json`.

7. **Compare view** (new `webapp/static/runs.html` + `webapp/static/js/runs.js`,
   linked from the main page) — list runs; select 2+; render a **config diff
   table** (rows = config keys, columns = runs, differing cells highlighted) and
   **overlaid charts** (smoothing Δ, overlap depth, RTT vs step) built with the
   existing `makeSeriesChart` from `charts.js`.

## Data flow

Live run → `TeeSink` → { `QueueSink` → WS → browser live view, `RecordingSink`
→ run dir }. After the run, compare view reads back through the run store API.

## Error handling

- `RecordingSink` disk errors (bad path, disk full, serialize failure): caught,
  logged once, sink continues as no-op. Never propagates to the loop or browser.
- Finalize in `SessionManager._run` `finally`: guarantees `summary.json` even on
  crash / estop, with the correct `end_reason`.
- Missing/failed `get_server_metadata()`: manifest `model` = null, run still
  records; UI shows "model: unknown".
- Corrupt/partial `events.jsonl` (crash before finalize): run store tolerates a
  missing `summary.json` (recompute lazily from events, or mark
  `end_reason=unknown`).

## Testing

- `RecordingSink`: events → correct JSONL lines; summary metric math; model
  metadata folded into manifest; `close()` writes summary with right end_reason;
  disk-error path degrades to no-op without raising.
- `TeeSink`: every callback reaches all inner sinks; one inner sink raising does
  not stop the others.
- Run store API: list/detail/rating against a temp runs dir; newest-first order;
  `?events=1` paging.
- Round-trip: run a fake live session through the Tee → RecordingSink, then load
  it back via the API and assert fields match.
- Regression: CLI path (NullSink) and non-live webapp kinds unchanged.

## Open items for the plan

- Exact metric accumulator reuse from `webapp/metrics.py` vs recomputation.
- Whether `events` are paged or streamed in `GET /api/runs/{id}`.
- Retention / cleanup of old run dirs (out of scope v1 unless trivial).
