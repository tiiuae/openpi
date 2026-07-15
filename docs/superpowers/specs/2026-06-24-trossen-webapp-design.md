# Trossen Control Web App — Design Spec

**Date:** 2026-06-24
**Status:** Approved (design); ready for implementation planning.

## Goal

A browser UI for the existing Trossen ↔ OpenPI control stack that lets a user:
- configure and launch a control session (no CLI flags),
- launch a dataset **replay** session (LeRobot v3.0 episode → IK → robot),
- watch live **logs**, **action charts**, **camera images**, and **metrics** (RTT, latency, loop rate),
- understand **action-ensemble smoothing** through dedicated real-time visualizations.

The web app **imports** the existing code and lives in its own folder. It only *observes* the control loop — it does not reimplement it.

## Decisions (locked)

- **Stack:** FastAPI backend (async + WebSocket), frontend = vanilla HTML/JS + Chart.js served as static files. No Node/build step.
- **Scope:** configure + run live episodes on the real robot (**test-mode default**, E-STOP), **and** drive dataset replay from the UI.
- **Smoothing visualizations:** overlap timeline, raw-vs-smoothed trace, buffer gauge + inference cadence. (Per-step weight bars excluded.)
- **Extras:** E-STOP + test/autonomous toggle, connection-health indicators, config presets save/load, latency p50/p95 + jitter metrics.

## Non-goals

- No reimplementation of the control loop, IK, or ensembling — the web app wraps them.
- No multi-user / auth / remote-internet exposure. Single operator on the robot LAN.
- No per-step ensemble weight-bar visualization (deferred).
- No change to CLI behavior: with the default null sink the existing entrypoints behave exactly as today.

## Folder layout

Self-contained, sibling to the code it imports (so imports stay flat — `ensemble`, `trossen_bridge`, `robot_control`, `dataset_replay`, `adapters`):

```
examples/trossen_ai/webapp/
  __init__.py
  server.py        # FastAPI app: static mount, REST endpoints, WebSocket, command dispatch
  session.py       # SessionManager: runs a bridge/replay session in a daemon thread; start/stop/estop
  telemetry.py     # TelemetrySink protocol + NullSink + QueueSink + event dataclasses
  metrics.py       # rolling accumulators: RTT, inference latency p50/p95, loop Hz, jitter, stale-drops
  config_store.py  # named config presets (JSON load/save/list)
  static/
    index.html
    app.js
    styles.css
  presets/         # saved presets (gitignored)
  tests/
    test_telemetry.py
    test_metrics.py
    test_config_store.py
    test_session.py
  README.md
```

Run command (robot runtime env, which has `lerobot_robot_trossen`):
```
<robot-env-python> -m uvicorn webapp.server:app --host 0.0.0.0 --port 8000   # from examples/trossen_ai
```

## Architecture: observe, don't touch

### TelemetrySink (the seam)

A `Protocol` in `telemetry.py`:

```
on_log(record)                       # one log line (level, msg, ts)
on_action(step, action, ts)          # executed action vector for a step
on_inference(rtt_ms, ts)             # one policy_client.infer() round trip
on_chunk(query_step, chunk, ts)      # a chunk added to the ensemble
on_overlap(step, count)              # overlapping predictions used at a step
on_images(images_jpeg, ts)           # latest camera frames (throttled, jpeg bytes)
on_status(kind, payload)             # session lifecycle / errors / connection health
```

- `NullSink` — every method a no-op. **Default everywhere**, so existing entrypoints and the 33 tests are unaffected.
- `QueueSink` — serializes events into a thread-safe queue that the WebSocket coroutine drains and forwards as JSON (images as base64). Applies throttling: images 5–10 fps, metrics aggregated ~2 Hz, action stream decimated client-side.

### Minimal instrumentation of existing code

All additions default to `NullSink()` so CLI is byte-for-byte behaviorally unchanged.

- `TrossenOpenPIBridge.__init__(..., sink: TelemetrySink = NullSink())`.
- `AsyncPolicyWorker.__init__(..., sink: TelemetrySink = NullSink())` (async RTT + chunk emit happen here).
- Emit points:
  - around `policy_client.infer(...)` → `on_inference(rtt_ms)`,
  - after `a_t` is resolved in `run_episode` → `on_action(step, a_t)`,
  - at each `ensemble.add_chunk(...)` call site (sync branch + worker) → `on_chunk(query_step, chunk)`,
  - where overlap is already computed → `on_overlap(step, count)`,
  - in `_build_observation`, throttled → `on_images(...)`.
- Add `buffer_size() -> int` to the `ActionEnsemble` ABC and the 3 implementations (cheap read) for the buffer gauge. `none`/no-ensemble path reports 0.
- A `logging.Handler` subclass forwards Python `LogRecord`s to `sink.on_log` — drives the live log box without touching existing log statements.

### Raw-vs-smoothed without an ensemble change

The sink keeps the most recent chunk seen via `on_chunk`. When `on_action(step)` fires, it derives the raw prediction `chunk[step - query_step]` (when in range) and sends both raw and blended values for the selected joint. No need to reintroduce `get_latest_raw`.

## SessionManager & data flow

`SessionManager` (in `session.py`) owns at most one running session.

- `start_live(config)` — builds `TrossenOpenPIBridge` (or EE variant) with a `QueueSink`, runs `run_episode` in a daemon thread.
- `start_replay(replay_config)` — builds the replay pipeline (`EpisodeReader` → `EEToJointsConverter` → `RobotController`) with a `QueueSink`, runs in a daemon thread, emitting the same action/overlap/image events where applicable.
- `stop()` — sets `is_running = False`; joins the thread with timeout.
- `estop()` — `stop()` + move arms to sleep (`RobotController.move_to_sleep_position` / bridge sleep path).
- Dependency injection: SessionManager takes a **runner factory** so tests inject a fake runner (no hardware).

**Flow:** UI command over WS → SessionManager starts session with `QueueSink` → bridge/worker emit events → queue → WebSocket → `app.js` updates panels.

## REST + WebSocket surface (server.py)

- `GET /` → `index.html`; `/static/*` assets.
- `GET /api/health` → best-effort robot / policy-server reachability + last RTT.
- `GET /api/presets`, `POST /api/presets`, `DELETE /api/presets/{name}` → config presets.
- `GET /api/episodes?dataset_dir=...` → episode count / indices (via `EpisodeReader`).
- `WS /ws/telemetry` → server→client: telemetry event stream; client→server commands: `start_live`, `start_replay`, `stop`, `estop`, `set_mode`.

## Frontend (single page, app.js)

Panels:
- **Config form** — all flags: `policy_host/port`, `control_freq`, `ensemble_type`, `cogact_mode`, `async_inference`, `max_steps`, adapter (joint/EE), `use_left/right_arm_only`, `starvla`, IK weights; replay: `dataset_dir`, `episode_index`. Preset dropdown (save/load/delete).
- **Controls** — Start / Replay / Stop / **E-STOP**; test⇄autonomous toggle (test default); confirm dialog before autonomous-on-hardware.
- **Log box** — streaming, level-colored, autoscroll, pause.
- **Charts (Chart.js, ring-buffered)**:
  - live per-joint executed actions (selectable joints),
  - raw-vs-smoothed overlay for one selected joint,
  - overlap timeline (predictions blended per step),
  - buffer gauge + inference cadence (chunk arrivals vs control Hz),
  - metrics: RTT live + p50/p95, loop Hz actual-vs-target, jitter.
- **Images** — latest camera frames (~5–10 fps).
- **Health badges** — robot / policy-server / RTT health.

## Metrics (metrics.py)

Rolling accumulators fed from telemetry events:
- RTT and inference latency → keep last N; report live + p50/p95.
- Loop Hz → from `on_action` timestamps; actual vs target (`control_freq`).
- Jitter → rolling stddev of executed action deltas.
- Async stale-drop counter → worker reports dropped pending observations.

## Error handling & safety

- The bridge connects to the policy server eagerly; connection failures (server or robot) are caught in SessionManager, emitted as `on_status` error events, surfaced in the UI (no crash).
- Exactly one session at a time; starting while running is rejected.
- WebSocket close → auto-`stop()` the session.
- **Test mode is the default**; autonomous-on-hardware requires explicit confirm. E-STOP always visible.

## Testing (off-hardware)

Unit tests (run in the `lerobot` env, no robot):
- `test_telemetry.py` — event ordering through QueueSink, raw-from-chunk derivation, throttling.
- `test_metrics.py` — percentile + jitter + loop-Hz math on synthetic event streams.
- `test_config_store.py` — preset save/load/list/delete round-trip.
- `test_session.py` — lifecycle (start/stop/estop, single-session guard) with an injected fake runner.
- Episode listing reuses the existing `EpisodeReader` tests.

Hardware path (real bridge/replay motion) stays **manual**. The existing 33 tests must remain green (null sink ⇒ no behavior change).

## Build order (for the plan)

1. `telemetry.py` (sink protocol, NullSink, QueueSink, events) + tests.
2. Instrument bridge/worker/ensemble with the sink + `buffer_size()`; confirm existing tests green.
3. `metrics.py` + tests.
4. `config_store.py` + tests.
5. `session.py` (SessionManager + runner-factory DI) + fake-runner tests.
6. `server.py` (REST + WebSocket) + static frontend (`index.html`, `app.js`, `styles.css`).
7. Smoothing/metrics panels wired to live data; manual rig verification.
8. README + `.gitignore` for `presets/`.
