# Trossen Eval Webapp — Overhaul Design

**Date:** 2026-06-25
**Branch:** ibrahim/feat/web-app-eval
**Scope:** `examples/trossen_ai/webapp/` + targeted changes in `trossen_bridge.py`,
`robot_control.py`. Reference UI: `get_inspired_from_web_wizard/`.

## Goal

Fix the stop/termination bugs and unnatural robot motion, then bring the eval
webapp up to the look, feel, and ergonomics of the dataset web wizard. 13 work
items, grouped below.

## Non-goals

- No changes to the openpi policy server or the `lerobot_robot_trossen` installed
  package (it is a dependency; we tune it through exposed parameters, not edits).
- No new policy/ensemble algorithms.
- CLI entrypoints (`main.py`, `main_ee.py`) keep working; the only behavior change
  reaching the CLI is the joint-limit removal (item 5) and the optional smooth-
  streaming toggle (item 13), both off/neutral by default where reasonable.

## Reference facts (verified in code)

- `webapp/server.py` — FastAPI: static files, preset REST, `/api/episodes`,
  one telemetry WebSocket `/ws/telemetry` handling `start_live`, `start_replay`,
  `stop`, `estop`.
- `webapp/session.py` — `SessionManager` runs one runner in a daemon thread.
  **Bug:** `start()` builds the runner *synchronously in the WS coroutine* before
  the thread starts.
- `webapp/runners.py` — `LiveRunner` builds `TrossenOpenPIBridge` in `__init__`,
  which connects the policy client in `__init__`.
- `packages/openpi-client/.../websocket_client_policy.py` — `_wait_for_server()`
  loops forever on `ConnectionRefusedError` (`time.sleep(5)`, logs
  "Still waiting for server...").
- Joint limits: hardcoded `JOINT_LIMIT` + `is_action_within_limits` in BOTH
  `trossen_bridge.py:29` and `robot_control.py:57`.
- `SLEEP_POSITION = zeros(14)`. **Home/stage pose** (from `trossen-ai` branch
  `main.py:169`, currently commented): `[0, π/3, π/6, π/5, 0,0,0, 0,0,0,0,0,0,0]`.
- Motion command path: `RobotController.execute_action` → `robot.send_action` →
  per-arm `widowxai_follower.send_action` →
  `driver.set_all_positions(goal, goal_time=min_time_to_move, blocking=False)` with
  `goal_feedforward_velocities=None` (→ zeros). `min_time_to_move = multiplier /
  loop_rate` (defaults 3.0 / 30 = 0.1 s); robot built with those defaults in
  `build_stationary_robot` (`robot_control.py:43-44`).
- StarVLA flag: when on, images preprocessed to 224×224 + BGR→RGB via PIL;
  off uses default training-size resize.

---

## Architecture

### Pages (multi-page, wizard-style)

FastAPI serves separate HTML pages (mirrors wizard landing/viewer/wizard):

- `/`        → **Live Control** page (policy eval).
- `/replay`  → **Dataset Replay** page.

Shared header with nav links + global Sleep/Home buttons + session badge.

### Static file layout

```
webapp/static/
  theme.css        # wizard CSS variables + shared component styles
  index.html       # Live Control page
  replay.html      # Dataset Replay page
  js/
    api.js         # fetch helpers (api, apiPost)
    ws.js          # telemetry WebSocket connect/send/reconnect
    charts.js      # wizard-style chart factory (palette, tooltip, zoom, grouping)
    config.js      # read/apply config form, presets, help tooltips
    filebrowser.js # modal folder browser (ports wizard)
    feedback.js    # feedback form submit
    live.js        # Live Control page wiring
    replay.js      # Dataset Replay page wiring
```

`charts.js` is shared by both pages. Splitting the current 159-line monolithic
`app.js` into focused modules keeps each file understandable in isolation.

### Theme (from `wizard.css`)

CSS variables adopted verbatim: `--bg #0d1117`, `--bg-card #161b22`,
`--bg-hover #21262d`, `--border #30363d`, `--text #e6edf3`, `--muted #8b949e`,
`--accent #e34c26`, `--ok #3fb950`, `--warn #d29922`, `--err #f85149`,
`--blue #4a9eff`; Consolas/Menlo mono font; card + button + modal styles.

---

## Work Items

### 1. Replay on its own page

Move all replay-only UI (dataset picker, episode index, replay charts/logs) from
`index.html` to `replay.html`. Live page keeps live-only config. Shared config
(IK weights for EE) lives in `config.js` and is read on whichever page needs it.

### 2. Live page layout — horizontal split, ordered

Top-to-bottom stack (full-width sections, not 3-column grid):
1. **Cameras** (image strip)
2. **Actions / charts**
3. **Logs**

Config sits in a collapsible left column (or a top "Configuration" card on
narrow screens). Implemented with CSS grid + a media query.

### 3. Logs = terminal parity

- Attach `SinkLogHandler` to the **root logger** for the session duration so
  every module's logging records reach the browser (today only records that
  propagate are forwarded; the session-scoped root attach guarantees parity).
- Replace bridge `print(...)` calls (e.g. in `move_to_sleep_position`,
  "Moving to sleep position…") with `logger.info(...)` so they also stream.
- Frontend: wizard-style log panel — ANSI→HTML coloring (`ansiToHtml` ported),
  per-level colors, Clear button, autoscroll, bounded line buffer.

### 4. Charts — wizard style, grouped & named series

- Port wizard `makeWizardChart`: shared `PALETTE`, floating external tooltip,
  zoom/pan plugin, named series, `maintainAspectRatio:false`, dark grid.
- **Replay page:** load the full episode (like wizard preview) and render
  grouped charts: per-arm joints (Left/Right via the `defineNamedArray`
  left_/right_ split), state vs action.
- **Live page:** stream into the same visual style using a bounded ring buffer
  of recent steps. Keep the eval-specific views but restyled: actions
  (grouped per arm), raw-vs-smoothed for a selected joint, blend weights (bar),
  RTT/loop metrics. Drop the ad-hoc Chart.js setup currently in `app.js`.
- Joint/series names come from `robot._joint_ft` keys (left_/right_ prefixes).

### 5. Remove hardcoded joint limits; trust firmware

- Delete `JOINT_LIMIT` arrays and `is_action_within_limits` from
  `trossen_bridge.py` and `robot_control.py`, and their call sites.
- Replace the pre-send software gate with a **firmware-error guard**: wrap the
  driver/`send_action` call in `try/except`. On any exception (firmware halts the
  arm on a joint/velocity/limit fault):
  - log the error (streams to web logs),
  - emit telemetry `status: {kind: "firmware_error", payload: {...}}`,
  - stop the loop (`is_running = False` / `self._stopped = True`),
  - move the arm to **sleep**, then disconnect.
- Net effect preserves "on fault → sleep" behavior, now driven by firmware
  rather than a hardcoded table.

### 6. Sleep / Home buttons

- Add `HOME_POSITION = np.array([0, np.pi/3, np.pi/6, np.pi/5, 0,0,0,
  0,0,0,0,0,0,0])` constant (shared location, e.g. `robot_control.py`).
- New WS actions `go_sleep` / `go_home`. Handler runs a short standalone
  movement (connect if needed → PCHIP interpolate to pose → disconnect) in the
  session thread; refuses if an eval/replay session is running (must Stop first).
- Two buttons in the shared header, available on both pages. Confirm dialog in
  autonomous mode (moves the real robot).

### 7. Help (ⓘ) tooltips

- Each config field label gets an `ⓘ` icon; hovering shows a CSS tooltip with a
  plain-language explanation. Help text in one `FIELD_HELP` map in `config.js`.
- Pure CSS/JS, no dependency.

### 8. Config reorganization + renaming

Grouped cards with approved labels (code key → UI label):

- **Connection:** `policy_host` → "Policy server host", `policy_port` → "Policy
  server port", `control_freq` → "Control rate (Hz)", `max_steps` → "Max steps".
- **Policy:** `adapter` → "Action space (Joint / End-effector)", `task_prompt` →
  "Task prompt", `starvla` → "StarVLA input mode (224×224 RGB)".
- **Action smoothing:** `ensemble_type` → "Action smoothing
  (Exponential / CogACT / None)", `cogact_mode` → "CogACT blend mode",
  `rate_of_inference` → "Inference interval (steps)", `async_inference` →
  "Async inference".
- **Arms:** `use_left_arm_only` → "Left arm only", `use_right_arm_only` →
  "Right arm only".
- **IK** (shown only when Action space = End-effector): `ik_orientation_weight`
  → "IK orientation weight", `ik_pos_tol_m` → "IK position tolerance (m)".
- **Motion tuning (Advanced):** see item 13.
- **Replay** fields move to `/replay`.

`readConfig`/`applyConfig` keep emitting the original code keys so the
backend/runners are unchanged.

### 9. Feedback section

- Form: name, email, feedback (textarea). `POST /api/feedback`.
- Backend writes `webapp/feedback/YYYY-MM-DD-HHMMSS.md`:

  ```markdown
  # Feedback — 2026-06-25 14:32:10

  - **Name:** ...
  - **Email:** ...

  <feedback body>
  ```
- Filename uses local timestamp; directory created on first write.
- Small card on both pages (or shared footer card).

### 10. Replay folder browser

- Port wizard `list_directory` as FastAPI `GET /api/files?path=`: returns
  `{path, parent, entries:[{name,type,is_dataset,path}]}`, traversing outside the
  repo via `Path(path).expanduser().resolve()`. `is_dataset` = `meta/info.json`
  exists.
- Port the modal browser UI (`filebrowser.js` + theme styles) with the green
  "dataset" badge. Replaces the free-text `dataset_dir` input on `/replay`.
- Selected directory fills `dataset_dir`; existing `/api/episodes` then lists
  episodes.

### 11 & 12. Stop bug + clean termination

Root cause: runner construction (blocking policy-client connect) runs in the WS
coroutine and blocks the event loop, so `stop` is never read and `self._runner`
stays `None`.

- **Build runner inside the session thread.** `SessionManager.start` stores the
  factory args and starts the thread immediately; the thread builds the runner
  (blocking connect happens here) then calls `run()`. WS stays responsive.
- **Cancellable / bounded connect.** Add a `connect_timeout` (config, default
  e.g. 15 s) to the bridge's policy-client creation. Implement by attempting the
  websocket connect with a deadline instead of the infinite
  `_wait_for_server` loop — wrap construction so it raises after the timeout
  (do not edit the shared client; add a bridge-side bounded-connect helper, e.g.
  a preflight `socket` reachability check + a stop-flag-aware retry loop). On
  timeout: emit `status: connect_failed`, log, end session cleanly.
- **Stoppable construction.** A shared stop flag is honored by the
  connect-retry loop and the run loop, so Stop interrupts a session that is still
  connecting.
- **Guaranteed cleanup.** `run()` wrapped in `try/finally` that always calls
  `robot.disconnect()` (covers the zombie-holding-the-arm symptom). `stop()`
  joins the thread with timeout and reports state to the UI; if join times out,
  surface a clear error rather than hanging.
- **Signal handling.** `server.py` installs `SIGINT`/`SIGTERM` handlers (or
  FastAPI shutdown hook) that stop the session and disconnect the robot, so
  CTRL+C frees the hardware instead of leaving a live thread.
- **UI:** Stop button disabled→"Stopping…"→idle based on `status` events; never
  silently stuck.

### 13. Unnatural / brief-brutal motion — root cause + tuning

Cause (verified upstream in `lerobot_robot_trossen`): each setpoint sent with
`goal_feedforward_velocities=None` (plans to arrive at rest at every waypoint →
stutter) and a fixed `goal_time = multiplier/loop_rate = 0.1 s` that mismatches
the real control period (e.g. 0.04 s at 25 Hz) and forces high-velocity slews on
far setpoints (the brutal jerk).

- **Expose knobs.** Add `min_time_to_move_multiplier` and `loop_rate` params to
  `build_stationary_robot` (default `loop_rate` to the session `control_freq`),
  surfaced in a **Motion tuning (Advanced)** config card.
- **Opt-in smooth-streaming mode.** New config toggle. When on,
  `RobotController.execute_action` (and the bridge's execute path) bypass
  `robot.send_action` and call the driver directly per arm:
  `arm.driver.set_all_positions(goal, goal_time=dt, blocking=False,
  goal_feedforward_velocities=(goal-cur)/dt)`, with `cur` from the latest
  observation and `dt = 1/control_freq`. Clamp/guard NaNs. Off = current
  behavior (unchanged).
- **Optional advanced knobs** in the same card (best-effort, behind "advanced"):
  `velocity_max` (`JointLimit` via `set_joint_limits`) and position/velocity PID
  gains.
- **Validation:** hardware-only. The toggle + exposed params let the user A/B and
  find the sweet spot; defaults preserve today's behavior so nothing regresses
  silently.

---

## Backend API summary (additions)

| Method | Path | Purpose |
|---|---|---|
| GET | `/replay` | Dataset Replay page |
| GET | `/api/files?path=` | Folder browser listing (item 10) |
| POST | `/api/feedback` | Save feedback markdown (item 9) |
| WS | `/ws/telemetry` action `go_sleep` | Move arm(s) to sleep (item 6) |
| WS | `/ws/telemetry` action `go_home` | Move arm(s) to home/stage (item 6) |

New telemetry `status` kinds: `firmware_error`, `connect_failed`, plus existing
`started`/`stopped`. Config gains keys: `min_time_to_move_multiplier`,
`loop_rate`, `smooth_streaming`, `connect_timeout` (+ optional advanced knobs).

## Testing

Off-robot unit tests (extend `webapp/tests/`, fake runner pattern already used):

- `start()` does not block the event loop: a fake runner that blocks in
  `__init__` still lets the WS process `stop`.
- Stop interrupts a session that is still "connecting" (stop flag honored).
- `connect_timeout` path emits `connect_failed` and ends cleanly.
- `/api/files` lists dirs, marks datasets, traverses to parent and outside repo.
- `/api/feedback` writes a dated markdown file with the submitted fields.
- Config rename round-trip: form labels change, emitted keys unchanged.

Hardware paths (motion smoothing, sleep/home, firmware-error→sleep, full UI)
stay syntax-checked in CI and manually validated per `webapp/HARDWARE.md`, which
gains a "Motion tuning" section.

## Build order (suggested)

1. Items 11/12 (stop/termination) + 5 (joint-limit removal / firmware guard) —
   correctness and safety first.
2. Item 13 (motion tuning exposure) — unblocks the user's hardware tuning.
3. Architecture split + theme (pages, modules, CSS) + items 1, 2, 8, 7.
4. Item 4 (charts) — replay first, then live.
5. Items 3 (logs), 6 (sleep/home), 9 (feedback), 10 (folder browser).
