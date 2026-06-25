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

## Two pages

The UI is split into two themed pages (wizard-style dark theme, shared `theme.css`),
served by the same FastAPI app:

- **`/` — Live.** Grouped config form (Connection, Policy, Action smoothing, Arms,
  IK, Motion tuning) with ⓘ help tooltips and savable presets; action chart;
  camera strip; RTT/latency/jitter metrics; terminal-parity log panel; feedback form.
- **`/replay` — Replay.** Folder browser (pick a dataset dir anywhere on disk via
  `/api/files`), episode picker, replayed-action chart, logs, feedback.

Both pages share a **Home / Sleep** header (send the arms to a fixed pose via the
`go_home`/`go_sleep` WS actions) and a **Stop / E-STOP** control. Autonomous moves
prompt for confirmation. See `HARDWARE.md` for motion-tuning knobs
(`smooth_streaming`, goal-time multiplier, loop rate, connect timeout).

## Frontend module layout

No bundler — pages load ES modules with `<script type="module">`. Under
`static/js/`:

| Module | Responsibility |
|---|---|
| `api.js` | `api` / `apiPost` fetch helpers |
| `ws.js` | reconnecting telemetry WebSocket + type-keyed handler registry |
| `config.js` | config groups (single source of truth), render/read/apply, presets, tooltips, EE-conditional visibility |
| `charts.js` | wizard chart factory + bounded streaming `LiveChart` |
| `filebrowser.js` | modal folder browser (calls `/api/files`) |
| `logs.js` | terminal-parity log panel |
| `controls.js` | stop/estop/sleep/home buttons + session/RTT badges + status handling |
| `feedback.js` | feedback form → `POST /api/feedback` |
| `live.js` | Live page wiring |
| `replay.js` | Replay page wiring |

`theme.css` holds the wizard CSS variables and shared components (cards, buttons,
modal, log panel, tooltips, header/nav).

## Tests (off-hardware, lerobot env)

```bash
PYBIN=/home/edgeai/miniconda3/envs/lerobot/bin/python
$PYBIN -m pytest webapp/tests/ tests/ -q
```
