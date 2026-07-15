# Trossen Control Web App

Browser UI over the existing Trossen ↔ OpenPI control stack. Observe-only: it
imports the bridge / ensemble / replay code and streams telemetry; it does not
reimplement the control loop.

## Checkpoint serving on Machine A

The FastAPI backend now owns one `ModelProcessManager`. The manager is a normal
Python object inside the web process; it is not another network service. When
requested, it starts `scripts/serve_policy.py` as one child process using the
root OpenPI environment.

```text
web API -> direct Python call -> manager -> inference child
                                 |            |
                                 |            +-- WebSocket: 127.0.0.1:8800
                                 +--------------- HTTP GET /healthz
```

Checkpoints are discovered below
`/home/ibrahim/storage/VLA_MODELS`. A selectable directory must contain the
FalconVLA configuration, normalization data, processor/tokenizer configuration,
and model weights. The API accepts a relative checkpoint ID, not an arbitrary
filesystem path.

The inference and web environments remain separate:

```bash
cd /path/to/openpi
git submodule update --init --recursive
uv sync

cd examples/trossen_ai
uv sync
```

For a local manual run on Machine A:

```bash
cd /path/to/openpi/examples/trossen_ai
OPENPI_CHECKPOINT_ROOT=/home/ibrahim/storage/VLA_MODELS \
OPENPI_REPO_ROOT=/path/to/openpi \
OPENPI_INFERENCE_PYTHON=/path/to/openpi/.venv/bin/python \
uv run uvicorn webapp.server:app \
  --host 127.0.0.1 --port 8000 --workers 1 --lifespan on
```

Use exactly one Uvicorn worker. The exclusive manager lock rejects a second web
backend before it can start another child on the same inference port.

### Checkpoint API

```bash
# Discover checkpoints.
curl http://127.0.0.1:8000/api/model/checkpoints

# Start loading one checkpoint. This returns immediately with state "starting".
curl -X POST http://127.0.0.1:8000/api/model/start \
  -H 'Content-Type: application/json' \
  -d '{"checkpoint_id":"my-checkpoint"}'

# Read readiness, child PID, errors, and the latest nvitop GPU sample.
curl http://127.0.0.1:8000/api/model/status

# Terminate and reap the owned inference child.
curl -X POST http://127.0.0.1:8000/api/model/stop
```

Optional proprio overrides must supply both `use_proprio` and `proprio_mode`;
omit both to use checkpoint auto-detection.

The local inference server's `GET /healthz` decides whether the model is ready.
`nvitop` separately samples GPU utilization, memory, temperature, power, and the
child's GPU memory while the child is running. A GPU-monitoring error is reported
under `gpu.error` and does not stop inference.

### systemd process cleanup

The production unit is `deploy/systemd/openpi-webapp.service`. Install it as a
user service after adjusting repository paths if needed:

```bash
mkdir -p ~/.config/systemd/user
cp /path/to/openpi/deploy/systemd/openpi-webapp.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now openpi-webapp.service
journalctl --user -u openpi-webapp.service -f
```

FastAPI performs bounded orderly shutdown: request a robot-session stop and
wait, terminate the inference process group, then retry the session join in case
closing inference released a blocked policy request. The manager uses `SIGKILL`
if its child ignores `SIGTERM`. systemd keeps the web backend and inference child
in one cgroup, so it can remove any remaining child if the backend crashes. If
the service must keep running after logout, ask IT whether user lingering may be
enabled.

The manual Uvicorn command has orderly shutdown handling, but a forced backend
death such as `kill -9` bypasses Python cleanup. Use the systemd unit in
production so the cgroup remains the final orphan-process safeguard.

This phase binds both services to loopback. Exposing the web app to Laptop B with
HTTPS/WSS and moving the physical robot gateway to Laptop B is a separate network
integration step; no public manager port has been added.

## Existing all-in-one robot-machine mode

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
