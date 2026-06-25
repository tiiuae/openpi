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
