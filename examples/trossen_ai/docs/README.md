# Trossen AI × OpenPI — Documentation

Entry point for the `examples/trossen_ai` codebase: a control stack that drives a
**Trossen bimanual (WidowX AI) follower** from an **OpenPI policy server**, plus a
dataset-replay tool and a browser **web app** for running and observing both.

> **Safety first:** the arm moves for real in `autonomous` mode. Read
> [Motion & Safety](motion-safety.md) and the [Hardware runbook](../webapp/HARDWARE.md)
> before any autonomous or replay run.

## Documents

| Doc | What's in it |
|---|---|
| [Architecture](architecture.md) | How the pieces fit: control core, EE/IK, dataset replay, web app; data flow; threading model; module-by-module reference. |
| [CLI — Live policy](cli-live.md) | Run a policy from the terminal (no web app): `main.py` (joint) / `main_ee.py` (end-effector), arguments, test vs autonomous. |
| [CLI — Dataset replay](cli-replay.md) | Replay a recorded episode from the terminal (no web app, no policy server): `replay_ee_dataset.py`, the sleep helper. |
| [Motion & Safety](motion-safety.md) | Firmware velocity limit, the IK branch-flip fault, the velocity limiter, smooth streaming, the firmware-fault guard, and all tuning knobs. |
| [Web App — overview](../webapp/README.md) | The two pages (Live / Replay) and the frontend module layout. |
| [Web App — hardware runbook](../webapp/HARDWARE.md) | Step-by-step run, first-run checklist, motion tuning, troubleshooting, off-robot testing. |

## The three ways to drive the arm

1. **CLI — live policy.** [`main.py`](../main.py) (joint space) / [`main_ee.py`](../main_ee.py) (end-effector space) build a [`TrossenOpenPIBridge`](../trossen_bridge.py) and run an episode against the policy server. → **[CLI: Live guide](cli-live.md)**
2. **CLI — dataset replay.** [`replay_ee_dataset.py`](../replay_ee_dataset.py) replays a recorded episode through IK onto the arm. → **[CLI: Replay guide](cli-replay.md)**
3. **Web app.** `python -m uvicorn webapp.server:app` serves a Live page (policy) and a Replay page (dataset), both streaming telemetry. See the [hardware runbook](../webapp/HARDWARE.md).

## Off-robot development

The hardware packages (`lerobot_robot_trossen`, `placo`) aren't needed to run the
test suite. Use the `lerobot` conda env:

```bash
PYBIN=/home/edgeai/miniconda3/envs/lerobot/bin/python
cd examples/trossen_ai
$PYBIN -m pytest webapp/tests tests -q
```

Hardware-bound modules (`trossen_bridge.py`, the live loop) are syntax/import
checked off-robot; their logic that *can* run without hardware (e.g. the velocity
limiter, IK math) is unit-tested. See [Architecture → Testing](architecture.md#testing).
