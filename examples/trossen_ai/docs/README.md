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
| [RobotController & the control chain](robot-controller.md) | Deep dive on how an action becomes motion: layers down to the motor firmware, every `RobotController` function/argument, and how each config knob (`goal_time`, smooth streaming, velocity cap, …) changes the motion. |
| [CLI — Live policy](cli-live.md) | Run a policy from the terminal (no web app): `cli.py live-joint` / `cli.py live-ee` (end-effector), arguments, test vs autonomous. |
| [CLI — Dataset replay](cli-replay.md) | Replay a recorded episode from the terminal (no web app, no policy server): `cli.py replay`, the sleep helper. |
| [Motion & Safety](motion-safety.md) | Firmware velocity limit, the IK branch-flip fault, the velocity limiter, smooth streaming, the firmware-fault guard, and all tuning knobs. |
| [Action smoothing — Temporal & CogACT](action_smoothing.md) | How overlapping action chunks are blended: temporal exp-decay ensembling vs CogACT consensus-by-agreement, the three CogACT modes, and where to watch it live. |
| [Motion tuning knobs (Live)](motion_tuning_live.md) | What goal-time multiplier, driver loop rate, and max joint speed do; how they interact; and why they act on execution, not the policy model. |
| [Async vs. inference interval](async_vs_inference_interval.md) | Why sync + interval=1 slows the control loop — how disabling async inference with `rate_of_inference = 1` collapses control-loop rate toward `1 / RTT`, and how async inference avoids it. |
| [Live vs `main.py` joint eval](live_vs_main_joint_eval.md) | Divergence audit: web-app Live joint run vs `trossen-ai` `main.py` — what's numerically identical and where execution/defaults differ. |
| [Web App — overview](../webapp/README.md) | The two pages (Live / Replay) and the frontend module layout. |
| [Web App — hardware runbook](../webapp/HARDWARE.md) | Step-by-step run, first-run checklist, motion tuning, troubleshooting, off-robot testing. |

## The three ways to drive the arm

1. **CLI — live policy.** [`cli.py live-joint`](../cli.py) (joint space) / [`cli.py live-ee`](../cli.py) (end-effector space) build a [`TrossenOpenPIBridge`](../trossen_bridge.py) and run an episode against the policy server. → **[CLI: Live guide](cli-live.md)**
2. **CLI — dataset replay.** [`cli.py replay`](../cli.py) replays a recorded episode through IK onto the arm. → **[CLI: Replay guide](cli-replay.md)**
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
