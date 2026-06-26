<!-- nav -->
**[← Docs Home](README.md)** · [CLI: Live](cli-live.md) · **CLI: Replay** · [Web App](../webapp/README.md) · [Motion & Safety](motion-safety.md)

---

# CLI — Dataset Replay

Replay one recorded episode from a **LeRobot v3.0 dataset** onto the arm, from a
terminal — no web app, **no policy server**.
[`replay_ee_dataset.py`](../replay_ee_dataset.py) reads the episode's absolute
end-effector actions (`action.ee_left` + `action.ee_right`, 8-D poses in the
robot-base frame), IK-decodes them to 14-D joint targets with the same converter
[`main_ee.py`](cli-live.md) uses, then streams them at the dataset fps — a PCHIP
move-to-start first, then per-step joint-velocity safety on every frame.

## Contents

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Arguments](#arguments)
- [How a replay runs](#how-a-replay-runs)
- [Sleep helper](#sleep-helper)
- [Troubleshooting](#troubleshooting)

> **Safety:** in `--mode autonomous` the arm moves for real. Read
> [Motion & Safety](motion-safety.md) and the
> [Hardware runbook](../webapp/HARDWARE.md) first.

## Prerequisites

1. A LeRobot v3.0 dataset directory. Default:
   `repo-root/dataset/converted_to_EE`. Override with `--dataset_dir`.
2. The arm powered on and reachable (left `192.168.1.5`, right `192.168.1.4`).
3. Run from the `examples/trossen_ai` directory.

## Quick start

```bash
cd examples/trossen_ai

# Decode + log only, no movement — always do this first:
uv run replay_ee_dataset.py --episode_index 0 --mode test

# Replay episode 0 on the real arm:
uv run replay_ee_dataset.py --episode_index 0 --mode autonomous

# A different dataset and episode:
uv run replay_ee_dataset.py --dataset_dir /data/my_ds --episode_index 3 --mode autonomous
```

## Arguments

Run `uv run replay_ee_dataset.py --help` for the live list.

| Flag | Default | Meaning |
|---|---|---|
| `--dataset_dir` | `repo/dataset/converted_to_EE` | LeRobot v3.0 dataset directory. |
| `--episode_index` | `0` | Which episode to replay. |
| `--control_freq` | dataset fps | Stream rate in Hz (defaults to the episode's fps). |
| `--mode` | `autonomous` | `autonomous` (move arms) or `test` (decode + log, no movement). |
| `--start_duration` | `5.0` | Seconds for the PCHIP move to the episode's first pose. |
| `--ik_orientation_weight` | `0.01` | placo IK orientation weight; raise for tighter rotation tracking. |
| `--ik_pos_tol_m` | `1e-3` | IK convergence/failure tolerance (m); above it → hold last joints. |

## How a replay runs

1. Read the episode; report frame count, fps, and task.
2. Seed IK from the arm's **actual current joints**, then decode every EE frame
   to joints (decode chains frame-to-frame for continuity).
3. PCHIP move-to-start to the first pose over `--start_duration`.
4. Stream the remaining frames at `--control_freq`. Each frame passes the
   joint-velocity safety check; a frame that would exceed limits aborts the
   replay and sends the arm to sleep.

## Sleep helper

After tests/replays the arm may be in a random pose. Park it safely:

```bash
uv run sleep.py
```

[`sleep.py`](../sleep.py) smoothly interpolates from the current pose to the
sleep position over ~5 s, then disconnects. No arguments.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `FileNotFoundError` / empty episode | Wrong `--dataset_dir` or `--episode_index`. Confirm with `--mode test`. |
| Aborts mid-replay: "exceeded joint limits" | An IK branch-flip produced an over-limit jump; the safety cap aborted. See [Motion & Safety](motion-safety.md). |
| Arm jerky at start | Increase `--start_duration`, or lower `--control_freq`. |
| `ModuleNotFoundError` | Run from `examples/trossen_ai`, and `uv sync` first. |

---

**[← Docs Home](README.md)** · [CLI: Live](cli-live.md) · **CLI: Replay** · [Web App](../webapp/README.md) · [Motion & Safety](motion-safety.md)
