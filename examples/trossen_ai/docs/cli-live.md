<!-- nav -->
**[← Docs Home](README.md)** · **CLI: Live** · [CLI: Replay](cli-replay.md) · [Web App](../webapp/README.md) · [Motion & Safety](motion-safety.md)

---

# CLI — Live Policy

Drive the Trossen bimanual arm directly from a terminal against a running
**OpenPI policy server** — no web app. Everything runs through the single
entry point [`cli.py`](../cli.py); two subcommands share the same control loop
([`trossen_bridge.py`](../trossen_bridge.py)) and differ only in the **action
space** the policy speaks:

| Subcommand | Action space | Use when |
|---|---|---|
| [`cli.py live-joint`](../cli.py) | **Joint** — raw 14-D joint targets | The policy was trained to output joints. |
| [`cli.py live-ee`](../cli.py) | **End-effector** — 8-D EE pose per arm, IK-decoded to joints | The policy outputs EE poses (see [end_effector_support.md](end_effector_support.md)). |

## Contents

- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Test mode vs autonomous](#test-mode-vs-autonomous)
- [Arguments](#arguments)
- [End-effector extras](#end-effector-extras)
- [Stopping & resetting the arm](#stopping--resetting-the-arm)
- [Troubleshooting](#troubleshooting)

> **Safety:** in `--mode autonomous` the arm moves for real. Read
> [Motion & Safety](motion-safety.md) and the
> [Hardware runbook](../webapp/HARDWARE.md) before your first autonomous run.

## Prerequisites

1. A reachable **policy server** (default `192.168.50.174:8800`). Start one with
   `uv run scripts/serve_policy.py …` from the repo root (see the top-level
   [README](../README.md)).
2. The arm powered on and reachable (left `192.168.1.5`, right `192.168.1.4`).
3. Run from the `examples/trossen_ai` directory so the local modules import.

## Quick start

```bash
cd examples/trossen_ai

# Joint-space policy, dry run (no movement) — always do this first:
uv run cli.py live-joint --mode test --task-prompt "grab red cube"

# Joint-space policy, real movement:
uv run cli.py live-joint --mode autonomous --task-prompt "grab red cube"

# End-effector policy (IK-decoded to joints):
uv run cli.py live-ee --mode autonomous --task-prompt "grab red cube"
```

Point at a non-default server:

```bash
uv run cli.py live-joint --policy-host 10.0.0.20 --policy-port 8800 --mode test
```

Run `uv run cli.py --help` (or `cli.py live-joint --help`, `cli.py live-ee --help`)
for the live list of subcommands and flags.

## Test mode vs autonomous

- `--mode test` — runs the full loop (connect, infer, ensemble, log) but
  **never sends motion** to the arm. Use it to confirm the policy server is
  reachable and the action stream looks sane.
- `--mode autonomous` — executes actions on the real arm. The firmware
  velocity-limit guard and the move-to-start ramp still apply
  (see [Motion & Safety](motion-safety.md)).

## Arguments

Shared by `live-joint` and `live-ee` (run `--help` for the live list):

| Flag | Default | Meaning |
|---|---|---|
| `--policy-host` | `192.168.50.174` | Policy server host. |
| `--policy-port` | `8800` | Policy server TCP port. |
| `--control-freq` | `25` | Control steps per second. |
| `--mode` | `autonomous` | `autonomous` (execute) or `test` (no movement). |
| `--task-prompt` | `"move the arm to the left"` | Natural-language instruction sent to the policy. |
| `--max-steps` | `1000` | Episode ends after this many control steps. |
| `--action-chunk-size` | `25` | Actions predicted per inference. |
| `--rate-of-inference` | `20` | Control steps between inferences. |
| `--ensemble-type` | `exp` | Action blending: `exp`, `cogact`, or `none`. |
| `--cogact-mode` | `cogact` | CogACT weighting (`cogact`/`latest`/`hybrid`); only with `--ensemble-type cogact`. |
| `--log-dir` | `None` | Directory for per-episode overlap JSON. |
| `--async-inference` | off | Background-thread inference (needs an ensemble; **joint mode only**). |
| `--starvla` | off | StarVLA 224×224 PIL image resizing. |
| `--use-left-arm-only` | off | Drive only the left arm; right holds. |
| `--use-right-arm-only` | off | Drive only the right arm; left holds. |

### End-effector extras

`live-ee` adds IK knobs (and rejects `--async-inference`, unsupported in EE mode):

| Flag | Default | Meaning |
|---|---|---|
| `--ik-orientation-weight` | `0.01` | placo IK orientation weight; raise for tighter rotation tracking. |
| `--ik-pos-tol-m` | `1e-3` | IK convergence/failure tolerance (m); above it after max iters → hold last joints. |

## Stopping & resetting the arm

- `Ctrl-C` ends the episode; the bridge cleans up on exit.
- To park the arm at the safe sleep pose afterwards:

  ```bash
  uv run scripts/sleep.py
  ```

  See [CLI: Replay → Sleep helper](cli-replay.md#sleep-helper).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Hangs at "Connecting to policy server" | Server not running or wrong `--policy-host`/`--policy-port`. Verify with `--mode test`. |
| "joint velocity limit exceeded" / arm sent to sleep | A large policy/IK jump. See [Motion & Safety](motion-safety.md). |
| `--async-inference is not supported in EE mode` | EE decoding is synchronous; drop the flag or use `live-joint`. |
| `ModuleNotFoundError` | Run from `examples/trossen_ai`, and `uv sync` first. |

---

**[← Docs Home](README.md)** · **CLI: Live** · [CLI: Replay](cli-replay.md) · [Web App](../webapp/README.md) · [Motion & Safety](motion-safety.md)
