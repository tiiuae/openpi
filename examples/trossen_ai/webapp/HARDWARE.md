# Trossen Control Web App — Hardware Setup, Run & Test Guide

End-to-end guide for the **robot machine** (the box with `lerobot_robot_trossen`
installed and the arms physically wired up). Off-hardware development uses the
`lerobot` conda env and the pytest suite instead — see the bottom section.

> The web app is **observe-only**. It imports the existing bridge / ensemble /
> replay code and streams telemetry to the browser. It does **not** reimplement
> the control loop, so anything you can do from the CLI you can do here, with the
> same safety behavior.

---

## 0. Prerequisites (on the robot machine)

- The robot runtime Python env that already runs the CLI (`cli.py`)
  — i.e. the one where `python -c "import lerobot_robot_trossen"` succeeds.
- The OpenPI policy server reachable on the network (default `192.168.50.174:8800`).
- Arms powered, e-stop within reach, **clear workspace**.

Confirm the env first:

```bash
python -c "import lerobot_robot_trossen; print('robot pkg OK')"
```

If that fails you are in the wrong env — activate the robot runtime env and retry.

---

## 1. Install the web dependencies (one-time)

Only two extra packages on top of the existing robot env:

```bash
pip install "fastapi>=0.110" "uvicorn[standard]>=0.27"
```

Nothing else changes — the control stack deps are already present in this env.

---

## 2. Run the server

```bash
cd examples/trossen_ai
python -m uvicorn webapp.server:app --host 0.0.0.0 --port 8000
```

Then open **`http://<robot-host>:8000`** in a browser on the same network.
(`0.0.0.0` lets you drive it from a laptop; use `127.0.0.1` to lock it to the
robot machine only.)

The robot import is **deferred to session start**, so the server boots even if a
dependency is momentarily missing — you only hit hardware when you press Start.

---

## 3. First-run checklist (do this in order)

Work up from zero-movement to live motion. **Do not skip test mode.**

1. **Health.** Page loads; `rtt` and `idle` badges visible top-right.
2. **Live in TEST mode** (no movement — the default mode in the dropdown):
   - Set Policy host/port, Control Hz, Ensemble = `exp`.
   - Press **Start Live**. Watch:
     - Log box streams INFO lines.
     - `rtt` badge turns green (< 100 ms) once inference replies.
     - **Actions** chart moves for the selected joint.
     - **Raw vs Smoothed** overlay shows orange (raw) vs blue (blended).
     - **Overlap timeline** climbs as chunks accumulate.
     - **Blend weights** bars appear (see §4).
     - **Buffer size** chart is bounded (does not grow forever).
     - **Cameras** show frames.
     - **Metrics** box shows RTT p50/p95, loop Hz, jitter, drops.
   - Press **Stop**. Session badge → `idle`.
3. **Presets.** Type a name, **Save**. Reload the page. Select it, **Load** —
   the form repopulates. **Delete** removes it.
4. **Replay (TEST mode).** Set Dataset dir = `../../dataset/converted_to_EE`,
   Episode = `0`, press **Replay Episode**. Same telemetry, sourced from the
   recorded episode instead of live inference.
5. **E-STOP.** During any session, press **E-STOP** — session stops and arms move
   to the sleep pose. Verify it actually halts before trusting autonomous.
6. **Autonomous (LAST).** Only with a clear workspace and a hand on the physical
   e-stop: switch Mode → `autonomous`, **Start Live**, confirm the dialog. The
   arms move for real.

---

## 4. Verifying the blend-weights chart (the new bit)

The **Blend weights (latest step)** bar chart shows the per-prediction weights the
ensemble used to blend the most recent action:

- One bar per overlapping prediction. **Bar 0 = oldest** prediction, last bar = newest.
- Bars sum to 1 (y-axis fixed 0–1).
- **`exp` ensemble:** bars decay oldest→newest (exponential, recency-weighted).
- **`cogact` ensemble:** bar heights reflect *agreement* — predictions that match
  the pack get tall bars, outliers get short ones (cosine-similarity consensus).
  Try the CogACT mode dropdown (`cogact` / `latest` / `hybrid`) and watch the bar
  shape change.
- Single overlap (early in an episode) → one full-height bar (weight 1.0).

If the chart stays empty: the ensemble is `none` (no blending), or no chunk has
produced an action yet — let the session run a few seconds.

---

## 5. Config field reference

All fields map 1:1 to the CLI flags.

| Field | Meaning |
|---|---|
| Policy host / port | OpenPI policy server address |
| Control Hz | Control loop frequency |
| Max steps | Episode length cap |
| Adapter | `joint` or `ee` (end-effector / IK) |
| Ensemble type | `exp`, `cogact`, or `none` |
| CogACT mode | `cogact` / `latest` / `hybrid` (only when ensemble = cogact) |
| Async | Async inference worker |
| Left/Right arm only | Single-arm modes |
| StarVLA | StarVLA policy path |
| Task prompt | Language instruction |
| IK orient w / pos tol | EE-adapter IK solver tuning |
| Dataset dir / Episode | Replay source |
| Mode | `test` (no movement) or `autonomous` |

---

## 5.1 Motion tuning (Advanced) & Sleep/Home

These knobs fix jerky/brutal arm motion and replace the old hardcoded joint-limit
table with a firmware-fault guard.

| Field | Meaning | Default |
|---|---|---|
| Smooth streaming | Send feed-forward joint velocities (`goal_feedforward_velocities`) so the arm carries momentum through waypoints instead of planning to stop at each one. Turn ON to fix stutter. | off |
| Goal-time multiplier (`min_time_to_move_multiplier`) | Driver `goal_time = multiplier / loop_rate`. Larger = smoother but laggier; smaller = snappier but jerkier. | 3.0 |
| Driver loop rate (`loop_rate`) | Driver control loop Hz. Match this to **Control Hz** to avoid mid-motion re-planning. | 25 |
| Connect timeout (`connect_timeout`) | Give up waiting for the policy server after this many seconds (bounded, stop-aware — Stop is honored while still connecting). | 15 |

**Tuning order:** start with Smooth streaming **on** and `loop_rate` == Control Hz.
If still laggy, lower the multiplier toward ~2.0; if jerky, raise it toward ~4.0.

**Firmware guard (replaces joint limits).** There is no longer a software
joint-limit table. If the firmware faults on an action (e.g. velocity exceeded),
the loop catches it, emits a `firmware_error` status to the UI, moves the arm to
the **sleep** pose, and stops the session. This is strictly safer than the old
table (which could pass an out-of-spec action the table didn't cover).

**Sleep / Home buttons** (header, top-right). Each sends the arm(s) to a fixed
pose via PCHIP-interpolated smooth motion, then disconnects:

- **Sleep** → `RobotController.SLEEP_POSITION` (parked/rest).
- **Home** → `HOME_POSITION` (stage pose: arms up & open, ready for task start).

They run through the same single-session guard, so they cannot race a live/replay
session — stop the session first. In `autonomous` mode the browser asks for a
confirm before moving the real robot.

---

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Server starts, Start Live errors with import failure | Wrong env — `lerobot_robot_trossen` not importable. Activate the robot runtime env. |
| `rtt` badge stays grey/red | Policy server unreachable. Check host/port and network. |
| No camera frames | Cameras not enumerated by the bridge; check the CLI path works first. |
| Weights chart empty | Ensemble = `none`, or session hasn't produced a blended action yet. |
| Buffer size chart grows unbounded | Report it — buffer should stay bounded; indicates an eviction bug. |
| Page loads but no telemetry | WebSocket blocked. The browser must reach `ws://<robot-host>:8000/ws/telemetry`. |

---

## 7. Off-hardware development (no robot needed)

On a dev box without the robot package, run the test suite in the `lerobot` env.
This covers everything except real motion, the robot import, and live uvicorn
serving.

```bash
PYBIN=/home/edgeai/miniconda3/envs/lerobot/bin/python
cd examples/trossen_ai
$PYBIN -m pytest webapp/tests/ tests/ -q
# -> 61 passed
```

Run pytest **from `examples/trossen_ai`** — its `conftest.py` puts the dir on
`sys.path`. The `lerobot` env is Python 3.10 (no `datetime.UTC`).

---

## Safety (read before autonomous)

- **Test mode is the default** — no arm movement. Stay in it until charts/cameras/
  RTT all look right.
- **Autonomous and Replay-in-autonomous move the real robot** and require an
  explicit confirm dialog.
- **E-STOP** stops the session and parks the arms at the sleep pose. Verify it
  works in test/replay before relying on it.
- Keep the **physical e-stop** in hand for autonomous runs.
