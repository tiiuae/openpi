# IK Branch-Flip Guard

When an absolute end-effector (EE) pose is IK-decoded to joint angles, the solver
(placo) can settle into an *alternate* arm configuration that reaches the **same**
EE pose through very different joint angles. Between two adjacent frames this shows
up as a one-step joint discontinuity — the "EE looks smooth, joints go aggressive"
branch flip. On hardware that is a velocity spike that faults the firmware (or
worse, slams the arm to the flipped pose).

There are **two independent defenses** against this, and they do *not* cover the
same paths. This matters before you run live.

---

## Two layers

### 1. In-IK branch-flip guard — `max_joint_jump_deg` (hold last good pose)

Implemented in `external/joint_to_ee/ee_to_joints.py`
(`EEToJointsConverter.__init__(..., max_joint_jump_deg=...)`, applied inside
`decode_chunk`). Per frame: if the new IK solve moves any revolute joint more than
`max_joint_jump_deg` vs the previous frame, the frame is **rejected** — it holds
the last good joints and re-seeds the next frame from them. Frame 0 is exempt (its
seed is the home/current pose, so a large first move is legitimate). `None`
disables it.

This rejects the discontinuity **at the source**, before it ever reaches the arm.

### 2. Per-step velocity cap — `max_joint_speed` (clamp to `max_speed * dt`)

Implemented in `robot_control.limit_joint_velocity` (default `MAX_JOINT_SPEED =
3.0` rad/s). It does not reject anything — it spreads any joint-space jump across
several control steps so no single step exceeds the velocity limit. This is the
backstop: it keeps a flip from faulting the firmware even if the flip wasn't
caught upstream.

---

## Related cleanup: the `±2π / branch` unwrap

At the end of `decode_chunk` (after the per-frame guard) there is a separate
post-processing pass:

```python
# Kill residual ±2π/branch wraps on the revolute joints: pick, per joint,
# the 2π-equivalent angle nearest the previous frame...
out[:, rev_idx] = np.unwrap(out[:, rev_idx].astype(np.float64), axis=0)
```

**Why it exists — angle aliasing.** A revolute joint angle is periodic: θ and
θ ± 2π (and θ ± 4π, …) command the **exact same physical orientation**. The IK
solver has no obligation to be consistent about which one it returns. So for a
sequence of frames where the arm barely moves, the solver can report, say,
`3.10 rad` on one frame and `-3.18 rad` (which is `3.10 - 2π`) on the next. The
**physical pose is identical**, but the *number* jumped by ~2π (~360°).

**Symptom.** In the joint-angle plots this is a single-frame vertical "line"
spike — a wrist that looks like it slammed 360° in one step while the EE pose was
actually unchanged. On hardware, commanded naively, it is a real over-limit
velocity jump (the firmware doesn't know the two angles are equivalent).

**Fix — `np.unwrap` along time (`axis=0`), per joint.** It walks each joint's
sequence and adds/subtracts whole multiples of 2π to each sample so consecutive
differences stay within (−π, π] — i.e. it picks, per frame, the 2π-equivalent
angle **nearest the previous frame**. The trajectory comes out continuous; the
physical poses are untouched (only an integer number of full turns is added).

- **Revolute joints only** (`rev_idx` = left + right arm joints): they're in
  radians and periodic, so 2π-equivalence applies.
- **Grippers excluded**: prismatic, measured in metres, not periodic — adding 2π
  to a gripper would be a real, wrong displacement.
- Needs ≥2 frames (`len(out) > 1`) — nothing to unwrap on a single frame.

### Branch-flip guard vs. ±2π unwrap — not the same thing

They both kill "the joints look aggressive while the EE looks smooth" spikes, but
the cause and the response differ:

| | Branch-flip guard (`max_joint_jump_deg`) | ±2π unwrap (`np.unwrap`) |
|---|---|---|
| **What changed** | Solver jumped to a *genuinely different* arm configuration (e.g. elbow-up → elbow-down) reaching the same EE | Same configuration, angle just *labeled* differently by a multiple of 2π |
| **Physical pose** | Really different joints | Identical — pure numbering artifact |
| **Response** | **Reject** the frame, hold last good pose | **Relabel** the angle to its continuous equivalent (keep it) |
| **When** | per-frame, inside the loop | once, over the whole chunk, after the loop |

The guard discards bad data; the unwrap repairs a representation of *good* data.

---

## Scope — where each layer is active

| Path | Branch-flip guard (`max_joint_jump_deg`) | Velocity cap (`max_joint_speed`) |
|------|:--:|:--:|
| **Replay — web UI** (`webapp/runners.py` `ReplayRunner`) | ✅ via `ik_max_joint_jump_deg` config key | ✅ |
| **Off-robot preview** (`webapp/episode_preview.py` `build_trajectory`) | ✅ | ✅ |
| **Live policy** (`webapp/runners.py` `LiveRunner` → `trossen_bridge`) | ✅ via `ik_max_joint_jump_deg` config key (EE adapter only) | ✅ (`trossen_bridge` line ~153) |
| **CLI `replay`** (`cli.py`) | ✅ via `--ik-max-joint-jump-deg` | ✅ (firmware/`execute_action`) |

All paths default to **disabled** (`0` / `None`). Set the knob `>0` to turn the
guard on.

### Live — how it applies

`LiveRunner._make_bridge` now threads `ik_max_joint_jump_deg` into the
`EEToJointsConverter` (only when `adapter == "ee"`). The EE adapter's
`decode_chunk` calls the same guarded converter method, so each policy inference
chunk gets the hold-last-good-pose rejection. The chunk is seeded per inference
from the **measured** joints, so frame 0 of every chunk is exempt (a legitimate
first move toward the new target); the guard only bites on intra-chunk flips.

A joint-space policy (`adapter != "ee"`) does no IK, so the guard does not apply —
its actions are already joint targets.

**Web UI:** the Live page exposes it as **"Branch-flip guard (deg)"** under the
**IK (End-effector only)** section (`config.js`). It is hidden unless *Action
space = End-effector*, since it only acts on the EE→IK path. Default `0` =
disabled.

---

## CLI parameter — added

`cli.py replay` exposes `--ik-max-joint-jump-deg` (default `0.0` = disabled). When
`>0` it is passed as `max_joint_jump_deg` into the `EEToJointsConverter`, giving
the CLI the same hold-last-good-pose rejection as the web UI.

```bash
uv run cli.py replay --mode test --ik-max-joint-jump-deg 20
```

Start with `--mode test` (decode + log, no movement) to confirm the guard is
catching flips before running `--mode autonomous` on the arm.

---

## Related

- Origin commit: `205b504` — *feat(trossen_ai): IK branch-flip guard + smooth
  feed-forward velocity*
- [Motion & Safety](motion-safety.md) — velocity cap, orientation weight, angle unwrap
- `docs/notes/ik-aggressive-joints-NOTES.md` — investigation notes on the flip behavior
