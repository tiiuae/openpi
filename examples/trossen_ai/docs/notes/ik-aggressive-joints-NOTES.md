# Why some episodes do a crazy joint move at the end (EE smooth, joints wild)

Personal investigation notes. NOT part of the docs. Kept here so you have the full
reasoning behind the two knobs that fix it: **IK orientation weight** and the
**joint-angle unwrap** in `decode_chunk`.

---

## 1. The symptom

- Dataset stores **absolute end-effector (EE) poses** per arm: position (x,y,z) +
  orientation quaternion (qw,qx,qy,qz) + gripper. 8 numbers/arm, 16/frame.
- The Replay path turns each EE pose back into **14 joint angles** via inverse
  kinematics (IK), because the robot is commanded in joint space.
- On a handful of episodes (e.g. **episode 5, ~step 545**) the **EE chart is
  perfectly smooth**, but the **joint chart shows a huge jump** — the arm reaches
  far out, a move that can hit the workspace/itself.

So the recorded motion is fine. The damage is introduced **by the EE→joints
conversion**, not by the data.

---

## 2. Root cause: redundant-DOF freedom + a near-singular wrist

The IK lives in `external/joint_to_ee/ee_to_joints.py` (`EEToJointsConverter`).
Two compounding effects:

### (a) Orientation was almost ignored — `orientation_weight = 0.01`

`_ik_arm` solves a weighted least-squares step:

```
minimize  position_weight * ||pos_error||²  +  orientation_weight * ||orient_error||²
```

With `position_weight = 1.0` and `orientation_weight = 0.01`, the solver cares
about position **100×** more than orientation. It will happily reach the target
**position** with almost any wrist orientation it likes.

A 6-DOF arm reaching a 3-DOF *position* target has **3 redundant DOFs**. If
orientation is barely constrained, those redundant joints (mainly
`forearm_roll`, `wrist_rotate`) are free to wander. Frame to frame they can swing
a lot while the fingertip position barely moves — exactly "EE smooth, joints
aggressive."

Near a **wrist singularity** (two wrist axes line up) the redundancy gets worse:
tiny EE changes map to giant joint changes, and the solver can hop to a different
**IK branch** (elbow-up vs elbow-down, wrist flipped 180°+). That branch hop is
the single big jump you see at the end of episode 5.

### (b) The per-frame seed mostly hides it… until it doesn't

`decode_chunk` seeds each frame's IK from the **previous frame's solution**
(`seed_l`/`seed_r` fed forward). That keeps the solution continuous *most* of the
time — the solver starts near where it ended, so it usually stays on the same
branch. But once the trajectory crosses a singular/boundary region, one frame's
solve jumps to another branch; that becomes the seed for the next frame, and the
arm is now committed to the wild configuration → the big end-of-episode move.

---

## 3. Fix knob #1 — raise `orientation_weight`

The recorded quaternion **is** in the data; we were throwing it away. Raising
`orientation_weight` forces the wrist to actually track the recorded orientation,
which removes the redundant freedom and keeps the solver on the intended branch.

- Exposed in the **Replay config card** as *IK orientation weight*, threaded into
  both the **Preview** endpoint and the **Replay** runner, so you can tune it and
  see the effect on the chart **without touching hardware**.
- You found **300** kills the crazy move on episode 5. That's a big number because
  position error is in **metres** (millimetre-scale, so ||pos_err||² is tiny) while
  orientation error is in **radians** (order ~1). To make orientation actually
  compete, its weight has to be large. 300 is not unreasonable given the unit
  mismatch.
- Trade-off: too high and the solver may sacrifice a millimetre of position
  accuracy to nail orientation. For replay that's fine — smoothness/safety beats
  sub-mm position.

## 4. Fix knob #2 — joint-angle unwrap (the residual small jumps)

After raising the weight you still saw **small line-like jumps** in the joint
chart. Those are **±2π wraps** on the continuous-rotation joints: the IK returns,
say, `+179°` on one frame and `-179°` on the next. Physically that's a **2°**
move, but numerically it's a **358° jump** — a thin vertical "line" in the chart,
and a real fast command to the firmware (a small but genuine aggressive twitch).

Fix (added at the end of `decode_chunk`):

```python
rev_idx = list(C.OBS_LEFT_JOINTS) + list(C.OBS_RIGHT_JOINTS)   # 6 joints/arm, no grippers
if len(out) > 1:
    out[:, rev_idx] = np.unwrap(out[:, rev_idx], axis=0)
```

`np.unwrap` walks each joint's time series and, whenever consecutive samples
differ by more than π, adds/subtracts multiples of 2π to make the series
continuous. It changes the **number**, never the **pose** (an angle and angle±2π
are the same physical configuration). Grippers (indices 6, 13) are **prismatic
(metres)**, so they're excluded.

Result: the ±2π wrap jumps vanish; the joint trajectory is continuous.

### Caveat on unwrap
- It only fixes **2π wraps**. A *genuine* branch flip (elbow-up→elbow-down, not a
  multiple of 2π) is **not** removed by unwrap — raising `orientation_weight` is
  what prevents those.
- In theory, unwrapping a joint that legitimately rides near its limit could push
  the unwrapped value past the joint limit; the firmware would clamp it. In
  practice the affected joints (`forearm_roll`, `wrist_rotate`) are
  continuous-rotation, so this is safe. Watch for it only if a *limited* joint
  ever shows post-unwrap values outside its range in the chart.

---

## 5. How to validate (no hardware needed)

The **Preview** uses the exact same `decode_chunk`, so:

1. Open Replay, point at an **EE-action dataset** (must have `action.ee_left` /
   `action.ee_right` columns — a plain joint dataset will error in the banner).
2. Set **IK orientation weight** in the config card (try 1 → 50 → 300).
3. Click **Preview**, scrub to the end of episode 5, watch the **joint chart**:
   - crazy branch-flip move → should disappear as you raise the weight;
   - small ±2π line jumps → already removed by unwrap.
4. The **velocity-spike markers** (red verticals) + banner tell you the peak
   rad/s, which joint, and which frame — use them to confirm the spikes are gone.

## 6. If small/branch jumps still remain
Next lever (not yet implemented): a **joint-space continuity clamp inside the IK
loop** — reject any single-step solve whose joint delta exceeds a threshold and
re-seed, so a branch flip can't be accepted in the first place. Ask and I'll add
it. The current velocity clamp (`limit_joint_velocity`) only *spreads* a jump over
frames for firmware safety; it doesn't *prevent* the arm from migrating to the bad
configuration.

---

## File map
- IK solve + seeding + unwrap: `external/joint_to_ee/ee_to_joints.py`
- velocity clamp + spike detection (preview): `webapp/episode_preview.py`
- velocity clamp (live replay): `robot_control.py` `limit_joint_velocity`
- runner that streams/executes: `webapp/runners.py` `ReplayRunner.run`
- config card + wiring: `webapp/static/replay.html`, `webapp/static/js/replay.js`
