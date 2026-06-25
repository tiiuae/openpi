# Motion & Safety

[← Docs home](README.md) · [← Architecture](architecture.md)

How the stack keeps arm motion smooth and within the firmware's limits, and how
to tune it. This is the most safety-relevant doc — read it before autonomous or
replay runs, alongside the [hardware runbook](../webapp/HARDWARE.md).

## 1. The firmware velocity limit

Each motor enforces a hard velocity limit in firmware. On the WidowX AI arms a
representative joint limit is **±3π ≈ 9.42 rad/s**. Exceed it and the controller
faults:

```
[Motor Interface] Joint 3 velocity limit exceeded:
  expected in range [-9.424778, 9.424778], motor reported -9.604396. Setting to idle.
```

When this happens the motor **latches to idle** — it will not accept motion
commands until re-enabled/power-cycled. (A subsequent move-to-sleep will also
fail with the *same* latched error; that is the latch being re-read, not a new
fault.)

## 2. The IK branch-flip fault (root cause of the "brutal replay")

Replaying a dataset converts recorded **end-effector** poses to **joint** angles
via IK ([`EEToJointsConverter`](../external/joint_to_ee/ee_to_joints.py)). The EE
path is smooth, but the IK solver occasionally jumps to an **alternate joint
configuration** for nearly the same pose — a *branch-flip*. Several joints leap at
once in a single control step.

Measured on a real episode (50 fps, dt = 20 ms):

| | Max commanded joint velocity |
|---|---|
| Left arm (smooth) | ~0.3 rad/s |
| Right arm at the flip frame | **16 / 21 / 13 rad/s** (joints R2/R3/R4) |

21 rad/s ≫ 9.42 → firmware fault. The flip is **non-deterministic** (depends on
IK seeding), so it can land on different frames between runs — which is exactly
why the fix is a general velocity cap, not a per-frame patch.

> An earlier change made it worse: the replay driver `loop_rate` had been set to
> the dataset fps (50) instead of the previous fixed 30, shrinking the driver
> `goal_time = multiplier / loop_rate` and pushing peak velocity over the limit.
> Replay's `loop_rate` default is back to 30.

## 3. The velocity limiter (the fix)

[`robot_control.limit_joint_velocity(prev14, target14, dt, max_speed)`](../robot_control.py)
clamps each commanded joint target so no joint moves faster than `max_speed`:

```
|target - prev|  ≤  max_speed * dt     (per joint)
```

The clamped target is fed back as `prev` on the next step, so a branch-flip is
**spread across several control steps** instead of commanded as one over-limit
jump. The arm still reaches the true target — verified offline on the real
episode: **raw 22.5 rad/s → clamped 3.0 rad/s, with zero final tracking error**.

Default `max_speed = MAX_JOINT_SPEED = 3.0` rad/s — well under the 9.42 firmware
limit (leaving margin for the driver's peak-velocity profile) and ~10× normal
recorded motion, so smooth playback is untouched and only a flip gets capped.

**Applied on both control paths:**

- **Replay** — [`ReplayRunner.run()`](../webapp/runners.py) clamps every frame before `execute_action`.
- **Live** — [`TrossenOpenPIBridge.execute_action()`](../trossen_bridge.py) clamps each policy action against the last command (`max_joint_speed`, `≤0` disables).

## 4. Smooth streaming (feed-forward velocity)

Separate from the cap: [`send_action_smooth()`](../robot_control.py) reduces the
*jerk* of normal motion. Plain `robot.send_action()` sends zero feed-forward
velocity, so the driver plans to arrive at rest at every waypoint → stutter.
Smooth streaming sends `ff = (goal − current)/dt` and `goal_time = dt`, so the arm
carries momentum through waypoints. Enable with **Smooth streaming** in the Live
config (off by default).

## 5. The firmware-fault guard

If a `send_action` raises (firmware fault), both `RobotController.execute_action`
and the bridge **catch it**, emit a `firmware_error` status to the UI, move the
arm to the **sleep** pose, and stop the session — instead of crashing the loop.
There is no longer a hardcoded software joint-limit table; the firmware is the
source of truth and the velocity limiter keeps commands inside it.

> If a fault still latches the motor to idle, re-enable/power-cycle the affected
> arm before the next run.

## 6. Tuning knobs

Set per session in the Live config form (Motion tuning) or as runner config keys.

| Knob | Key | Default | Effect |
|---|---|---|---|
| Max joint speed | `max_joint_speed` | 3.0 rad/s | Hard per-joint velocity cap. Lower = gentler/safer; `0` disables. **The primary anti-fault safeguard.** |
| Smooth streaming | `smooth_streaming` | off | Feed-forward velocity → momentum through waypoints; fixes stutter. |
| Goal-time multiplier | `min_time_to_move_multiplier` | 3.0 | Driver `goal_time = multiplier / loop_rate`. Larger = smoother/laggier; smaller = snappier/jerkier. |
| Driver loop rate | `loop_rate` | 30 (replay) / control rate (live) | Higher = shorter `goal_time` = snappier (and more aggressive). |
| Connect timeout | `connect_timeout` | 15 s | Bounded, stop-aware wait for the policy server. |

**Recommended tuning order:** keep `max_joint_speed` at 3.0 (lower to ~2.0 if
still too quick), turn `smooth_streaming` on, then adjust the multiplier for feel.

## 7. Pre-run safety checklist

1. **Clear workspace**, physical e-stop in hand.
2. Stay in **test mode** until charts/telemetry look right (no movement).
3. After any firmware fault: **re-enable/power-cycle the faulted arm** before retrying.
4. Start gentle: low `max_joint_speed`, then raise if needed.
5. Use **Sleep** / **Home** (single-session guarded) to park the arms between runs.

See the [hardware runbook](../webapp/HARDWARE.md) for the full first-run checklist.
