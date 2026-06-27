# RobotController & the control chain

[← Docs home](README.md) · [Architecture](architecture.md) · [Motion & Safety](motion-safety.md)

This explains **how an action becomes arm motion** — every layer from a 14-D joint
vector down to the motor firmware — plus every `RobotController` function, its
arguments, and exactly how each config knob changes the motion.

---

## 1. The layers

```
 your 14-D action  (left 0:7 | right 7:14)
        │
        ▼
 RobotController            robot_control.py   ── 14-D joint motion + safety
        │  execute_action()
        ▼
 BiWidowXAIFollower (robot) lerobot_robot_trossen ── splits 14-D → left / right
        │  send_action(dict)                          and exposes get_observation()
        ▼
 WidowXAIFollower ×2        widowxai_follower.py ── one per arm; turns a joint
        │  driver.set_all_positions(...)               dict into a driver call
        ▼
 TrossenArmDriver           trossen_arm (C++ .pyi) ── talks to the arm over IP
        │  set_all_positions(goal, goal_time, ff…)
        ▼
 Motor firmware             on the arm           ── interpolates to the goal,
                                                     enforces velocity/limit faults
```

- **`RobotController`** works in **14-D joint space**: left arm = indices `0:7`,
  right arm = `7:14`. Each arm has **7 joints = 6 revolute (rad) + 1 gripper
  (prismatic, metres)**.
- It never talks to a motor directly. It calls the lerobot robot
  (`robot.send_action`) or, on the smooth path, each arm's driver
  (`robot.left_arm.driver.set_all_positions`).
- The robot object is built by [`build_stationary_robot()`](../robot_control.py),
  which is also where the driver-tuning config (`min_time_to_move_multiplier`,
  `loop_rate`, cameras, arm IPs) is set.

---

## 2. The key idea: the firmware moves to a *goal* over a *goal_time*

The lowest layer is the driver call (from `trossen_arm.pyi`):

```python
driver.set_all_positions(
    goal_positions,              # list[float] — rad per arm joint, m for gripper
    goal_time=2.0,               # seconds to REACH the goal
    blocking=True,               # wait until reached?
    goal_feedforward_velocities=None,   # rad/s, default zeros
    goal_feedforward_accelerations=None,
)
```

**The firmware generates the trajectory.** You give it a target and a `goal_time`;
it interpolates from the *current* position to the goal so it arrives in
`goal_time`. The commanded joint velocity is therefore roughly:

```
velocity ≈ (goal − current) / goal_time          (peak is higher — the profile
                                                   accelerates then decelerates)
```

That single relationship explains almost everything about "aggressiveness" and the
velocity faults:

- **Small `goal_time`** → the firmware tries to cover the same distance in less
  time → **higher velocity** → snappier, and past `±9.42 rad/s` it **faults**
  (`Joint N velocity limit exceeded`). See [Motion & Safety](motion-safety.md).
- **`goal_feedforward_velocities`** lets the arm *carry speed through* a waypoint
  instead of planning to stop at it (used by smooth streaming).
- **`blocking=True`** waits until the goal is reached (used for the slow 2 s
  staged/sleep moves); **`blocking=False`** returns immediately so you can stream a
  new goal every control step (used for live/replay).

Where does `goal_time` come from for a normal `send_action`? In the follower:

```python
# widowxai_follower.py
self.min_time_to_move = config.min_time_to_move_multiplier / config.loop_rate
...
def send_action(self, action):
    ...
    self.driver.set_all_positions(goal_positions=[...],
                                  goal_time=self.min_time_to_move,
                                  blocking=False)
```

So:

```
goal_time (plain send_action)  =  min_time_to_move_multiplier / loop_rate
```

This is the most important tuning relationship in the whole stack.

---

## 3. `build_stationary_robot(...)` — where the driver tuning is set

```python
def build_stationary_robot(*, connect=True, with_cameras=True,
                           min_time_to_move_multiplier=3.0, loop_rate=30):
```

| Argument | Meaning / effect on control |
|---|---|
| `connect` | If `True`, calls `robot.connect()` (configures both drivers to **position mode** and moves to the staged pose) before returning. `False` builds the object without touching hardware. |
| `with_cameras` | Attach the three OpenCV cameras. Replay/movers pass `False` (no images needed); the live loop keeps them. |
| `min_time_to_move_multiplier` | Numerator of `goal_time = multiplier / loop_rate`. **Larger → larger goal_time → slower, smoother** motion (and laggier); smaller → snappier/jerkier. |
| `loop_rate` | Denominator of `goal_time`. **Higher → smaller goal_time → faster/snappier** (and more aggressive — this is what made replay fault when it was set to fps=50). |

`connect()` also runs `driver.configure(model=wxai_v0, …, clear_error=True)` and
`set_all_modes(Mode.position)`, and `disconnect()` moves the arm to the staged pose
then to all-zeros (sleep) with `goal_time=2.0, blocking=True` before cleanup.

---

## 4. `RobotController` — constructor & state

```python
RobotController(robot, control_frequency=50, test_mode="autonomous",
                smooth_streaming=False)
```

| Argument | Meaning / effect on control |
|---|---|
| `robot` | A connected `build_stationary_robot()` object (the bimanual follower). |
| `control_frequency` | Control steps per second. Sets `self.dt = 1/control_frequency`. **`dt` is both** the loop period **and** the `goal_time` used on the *smooth* path. Higher = finer, faster updates. |
| `test_mode` | `"test"` = dry run, **no motion** (`execute_action` only logs). `"autonomous"` = actually send to the arm. |
| `smooth_streaming` | Selects the send path in `execute_action` (see §5–6). |

**`current_joints14()`** → reads `robot.get_observation()`, pulls every `*.pos`
key, returns them as a 14-D array (the live joint positions; what "current" means
in the velocity formula above).

---

## 5. `execute_action(action) -> bool` — the heart

```python
def execute_action(self, action) -> bool:
    full_action = np.asarray(action).flatten()      # 14-D
    if self.test_mode == "test":
        log(...); return True                       # dry run, no motion
    try:
        if self.smooth_streaming:
            send_action_smooth(self.robot, full_action, self.dt)   # path A
        else:
            joint_features = list(self.robot._joint_ft.keys())     # path B
            action_dict = {k: full_action[i] for i, k in enumerate(joint_features)}
            self.robot.send_action(action_dict)
    except Exception as exc:                         # firmware fault
        log(exc); self.move_to_sleep_position(duration=10.0)
        return False
    return True
```

- **Argument `action`**: any 14-D joint vector (list/array). Flattened to 14 floats.
- **Returns** `True` on success, `False` if the firmware faulted (in which case the
  arm is moved to the sleep pose — the [firmware-fault guard](motion-safety.md#5-the-firmware-fault-guard)).
- **Path B (default, `smooth_streaming=False`)**: builds `{joint_feature: value}`
  and calls `robot.send_action`. The follower clips to `max_relative_target` (if
  set in config), then `set_all_positions(goal, goal_time = multiplier/loop_rate,
  blocking=False, ff=0)`. Zero feed-forward → the arm plans to **arrive at rest**
  at every waypoint (the source of stutter).
- **Path A (`smooth_streaming=True`)**: calls `send_action_smooth` (§7).

> The velocity cap (`limit_joint_velocity`, see [Motion & Safety §3](motion-safety.md#3-the-velocity-limiter-the-fix))
> is applied **by the callers** (`ReplayRunner`, the bridge) *before* `execute_action`,
> so the `action` reaching here is already velocity-bounded.

---

## 6. The two send paths, side by side

| | Path B — `send_action` (default) | Path A — `send_action_smooth` |
|---|---|---|
| `goal_time` | `multiplier / loop_rate` | `dt` (= `1/control_frequency`) |
| feed-forward velocity | **0** (arrive at rest → stutter) | `(goal − current)/dt` (carry momentum) |
| who is called | `robot.send_action` → both arms | `left_arm.driver` & `right_arm.driver` directly |
| `max_relative_target` clip | yes (if configured) | no |
| use | normal/default | enable for smooth, continuous motion |

---

## 7. `send_action_smooth(robot, action14, dt)`

```python
def send_action_smooth(robot, action14, dt):
    current = <robot's current .pos vector>
    ff = (action14 - current) / dt                  # feed-forward velocity rad/s
    ff = nan_to_num(ff)
    n = len(robot.left_arm.config.joint_names)       # 7
    robot.left_arm.driver.set_all_positions(action14[:n],  goal_time=dt,
        blocking=False, goal_feedforward_velocities=ff[:n])
    robot.right_arm.driver.set_all_positions(action14[n:2n], goal_time=dt,
        blocking=False, goal_feedforward_velocities=ff[n:2n])
```

| Argument | Meaning |
|---|---|
| `robot` | the bimanual follower (needs `left_arm`/`right_arm` with `.driver`/`.config`). |
| `action14` | 14-D joint target. |
| `dt` | control period; used as **both** `goal_time` and the divisor for feed-forward velocity. |

Why it's smoother: with `goal_time = dt` and a matching feed-forward velocity, the
arm is told *"be here, moving at this speed, in exactly one control period"* — so it
flows through waypoints instead of decelerating to rest at each one.

---

## 8. `move_to_start_position(goal_position, duration=5.0)` & `move_to_sleep_position(duration=10.0)`

Both build a **PCHIP** (smooth, monotone) interpolation from the current pose to a
target and step it at `self.dt`:

| Argument | Meaning |
|---|---|
| `goal_position` (start only) | 14-D target pose to ramp to (e.g. the first replay frame, or `HOME_POSITION`). |
| `duration` | seconds for the whole ramp. Larger = gentler. Start ramp default 5 s; sleep default 10 s. |

- `move_to_start_position` calls `execute_action` each step (so it honors
  `test_mode`/`smooth_streaming`).
- `move_to_sleep_position` sends positions **directly** via `robot.send_action`
  (it's the safety fallback, used by the firmware guard and E-STOP) toward
  `SLEEP_POSITION` (all zeros). It runs even from the fault handler.

`disconnect()` → `robot.disconnect()` (which itself ramps to staged then sleep and
cleans up the driver).

---

## 9. Module-level constants & helpers

| Name | Meaning |
|---|---|
| `SLEEP_POSITION` | `np.zeros(14)` — the parked pose. |
| `HOME_POSITION` | a "stage" pose (arms up & open) for the Home button. |
| `MAX_JOINT_SPEED = 3.0` | default rad/s cap used by `limit_joint_velocity`. |
| `limit_joint_velocity(prev14, target14, dt, max_speed)` | clamps `|target−prev| ≤ max_speed·dt` per joint; see [Motion & Safety §3](motion-safety.md#3-the-velocity-limiter-the-fix). |

---

## 10. How each config knob reaches the motors (summary)

| Config key | Enters at | What it changes | Net effect |
|---|---|---|---|
| `control_freq` | `RobotController.control_frequency` → `dt` | loop period; smooth-path `goal_time` | higher = finer/faster updates |
| `min_time_to_move_multiplier` | follower `min_time_to_move` numerator | plain-path `goal_time = mult/loop_rate` | higher = slower, smoother |
| `loop_rate` | follower `min_time_to_move` denominator | plain-path `goal_time` | higher = snappier, more aggressive |
| `smooth_streaming` | `execute_action` path select | feed-forward velocity + `goal_time=dt` | on = momentum through waypoints, less stutter |
| `max_joint_speed` | caller, before `execute_action` | clamps per-step joint delta | hard velocity cap; prevents faults |
| `test_mode` | `execute_action` gate | send vs log-only | `test` = no motion |
| `max_relative_target` | follower `send_action` (config) | clips goal vs present | bounds per-step jump (plain path only) |

### Worked example (the replay fault)

An IK branch-flip asks for **Δ = 0.42 rad** on one joint in one step:

| `goal_time` | avg velocity | ~peak (≈2×) | result |
|---|---|---|---|
| `3/30 = 0.10 s` (old) | 4.2 rad/s | ~8.4 | under 9.42 → ok |
| `3/50 = 0.06 s` (regressed) | 7.0 rad/s | ~14 | **over 9.42 → fault** |
| with `max_joint_speed=3.0` | Δ capped to `3.0·dt` | spread over steps | safe |

This is why the fix both restored `loop_rate=30` for replay **and** added the
velocity cap. Full story: [Motion & Safety](motion-safety.md).

---

## See also
- [Architecture](architecture.md) — where `RobotController` sits in the whole stack.
- [Motion & Safety](motion-safety.md) — firmware limits, the velocity limiter, tuning.
- [Hardware runbook](../webapp/HARDWARE.md) — running it on the real arm.
