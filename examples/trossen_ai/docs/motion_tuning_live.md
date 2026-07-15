# Goal-time Multiplier, Driver Loop Rate & Max Joint Speed (Live mode)

What the three **Motion tuning (Advanced)** knobs do, how they interact, and the
key question: **are they functional in Joint action space, and do they affect my
policy model?**

**Short answer:** All three are **execution-side** (driver / safety) parameters.
They shape *how the commanded joint targets are sent to the arm firmware*, **after**
the policy has predicted and after smoothing/decoding. They are **fully functional
in Joint action space** (they act on the final 14-D joint command, which exists in
every action space). They **do not change your policy model** — not its weights,
not its architecture, not what it predicts. The only indirect effect is
closed-loop: if a knob changes the *executed* motion, the next observation the arm
reports back differs — but the model itself is untouched.

Source of truth: [`trossen_bridge.py`](../trossen_bridge.py),
[`robot_control.py`](../robot_control.py).

---

## The three knobs

| UI field | Config key | Default |
|----------|-----------|---------|
| Goal-time multiplier | `min_time_to_move_multiplier` | `3.0` |
| Driver loop rate (Hz) | `loop_rate` | `25` |
| Max joint speed (rad/s) | `max_joint_speed` | `3.0` |

All three enter the bridge in `LiveRunner._make_bridge` and are used only in
`test_mode == "autonomous"` (the real robot). In dry-run they do nothing —
there is no firmware to command.

---

## 1. Goal-time multiplier (`min_time_to_move_multiplier`)

`goal_time` is the **time horizon the firmware plans each move over**: it plans a
trajectory from the arm's current (position, velocity) to the commanded target,
arriving in `goal_time` seconds. The multiplier scales that horizon.

There are **two execution paths**, and the multiplier is used differently in each:

### Path A — plain send (`smooth_streaming = OFF`, the Live default)

`robot.send_action(...)` uses the horizon baked into the **driver config** at
robot build (`build_stationary_robot(min_time_to_move_multiplier=…, loop_rate=…)`):

```
driver goal_time  =  min_time_to_move_multiplier / loop_rate
```

So here the multiplier and **loop rate together** set the horizon. Example:
`3.0 / 25 Hz = 0.12 s` per commanded move.

### Path B — smooth streaming (`smooth_streaming = ON`)

`send_action_smooth(...)` computes the horizon from the **control period**:

```
goal_time  =  min_time_to_move_multiplier · dt        (dt = 1 / control_freq)
```

then **clamps it to ≥ ~0.21 s**. This matters: the Trossen firmware only runs the
**quintic** interpolation that honours the feed-forward velocity when
`goal_time > 0.2 s`. Below that it silently drops to linear interpolation and
**ignores** the feed-forward velocity — the old "accelerate-then-brake" chatter.
So with smooth streaming, the multiplier scales the horizon but never below the
quintic threshold.

### Effect either way

| Multiplier | Motion |
|-----------|--------|
| Larger | Longer horizon → smoother, but the arm **lags** the target more (it's always heading toward a pose it plans to reach further in the future). |
| Smaller | Shorter horizon → snappier / more reactive, but **jerkier**; too small and moves become stop-start. |

---

## 2. Driver loop rate (`loop_rate`)

Passed into the driver config at robot build. It sets the driver's internal
command cadence and, in **Path A**, is the denominator of the goal-time formula
(`goal_time = multiplier / loop_rate`).

**Rule: match `loop_rate` to your Control rate (`control_freq`).** If they differ,
the driver's planning cadence and the loop's command cadence disagree, so the
driver **re-plans mid-motion** — you issue a new target before the previous plan
finishes, producing stutter. Matching them makes each command a clean single plan.

Note the asymmetry between the paths:
- Path A horizon depends on **`loop_rate`** (`/ loop_rate`).
- Path B horizon depends on **`control_freq`** (`· dt = / control_freq`).

If `loop_rate == control_freq`, the two formulas agree and there's no surprise.

---

## 3. Max joint speed (`max_joint_speed`, rad/s)

A **safety velocity cap** applied to **every** commanded action in autonomous
mode, before it's sent (`limit_joint_velocity`), on **both** paths:

```
|target − prev_command|  per joint  ≤  max_joint_speed · dt
```

A one-step joint jump (a policy discontinuity, or an IK branch-flip in EE mode) is
**spread across several control steps** instead of commanding an over-limit
velocity that trips the firmware limit (~9.4 rad/s → "joint velocity limit
exceeded"). The clamped target is fed back as `prev` so the arm keeps migrating
toward the true target over the next few steps.

| `max_joint_speed` | Effect |
|-------------------|--------|
| Lower | Gentler; large jumps ramp in slowly. Too low → the arm can't keep up with fast intended motion (it lags the policy). |
| Higher | Follows the policy more literally; less protection against spikes. |
| `0` (or ≤ 0) | Disabled — no cap. |

Because the clamp uses `dt = 1/control_freq`, the *per-step* allowance scales with
your control rate: a lower control rate means a larger allowed jump per step for
the same `max_joint_speed`.

---

## 4. Are they functional in **Joint** action space?

**Yes — all three.** Action space (`Joint` vs `End-effector`) only decides how the
policy's output becomes a 14-D joint vector:

- **Joint:** the policy's 14-D joints are used directly.
- **End-effector:** the policy's EE poses are solved to 14-D joints via IK first.

Either way you end up with a 14-D joint command, and **all three knobs act on that
final joint command** at send time. Nothing about them is EE-specific. (The
branch-flip guard under *IK (End-effector only)* is the only EE-specific knob —
that one genuinely does nothing in Joint space.)

---

## 5. Do they affect my **policy model**?

**No.** They are downstream of inference:

```
policy.infer(obs)  →  decode (IK if EE)  →  temporal/CogACT smoothing
                   →  [max_joint_speed clamp]  →  send with [goal_time / loop_rate]
                                                   └── these three knobs live here
```

- They do **not** change the observation the model receives.
- They do **not** change the model's predicted actions.
- They do **not** touch weights, architecture, or the inference call.

They only change **how the arm physically tracks** the commanded joints.

**One caveat — closed loop.** The robot is closed-loop: each step reads the arm's
current state and feeds it into the next observation. If `max_joint_speed` clamps a
command, or a large `goal_time` makes the arm lag, the *actual* pose the arm
reaches differs from the predicted one, so the **next observation** the policy sees
is different. That can change subsequent predictions — but that's the model
reacting to a different world state, **not** the knobs modifying the model. If you
want the executed motion to match the policy's intent as closely as possible
(e.g. when evaluating the policy itself), keep `max_joint_speed` high enough not to
bite on normal motion and `goal_time` short enough to avoid lag.

---

## 6. Recommended starting points (Live, real robot)

- `loop_rate` **= Control rate** (`control_freq`) — always match them first.
- `min_time_to_move_multiplier ≈ 3.0` — raise for smoother/laggier, lower for
  snappier/jerkier.
- `max_joint_speed ≈ 3.0` rad/s — well under the firmware limit, ~10× normal
  motion, so it only bites on spikes. Lower only if the arm moves too aggressively.
- `smooth_streaming` OFF unless you specifically want feed-forward-velocity motion;
  when ON, remember the ~0.2 s quintic floor on `goal_time`.
