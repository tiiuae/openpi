# Web-App Live (Joint) vs `trossen-ai` `main.py` — Divergence Audit

**Question:** When I run a **joint** evaluation on the real robot through the
web-app **Live** page (branch `ibrahim/feat/web-app-eval`), is it doing the same
thing as `python main.py` on the **`trossen-ai`** branch?

**Short answer:** The **policy input/output path is numerically equivalent**
(observation build, state vector, image preprocessing, chunk decode, temporal
ensemble math). All real differences are in the **motion/safety execution
layer**, in a few **default values**, and in **which knobs the Joint preset
turns on**. None of them change *what the policy sees or predicts*; several
change *how the predicted joints are commanded to the arm*.

Reference files:
- Old: `trossen-ai:examples/trossen_ai/main.py` (single `TrossenOpenPIBridge`)
- New: `examples/trossen_ai/trossen_bridge.py` + `webapp/runners.py` (`LiveRunner`)
  + `robot_control.py` + `ensemble/__init__.py` + `adapters.py`
- Preset the Live page loads for joint eval:
  `webapp/presets/Resnet-Joint.json`

---

## 1. What is IDENTICAL (no divergence)

For the joint path (`adapter="joint"`, autonomous, sync inference):

| Stage | Old `main.py` | New Live joint | Same? |
|---|---|---|---|
| **State vector** | `np.array([obs[k] for k in *.pos])` | `JointAdapter.build_state` → `extract_joints` (same `*.pos` order) | ✅ identical |
| **Image preprocess** | `cv2.resize(224², LANCZOS4)` → `BGR2RGB` → `transpose(2,0,1)` | same, byte-for-byte | ✅ identical |
| **Chunk decode** | `response["actions"][:, :action_dim]` (action_dim=14) | `JointAdapter.decode_chunk` = `raw[:, :14]` | ✅ identical |
| **Inference cadence (sync)** | new chunk every `rate_of_inference` steps | same loop, same condition | ✅ identical |
| **Ensemble math** | `ExponentialEnsemble`, weights `exp(-decay·k)`, k=0 oldest | `TemporalEnsemble`, same weight formula | ✅ same result* |
| **Start ramp** | PCHIP from measured pose → first action over 5 s | same PCHIP, same 5 s | ✅ identical |
| **Control-loop timing** | `dt = 1/control_freq`, sleep remainder | same | ✅ identical |

\* The ensemble was **rewritten** (`ExponentialEnsemble` → `TemporalEnsemble`).
Same exponential-decay weighting and same overlap selection, so the blended
action is the same. The rewrite only changed the *buffer internals* (keeps whole
chunks + thread-safe lock, fixing a premature-eviction bug in the old per-step
`defaultdict` buffer). In normal runs the numeric output matches; the only place
they could differ is a pathological eviction edge case the old code got wrong.

**The prompt, state, and images the policy server receives are the same. The
policy's raw predicted joints are decoded the same way.** So the *evaluation
signal* is not divergent.

---

## 2. What is DIFFERENT — execution / safety layer (NEW in Live)

These are added downstream of the policy. They can alter the *commanded* joint
values vs the raw prediction that old `main.py` sent straight through.

### 2.1 Per-joint velocity clamp — **ACTIVE in the Joint preset** ⚠️
- New: `execute_action` calls `limit_joint_velocity(prev_cmd, target, dt, max_joint_speed)`
  before every autonomous `send_action` (`trossen_bridge.py:154`).
- Old `main.py`: **no clamp** — raw joints sent as-is.
- Preset `Resnet-Joint.json` sets `max_joint_speed: 5` → clamp is **on**.
- **Effect:** if the policy/ensemble commands a per-joint step larger than
  `5 rad/s · dt` (`= 0.2 rad` @ 25 Hz), the new code caps it and spreads the
  jump over several steps; old `main.py` would send the full jump. Under normal
  smooth policy output the deltas are far below the cap, so **outputs match**;
  it only bites on a discontinuity/spike. Set `max_joint_speed: 0` to fully
  reproduce old behavior.

### 2.2 Smooth streaming / feed-forward velocity — **OFF in the preset** ✅
- New: optional `send_action_smooth` (firmware quintic + goal-feedforward vel).
- Preset sets `smooth_streaming: false` → both old and new call the plain
  `robot.send_action(action_dict)`. **No divergence** as configured.

### 2.3 "Hold last action" fallback vs zeros ⚠️ (rare)
- When the ensemble has **no overlap** for a step:
  - Old `main.py`: `a_t = np.zeros(action_dim)` → commands **all-zero joints**.
  - New: `HoldLastAction` → repeats the **last good action** (or skips the step
    if none yet).
- Only triggers on a missing prediction (mostly step 0 / async warm-up). New
  behavior is safer; old behavior could jerk toward zero. Different but edge-case.

### 2.4 Firmware-error handling ⚠️ (behavioral, not numeric)
- New: `send_action` wrapped in try/except → on fault, logs, emits
  `firmware_error`, ramps to sleep, stops the episode.
- Old `main.py`: unhandled → the process crashes.
- Doesn't change commanded values on a healthy run; changes what happens on a fault.

### 2.5 Stop responsiveness
- New: `move_to_start_position` loop has `and self.is_running`; loop checks
  `is_running` after each `execute_action`. Lets the web **Stop** button abort mid-ramp.
- Old: ramps/loops run to completion. No effect on eval values.

---

## 3. What is DIFFERENT — default / config values

Compare old `main.py` argparse defaults **and** its hard-coded robot config
against the `Resnet-Joint.json` preset the Live page submits.

| Knob | Old `main.py` | Live Joint preset | Divergent? |
|---|---|---|---|
| `control_freq` | 25 | 25 | ✅ same |
| `rate_of_inference` | 20 | 20 | ✅ same |
| `max_steps` | 1000 | 1000 | ✅ same |
| `action_chunk_size` | 25 (unused — chunk len comes from server) | not sent (bridge default 10, also unused) | ➖ irrelevant |
| **ensemble / smoothing** | `--ensemble_type` default **`exp` (ON)** | `smoothing: false` (**OFF**) | ⚠️ **see 3.1** |
| **`min_time_to_move_multiplier`** | **3.0** (robot config) | **10** | ⚠️ different motion |
| **`loop_rate`** (driver) | **30** (robot config) | **25** | ⚠️ different motion |
| `max_joint_speed` | n/a (no clamp) | 5 | ⚠️ see 2.1 |
| arm selection | both arms (unless `--use_*_arm_only`) | `use_right_arm_only: true` | ⚠️ **see 3.2** |

### 3.1 Ensemble ON vs OFF — biggest eval-signal divergence ⚠️⚠️
- Old `main.py` **defaults to `--ensemble_type exp`** (temporal smoothing ON)
  unless you passed `--ensemble_type none`.
- The Joint preset ships `smoothing: false` → **no temporal ensemble**; the loop
  uses the **latest chunk's action directly**.
- **This is a genuine behavioral difference in the commanded trajectory.** If
  your old `main.py` joint eval ran with the default `exp` ensemble, and now you
  run Live with the preset as-is, the arm is *less smoothed* now.
- **To match old `main.py`:** tick **Temporal smoothing ON** with **decay 1.0**
  in the Live form (or edit the preset `"smoothing": true`).

### 3.2 Single-arm vs both arms ⚠️
- Old `main.py` default moves **both arms**.
- Preset sets `use_right_arm_only: true` → left arm frozen at its captured pose,
  only right-arm joints (7:14) are driven.
- If your original eval drove **both** arms, this is a real difference — turn off
  "right arm only" (or set both flags false) to reproduce.

### 3.3 Driver goal-time / loop-rate (motion feel, not target values)
- `min_time_to_move_multiplier` (3.0 → 10) and `loop_rate` (30 → 25) are passed
  into `BiWidowXAIFollowerRobotConfig`. They change the firmware's per-command
  interpolation horizon → the arm tracks the **same target joints** with a
  different smoothness/lag profile. Target values are unchanged; motion feel is not.

---

## 4. Cameras — verify the physical mapping ⚠️
Same 3 cameras, but addressed differently:

| Slot | Old `main.py` | New `build_stationary_robot` |
|---|---|---|
| `cam_high` | `/dev/cam_high` | `/dev/video16` |
| `cam_right_wrist` | `/dev/cam_wrist_right` | `/dev/video10` |
| `cam_left_wrist` | `/dev/cam_wrist_left` | `/dev/video4` |

Old used **udev symlinks**, new uses **raw `/dev/videoN`**. If the raw indices
map to the same physical cameras as the symlinks, images are identical. **If not,
camera-to-slot assignment is swapped → the policy sees different views → divergent
eval.** Worth a one-time check that `/dev/video16/10/4` == `cam_high/right/left`.

---

## 5. Bottom line / checklist to reproduce old `main.py` joint eval

The evaluation is **not divergent in the policy path**. To make the Live joint
run command-for-command equivalent to a default `trossen-ai main.py` run:

- [ ] **Temporal smoothing: ON**, decay **1.0** (preset ships it OFF — biggest diff)
- [ ] **Right-arm-only: OFF** (preset ships it ON) — unless you truly want one arm
- [ ] `max_joint_speed: 0` to disable the new velocity clamp (or leave ≥5; only
      matters on spikes)
- [ ] `smooth_streaming: OFF` (already off in preset ✅)
- [ ] Accept minor motion-feel differences from `min_time_to_move_multiplier`
      (10 vs 3.0) and `loop_rate` (25 vs 30), or set them to 3.0 / 30 to match
- [ ] Confirm `/dev/video16,10,4` map to high / right-wrist / left-wrist
- [ ] `control_freq 25`, `rate_of_inference 20`, `max_steps 1000` (already match ✅)

Everything else — state, images, prompt, chunk decode, ensemble weighting — is
the same, so the numbers coming out of the policy server are the same.
