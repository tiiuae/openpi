# End-Effector (EE) Support for the Trossen OpenPI Eval Bridge

Status: **design / explainer** (pre-implementation). This document explains the
coordinate-frame handling and details the two candidate approaches for executing
an EE-space policy on the bimanual WidowX-AI arms. No code has been written yet.

---

## 1. Context

Today [`main.py`](main.py) drives the arms in **joint space**: the policy server
returns a 14-D joint action (7 per arm = 6 arm joints + 1 gripper), and the
bridge sends it straight to the robot via lerobot's `send_action`.

We now have a policy trained in **end-effector (EE) space**. Per arm it outputs
an **absolute 8-D pose**:

```
[ x, y, z, qw, qx, qy, qz, gripper_norm ]
   └──pos──┘ └────quat────┘ └─0..1 grip─┘
```

So the full bimanual model output is **16-D** (8 left + 8 right). The arms only
accept joint commands (or arm-frame cartesian commands). Something must convert
the model's EE output into a command the hardware understands. This document is
about *what that conversion is* and *where it should happen*.

The EE labels were produced by the dataset-enrichment pipeline in
[`external/joint_to_ee/`](external/joint_to_ee/). Understanding how those labels
were made is the key to inverting them correctly.

---

## 2. Coordinate frames (the important part)

There are **two** frames in play, and the difference between them is the whole
reason the "undo mount" step exists.

```
        robot_base_link  (shared frame — torso / mobile base origin)
              │
              ├──  T_mount_left  = translate( +0.331, +0.300, +0.831 )
              │         │
              │     left_arm base_link  ──FK(joint_0..5)──►  left EE (ee_gripper_link)
              │
              └──  T_mount_right = translate( +0.331, -0.300, +0.831 )
                        │
                    right_arm base_link ──FK(joint_0..5)──►  right EE (ee_gripper_link)
```

- **Arm-base frame** — each arm's own `base_link`. This is what the arm's
  kinematics (and firmware) natively use.
- **Robot-base frame** — the single shared frame for the whole robot. Each arm
  is mounted at a fixed offset from it.

### How the dataset labels were built (FK forward path)

In [`kinematics.py`](external/joint_to_ee/kinematics.py) and
[`enrich.py`](external/joint_to_ee/enrich.py):

```python
# 1. FK in the ARM-BASE frame (placo, takes degrees)
tf_arm = arm_fk(kin, q_joints)            # 4x4 SE3, EE in arm base_link

# 2. Lift into the ROBOT-BASE frame
tf_robot = T_mount @ tf_arm               # apply_mount(): pre-multiply

# 3. Serialize to 8-D pose
pose8 = [x, y, z, qw, qx, qy, qz, gripper_norm]
```

The mount is built from **translation only** (`constants.py` stores
`LEFT_MOUNT_XYZ` / `RIGHT_MOUNT_XYZ` as xyz; `apply_mount` puts them in
`T[:3, 3]` with identity rotation). The enrichment for our model was run with
`ref_frame == "robot_base"`, which is how the model was trained.

> **Consequence:** because `T_mount` is a pure translation, lifting arm-base →
> robot-base **only shifts position**. The orientation quaternion is *identical*
> in both frames. So the inverse correction touches `xyz` only.

### Answering the two questions

**Q1 — Why undo the mount (robot-base → arm-base) before control?**

Both control APIs we could use operate in the **arm-base frame**:

- placo `RobotKinematics.inverse_kinematics(...)` is built from the arm URDF
  (`joint_0..5`, target `ee_gripper_link`); its FK returns arm-base poses, so
  its IK expects an arm-base target.
- trossen `driver.set_cartesian_positions(...)` commands poses "measured in the
  base frame" — that is each arm's own base.

The model emits poses in **robot-base**. Feeding a robot-base pose to an
arm-base controller would ask the arm to reach a point offset by the mount
(~0.33 m forward, ±0.30 m sideways, ~0.83 m up) from the intended target. So we
must reverse step 2 of the forward path:

```python
tf_arm = inv(T_mount) @ tf_robot          # general form
# mount is pure translation, so equivalently:
xyz_arm  = xyz_robot - MOUNT_XYZ          # subtract offset
quat_arm = quat_robot                     # unchanged
```

**Q2 — The model was trained in robot-base frame. Does that hurt?**

No — not for policy quality. The frame choice is just a convention the model
learned consistently. It only adds **one deterministic, lossless correction** at
eval time: subtract the *same* mount constant the enrichment used. Properties:

- **Lossless / exact** round-trip *if and only if* we reuse the identical
  `LEFT/RIGHT_MOUNT_XYZ` constants (and the same URDF) from enrichment.
- **Position-only** correction (pure translation), so orientation is untouched —
  fewer things to get wrong.
- **High blast radius if skipped:** omitting it puts targets ~0.3–0.8 m off →
  the arm slams toward an unreachable/garbage pose. This is the single most
  important line in the conversion.

> Caveat worth recording: the enrichment treated the mount as pure translation
> ("verified against mobile_ai.urdf"). If the real mount had any rotation, the
> pipeline ignored it — and so must we, because round-trip consistency requires
> matching the *pipeline*, not physical reality. Use the pipeline's constants,
> not freshly measured ones.

---

## 3. The shared front-end (identical for both options)

Whatever we do downstream, every approach starts the same way, per arm, per
predicted EE pose:

```python
def ee_pose8_to_arm_se3(pose8, mount_xyz):
    x, y, z, qw, qx, qy, qz, grip_norm = pose8
    # 1. undo mount: robot-base -> arm-base (position only)
    pos_arm = np.array([x, y, z]) - mount_xyz
    # 2. quat (w,x,y,z) -> rotation matrix  (scipy wants x,y,z,w order)
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    # 3. assemble SE3
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos_arm
    # 4. denormalize gripper: 0..1 -> carriage meters (0 .. 0.044)
    grip_m = grip_norm * C.GRIPPER_OPEN          # 0.044
    return T, grip_m
```

After this point the two options diverge: **what do we do with `(T, grip_m)`?**

---

## 4. Option 1 — Software IK (EE → joints), reuse the joint pipeline

Convert each EE pose to joint angles in Python using the **same** placo/URDF
kinematics that generated the labels, then feed the result through the existing
joint-space machinery unchanged.

### Pipeline

```
model EE chunk (N x 16)
   └─ per arm, per step:
        ee_pose8_to_arm_se3()                 # shared front-end
        joints_deg = kin.inverse_kinematics(  # placo, arm-base frame
                         current_joints_deg,   #   seed = current pose
                         T_arm)                #   6 joints out (deg)
        joints_rad = deg2rad(joints_deg)       # send path uses radians
        arm_action = [*joints_rad, grip_m]     # 7-D per arm
   └─ concat L+R -> joint chunk (N x 14)
        │
        ▼
   EXISTING pipeline: ensemble -> limit check -> send_action / sleep-safety
```

Key facts that make this clean:

- `inverse_kinematics(current_joint_pos_deg, desired_ee_4x4)` is already in
  lerobot. It **takes degrees**, is **seeded with the current joints** (good for
  continuity / picking the right IK branch), and **preserves the gripper slot**.
- The arm's joint state (`*.pos`) and `send_action` go straight through the raw
  driver in **radians** for the arm joints and **meters** for the gripper
  carriage (confirmed in `widowxai_follower.py`). FK does `rad2deg(q)`, so state
  is radians; IK returns degrees → we convert **deg → rad** before sending.
- Gripper round-trips through `normalize_gripper`'s inverse:
  `grip_m = grip_norm * 0.044`.

### Where the new code lives

- Add an inverse helper next to the FK code, e.g.
  `external/joint_to_ee/ee_to_joints.py`: `ee_pose8_to_arm_se3()`, an
  `ik_pose8_to_arm_joints()` wrapper, and a chunk-level
  `ee_chunk_to_joint_chunk()`.
- Convert the **whole chunk to joints once, on receipt**, then hand the `N x 14`
  joint chunk to the existing ensemble + execute path. `action_dim` stays 14
  downstream — nothing else in `main.py` changes.

### Pros

- **Reuses the entire existing stack**: joint-limit safety check, sleep
  fallback, `move_to_start` PCHIP ramp, and the **action ensemble in joint
  space** (which is correct — see §6).
- **Round-trip kinematic consistency**: FK produced the labels, IK inverts them
  with the *same* model → minimal systematic error between training and eval.
- **Canonical lerobot path**: their inference postprocessor is literally
  "unnormalize → absolute → (optionally IK to joints)" via
  `InverseKinematicsEEToJoints`. lerobot does software IK, *not* firmware
  cartesian.
- New code is isolated to one conversion module; `main.py` logic is untouched.

### Cons / risks

- **Latency**: 2 placo IK solves per step (~1–5 ms each). Must confirm it fits
  the 25 Hz (40 ms) budget. Mitigation: solve the full chunk once on receipt
  (amortized over `rate_of_inference` steps), and/or reuse the existing
  `--async_inference` worker.
- **IK failure / jumps**: near singularities or for unreachable targets, IK can
  fail or snap to a far branch. Mitigations: seed with current joints (already
  done), add a fallback that holds the previous valid joints, and keep the
  existing velocity-limit check as a backstop.
- **Orientation tracking**: `inverse_kinematics` defaults
  `orientation_weight=0.01` (loose). May under-track rotation; tunable per task.

---

## 5. Option 2 — Firmware cartesian (`set_cartesian_positions`)

Send the arm-base EE pose directly to each arm's cartesian controller and let
the **firmware** do the IK. Gripper is commanded separately.

### Pipeline

```
model EE chunk (N x 16)
   └─ per arm, per step:
        ee_pose8_to_arm_se3()                          # shared front-end
        rotvec = Rotation.from_matrix(R).as_rotvec()   # quat -> angle-axis (3)
        pose6  = [x, y, z, *rotvec]                     # ArrayDouble6, arm-base
        driver.set_cartesian_positions(
            pose6, InterpolationSpace.cartesian_space,
            goal_time=dt, blocking=False)
        # gripper: SEPARATE command (cartesian call has no gripper slot)
        driver.<gripper command>(grip_m)
```

Note the format change: trossen cartesian is **6-D** `[x, y, z, wx, wy, wz]`
where the rotation is an **angle-axis (rotation vector)**, in **meters and
radians**, in the **arm-base frame** — *not* the model's quaternion, and *no*
gripper channel. The driver is reachable at `robot.left_arm.driver` /
`robot.right_arm.driver` (raw `trossen_arm.TrossenArmDriver`).

### Pros

- **No Python IK in the loop** — IK runs on the arm firmware.
- **Native cartesian-space interpolation** between targets.

### Cons / risks

- **Bypasses lerobot entirely.** We lose `send_action`, the joint-limit safety
  check, and the sleep fallback — all of which currently protect the hardware.
- **Breaks the action ensemble.** The existing `action_ensemble` averages
  **joint vectors**. In EE space it would be averaging quaternions/rotvecs,
  which is mathematically wrong without slerp / Lie-group handling. Re-deriving a
  correct EE-space ensemble is real work.
- **Different IK than the training labels.** Firmware's solver ≠ placo. Even with
  perfect frames, the joints the arm actually picks differ from the distribution
  the policy was trained against → systematic mismatch.
- **Gripper out-of-band.** Separate command per arm → 2 calls/arm, plus control
  **mode juggling** (`set_all_modes` cartesian vs position) and sync risk.
- **Startup ramp**: `move_to_start` (joint-space PCHIP) no longer applies; a new
  cartesian startup is needed to avoid a large first-step jump.
- **Streaming semantics unproven.** `set_cartesian_positions` is designed for
  goal moves; pushing it at 25 Hz, with the safety net removed and harder
  debugging, is the riskiest path.

---

## 6. Why the ensemble must stay in joint space

The current `action_ensemble` (exp decay / cogact) blends overlapping action
chunks element-wise. That is valid for joint vectors (a Euclidean space). It is
**not** valid for orientations: linearly averaging quaternions or rotation
vectors does not produce a correct mean rotation. Option 1 keeps the ensemble
operating on joints (after IK), so it stays correct and unchanged. Option 2
would force either a wrong ensemble or a non-trivial rewrite.

This is a major reason the recommendation favors Option 1.

---

## 7. Recommendation

**Option 1 (software IK + existing joint pipeline).** It is the lerobot-canonical
path, preserves round-trip kinematic consistency with the training labels, and
reuses the ensemble and all hardware-safety logic with essentially no change to
`main.py`. The only genuinely new code is one EE→joint conversion module.

Option 2 should be kept documented as the alternative. It only becomes
attractive if Python IK latency proves to be a hard bottleneck *and* we are
willing to rebuild the ensemble + safety layer in EE space.

---

## 8. Open questions (to resolve before/while writing the plan)

1. **Latency budget**: measure placo IK solve time (×2/step) against 40 ms.
   Chunk-level conversion likely makes this a non-issue, but confirm.
2. **Packaging**: new `main_ee.py` (keep `main.py` joint-only) vs an
   `--action_space {joint,ee}` flag on the existing `main.py`.
3. **Gripper units**: confirm the follower's gripper `.pos` is the carriage
   position in meters (0–0.044) end-to-end, matching `normalize_gripper`.
4. **IK weights**: default `orientation_weight=0.01` — is tighter orientation
   tracking needed for the target tasks?
5. **IK-failure policy**: hold-last-joints vs skip vs sleep — pick a default.
```
