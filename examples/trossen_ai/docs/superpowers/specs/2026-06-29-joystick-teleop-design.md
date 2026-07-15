# Joystick EE Teleoperation — Design

**Date:** 2026-06-29
**Status:** Approved (pending implementation plan)
**Branch:** ibrahim/feat/web-app-eval

## Goal

Drive the Trossen bimanual arm by joystick/keyboard through the current
end-effector IK, as a manual test path before running a policy model live. Deliver
it as a **third web page** (`/teleop`) *and* a **CLI command**, sharing one control
core with zero duplicated control logic.

The operator selects which of the two arms to control, switches the controlled arm
with a button, and sees on-screen: the robot in **3D** (URDF) plus the **3 camera
views** in real time, and the full **keybindings** table.

## Decisions (locked)

- **Input:** gamepad **and** keyboard (gamepad primary, keyboard fallback). Both on
  web and CLI.
- **Control DoF:** full 6-DoF end-effector (X/Y/Z translate + roll/pitch/yaw rotate)
  **+ gripper**.
- **Motion model:** velocity — hold to move, release to stop. Integrated per tick.
- **CLI gamepad reader:** `evdev` (Linux `/dev/input`, matches the robot box).
- **Translation frame:** robot-base frame (world X/Y/Z).
- **Three run modes** (see below), default **detached**.

## Run modes

| Mode | Robot connect | Cameras | `execute_action` | Use |
|---|---|---|---|---|
| **detached** | none | none (3D only) | no | Safe IK-flip dry test; runs off-robot (dev env) |
| **test** | yes | live JPEG feeds | no-op | Cameras + 3D, arm does not move |
| **autonomous** | yes | live | real motion | Live drive (confirm-gated) |

Detached mode is the headline safety feature: it runs the *exact* IK + seed-feedback
path as real motion, streaming joints to the 3D model, but never connects or commands
hardware — so an IK branch-flip is visible in 3D and commands nothing. Because it
imports no `lerobot_robot_trossen`, it also runs in the off-robot `lerobot` dev env
and in the pytest suite (same precedent as `/api/episode_trajectory`).

## Architecture

```
            TeleopInputSource (poll -> TeleopCommand)
            |- WebTeleopInput   (WS-fed latch, thread-safe)
            |- EvdevTeleopInput (CLI gamepad + keyboard)
                        |
                        v
   TeleopController.run()  <-- reuses RobotController, EEToJointsConverter,
   per tick:                   limit_joint_velocity, camera-jpeg helper
     cmd = input.poll()
     integrate target pose8[active arm] += vel*dt   (xyz, quat, grip_norm)
     joints14 = converter.decode_chunk(target16, seed14)   (IK, 1 frame)
     joints14 = limit_joint_velocity(last, joints14, dt)
     if controller: controller.execute_action(joints14)    (skipped if detached)
     sink.on_action(step, joints14, ts)
     if controller and cameras: sink.on_images(jpegs, ts)
     seed14 = joints14   (feed-back for IK continuity / flip detection)
                        |
        +---------------+----------------+
   Web: TeleopRunner (Runner)       CLI: cli.py `teleop` (Typer)
   wired into runner_factory        builds robot+controller+evdev input
   + WS start_teleop/teleop_input   console/NullSink
```

### New / changed modules

- **`teleop.py`** (next to `trossen_bridge.py`) — the shared core:
  - `TeleopCommand` dataclass: `lin[3]`, `ang[3]`, `grip` (float), `arm`
    ('left'|'right'), and edge events `switch_arm`, `go_home`, `go_sleep` (bool,
    consumed once per press).
  - `TeleopInputSource` Protocol: `poll() -> TeleopCommand` (non-blocking, returns
    latest latched command).
  - `integrate_pose16(target16, cmd, dt, max_lin, max_ang, grip_rate) -> target16` —
    pure function: xyz += lin*dt on the active arm; quat updated by a small-angle
    increment from ang*dt; grip_norm += grip*dt clamped [0,1]; inactive arm
    unchanged. Unit-tested directly.
  - `axes_to_command(...)` helpers for the velocity scaling/deadzone math (shared by
    both input adapters).
  - `TeleopController`: holds optional controller (`None` => detached), converter,
    `target16`, `active_arm`, `seed14`. `step(cmd, dt)` runs one tick;
    `run(stop_flag)` loops at `control_freq`. Seeds `target16` from
    `converter.joints14_to_ee16(start14)` where `start14` is the live pose (test/
    autonomous) or `HOME_POSITION` (detached).
- **`webapp/teleop_input.py`** — `WebTeleopInput(TeleopInputSource)`: thread-safe
  latch. WS handler writes the latest command + arm-switch edge; `poll()` returns the
  current latched `TeleopCommand` (velocity persists while held).
- **`teleop_evdev.py`** — `EvdevTeleopInput(TeleopInputSource)`: background thread
  reads `/dev/input` gamepad events via `evdev`, plus terminal keyboard fallback;
  maps device codes -> logical inputs -> `TeleopCommand` via the shared
  `axes_to_command` math. The index->logical lookup is the only device-specific part.
- **camera-jpeg helper**: factor the `cv2.imencode` dict-build out of
  `trossen_bridge._build_observation` into a shared function; bridge and teleop both
  call it (removes the only duplication).

### Web wiring

- Route `GET /teleop` -> `static/teleop.html` (mirrors the `/replay` route in
  `webapp/server.py`). Nav link added to all three pages.
- `runner_factory` gains `kind == "teleop"` -> `TeleopRunner` (Runner interface:
  run/stop/estop), wrapping `TeleopController` + `WebTeleopInput`. Detached branch
  imports no hardware module.
- WS gains commands: `start_teleop` (config incl. mode), `teleop_input` (per-tick
  command payload), `switch_arm`. Reuses existing `stop`, `estop`, `go_sleep`,
  `go_home`.
- **`static/teleop.html` + `static/js/teleop.js`**:
  - Reuse `UrdfView` (3D) driven by streamed `on_action` joints.
  - 3 camera `<img>` tiles driven by `on_images` (live.js pattern); detached mode
    shows "detached — no camera" placeholders.
  - Reuse header: mode selector (detached/test/autonomous), Start/Stop/E-STOP/Sleep/
    Home, status badges, log box.
  - New: Gamepad API + keydown/keyup poll loop -> throttled WS `teleop_input` send
    (~30-50 Hz); arm selector + switch button (pad X / Tab / on-screen) with
    active-arm highlight; static **keybindings panel** rendering the tables below.

### CLI wiring

- `cli.py` Typer `teleop` command: builds robot via `build_stationary_robot`
  (test/autonomous) or none (detached), constructs `TeleopController` +
  `EvdevTeleopInput`, console/NullSink. Same knobs as web: `--mode`, `--control-freq`,
  IK tol (`--ik-orientation-weight`, `--ik-pos-tol-m`), velocity scales, deadzone.

## Axis & key mapping

One **logical mapping**; each backend resolves its own index->logical. Base-frame,
velocity (hold-to-move). Sign = positive direction.

### Gamepad (Xbox-style)

| Control | Logical | Action |
|---|---|---|
| Left stick up/down | `LS_Y` | translate +X / -X (fwd/back) |
| Left stick left/right | `LS_X` | translate +Y / -Y (left/right) |
| RB / LB | bumpers | translate +Z / -Z (up/down) |
| Right stick up/down | `RS_Y` | pitch + / - |
| Right stick left/right | `RS_X` | yaw + / - |
| RT / LT | triggers | roll + / - |
| A / B | buttons | gripper close / open |
| X | button | switch active arm (L<->R), edge |
| D-pad up | button | move to **Home** pose, then resume teleop, edge |
| D-pad down | button | move to **Sleep** pose, then resume teleop, edge |
| Back / Select | button | E-STOP |
| Start | button | Stop session |

Gamepad API indices: `axes[0]=LS_X, axes[1]=LS_Y, axes[2]=RS_X, axes[3]=RS_Y`;
`buttons[4/5]=LB/RB, [6/7]=LT/RT, [0..3]=A/B/X/Y, [8/9]=Back/Start,
[12/13]=D-up/D-down`.
evdev codes: `ABS_X/ABS_Y/ABS_RX/ABS_RY/ABS_Z/ABS_RZ`, `ABS_HAT0Y` (D-pad) + `BTN_*`.

### Keyboard (fallback)

| Keys | Action |
|---|---|
| W / S | translate +X / -X |
| A / D | translate +Y / -Y |
| Q / E | translate +Z / -Z |
| I / K | pitch + / - |
| J / L | yaw + / - |
| U / O | roll + / - |
| Z / C | gripper close / open |
| Tab | switch active arm |
| H | move to Home pose, then resume teleop |
| P | move to Sleep (park) pose, then resume teleop |
| Space | E-STOP |
| Esc | Stop session |

### Scaling / behavior

- Analog axis -> velocity = `axis × max_*` after deadzone; held digital input = full
  rate.
- Defaults (tunable, header + CLI): `max_lin` 0.05 m/s, `max_ang` 0.5 rad/s,
  `grip_rate` 0.5 /s, `deadzone` 0.1.
- Per tick: `target_pose8[arm] += vel × dt`; xyz linear, quat small-angle increment,
  `grip_norm` clamped [0,1]; inactive arm holds.
- **Home / Sleep edges** (in-session, distinct from the header movers that *end* the
  session): on `go_home` / `go_sleep`, `TeleopController` drives **both** arms to the
  goal joints (`HOME_POSITION` / `RobotController.SLEEP_POSITION`) —
  test/autonomous via `controller.move_to_start_position` /
  `move_to_sleep_position` (PCHIP smooth, reusing `RobotController`); detached snaps
  `seed14` to the goal — then re-seeds `target16 = converter.joints14_to_ee16(goal)`
  and **resumes** teleop from there. Velocity input is ignored during the move.

## Data flow

```
browser input (pad/keys) --WS {action:teleop_input, cmd}--> WebTeleopInput latch
   -> TeleopController.step reads -> IK decode -> velocity-limit -> (execute if not detached)
   -> sink.on_action / on_images -> WS /ws/telemetry -> browser
   -> UrdfView.setFrameJoints + camera <img> update
```

CLI: evdev/keyboard thread -> EvdevTeleopInput latch -> same TeleopController ->
console sink (+ real robot in test/autonomous).

## Error handling & safety

- **Default detached**: nothing moves, no hardware needed.
- **test**: connects + shows cameras, `execute_action` is a no-op.
- **autonomous**: real motion behind a browser confirm dialog; E-STOP -> sleep pose;
  firmware-fault guard via `RobotController.execute_action` (moves to sleep, stops on
  fault); single-session guard (`SessionManager`) prevents racing another session.
- IK non-convergence / unreachable: `decode_chunk` holds the last valid joints (its
  existing behavior); detached mode lets the operator see this without risk.
- `limit_joint_velocity` caps per-step joint speed so an IK branch-flip can't command
  an over-limit jump in test/autonomous.

## Testing (off-robot, extends `webapp/tests/`)

- `integrate_pose16` pure-fn: translation/rotation/gripper integration, clamps,
  deadzone, inactive-arm-holds.
- `axes_to_command`: deadzone + scaling, sign conventions.
- `TeleopController.step` detached path with a fake converter: joint stream + IK-seed
  feedback continuity, no hardware imports.
- `TeleopController.step` test/autonomous path with a fake controller: asserts
  `execute_action` called (autonomous) / not called (test), images emitted when
  cameras present.
- `go_home` / `go_sleep` edges re-seed `target16` to `FK(goal)` and resume (detached
  path, no hardware): assert `seed14` and `target16` match the goal pose.
- `WebTeleopInput` latch concurrency (writer thread vs `poll`).
- `EvdevTeleopInput` code->command mapping via a pure mapping fn (no device needed).
- Server: `/teleop` route returns HTML; `start_teleop` + `teleop_input` over WS via a
  fake runner factory (mirrors `test_server.py`).

## Out of scope (YAGNI)

- Recording teleop episodes to a dataset.
- Force feedback / haptics.
- Operator/camera-relative translation frame (base frame only for now).
- Simultaneous dual-arm control (one active arm at a time).
