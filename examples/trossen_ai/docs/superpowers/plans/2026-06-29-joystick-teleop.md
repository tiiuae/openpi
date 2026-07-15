# Joystick EE Teleoperation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third web page (`/teleop`) and a CLI command that drive the Trossen bimanual arm in 6-DoF via the current EE IK using a gamepad/keyboard, sharing one control core, with a detached (3D-only, no-hardware) safety mode.

**Architecture:** One shared `TeleopController` core (FK current pose → per-tick: read input velocity → integrate target EE pose → IK 1 frame → velocity-limit → optionally execute → stream joints/images). Two thin input sources (web WS latch, CLI evdev/keyboard) and two entry points (web `TeleopRunner` + WS, CLI Typer command) wrap the same core. Detached mode imports no hardware and runs in the off-robot pytest env.

**Tech Stack:** Python (numpy, scipy, placo IK via existing `EEToJointsConverter`), FastAPI + WebSocket, Typer CLI, `evdev` (CLI gamepad), vanilla JS + three.js/urdf-loader (existing `UrdfView`).

**Spec:** `docs/superpowers/specs/2026-06-29-joystick-teleop-design.md`

**Run tests with:** `PYBIN=/home/edgeai/miniconda3/envs/lerobot/bin/python; cd examples/trossen_ai; $PYBIN -m pytest webapp/tests/ tests/ -q` (run from `examples/trossen_ai`).

---

## File structure

- Create `examples/trossen_ai/camera_utils.py` — `encode_camera_jpegs(obs, cam_keys)` shared by bridge + teleop.
- Create `examples/trossen_ai/teleop.py` — `TeleopCommand`, `TeleopInputSource`, `scale_axes`, `integrate_pose16`, `TeleopController`.
- Create `examples/trossen_ai/teleop_evdev.py` — `state_to_command` pure mapper + `EvdevTeleopInput`.
- Create `examples/trossen_ai/webapp/teleop_input.py` — `WebTeleopInput` (thread-safe latch).
- Create `examples/trossen_ai/webapp/teleop_runner.py` — `TeleopRunner` (Runner interface).
- Modify `examples/trossen_ai/webapp/server.py` — `/teleop` route, `teleop` runner-factory branch, WS `start_teleop`/`teleop_input`/`switch_arm`.
- Modify `examples/trossen_ai/trossen_bridge.py` — use `encode_camera_jpegs`.
- Modify `examples/trossen_ai/cli.py` — `teleop` command.
- Create `examples/trossen_ai/webapp/static/teleop.html` + `static/js/teleop.js` + `static/js/gamepad.js`.
- Modify `static/index.html`, `static/replay.html` — add `/teleop` nav link.
- Modify `examples/trossen_ai/pyproject.toml` — add `evdev` extra.
- Tests: `webapp/tests/test_camera_utils.py`, `test_teleop_core.py`, `test_teleop_controller.py`, `test_teleop_input.py`, `test_teleop_evdev.py`, `test_cli_teleop.py`, and additions to `test_server.py`.

---

## Task 1: Shared camera-JPEG helper

**Files:**
- Create: `examples/trossen_ai/camera_utils.py`
- Modify: `examples/trossen_ai/trossen_bridge.py:191-193`
- Test: `examples/trossen_ai/webapp/tests/test_camera_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# webapp/tests/test_camera_utils.py
import numpy as np
from camera_utils import encode_camera_jpegs


def test_encodes_only_named_cameras_as_jpeg_bytes():
    obs = {
        "cam_high": np.zeros((4, 4, 3), dtype=np.uint8),
        "cam_low": np.full((4, 4, 3), 255, dtype=np.uint8),
        "left.pos": 0.1,  # non-camera key must be ignored
    }
    out = encode_camera_jpegs(obs, ["cam_high", "cam_low"])
    assert set(out) == {"cam_high", "cam_low"}
    # JPEG SOI marker
    assert out["cam_high"][:2] == b"\xff\xd8"
    assert isinstance(out["cam_low"], bytes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PYBIN -m pytest webapp/tests/test_camera_utils.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'camera_utils'`.

- [ ] **Step 3: Write minimal implementation**

```python
# camera_utils.py
"""Encode robot-observation camera frames to JPEG bytes.

Shared by the policy bridge and the teleop controller so the cv2.imencode
logic lives in exactly one place.
"""
from __future__ import annotations

import cv2


def encode_camera_jpegs(obs: dict, cam_keys) -> dict[str, bytes]:
    """Return {camera_name: jpeg_bytes} for each key in cam_keys present in obs."""
    out: dict[str, bytes] = {}
    for cam in cam_keys:
        if cam in obs:
            out[cam] = cv2.imencode(".jpg", obs[cam])[1].tobytes()
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PYBIN -m pytest webapp/tests/test_camera_utils.py -q`
Expected: PASS.

- [ ] **Step 5: Use the helper in the bridge**

In `trossen_bridge.py`, replace the inline encode at lines 191-193:

```python
        self.sink.on_images(
            {cam: cv2.imencode(".jpg", observation_dict[cam])[1].tobytes() for cam in cameras}, time.time()
        )
```

with:

```python
        from camera_utils import encode_camera_jpegs
        self.sink.on_images(encode_camera_jpegs(observation_dict, cameras), time.time())
```

- [ ] **Step 6: Run the full suite (no regression)**

Run: `$PYBIN -m pytest webapp/tests/ tests/ -q`
Expected: PASS (same count as before + 1).

- [ ] **Step 7: Commit**

```bash
git add examples/trossen_ai/camera_utils.py examples/trossen_ai/trossen_bridge.py examples/trossen_ai/webapp/tests/test_camera_utils.py
git commit -m "feat(trossen_ai): shared encode_camera_jpegs helper"
```

---

## Task 2: Teleop core data types + pure math

**Files:**
- Create: `examples/trossen_ai/teleop.py`
- Test: `examples/trossen_ai/webapp/tests/test_teleop_core.py`

EE pose layout (from `external/joint_to_ee/ee_frames.py`): `pose8 = [x, y, z, qw, qx, qy, qz, grip_norm]` in robot-base frame. `pose16 = [left pose8, right pose8]`. Left arm offset 0, right arm offset 8.

- [ ] **Step 1: Write the failing test**

```python
# webapp/tests/test_teleop_core.py
import numpy as np
from teleop import TeleopCommand, scale_axes, integrate_pose16


def _identity_pose16():
    p = np.zeros(16, dtype=float)
    p[3] = 1.0          # left qw
    p[8 + 3] = 1.0      # right qw
    return p


def test_scale_axes_applies_deadzone_and_scaling():
    # raw6 = [tx, ty, tz, roll, pitch, yaw] in [-1, 1]
    lin, ang = scale_axes([1.0, 0.05, 0.0, 0.0, 0.0, -1.0],
                          max_lin=0.05, max_ang=0.5, deadzone=0.1)
    assert lin[0] == 0.05          # full +X
    assert lin[1] == 0.0           # 0.05 < deadzone -> 0
    assert ang[2] == -0.5          # full -yaw
    np.testing.assert_allclose(ang[:2], [0.0, 0.0])


def test_integrate_translates_active_arm_only():
    p = _identity_pose16()
    cmd = TeleopCommand(lin=np.array([0.1, 0.0, 0.0]), ang=np.zeros(3),
                        grip=0.0, arm="right")
    out = integrate_pose16(p, cmd, dt=0.5)
    # right arm x advanced by 0.1 * 0.5, left arm untouched
    assert out[8 + 0] == 0.05
    assert out[0] == 0.0


def test_integrate_clamps_gripper_0_1():
    p = _identity_pose16()
    p[7] = 0.9  # left grip near open
    cmd = TeleopCommand(lin=np.zeros(3), ang=np.zeros(3), grip=1.0, arm="left")
    out = integrate_pose16(p, cmd, dt=1.0)  # 0.9 + 1.0 -> clamp 1.0
    assert out[7] == 1.0


def test_integrate_rotation_keeps_unit_quaternion():
    p = _identity_pose16()
    cmd = TeleopCommand(lin=np.zeros(3), ang=np.array([0.0, 0.0, 1.0]),
                        grip=0.0, arm="left")
    out = integrate_pose16(p, cmd, dt=0.1)
    q = out[3:7]
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    assert not np.allclose(q, [1.0, 0.0, 0.0, 0.0])  # actually rotated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PYBIN -m pytest webapp/tests/test_teleop_core.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'teleop'`.

- [ ] **Step 3: Write minimal implementation**

```python
# teleop.py
"""Shared joystick/keyboard end-effector teleoperation core.

The control loop is identical for the web and CLI front-ends: only the input
source (WebTeleopInput / EvdevTeleopInput) and the telemetry sink differ. The
loop reuses the existing EE IK (EEToJointsConverter), velocity safety
(limit_joint_velocity) and motion/safety (RobotController).

EE pose16 = [left pose8, right pose8], pose8 = [x, y, z, qw, qx, qy, qz, grip_norm]
in robot-base frame (see external/joint_to_ee/ee_frames.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

ARM_OFFSET = {"left": 0, "right": 8}


@dataclass
class TeleopCommand:
    """One control tick of operator intent (physical velocities, base frame)."""
    lin: np.ndarray                 # (3,) m/s
    ang: np.ndarray                 # (3,) rad/s (roll, pitch, yaw)
    grip: float = 0.0               # normalized gripper rate (1/s), + = open
    arm: str = "left"               # active arm: 'left' | 'right'
    switch_arm: bool = False        # edge: toggle active arm
    go_home: bool = False           # edge: move to HOME pose, then resume
    go_sleep: bool = False          # edge: move to SLEEP pose, then resume


class TeleopInputSource(Protocol):
    def poll(self) -> TeleopCommand: ...


def _deadzone(v: float, dz: float) -> float:
    return 0.0 if abs(v) < dz else float(v)


def scale_axes(raw6, max_lin: float, max_ang: float, deadzone: float):
    """Map 6 normalized axes [-1,1] (tx,ty,tz,roll,pitch,yaw) to (lin[3], ang[3])."""
    a = [_deadzone(float(x), deadzone) for x in raw6]
    lin = np.array(a[:3], dtype=float) * max_lin
    ang = np.array(a[3:6], dtype=float) * max_ang
    return lin, ang


def integrate_pose16(pose16: np.ndarray, cmd: TeleopCommand, dt: float) -> np.ndarray:
    """Advance the active arm's pose8 by the commanded velocity over dt.

    Translation: xyz += lin*dt. Rotation: base-frame small-angle increment
    pre-multiplied onto the current orientation. Gripper: grip_norm += grip*dt,
    clamped [0,1]. The inactive arm is returned unchanged.
    """
    out = np.asarray(pose16, dtype=float).copy()
    o = ARM_OFFSET[cmd.arm]
    out[o:o + 3] += np.asarray(cmd.lin, dtype=float) * dt
    rotvec = np.asarray(cmd.ang, dtype=float) * dt
    if np.any(rotvec):
        q_wxyz = out[o + 3:o + 7]
        cur = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        new = Rotation.from_rotvec(rotvec) * cur
        x, y, z, w = new.as_quat()  # scipy returns x,y,z,w
        out[o + 3:o + 7] = [w, x, y, z]
    out[o + 7] = float(np.clip(out[o + 7] + cmd.grip * dt, 0.0, 1.0))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PYBIN -m pytest webapp/tests/test_teleop_core.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/trossen_ai/teleop.py examples/trossen_ai/webapp/tests/test_teleop_core.py
git commit -m "feat(trossen_ai): teleop command types + pose integration math"
```

---

## Task 3: TeleopController — detached step + run loop

**Files:**
- Modify: `examples/trossen_ai/teleop.py`
- Test: `examples/trossen_ai/webapp/tests/test_teleop_controller.py`

The controller is constructed with an `EEToJointsConverter`, a telemetry sink, an input source, and an optional `controller` (a `RobotController`, or `None` for detached). It seeds `target16` from a starting 14-D joint vector.

- [ ] **Step 1: Write the failing test (detached path)**

```python
# webapp/tests/test_teleop_controller.py
import numpy as np
from teleop import TeleopController, TeleopCommand


class FakeConverter:
    """Identity-ish stand-in: avoids placo. Records IK seeds it was given."""
    def __init__(self):
        self.seeds = []

    def joints14_to_ee16(self, joints14):
        p = np.zeros(16, dtype=float)
        p[3] = 1.0
        p[8 + 3] = 1.0
        # encode the first joint into x so we can detect re-seeding
        p[0] = float(joints14[0])
        return p

    def decode_chunk(self, chunk16, seed14):
        self.seeds.append(np.asarray(seed14, dtype=float).copy())
        # map pose16[0] (left x) back into joint 0 so motion is observable
        out = np.zeros((len(chunk16), 14), dtype=float)
        out[:, 0] = chunk16[:, 0]
        return out


class RecordingSink:
    def __init__(self):
        self.actions = []
        self.images = []
        self.status = []

    def on_action(self, step, action, ts): self.actions.append((step, np.asarray(action)))
    def on_images(self, images, ts): self.images.append(images)
    def on_status(self, kind, payload): self.status.append((kind, payload))
    def on_log(self, *a): pass


class OneShotInput:
    def __init__(self, cmd): self._cmd = cmd
    def poll(self): return self._cmd


def test_detached_step_streams_joints_and_no_images():
    conv, sink = FakeConverter(), RecordingSink()
    cmd = TeleopCommand(lin=np.array([1.0, 0, 0]), ang=np.zeros(3), grip=0.0, arm="left")
    ctrl = TeleopController(conv, sink, OneShotInput(cmd), controller=None,
                            start14=np.zeros(14), control_freq=10)
    ctrl.step(cmd, dt=1.0)
    assert len(sink.actions) == 1
    # left x advanced by lin*dt = 1.0 -> joint 0 ~ 1.0
    assert abs(sink.actions[0][1][0] - 1.0) < 1e-6
    assert sink.images == []          # detached: never streams camera frames


def test_detached_step_feeds_decoded_joints_back_as_next_seed():
    conv, sink = FakeConverter(), RecordingSink()
    cmd = TeleopCommand(lin=np.array([0.5, 0, 0]), ang=np.zeros(3), grip=0.0, arm="left")
    ctrl = TeleopController(conv, sink, OneShotInput(cmd), controller=None,
                            start14=np.zeros(14), control_freq=10)
    ctrl.step(cmd, dt=1.0)
    ctrl.step(cmd, dt=1.0)
    # second decode_chunk seed == first decoded joints (continuity)
    assert conv.seeds[1][0] == sink.actions[0][1][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PYBIN -m pytest webapp/tests/test_teleop_controller.py -q`
Expected: FAIL — `TypeError`/`AttributeError` (`TeleopController` has no such constructor yet).

- [ ] **Step 3: Add `TeleopController` to `teleop.py`**

Append to `teleop.py`:

```python
import logging
import time

from robot_control import HOME_POSITION, limit_joint_velocity

logger = logging.getLogger(__name__)

SLEEP_POSITION = np.zeros(14)  # mirrors RobotController.SLEEP_POSITION


class TeleopController:
    """Per-tick EE teleop loop, hardware-optional.

    Args:
        converter:  EEToJointsConverter (FK seed + per-frame IK decode).
        sink:       TelemetrySink (on_action / on_images / on_status).
        input_src:  TeleopInputSource (poll() -> TeleopCommand).
        controller: RobotController, or None for detached (3D-only) mode.
        start14:    Seed joints for the initial EE target (live pose or HOME).
        cam_keys:   Camera observation keys to stream (test/autonomous only).
    """

    def __init__(self, converter, sink, input_src, controller=None, *,
                 start14=None, control_freq: int = 25, max_lin: float = 0.05,
                 max_ang: float = 0.5, grip_rate: float = 0.5,
                 max_joint_speed: float = 3.0, cam_keys=None):
        self.converter = converter
        self.sink = sink
        self.input = input_src
        self.controller = controller
        self.control_freq = int(control_freq)
        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)
        self.grip_rate = float(grip_rate)
        self.max_joint_speed = float(max_joint_speed)
        self.cam_keys = list(cam_keys or [])
        self.active_arm = "left"
        self.step_i = 0
        start = np.zeros(14) if start14 is None else np.asarray(start14, float).flatten()
        self.seed14 = start.copy()
        self.last_joints = start.copy()
        self.target16 = np.asarray(converter.joints14_to_ee16(start), float).flatten()

    @property
    def detached(self) -> bool:
        return self.controller is None

    def _reseed_to(self, goal14: np.ndarray) -> None:
        goal14 = np.asarray(goal14, float).flatten()
        self.seed14 = goal14.copy()
        self.last_joints = goal14.copy()
        self.target16 = np.asarray(self.converter.joints14_to_ee16(goal14), float).flatten()

    def _goto(self, goal14: np.ndarray, label: str) -> None:
        self.sink.on_status("teleop_move", {"target": label})
        if not self.detached:
            self.controller.move_to_start_position(np.asarray(goal14, float).flatten(),
                                                   duration=5.0)
        self._reseed_to(goal14)

    def step(self, cmd: TeleopCommand, dt: float) -> None:
        if cmd.go_home:
            self._goto(HOME_POSITION, "home")
            return
        if cmd.go_sleep:
            self._goto(SLEEP_POSITION, "sleep")
            return
        if cmd.switch_arm:
            self.active_arm = "right" if self.active_arm == "left" else "left"
        cmd.arm = self.active_arm

        self.target16 = integrate_pose16(self.target16, cmd, dt)
        joints = self.converter.decode_chunk(self.target16[None, :], self.seed14)[0]
        joints = limit_joint_velocity(self.last_joints, joints, dt, self.max_joint_speed)

        if not self.detached:
            ok = self.controller.execute_action(joints)
            if not ok:
                self.sink.on_status("firmware_error", {"step": self.step_i})
            if self.cam_keys:
                from camera_utils import encode_camera_jpegs
                obs = self.controller.robot.get_observation()
                self.sink.on_images(encode_camera_jpegs(obs, self.cam_keys), time.time())

        self.sink.on_action(self.step_i, np.asarray(joints), time.time())
        self.last_joints = np.asarray(joints, float).flatten()
        self.seed14 = self.last_joints.copy()
        self.step_i += 1

    def run(self, should_stop) -> None:
        dt = 1.0 / self.control_freq
        self.sink.on_status("teleop_started", {"detached": self.detached,
                                               "arm": self.active_arm})
        try:
            while not should_stop():
                t0 = time.perf_counter()
                self.step(self.input.poll(), dt)
                rest = dt - (time.perf_counter() - t0)
                if rest > 0:
                    time.sleep(rest)
        finally:
            self.sink.on_status("teleop_stopped", {})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PYBIN -m pytest webapp/tests/test_teleop_controller.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/trossen_ai/teleop.py examples/trossen_ai/webapp/tests/test_teleop_controller.py
git commit -m "feat(trossen_ai): TeleopController detached step + run loop"
```

---

## Task 4: Home/Sleep edges + hardware-path assertions

**Files:**
- Modify: `examples/trossen_ai/webapp/tests/test_teleop_controller.py`

The `_goto` behavior is already implemented in Task 3; this task locks it with tests, including the test/autonomous path via a fake controller.

- [ ] **Step 1: Write the failing tests**

Append to `test_teleop_controller.py`:

```python
class FakeRobot:
    def __init__(self):
        self._obs = {"cam_high": np.zeros((2, 2, 3), dtype=np.uint8), "j.pos": 0.0}
    def get_observation(self):
        return self._obs


class FakeRobotController:
    """Stands in for RobotController in test/autonomous paths."""
    def __init__(self, execute_ok=True):
        self.robot = FakeRobot()
        self.executed = []
        self.moves = []
        self._ok = execute_ok
    def execute_action(self, joints, feedforward_velocity=None):
        self.executed.append(np.asarray(joints)); return self._ok
    def move_to_start_position(self, goal, duration=5.0):
        self.moves.append(np.asarray(goal))


def test_go_home_reseeds_target_to_fk_of_home():
    import teleop
    conv, sink = FakeConverter(), RecordingSink()
    ctrl = TeleopController(conv, sink, OneShotInput(None), controller=None,
                            start14=np.zeros(14), control_freq=10)
    home_cmd = TeleopCommand(lin=np.zeros(3), ang=np.zeros(3), go_home=True)
    ctrl.step(home_cmd, dt=0.1)
    # seed14 now equals HOME_POSITION, target16 == FK(HOME)
    np.testing.assert_allclose(ctrl.seed14, teleop.HOME_POSITION)
    assert ctrl.target16[0] == float(teleop.HOME_POSITION[0])
    assert ("teleop_move", {"target": "home"}) in sink.status


def test_autonomous_step_executes_and_streams_images():
    conv, sink = FakeConverter(), RecordingSink()
    fc = FakeRobotController()
    cmd = TeleopCommand(lin=np.array([0.1, 0, 0]), ang=np.zeros(3), arm="left")
    ctrl = TeleopController(conv, sink, OneShotInput(cmd), controller=fc,
                            start14=np.zeros(14), control_freq=10,
                            cam_keys=["cam_high"])
    ctrl.step(cmd, dt=1.0)
    assert len(fc.executed) == 1            # real execute attempted
    assert len(sink.images) == 1            # camera frame streamed
    assert "cam_high" in sink.images[0]


def test_firmware_fault_emits_status():
    conv, sink = FakeConverter(), RecordingSink()
    fc = FakeRobotController(execute_ok=False)
    cmd = TeleopCommand(lin=np.array([0.1, 0, 0]), ang=np.zeros(3), arm="left")
    ctrl = TeleopController(conv, sink, OneShotInput(cmd), controller=fc,
                            start14=np.zeros(14), control_freq=10)
    ctrl.step(cmd, dt=1.0)
    assert any(k == "firmware_error" for k, _ in sink.status)
```

- [ ] **Step 2: Run tests**

Run: `$PYBIN -m pytest webapp/tests/test_teleop_controller.py -q`
Expected: PASS (5 tests total). `FakeConverter.joints14_to_ee16` encodes joint 0 into pose x, so the `target16[0]` assertion holds with no code change.

- [ ] **Step 3: Commit**

```bash
git add examples/trossen_ai/webapp/tests/test_teleop_controller.py
git commit -m "test(trossen_ai): teleop home/sleep edges + hardware path"
```

---

## Task 5: WebTeleopInput latch

**Files:**
- Create: `examples/trossen_ai/webapp/teleop_input.py`
- Test: `examples/trossen_ai/webapp/tests/test_teleop_input.py`

The WS handler writes the latest command dict; `poll()` returns a `TeleopCommand`. Velocity persists while held; edges (`switch_arm`/`go_home`/`go_sleep`) fire once then auto-clear.

- [ ] **Step 1: Write the failing test**

```python
# webapp/tests/test_teleop_input.py
import numpy as np
from webapp.teleop_input import WebTeleopInput


def test_poll_returns_latest_velocity():
    src = WebTeleopInput(max_lin=0.05, max_ang=0.5, grip_rate=0.5, deadzone=0.1)
    src.update({"axes": [1.0, 0, 0, 0, 0, 0], "grip": 0.0})
    cmd = src.poll()
    assert cmd.lin[0] == 0.05
    # held: a second poll without update keeps the velocity
    assert src.poll().lin[0] == 0.05


def test_edges_fire_once_then_clear():
    src = WebTeleopInput()
    src.update({"axes": [0, 0, 0, 0, 0, 0], "grip": 0.0, "switch_arm": True})
    assert src.poll().switch_arm is True
    assert src.poll().switch_arm is False   # auto-cleared


def test_zero_input_when_never_updated():
    src = WebTeleopInput()
    cmd = src.poll()
    assert np.allclose(cmd.lin, 0) and np.allclose(cmd.ang, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PYBIN -m pytest webapp/tests/test_teleop_input.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp.teleop_input'`.

- [ ] **Step 3: Write minimal implementation**

```python
# webapp/teleop_input.py
"""Thread-safe latch turning browser WS messages into TeleopCommands.

The WebSocket handler calls update() on every 'teleop_input'/'switch_arm'
message; the TeleopController thread calls poll() each control tick. Velocities
persist while held; edge events fire exactly once.
"""
from __future__ import annotations

import threading

from teleop import TeleopCommand, scale_axes


class WebTeleopInput:
    def __init__(self, max_lin: float = 0.05, max_ang: float = 0.5,
                 grip_rate: float = 0.5, deadzone: float = 0.1) -> None:
        self._lock = threading.Lock()
        self._axes = [0.0] * 6
        self._grip = 0.0
        self._switch = False
        self._home = False
        self._sleep = False
        self.max_lin, self.max_ang = max_lin, max_ang
        self.grip_rate, self.deadzone = grip_rate, deadzone

    def update(self, msg: dict) -> None:
        with self._lock:
            if "axes" in msg:
                a = list(msg["axes"])[:6]
                self._axes = (a + [0.0] * 6)[:6]
            if "grip" in msg:
                self._grip = float(msg["grip"])
            self._switch = self._switch or bool(msg.get("switch_arm"))
            self._home = self._home or bool(msg.get("go_home"))
            self._sleep = self._sleep or bool(msg.get("go_sleep"))

    def poll(self) -> TeleopCommand:
        with self._lock:
            lin, ang = scale_axes(self._axes, self.max_lin, self.max_ang, self.deadzone)
            grip = 0.0 if abs(self._grip) < self.deadzone else self._grip * self.grip_rate
            cmd = TeleopCommand(lin=lin, ang=ang, grip=grip,
                                switch_arm=self._switch, go_home=self._home,
                                go_sleep=self._sleep)
            self._switch = self._home = self._sleep = False  # consume edges
            return cmd
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PYBIN -m pytest webapp/tests/test_teleop_input.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/trossen_ai/webapp/teleop_input.py examples/trossen_ai/webapp/tests/test_teleop_input.py
git commit -m "feat(trossen_ai): WebTeleopInput thread-safe latch"
```

---

## Task 6: TeleopRunner + server wiring

**Files:**
- Create: `examples/trossen_ai/webapp/teleop_runner.py`
- Modify: `examples/trossen_ai/webapp/server.py`
- Test: `examples/trossen_ai/webapp/tests/test_server.py`

`TeleopRunner` implements the Runner interface (run/stop/estop). It owns a `WebTeleopInput` so the WS handler can push input into the live session. The server routes `teleop_input` messages to the live runner via `session._runner`.

- [ ] **Step 1: Write the failing tests (route + WS start)**

Append to `webapp/tests/test_server.py`:

```python
def test_teleop_page_served():
    client = TestClient(create_app())
    r = client.get("/teleop")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_ws_start_teleop_uses_factory():
    started = {}

    class FakeRunner:
        def __init__(self, kind, config, sink):
            started["kind"] = kind
            started["config"] = config
            self._sink = sink
        def run(self):
            self._sink.on_status("teleop_started", {"detached": True})
        def stop(self): pass
        def estop(self): pass

    app = create_app(runner_factory=lambda k, c, s: FakeRunner(k, c, s))
    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.send_json({"action": "start_teleop",
                      "config": {"mode": "detached", "control_freq": 10}})
        for _ in range(20):
            evt = ws.receive_json()
            if evt.get("type") == "status" and evt.get("kind") == "teleop_started":
                break
        assert started["kind"] == "teleop"
        assert started["config"]["mode"] == "detached"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PYBIN -m pytest webapp/tests/test_server.py -q -k teleop`
Expected: FAIL — `/teleop` 404 and no `start_teleop` handling.

- [ ] **Step 3: Implement `TeleopRunner`**

```python
# webapp/teleop_runner.py
"""Runner wrapping TeleopController for a web session.

Owns a WebTeleopInput so the WS layer can stream operator input into the live
session. Detached mode imports no hardware module. Cheap to construct; all
blocking work (robot connect, control loop) runs in run() on the session thread.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class TeleopRunner:
    def __init__(self, kind: str, config: dict, sink) -> None:
        from webapp.teleop_input import WebTeleopInput
        self._config = config
        self._sink = sink
        self._stopped = threading.Event()
        self._controller = None          # RobotController (test/autonomous)
        self._teleop = None              # TeleopController
        self.input = WebTeleopInput(
            max_lin=float(config.get("max_lin", 0.05)),
            max_ang=float(config.get("max_ang", 0.5)),
            grip_rate=float(config.get("grip_rate", 0.5)),
            deadzone=float(config.get("deadzone", 0.1)),
        )

    def _build_converter(self):
        from external.joint_to_ee.ee_to_joints import EEToJointsConverter
        from external.joint_to_ee.kinematics import make_kinematics
        return EEToJointsConverter(
            make_kinematics(),
            orientation_weight=float(self._config.get("ik_orientation_weight", 0.01)),
            pos_tol_m=float(self._config.get("ik_pos_tol_m", 1e-3)),
        )

    def run(self) -> None:
        from robot_control import HOME_POSITION
        from teleop import TeleopController

        cfg = self._config
        mode = cfg.get("mode", "detached")
        freq = int(cfg.get("control_freq", 25))
        converter = self._build_converter()

        controller = None
        cam_keys: list[str] = []
        start14 = np.asarray(HOME_POSITION, float)

        if mode != "detached":
            from robot_control import RobotController, build_stationary_robot
            robot = build_stationary_robot(with_cameras=True)
            cam_keys = list(robot._cameras_ft.keys())  # noqa: SLF001
            controller = RobotController(robot, control_frequency=freq, test_mode=mode)
            start14 = controller.current_joints14()
            if self._stopped.is_set():
                controller.disconnect()
                self._sink.on_status("stopped", {"reason": "cancelled"})
                return

        self._controller = controller
        self._teleop = TeleopController(
            converter, self._sink, self.input, controller=controller,
            start14=start14, control_freq=freq,
            max_lin=float(cfg.get("max_lin", 0.05)),
            max_ang=float(cfg.get("max_ang", 0.5)),
            grip_rate=float(cfg.get("grip_rate", 0.5)),
            max_joint_speed=float(cfg.get("max_joint_speed", 3.0)),
            cam_keys=cam_keys,
        )
        try:
            self._teleop.run(should_stop=self._stopped.is_set)
        finally:
            if controller is not None:
                controller.disconnect()

    def stop(self) -> None:
        self._stopped.set()

    def estop(self) -> None:
        self._stopped.set()
        if self._controller is not None:
            self._controller.move_to_sleep_position(duration=10.0)
```

- [ ] **Step 4: Wire the server**

In `webapp/server.py`, add the `teleop` branch to the default `runner_factory` (inside `create_app`, alongside the `sleep`/`home` branches):

```python
            if kind == "teleop":
                from webapp.teleop_runner import TeleopRunner
                return TeleopRunner(kind, config, sink)
```

Add the route next to `/replay`:

```python
    @app.get("/teleop")
    def teleop_page():
        return FileResponse(STATIC_DIR / "teleop.html")
```

In the `telemetry_ws` handler, add the new command branches inside the `while True` dispatch (after the existing `estop` branch):

```python
                elif action == "start_teleop":
                    start("teleop", cmd["config"])
                elif action in ("teleop_input", "switch_arm"):
                    runner = getattr(session, "_runner", None)
                    if runner is not None and hasattr(runner, "input"):
                        payload = {"switch_arm": True} if action == "switch_arm" else cmd.get("payload", {})
                        runner.input.update(payload)
```

> Routing note: `session._runner` is set in `SessionManager._run` before `runner.run()` blocks, so the live `TeleopRunner` is reachable while teleop is active. `WebTeleopInput.update` is a cheap no-op until the loop's next `poll()`, and the browser resends input at 30-50 Hz — so a `None` runner on the very first message is harmless. No `SessionManager` change is required.

- [ ] **Step 5: Run tests to verify they pass**

Run: `$PYBIN -m pytest webapp/tests/test_server.py -q -k teleop`
Expected: PASS (2 tests).

- [ ] **Step 6: Run the full suite**

Run: `$PYBIN -m pytest webapp/tests/ tests/ -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add examples/trossen_ai/webapp/teleop_runner.py examples/trossen_ai/webapp/server.py examples/trossen_ai/webapp/tests/test_server.py
git commit -m "feat(trossen_ai): TeleopRunner + /teleop route + WS teleop wiring"
```

---

## Task 7: evdev input mapper (CLI gamepad)

**Files:**
- Create: `examples/trossen_ai/teleop_evdev.py`
- Test: `examples/trossen_ai/webapp/tests/test_teleop_evdev.py`

The pure mapper turns a snapshot of axis/button state into a `TeleopCommand`; it is unit-tested with no device. The `EvdevTeleopInput` reader thread is thin glue and is not unit-tested.

- [ ] **Step 1: Write the failing test**

```python
# webapp/tests/test_teleop_evdev.py
from teleop_evdev import state_to_command


def test_left_stick_maps_to_translation():
    # axis state normalized [-1,1]; stick up = -1 -> +X
    state = {"LS_Y": -1.0, "LS_X": 0.0, "RS_X": 0.0, "RS_Y": 0.0,
             "LT": 0.0, "RT": 0.0, "Z_UP": 0, "Z_DOWN": 0,
             "GRIP_CLOSE": 0, "GRIP_OPEN": 0}
    cmd = state_to_command(state, max_lin=0.05, max_ang=0.5, grip_rate=0.5,
                           deadzone=0.1, arm="left")
    assert cmd.lin[0] == 0.05      # +X
    assert cmd.lin[1] == 0.0


def test_bumpers_map_to_z_and_grip_buttons_to_rate():
    state = {"LS_Y": 0.0, "LS_X": 0.0, "RS_X": 0.0, "RS_Y": 0.0,
             "LT": 0.0, "RT": 0.0, "Z_UP": 1, "Z_DOWN": 0,
             "GRIP_CLOSE": 1, "GRIP_OPEN": 0}
    cmd = state_to_command(state, max_lin=0.05, max_ang=0.5, grip_rate=0.5,
                           deadzone=0.1, arm="right")
    assert cmd.lin[2] == 0.05      # +Z from Z_UP
    assert cmd.grip == -0.5        # close = negative grip rate
    assert cmd.arm == "right"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PYBIN -m pytest webapp/tests/test_teleop_evdev.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'teleop_evdev'`.

- [ ] **Step 3: Write minimal implementation**

```python
# teleop_evdev.py
"""CLI gamepad/keyboard input via evdev.

state_to_command (pure) maps a normalized axis/button snapshot to a
TeleopCommand and is fully unit-tested. EvdevTeleopInput runs a background
thread that maintains that snapshot from /dev/input events; evdev is imported
lazily inside the reader thread so this module loads without evdev present.
"""
from __future__ import annotations

import threading

from teleop import TeleopCommand, scale_axes

# Logical axis snapshot keys (analog in [-1,1], buttons 0/1):
#   LS_X, LS_Y, RS_X, RS_Y, LT, RT  (analog)
#   Z_UP, Z_DOWN, GRIP_CLOSE, GRIP_OPEN  (buttons)


def state_to_command(state: dict, *, max_lin: float, max_ang: float,
                     grip_rate: float, deadzone: float, arm: str,
                     switch_arm: bool = False, go_home: bool = False,
                     go_sleep: bool = False) -> TeleopCommand:
    # translation: LS up(-Y)=+X, LS right(+X)=+Y, bumpers=+/-Z
    raw6 = [
        -state.get("LS_Y", 0.0),                       # tx
        state.get("LS_X", 0.0),                        # ty
        float(state.get("Z_UP", 0)) - float(state.get("Z_DOWN", 0)),  # tz
        state.get("RT", 0.0) - state.get("LT", 0.0),   # roll
        -state.get("RS_Y", 0.0),                       # pitch
        state.get("RS_X", 0.0),                        # yaw
    ]
    lin, ang = scale_axes(raw6, max_lin, max_ang, deadzone)
    grip = (float(state.get("GRIP_OPEN", 0)) - float(state.get("GRIP_CLOSE", 0))) * grip_rate
    return TeleopCommand(lin=lin, ang=ang, grip=grip, arm=arm,
                         switch_arm=switch_arm, go_home=go_home, go_sleep=go_sleep)


class EvdevTeleopInput:
    """Background-thread gamepad reader. Lazy-imports evdev."""

    def __init__(self, device_path: str | None = None, *, max_lin: float = 0.05,
                 max_ang: float = 0.5, grip_rate: float = 0.5, deadzone: float = 0.1):
        self._lock = threading.Lock()
        self._state = {k: 0.0 for k in
                       ("LS_X", "LS_Y", "RS_X", "RS_Y", "LT", "RT",
                        "Z_UP", "Z_DOWN", "GRIP_CLOSE", "GRIP_OPEN")}
        self._arm = "left"
        self._edges = {"switch_arm": False, "go_home": False, "go_sleep": False}
        self._device_path = device_path
        self._cfg = dict(max_lin=max_lin, max_ang=max_ang,
                         grip_rate=grip_rate, deadzone=deadzone)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def poll(self) -> TeleopCommand:
        with self._lock:
            edges = dict(self._edges)
            self._edges = {k: False for k in self._edges}
            cmd = state_to_command(self._state, arm=self._arm,
                                   switch_arm=edges["switch_arm"],
                                   go_home=edges["go_home"],
                                   go_sleep=edges["go_sleep"], **self._cfg)
        if edges["switch_arm"]:
            self._arm = "right" if self._arm == "left" else "left"
        return cmd

    def _reader(self) -> None:  # pragma: no cover - needs a device
        from evdev import InputDevice, ecodes, list_devices

        path = self._device_path or (list_devices()[0] if list_devices() else None)
        if path is None:
            return
        dev = InputDevice(path)
        ABS = {ecodes.ABS_X: ("LS_X", 32767), ecodes.ABS_Y: ("LS_Y", 32767),
               ecodes.ABS_RX: ("RS_X", 32767), ecodes.ABS_RY: ("RS_Y", 32767),
               ecodes.ABS_Z: ("LT", 255), ecodes.ABS_RZ: ("RT", 255)}
        BTN = {ecodes.BTN_TR: "Z_UP", ecodes.BTN_TL: "Z_DOWN",
               ecodes.BTN_SOUTH: "GRIP_CLOSE", ecodes.BTN_EAST: "GRIP_OPEN"}
        for ev in dev.read_loop():
            if self._stop.is_set():
                break
            with self._lock:
                if ev.type == ecodes.EV_ABS and ev.code in ABS:
                    name, scale = ABS[ev.code]
                    self._state[name] = max(-1.0, min(1.0, ev.value / scale))
                elif ev.type == ecodes.EV_ABS and ev.code == ecodes.ABS_HAT0Y:
                    self._edges["go_home"] |= ev.value < 0
                    self._edges["go_sleep"] |= ev.value > 0
                elif ev.type == ecodes.EV_KEY and ev.code in BTN:
                    self._state[BTN[ev.code]] = float(ev.value)
                elif ev.type == ecodes.EV_KEY and ev.code == ecodes.BTN_WEST and ev.value == 1:
                    self._edges["switch_arm"] = True
```

> The evdev code→logical map is Xbox-style; verify `ABS_*`/`BTN_*` against your actual pad with `evtest` and adjust the `ABS`/`BTN` dicts if needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `$PYBIN -m pytest webapp/tests/test_teleop_evdev.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/trossen_ai/teleop_evdev.py examples/trossen_ai/webapp/tests/test_teleop_evdev.py
git commit -m "feat(trossen_ai): evdev gamepad input mapper for CLI teleop"
```

---

## Task 8: CLI `teleop` command

**Files:**
- Modify: `examples/trossen_ai/cli.py`
- Test: `examples/trossen_ai/webapp/tests/test_cli_teleop.py`

- [ ] **Step 1: Write the failing test (help, no hardware)**

```python
# webapp/tests/test_cli_teleop.py
from typer.testing import CliRunner
from cli import app

runner = CliRunner()


def test_teleop_in_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "teleop" in res.output


def test_teleop_help_lists_mode_flag():
    res = runner.invoke(app, ["teleop", "--help"])
    assert res.exit_code == 0
    assert "--mode" in res.output
    assert "detached" in res.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PYBIN -m pytest webapp/tests/test_cli_teleop.py -q`
Expected: FAIL — no `teleop` command registered.

- [ ] **Step 3: Add the command to `cli.py`**

Append before `if __name__ == "__main__":`:

```python
@app.command("teleop")
def teleop(
    mode: str = typer.Option("detached", help="detached (3D only, no robot) | test (cameras, no motion) | autonomous (real motion)"),
    control_freq: int = typer.Option(25, help="Control loop frequency in Hz"),
    arm: str = typer.Option("left", help="Initial active arm: left | right"),
    device: Optional[str] = typer.Option(None, help="evdev gamepad path (default: first /dev/input device)"),
    max_lin: float = typer.Option(0.05, help="Max EE linear velocity (m/s) at full stick"),
    max_ang: float = typer.Option(0.5, help="Max EE angular velocity (rad/s) at full stick"),
    grip_rate: float = typer.Option(0.5, help="Gripper rate (normalized 1/s) while held"),
    deadzone: float = typer.Option(0.1, help="Analog stick deadzone"),
    max_joint_speed: float = typer.Option(3.0, help="Per-joint velocity cap (rad/s)"),
    ik_orientation_weight: float = typer.Option(0.01, help="placo IK orientation weight"),
    ik_pos_tol_m: float = typer.Option(1e-3, help="IK convergence/failure tolerance (m)"),
) -> None:
    """Joystick/keyboard end-effector teleoperation through the current EE IK.

    Detached mode runs off-robot (3D model only) — useful to dry-test IK before
    touching hardware. test/autonomous build the real robot (cameras on);
    autonomous moves it for real. Same control core as the /teleop web page.
    """
    import logging
    import signal

    import numpy as np

    from external.joint_to_ee.ee_to_joints import EEToJointsConverter
    from external.joint_to_ee.kinematics import make_kinematics
    from robot_control import HOME_POSITION
    from teleop import TeleopController
    from teleop_evdev import EvdevTeleopInput
    from webapp.telemetry import NullSink

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    converter = EEToJointsConverter(
        make_kinematics(), orientation_weight=ik_orientation_weight, pos_tol_m=ik_pos_tol_m)

    controller = None
    cam_keys: list[str] = []
    start14 = np.asarray(HOME_POSITION, float)
    if mode != "detached":
        from robot_control import RobotController, build_stationary_robot
        robot = build_stationary_robot(with_cameras=True)
        cam_keys = list(robot._cameras_ft.keys())  # noqa: SLF001
        controller = RobotController(robot, control_frequency=control_freq, test_mode=mode)
        start14 = controller.current_joints14()

    inp = EvdevTeleopInput(device_path=device, max_lin=max_lin, max_ang=max_ang,
                           grip_rate=grip_rate, deadzone=deadzone)
    inp.start()
    teleop_ctrl = TeleopController(
        converter, NullSink(), inp, controller=controller, start14=start14,
        control_freq=control_freq, max_lin=max_lin, max_ang=max_ang,
        grip_rate=grip_rate, max_joint_speed=max_joint_speed, cam_keys=cam_keys)
    teleop_ctrl.active_arm = arm

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    typer.echo(f"Teleop running (mode={mode}, arm={arm}). Ctrl-C to stop.")
    try:
        teleop_ctrl.run(should_stop=lambda: stop["flag"])
    finally:
        inp.stop()
        if controller is not None:
            controller.disconnect()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PYBIN -m pytest webapp/tests/test_cli_teleop.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/trossen_ai/cli.py examples/trossen_ai/webapp/tests/test_cli_teleop.py
git commit -m "feat(trossen_ai): CLI teleop command (shared TeleopController)"
```

---

## Task 9: Teleop web page (HTML + JS)

**Files:**
- Create: `examples/trossen_ai/webapp/static/teleop.html`
- Create: `examples/trossen_ai/webapp/static/js/gamepad.js`
- Create: `examples/trossen_ai/webapp/static/js/teleop.js`
- Modify: `examples/trossen_ai/webapp/static/index.html:15`, `static/replay.html:23`
- Modify: `examples/trossen_ai/webapp/static/theme.css`

Serving is already covered by `test_teleop_page_served` (Task 6). This task is the browser UI; verify manually in a browser.

- [ ] **Step 1: Create `teleop.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trossen Control — Teleop</title>
  <link rel="stylesheet" href="/static/theme.css">
  <script type="importmap">
  {
    "imports": {
      "three":               "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js",
      "three/addons/":       "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/",
      "three/examples/jsm/": "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/",
      "urdf-loader":         "/static/vendor/URDFLoader.js"
    }
  }
  </script>
</head>
<body>
  <header class="app-header">
    <div class="app-nav"><strong style="color:var(--text);margin-right:16px">Trossen Control</strong>
      <a href="/">Live</a><a href="/replay">Replay</a><a href="/teleop" class="active">Teleop</a></div>
    <div class="header-actions">
      <span class="badge" id="badge-session">idle</span>
      <span class="badge" id="badge-arm">arm: left</span>
      <span class="badge" id="badge-pad">no gamepad</span>
    </div>
  </header>

  <div class="live-grid">
    <aside id="config-col">
      <section class="card"><h2>Teleop</h2>
        <div class="field-inline"><label>Mode</label>
          <select id="cf_mode">
            <option value="detached">Detached — 3D only (no robot)</option>
            <option value="test">Test — cameras, no motion</option>
            <option value="autonomous">Autonomous — real motion</option>
          </select></div>
        <div class="field-inline"><label>Control rate (Hz)</label>
          <input type="number" id="cf_control_freq" value="25" min="1" step="1"></div>
        <div class="field-inline"><label>Max linear (m/s)</label>
          <input type="number" id="cf_max_lin" value="0.05" step="0.01" min="0"></div>
        <div class="field-inline"><label>Max angular (rad/s)</label>
          <input type="number" id="cf_max_ang" value="0.5" step="0.1" min="0"></div>
        <div class="field-inline"><label>Gripper rate (1/s)</label>
          <input type="number" id="cf_grip_rate" value="0.5" step="0.1" min="0"></div>
        <div class="field-inline"><label>Deadzone</label>
          <input type="number" id="cf_deadzone" value="0.1" step="0.05" min="0" max="0.9"></div>
        <div class="field-inline"><label>IK orientation weight</label>
          <input type="number" id="cf_ik_orientation_weight" value="0.01" step="0.01" min="0"></div>
        <div class="field-inline"><label>IK position tol (m)</label>
          <input type="number" id="cf_ik_pos_tol_m" value="0.001" step="0.001" min="0"></div>
        <div class="run-controls">
          <button class="btn-primary" id="btn-start">Start Teleop</button>
          <button class="btn-outline" id="btn-switch">Switch arm</button>
          <button class="btn-outline" id="btn-stop">Stop</button>
          <button class="btn-danger" id="btn-estop">E-STOP</button></div>
      </section>
      <section class="card"><h2>Key bindings</h2>
        <div id="keybindings"></div></section>
    </aside>
    <main class="live-main">
      <section class="card">
        <div class="urdf-wrap"><canvas id="urdf-canvas"></canvas>
          <div class="cam-views" id="cam-views">
            <button class="btn-sm" data-view="iso">Iso</button>
            <button class="btn-sm" data-view="front">Front</button>
            <button class="btn-sm" data-view="top">Top</button>
            <button class="btn-sm" data-view="left">L</button>
            <button class="btn-sm" data-view="right">R</button>
          </div>
        </div>
        <div class="images-row" id="images-box"></div>
      </section>
      <section class="card"><div class="log-header"><h2>Logs</h2><button class="btn-sm" id="btn-clear-log">Clear</button></div>
        <div class="log-panel" id="log-box"></div></section>
    </main>
  </div>
  <script type="module" src="/static/js/teleop.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `gamepad.js` (Gamepad API + keyboard → axes/edges)**

```js
// Reads the Gamepad API and keyboard into a normalized teleop input snapshot.
// axes6 = [tx, ty, tz, roll, pitch, yaw] in [-1,1]; grip in [-1,1] (+ open).
// Edges (switch_arm/go_home/go_sleep) latch until consumed by readEdges().
const keys = new Set();
window.addEventListener("keydown", (e) => { keys.add(e.key.toLowerCase()); if (e.key === "Tab") e.preventDefault(); });
window.addEventListener("keyup",   (e) => keys.delete(e.key.toLowerCase()));

let edges = { switch_arm: false, go_home: false, go_sleep: false };
let prevPadButtons = [];
let prevTabKey = false, prevHomeKey = false, prevSleepKey = false;

const ax = (k) => (keys.has(k) ? 1 : 0);
const clamp = (v) => Math.max(-1, Math.min(1, v));

export function hasGamepad() { return [...navigator.getGamepads()].some((p) => p); }

export function readInput(deadzone) {
  const pad = [...navigator.getGamepads()].find((p) => p) || null;
  const dz = (v) => (Math.abs(v) < deadzone ? 0 : v);
  let lsx = 0, lsy = 0, rsx = 0, rsy = 0, lt = 0, rt = 0;
  let zUp = ax("q"), zDown = ax("e"), gOpen = ax("c"), gClose = ax("z");
  let roll = ax("u") - ax("o");

  if (pad) {
    lsx = dz(pad.axes[0] || 0); lsy = dz(pad.axes[1] || 0);
    rsx = dz(pad.axes[2] || 0); rsy = dz(pad.axes[3] || 0);
    lt = pad.buttons[6]?.value || 0; rt = pad.buttons[7]?.value || 0;
    zUp = zUp || (pad.buttons[5]?.pressed ? 1 : 0);   // RB
    zDown = zDown || (pad.buttons[4]?.pressed ? 1 : 0); // LB
    gClose = gClose || (pad.buttons[0]?.pressed ? 1 : 0); // A
    gOpen = gOpen || (pad.buttons[1]?.pressed ? 1 : 0);   // B
    roll += rt - lt;
    const btn = (i) => pad.buttons[i]?.pressed;
    if (btn(2) && !prevPadButtons[2]) edges.switch_arm = true;   // X
    if (btn(12) && !prevPadButtons[12]) edges.go_home = true;    // D-up
    if (btn(13) && !prevPadButtons[13]) edges.go_sleep = true;   // D-down
    prevPadButtons = pad.buttons.map((b) => b.pressed);
  }
  const tab = keys.has("tab"), h = keys.has("h"), p = keys.has("p");
  if (tab && !prevTabKey) edges.switch_arm = true;
  if (h && !prevHomeKey) edges.go_home = true;
  if (p && !prevSleepKey) edges.go_sleep = true;
  prevTabKey = tab; prevHomeKey = h; prevSleepKey = p;

  const tx = dz(-lsy) + (ax("w") - ax("s"));
  const ty = dz(lsx) + (ax("d") - ax("a"));
  const tz = (zUp ? 1 : 0) - (zDown ? 1 : 0);
  const pitch = dz(-rsy) + (ax("i") - ax("k"));
  const yaw = dz(rsx) + (ax("j") - ax("l"));
  const grip = (gOpen ? 1 : 0) - (gClose ? 1 : 0);

  return {
    axes: [clamp(tx), clamp(ty), clamp(tz), clamp(roll), clamp(pitch), clamp(yaw)],
    grip: clamp(grip),
    hasPad: !!pad,
  };
}

export function readEdges() {
  const e = edges;
  edges = { switch_arm: false, go_home: false, go_sleep: false };
  return e;
}
```

> The Gamepad API button index map (LB=4, RB=5, LT=6, RT=7, A=0, B=1, X=2, D-up=12, D-down=13) is the W3C "standard" mapping; confirm against your actual controller and adjust if your pad reports a non-standard `mapping`.

- [ ] **Step 3: Create `teleop.js`**

```js
import { connect, send, onMessage, onOpen } from "./ws.js";
import { setupLogs } from "./logs.js";
import { UrdfView } from "./urdf_view.js";
import { readInput, readEdges, hasGamepad } from "./gamepad.js";

const $ = (id) => document.getElementById(id);
let view = null, active = false, sendTimer = null;

function readConfig() {
  return {
    mode: $("cf_mode").value,
    control_freq: +$("cf_control_freq").value,
    max_lin: +$("cf_max_lin").value,
    max_ang: +$("cf_max_ang").value,
    grip_rate: +$("cf_grip_rate").value,
    deadzone: +$("cf_deadzone").value,
    ik_orientation_weight: +$("cf_ik_orientation_weight").value,
    ik_pos_tol_m: +$("cf_ik_pos_tol_m").value,
  };
}

const BINDINGS = [
  ["Left stick / WASD", "translate X-Y"],
  ["RB·LB / Q·E", "translate Z up/down"],
  ["Right stick / IJKL", "pitch / yaw"],
  ["RT·LT / U·O", "roll"],
  ["A·B / Z·C", "gripper close / open"],
  ["X / Tab", "switch active arm"],
  ["D-pad ↑ / H", "Home pose"],
  ["D-pad ↓ / P", "Sleep pose"],
  ["Back / Space", "E-STOP"],
  ["Start / Esc", "Stop session"],
];

function renderKeybindings() {
  $("keybindings").innerHTML =
    "<table class='kb'>" +
    BINDINGS.map(([k, a]) => `<tr><td><kbd>${k}</kbd></td><td>${a}</td></tr>`).join("") +
    "</table>";
}

function startSendLoop(cfg) {
  const period = 1000 / Math.max(10, Math.min(50, cfg.control_freq));
  sendTimer = setInterval(() => {
    if (!active) return;
    const inp = readInput(cfg.deadzone);
    const edges = readEdges();
    $("badge-pad").textContent = inp.hasPad ? "gamepad ✓" : "keyboard only";
    send({ action: "teleop_input", payload: { axes: inp.axes, grip: inp.grip, ...edges } });
  }, period);
}

document.addEventListener("DOMContentLoaded", async () => {
  setupLogs();
  renderKeybindings();
  view = new UrdfView($("urdf-canvas"));
  await view.load();
  $("cam-views").addEventListener("click", (e) => {
    const v = e.target?.dataset?.view; if (v) view.snapView(v);
  });

  onMessage("action", (e) => view.setFrameJoints(e.action));
  onMessage("images", (e) => {
    const ib = $("images-box"); ib.innerHTML = "";
    for (const [name, b64] of Object.entries(e.images)) {
      const img = new Image(); img.src = "data:image/jpeg;base64," + b64; img.title = name; ib.appendChild(img);
    }
  });
  onMessage("status", (e) => {
    if (e.kind === "teleop_started") $("badge-session").textContent = "teleop";
    if (e.kind === "teleop_stopped" || e.kind === "stopped") { $("badge-session").textContent = "idle"; active = false; }
    if (e.kind === "teleop_move") $("badge-arm").textContent = "moving → " + e.payload.target;
  });

  $("btn-start").addEventListener("click", () => {
    const cfg = readConfig();
    if (cfg.mode === "autonomous" && !confirm("Autonomous mode moves the REAL robot. Continue?")) return;
    active = true; $("badge-session").textContent = "teleop";
    send({ action: "start_teleop", config: cfg });
    if (!sendTimer) startSendLoop(cfg);
  });
  $("btn-switch").addEventListener("click", () => {
    send({ action: "switch_arm" });
    const b = $("badge-arm"); b.textContent = b.textContent.includes("left") ? "arm: right" : "arm: left";
  });
  $("btn-stop").addEventListener("click", () => { active = false; send({ action: "stop" }); });
  $("btn-estop").addEventListener("click", () => { active = false; send({ action: "estop" }); });
  window.addEventListener("keydown", (e) => {
    if (e.key === " ") { active = false; send({ action: "estop" }); }
    if (e.key === "Escape") { active = false; send({ action: "stop" }); }
  });

  onOpen(() => { $("badge-pad").textContent = hasGamepad() ? "gamepad ✓" : "keyboard only"; });
  connect();
});
```

- [ ] **Step 4: Add nav links on the other two pages**

In `static/index.html` line 15, change:

```html
      <a href="/" class="active">Live</a><a href="/replay">Replay</a>
```
to:
```html
      <a href="/" class="active">Live</a><a href="/replay">Replay</a><a href="/teleop">Teleop</a>
```

In `static/replay.html` line 23, change:

```html
      <a href="/">Live</a><a href="/replay" class="active">Replay</a></div>
```
to:
```html
      <a href="/">Live</a><a href="/replay" class="active">Replay</a><a href="/teleop">Teleop</a></div>
```

- [ ] **Step 5: Add minimal CSS for the keybindings table + image row**

Append to `static/theme.css`:

```css
table.kb { width: 100%; border-collapse: collapse; font-size: 12px; }
table.kb td { padding: 3px 6px; border-bottom: 1px solid var(--border, #21262d); vertical-align: top; }
table.kb kbd { background:#21262d; border-radius:4px; padding:1px 5px; color:var(--text); }
.images-row { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
.images-row img { width: 30%; border-radius:6px; background:#0c0f14; }
```

- [ ] **Step 6: Run the suite (route still served, no regression)**

Run: `$PYBIN -m pytest webapp/tests/ tests/ -q`
Expected: PASS.

- [ ] **Step 7: Manual browser check (detached, no hardware)**

```bash
cd examples/trossen_ai
$PYBIN -m uvicorn webapp.server:app --port 8000   # needs uvicorn/fastapi in this env
```
Open `http://localhost:8000/teleop`. Verify: page loads, 3D robot renders, keybindings table shows, badges visible. Press **Start Teleop** (mode=detached). With a gamepad or WASD/IJKL, the 3D arm moves; Tab switches arm; H/P send it to Home/Sleep; the badge updates. (Detached needs no robot.)

- [ ] **Step 8: Commit**

```bash
git add examples/trossen_ai/webapp/static/teleop.html examples/trossen_ai/webapp/static/js/gamepad.js examples/trossen_ai/webapp/static/js/teleop.js examples/trossen_ai/webapp/static/index.html examples/trossen_ai/webapp/static/replay.html examples/trossen_ai/webapp/static/theme.css
git commit -m "feat(trossen_ai): /teleop page — 3D view, cameras, gamepad/keyboard, keybindings"
```

---

## Task 10: Dependency + docs

**Files:**
- Modify: `examples/trossen_ai/pyproject.toml`
- Modify: `examples/trossen_ai/webapp/HARDWARE.md`

- [ ] **Step 1: Add the evdev dependency**

In `pyproject.toml`, locate the `[project]` `dependencies` array and add (CLI gamepad only; the web path needs no extra deps):

```toml
    "evdev>=1.6; sys_platform == 'linux'",
```

- [ ] **Step 2: Document the page + command**

Append a section to `webapp/HARDWARE.md`:

```markdown
## 8. Joystick teleoperation (`/teleop` and `cli.py teleop`)

Drive the arm in 6-DoF by gamepad/keyboard through the current EE IK — a manual
test path before running a policy live.

- **Web:** open `/teleop`. Pick a **Mode**: `detached` (3D model only, no robot —
  safe IK-flip dry test), `test` (cameras on, no motion), `autonomous` (real
  motion, confirm-gated). Press **Start Teleop**. The 3D view + 3 camera tiles
  update live; the key-bindings panel lists every control.
- **CLI:** `python cli.py teleop --mode detached` (add `--device /dev/input/eventN`
  to pick a gamepad). `--mode autonomous` moves the real arm.
- **Bindings:** left stick / WASD = translate X-Y; RB·LB / Q·E = Z; right stick /
  IJKL = pitch·yaw; RT·LT / U·O = roll; A·B / Z·C = gripper; X / Tab = switch arm;
  D-pad↑·H = Home; D-pad↓·P = Sleep; Back / Space = E-STOP; Start / Esc = Stop.
- **Safety:** detached touches no hardware; autonomous is confirm-gated, honors
  E-STOP (→ sleep) and the firmware-fault guard, and cannot race another session.
```

- [ ] **Step 3: Run the full suite one final time**

Run: `$PYBIN -m pytest webapp/tests/ tests/ -q`
Expected: PASS (all prior + new teleop tests).

- [ ] **Step 4: Commit**

```bash
git add examples/trossen_ai/pyproject.toml examples/trossen_ai/webapp/HARDWARE.md
git commit -m "docs(trossen_ai): document joystick teleop; add evdev dep"
```

---

## Self-review notes

- **Spec coverage:** third page (T9), CLI (T8), shared core (T2-T3), 3D view (T9 reuses `UrdfView`), 3 camera views (T1+T3+T9), arm select + switch button/edges (T3,T5,T7,T9), keybindings panel (T9), detached/test/autonomous modes (T3,T6,T8), Home/Sleep bindings (T4,T7,T9), full axis map (T7,T9), velocity model + scaling (T2), safety/E-STOP/firmware guard (T6 reuses RobotController), tests off-robot (T2-T8).
- **Reuse / no duplication:** one `TeleopController`, one `scale_axes`/`integrate_pose16`, one `encode_camera_jpegs`; web and CLI differ only in input source + sink.
- **Type consistency:** `TeleopCommand` fields (`lin`, `ang`, `grip`, `arm`, `switch_arm`, `go_home`, `go_sleep`) used identically across T2/T3/T5/T7; `scale_axes(raw6, max_lin, max_ang, deadzone)` signature identical in T2/T5/T7; `poll()`/`update()` consistent.
- **Integration risk to watch (T6):** routing `teleop_input` reads `session._runner`. It is set before `run()` blocks, so it is reachable during a live session; a missed first message is harmless (browser resends at 30-50 Hz). If a future `SessionManager` change clears `_runner`, expose the runner explicitly from `session.start`.
- **Hardware-specific verification:** the gamepad button/axis index maps (Gamepad API in T9, evdev codes in T7) are the standard Xbox layout — verify against the actual controller and adjust the index/code dicts if the pad reports a non-standard mapping.
```
