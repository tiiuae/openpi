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
