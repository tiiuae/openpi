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
