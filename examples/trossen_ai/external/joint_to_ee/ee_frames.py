"""Pure frame + gripper math for EE -> arm-base conversion (no placo / URDF).

Inverts the forward path in `enrich.py` / `kinematics.py`: model EE poses are in
the ROBOT-BASE frame (mount applied); arm-base controllers need the mount removed.
Mount is a pure translation, so undoing it subtracts xyz and leaves rotation intact.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from . import constants as C


def denormalize_gripper(grip_norm: float) -> float:
    """Inverse of `poses.normalize_gripper`: 0..1 -> carriage meters [0, GRIPPER_OPEN]."""
    g = float(np.clip(grip_norm, 0.0, 1.0))
    return g * (C.GRIPPER_OPEN - C.GRIPPER_CLOSED) + C.GRIPPER_CLOSED


def pose8_to_arm_se3(pose8: np.ndarray, mount_xyz) -> tuple[np.ndarray, float]:
    """[x,y,z,qw,qx,qy,qz,grip_norm] (robot-base) -> (4x4 SE3 in arm-base, grip_m).

    Undoes the mount translation (robot-base -> arm-base) and converts the
    quaternion (w,x,y,z) into a rotation matrix.
    """
    pose8 = np.asarray(pose8, dtype=np.float64).flatten()
    x, y, z, qw, qx, qy, qz, grip_norm = pose8
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy wants x,y,z,w
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z], dtype=np.float64) - np.asarray(mount_xyz, dtype=np.float64)
    return T, denormalize_gripper(grip_norm)
