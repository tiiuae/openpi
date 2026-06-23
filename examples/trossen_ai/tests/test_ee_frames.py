import numpy as np
from scipy.spatial.transform import Rotation

from external.joint_to_ee import constants as C
from external.joint_to_ee import ee_frames


def test_denormalize_gripper_endpoints_and_clip():
    assert ee_frames.denormalize_gripper(0.0) == 0.0
    assert np.isclose(ee_frames.denormalize_gripper(1.0), C.GRIPPER_OPEN)
    assert np.isclose(ee_frames.denormalize_gripper(0.5), C.GRIPPER_OPEN / 2)
    # out-of-range clips into [0, GRIPPER_OPEN]
    assert ee_frames.denormalize_gripper(-0.3) == 0.0
    assert np.isclose(ee_frames.denormalize_gripper(2.0), C.GRIPPER_OPEN)


def test_pose8_to_arm_se3_identity_quat_subtracts_mount():
    mount = np.array([0.331, 0.300, 0.831])
    pose8 = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0])  # identity quat, grip=1
    T, grip_m = ee_frames.pose8_to_arm_se3(pose8, mount)
    # rotation is identity
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-6)
    # translation = xyz - mount
    assert np.allclose(T[:3, 3], np.array([1.0, 2.0, 3.0]) - mount, atol=1e-6)
    assert np.isclose(grip_m, C.GRIPPER_OPEN)


def test_pose8_to_arm_se3_preserves_rotation():
    mount = np.zeros(3)
    R = Rotation.from_euler("xyz", [10, 20, 30], degrees=True)
    qx, qy, qz, qw = R.as_quat()  # scipy order x,y,z,w
    pose8 = np.array([0.1, 0.2, 0.3, qw, qx, qy, qz, 0.0])
    T, _ = ee_frames.pose8_to_arm_se3(pose8, mount)
    assert np.allclose(T[:3, :3], R.as_matrix(), atol=1e-6)
