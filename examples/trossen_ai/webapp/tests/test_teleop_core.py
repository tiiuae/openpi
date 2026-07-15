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
