import numpy as np
import pytest

from external.joint_to_ee import constants as C
from external.joint_to_ee.kinematics import make_kinematics

pytestmark = pytest.mark.integration

# A reachable, non-singular bimanual joint vector (radians); grippers half-open (m).
_Q14 = np.zeros(14, dtype=np.float64)
_Q14[C.OBS_LEFT_JOINTS] = [0.1, 0.3, 0.4, 0.0, 0.2, 0.0]
_Q14[C.LEFT_GRIPPER_IDX] = 0.02
_Q14[C.OBS_RIGHT_JOINTS] = [-0.1, 0.3, 0.4, 0.0, 0.2, 0.0]
_Q14[C.RIGHT_GRIPPER_IDX] = 0.02


@pytest.fixture(scope="module")
def converter():
    from external.joint_to_ee.ee_to_joints import EEToJointsConverter
    return EEToJointsConverter(make_kinematics())


def test_joints14_to_ee16_shape_and_frame(converter):
    state16 = converter.joints14_to_ee16(_Q14)
    assert state16.shape == (16,)
    # left EE x should include the +0.331 robot-base mount offset
    assert state16[0] > 0.331 - 0.5  # sanity: mounted forward of arm base


def test_fk_then_ik_round_trip_recovers_joints(converter):
    # FK both arms (robot-base, mounted) -> EE16, then decode back to joints.
    ee16 = converter.joints14_to_ee16(_Q14)
    out14 = converter.decode_chunk(ee16[None, :], _Q14)[0]
    # arm joints recovered within IK tolerance
    assert np.allclose(out14[C.OBS_LEFT_JOINTS], _Q14[C.OBS_LEFT_JOINTS], atol=2e-2)
    assert np.allclose(out14[C.OBS_RIGHT_JOINTS], _Q14[C.OBS_RIGHT_JOINTS], atol=2e-2)
    # gripper passes through denormalized: ee gripper was normalized 0.02/0.044
    assert np.isclose(out14[C.LEFT_GRIPPER_IDX], 0.02, atol=1e-3)


def test_decode_chunk_shape(converter):
    ee16 = converter.joints14_to_ee16(_Q14)
    chunk = np.repeat(ee16[None, :], 4, axis=0)
    out = converter.decode_chunk(chunk, _Q14)
    assert out.shape == (4, 14)


def test_unreachable_pose_holds_last_valid(converter):
    # A target 10 m away is unreachable -> decode must hold the seed (current) joints.
    ee16 = converter.joints14_to_ee16(_Q14).copy()
    ee16[0] += 10.0  # left EE x way out of range
    out14 = converter.decode_chunk(ee16[None, :], _Q14)[0]
    assert np.allclose(out14[C.OBS_LEFT_JOINTS], _Q14[C.OBS_LEFT_JOINTS], atol=1e-6)
