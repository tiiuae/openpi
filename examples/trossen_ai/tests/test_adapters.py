import numpy as np

from adapters import ActionSpaceAdapter, EEAdapter, JointAdapter, extract_joints


def _fake_obs():
    # insertion order mirrors get_observation(): left joints then right joints
    names = [f"left_joint_{i}.pos" for i in range(6)] + ["left_gripper.pos"] \
            + [f"right_joint_{i}.pos" for i in range(6)] + ["right_gripper.pos"]
    obs = {n: float(i) for i, n in enumerate(names)}
    obs["cam_high"] = np.zeros((4, 4, 3))  # non-.pos keys ignored
    return obs


def test_extract_joints_orders_14():
    obs = _fake_obs()
    q = extract_joints(obs)
    assert q.shape == (14,)
    assert np.allclose(q, np.arange(14))


def test_joint_adapter_build_state_is_joints():
    adapter = JointAdapter()
    state = adapter.build_state(_fake_obs())
    assert state.shape == (14,)
    assert np.allclose(state, np.arange(14))


def test_joint_adapter_decode_chunk_is_identity_slice():
    adapter = JointAdapter()
    raw = np.arange(3 * 20, dtype=np.float32).reshape(3, 20)  # model may pad >14
    out = adapter.decode_chunk(raw, current_joints14=np.zeros(14))
    assert out.shape == (3, 14)
    assert np.allclose(out, raw[:, :14])


def test_base_and_joint_adapter_ee_chunk_is_none():
    raw = np.arange(3 * 20, dtype=np.float32).reshape(3, 20)
    # base class returns None (not an EE space)
    assert ActionSpaceAdapter().ee_chunk(raw) is None
    # JointAdapter inherits the None default
    assert JointAdapter().ee_chunk(raw) is None


def test_ee_adapter_ee_chunk_slices_first_16_cols():
    # ee_chunk does not touch the converter, so a stub is fine here.
    adapter = EEAdapter(converter=None)
    raw = np.arange(4 * 20, dtype=np.float32).reshape(4, 20)  # wider than 16
    out = adapter.ee_chunk(raw)
    assert out.shape == (4, 16)
    assert np.allclose(out, raw[:, :16])


import pytest


@pytest.mark.integration
def test_ee_adapter_state_is_16_and_decodes_to_joints():
    from external.joint_to_ee import constants as C
    from external.joint_to_ee.kinematics import make_kinematics
    from external.joint_to_ee.ee_to_joints import EEToJointsConverter
    from adapters import EEAdapter

    q14 = np.zeros(14)
    q14[C.OBS_LEFT_JOINTS] = [0.1, 0.3, 0.4, 0.0, 0.2, 0.0]
    q14[C.OBS_RIGHT_JOINTS] = [-0.1, 0.3, 0.4, 0.0, 0.2, 0.0]

    converter = EEToJointsConverter(make_kinematics())
    adapter = EEAdapter(converter)
    assert adapter.state_dim == 16

    # build a fake obs dict from q14 so build_state can extract+FK it
    names = [f"left_joint_{i}.pos" for i in range(6)] + ["left_gripper.pos"] \
            + [f"right_joint_{i}.pos" for i in range(6)] + ["right_gripper.pos"]
    obs = {n: float(q14[i]) for i, n in enumerate(names)}

    state = adapter.build_state(obs)
    assert state.shape == (16,)

    # round-trip: decode the FK'd state back to joints
    out = adapter.decode_chunk(state[None, :], q14)
    assert out.shape == (1, 14)
    assert np.allclose(out[0, C.OBS_LEFT_JOINTS], q14[C.OBS_LEFT_JOINTS], atol=2e-2)
