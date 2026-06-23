import numpy as np

from adapters import JointAdapter, extract_joints


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
