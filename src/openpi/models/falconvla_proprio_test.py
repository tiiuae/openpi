"""Single-arm heads, proprio assembly and proprio delivery in FalconVLA inference.

No checkpoint is loaded: the helpers are tested directly, and `inference` runs against a stub
model/processor so the arm offset, proprio vector and action widening are checked end to end.
"""

import numpy as np
import pytest
import torch

from openpi.models import falconvla
from openpi.models.falconvla_config import FalconVLAConfig
from openpi.training import config as _config

# 7 channels per arm; the right arm is 7:14.
STATE = np.arange(14, dtype=np.float64) / 10.0


def _stats(dim, *, frozen=()):
    q01 = np.full(dim, -1.0)
    q99 = np.full(dim, 2.0)
    for i in frozen:
        q01[i] = q99[i] = 0.5
    return {"mean": np.full(dim, 0.5), "q01": q01, "q99": q99, "min": q01, "max": q99}


# ---- helpers -----------------------------------------------------------------------------------


def test_arm_offset():
    assert falconvla._arm_offset("both", 14) == 0
    assert falconvla._arm_offset("left", 14) == 0
    assert falconvla._arm_offset("right", 14) == 7
    with pytest.raises(ValueError, match="arm must be"):
        falconvla._arm_offset("middle", 14)


def test_build_proprio_takes_the_right_arm_slice():
    proprio, pinned = falconvla._build_proprio(STATE, _stats(7), proprio_dim=7, offset=7, static_indices=None)
    np.testing.assert_allclose(proprio, STATE[7:14])
    assert not pinned.any()


def test_build_proprio_keeps_the_mean_where_the_robot_has_no_value():
    """19-d proprio from a 14-d robot: channels 14:19 stay at the training mean, not zero."""
    proprio, _ = falconvla._build_proprio(STATE, _stats(19), proprio_dim=19, offset=0, static_indices=None)
    np.testing.assert_allclose(proprio[:14], STATE)
    np.testing.assert_allclose(proprio[14:], 0.5)


def test_build_proprio_pins_static_and_frozen_channels():
    proprio, pinned = falconvla._build_proprio(
        STATE, _stats(14, frozen=(10,)), proprio_dim=14, offset=0, static_indices=(0, 1)
    )
    assert pinned.tolist() == [i in (0, 1, 10) for i in range(14)]
    np.testing.assert_allclose(proprio[[0, 1, 10]], 0.5)
    np.testing.assert_allclose(proprio[2:10], STATE[2:10])


def test_build_proprio_rejects_a_dim_mismatch():
    with pytest.raises(ValueError, match="proprio_dim=7"):
        falconvla._build_proprio(STATE, _stats(14), proprio_dim=7, offset=0, static_indices=None)


def test_single_arm_actions_hold_the_other_arm():
    actions = np.ones((25, 7))
    right = falconvla._to_robot_actions(actions, offset=7, robot_action_dim=14, hold=STATE)
    assert right.shape == (25, 14)
    np.testing.assert_allclose(right[:, :7], np.tile(STATE[:7], (25, 1)))
    np.testing.assert_allclose(right[:, 7:], 1.0)

    left = falconvla._to_robot_actions(actions, offset=0, robot_action_dim=14, hold=STATE)
    np.testing.assert_allclose(left[:, :7], 1.0)
    np.testing.assert_allclose(left[:, 7:], np.tile(STATE[7:], (25, 1)))


def test_wide_actions_are_truncated():
    actions = np.arange(25 * 16, dtype=np.float64).reshape(25, 16)
    out = falconvla._to_robot_actions(actions, offset=0, robot_action_dim=14, hold=STATE)
    np.testing.assert_array_equal(out, actions[:, :14])


def test_proprio_token_ids_ascend_from_the_block_base():
    config = type("C", (), {"proprio_num_bins": 256, "proprio_token_id_max": 131071})()
    ids = falconvla._proprio_token_ids(np.array([-1.0, 0.0, 1.0, 5.0]), config)
    base = 131071 - 256 + 1
    assert ids[0] == base
    assert ids[2] == ids[3] == 131071
    assert base < ids[1] < 131071


def test_splice_before_thinking():
    inputs = {"input_ids": torch.tensor([[1, 2, 99, 3]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])}
    falconvla._splice_before_thinking(inputs, np.array([7, 8]), thinking_token_id=99)
    assert inputs["input_ids"].tolist() == [[1, 2, 7, 8, 99, 3]]
    assert inputs["attention_mask"].tolist() == [[1] * 6]
    with pytest.raises(ValueError, match="no <thinking>"):
        falconvla._splice_before_thinking({"input_ids": torch.tensor([[1, 2]])}, np.array([7]), 99)


def test_proprio_is_kwarg_and_internal_normalization_detection():
    class ResNet:
        def predict_action(self, inputs, proprio_inputs=None, **kwargs): ...

    class FmDit:
        def predict_action(self, inputs, **kwargs): ...

        def _normalize_proprio(self, x): ...

    assert falconvla._proprio_is_kwarg(ResNet())
    assert not falconvla._normalizes_internally(ResNet())
    assert not falconvla._proprio_is_kwarg(FmDit())
    assert falconvla._normalizes_internally(FmDit())


# ---- inference end to end, against a stub model ------------------------------------------------


class _Tokenizer:
    unk_token_id = 0

    def __init__(self, has_prop_tokens):
        self._has = has_prop_tokens

    def convert_tokens_to_ids(self, token):
        return 5 if self._has else self.unk_token_id


class _Builder:
    last = None

    def __init__(self, task_label):
        self.proprio = None
        _Builder.last = self

    def add_main_image(self, image): ...
    def add_wrist_image(self, image): ...
    def add_secondary_image(self, image): ...

    def add_proprio(self, text):
        self.proprio = text

    def build_inputs(self, processor):
        return {"input_ids": torch.tensor([[1, 2, 3]])}, None


class _ResNetModel:
    """A FiLM head: takes `proprio_inputs=` and expects it already normalized."""

    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.config = type("C", (), {"proprio_mode": "film"})()
        self.seen = None

    def predict_action(self, inputs, unnorm_key, horizon, do_sample, proprio_inputs=None):
        self.seen = proprio_inputs
        return torch.ones(1, horizon, self.action_dim)


@pytest.fixture
def make_vla(monkeypatch):
    """Builds a FalconVLA through its real __init__, with the checkpoint replaced by stubs."""
    stats = _stats(7)
    monkeypatch.setattr(falconvla, "ActionPromptBuilder", _Builder)
    monkeypatch.setattr(falconvla, "fetch_proprio_stats", lambda model, key: stats)
    monkeypatch.setattr(falconvla, "_warn_if_action_shape_mismatch", lambda *args: None)

    def make(config, *, has_prop_tokens=False):
        processor = type("P", (), {"tokenizer": _Tokenizer(has_prop_tokens)})()
        model = _ResNetModel(config.action_dim)
        monkeypatch.setattr(falconvla, "load_falcon_model", lambda *args: (processor, model))
        return falconvla.FalconVLA(config)

    return make


def _obs(state=STATE):
    image = np.zeros((3, 48, 64), dtype=np.uint8)
    obs = {"prompt": "pick", "images": {"cam_high": image, "cam_right_wrist": image, "cam_left_wrist": image}}
    if state is not None:
        obs["state"] = state
    return obs


def test_right_arm_head_end_to_end(make_vla):
    config = FalconVLAConfig(
        unnorm_key="k",
        action_dim=7,
        action_horizon=25,
        use_proprio=True,
        proprio_mode="film",
        arm="right",
        device="cpu",
    )
    vla = make_vla(config)
    actions = vla.inference(_obs())

    assert actions.shape == (25, 14)
    np.testing.assert_allclose(actions[:, :7], np.tile(STATE[:7], (25, 1)))  # left arm holds
    np.testing.assert_allclose(actions[:, 7:], 1.0)  # right arm follows the head
    # The head saw the right arm's state, normalized on [q01, q99] = [-1, 2].
    expected = 2 * (STATE[7:14] + 1) / 3 - 1
    np.testing.assert_allclose(vla.model.seen.reshape(-1).numpy(), expected, atol=1e-6)
    assert tuple(vla.model.seen.shape) == (1, 1, 7)


def test_tokens_mode_without_prop_tokens_falls_back_to_continuous(make_vla):
    config = FalconVLAConfig(
        unnorm_key="k",
        action_dim=7,
        action_horizon=25,
        use_proprio=True,
        proprio_mode="tokens",
        arm="right",
        device="cpu",
    )
    vla = make_vla(config, has_prop_tokens=False)
    vla.inference(_obs())
    assert _Builder.last.proprio is None
    assert vla.model.seen is not None


def test_tokens_mode_with_prop_tokens_injects_text_tokens(make_vla):
    config = FalconVLAConfig(
        unnorm_key="k",
        action_dim=7,
        action_horizon=25,
        use_proprio=True,
        proprio_mode="tokens",
        arm="right",
        device="cpu",
    )
    vla = make_vla(config, has_prop_tokens=True)
    vla.inference(_obs())
    assert _Builder.last.proprio.count("<prop") == 7
    assert vla.model.seen is None


def test_single_arm_head_without_state_is_refused(make_vla):
    config = FalconVLAConfig(
        unnorm_key="k", action_dim=7, action_horizon=25, use_proprio=False, arm="right", device="cpu"
    )
    with pytest.raises(ValueError, match="needs the robot's 14-d state"):
        make_vla(config).inference(_obs(state=None))


def test_single_arm_head_must_name_its_arm(make_vla):
    config = FalconVLAConfig(unnorm_key="k", action_dim=7, action_horizon=25, use_proprio=False, device="cpu")
    with pytest.raises(ValueError, match="set arm='left' or arm='right'"):
        make_vla(config)


# ---- registry ----------------------------------------------------------------------------------


def _falcon_configs():
    return [c for c in _config._CONFIGS if isinstance(c.model, FalconVLAConfig)]


def test_config_names_are_unique():
    names = [c.name for c in _config._CONFIGS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("train_config", _falcon_configs(), ids=lambda c: c.name)
def test_single_arm_configs_name_their_arm(train_config):
    model = train_config.model
    if model.action_dim < model.robot_action_dim:
        assert model.arm in ("left", "right")
    assert _config.get_config(train_config.name) is train_config
