"""Tests for serve_policy --debug_dir image capture.

Runs openpi's real AlohaInputs and ResizeImages transforms, so model_input_*.png
is checked against what a pi0/pi05 model really receives. A pure-red camera
frame is used because a channel swap turns it pure blue.
"""

import json

import numpy as np
from PIL import Image
import pytest

from openpi import transforms as _transforms
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.serving.debug_capture import DebugCapturePolicy

RED, GREEN, BLUE = 0, 1, 2
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class _TransformOnlyPolicy(_policy.Policy):
    """A real openpi Policy's input transform, with the model call stubbed out."""

    def __init__(self) -> None:  # skips loading a model
        self._input_transform = _transforms.compose(
            [aloha_policy.AlohaInputs(adapt_to_pi=False), _transforms.ResizeImages(224, 224)]
        )
        self._metadata = {"test": True}
        self.seen = None

    def infer(self, obs: dict, **kwargs) -> dict:
        self.seen = self._input_transform(dict(obs))
        return {"actions": np.zeros((50, 14), dtype=np.float32)}


class _PlainPolicy:
    """Stands in for FalconVLA / REST policies: no openpi input transform."""

    @property
    def metadata(self) -> dict:
        return {}

    def infer(self, obs: dict, **kwargs) -> dict:
        return {"actions": np.zeros((25, 14), dtype=np.float32)}


def _red_chw(height=480, width=640, order="rgb") -> np.ndarray:
    frame = np.zeros((3, height, width), dtype=np.uint8)
    frame[RED if order == "rgb" else BLUE] = 255
    return frame


def _obs(order="rgb") -> dict:
    return {
        "state": np.zeros(14, dtype=np.float32),
        "images": {cam: _red_chw(order=order) for cam in CAMERAS},
        "prompt": "pick up the cup",
    }


def _call_dir(tmp_path):
    (run,) = tmp_path.iterdir()
    return run / "call_00000"


def _load(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def test_wire_images_are_saved_as_received(tmp_path):
    policy = DebugCapturePolicy(_TransformOnlyPolicy(), str(tmp_path))
    policy.infer(_obs())
    policy.flush()
    for cam in CAMERAS:
        image = _load(_call_dir(tmp_path) / f"wire_{cam}.png")
        assert image.shape == (480, 640, 3)
        assert image[..., RED].min() == 255
        assert image[..., BLUE].max() == 0


def test_model_input_is_the_letterboxed_rgb_image_the_model_gets(tmp_path):
    inner = _TransformOnlyPolicy()
    policy = DebugCapturePolicy(inner, str(tmp_path))
    policy.infer(_obs())
    policy.flush()
    image = _load(_call_dir(tmp_path) / "model_input_base_0_rgb.png")
    assert image.shape == (224, 224, 3)
    content = image[28:196]
    assert content[..., RED].min() >= 254
    assert content[..., BLUE].max() <= 1
    assert image[:28].max() <= 1  # the training letterbox: black above
    assert image[196:].max() <= 1  # and below
    # And it is byte-identical to what the model was handed.
    np.testing.assert_array_equal(image, np.asarray(inner.seen["image"]["base_0_rgb"]))


def test_every_model_camera_is_saved(tmp_path):
    policy = DebugCapturePolicy(_TransformOnlyPolicy(), str(tmp_path))
    policy.infer(_obs())
    policy.flush()
    names = {p.name for p in _call_dir(tmp_path).glob("model_input_*.png")}
    assert names == {"model_input_base_0_rgb.png", "model_input_left_wrist_0_rgb.png", "model_input_right_wrist_0_rgb.png"}


def test_a_bgr_client_is_visible_in_the_saved_images(tmp_path):
    """The point of the feature: an old BGR client shows up as a blue scene."""
    policy = DebugCapturePolicy(_TransformOnlyPolicy(), str(tmp_path))
    policy.infer(_obs(order="bgr"))
    policy.flush()
    content = _load(_call_dir(tmp_path) / "model_input_base_0_rgb.png")[28:196]
    assert content[..., BLUE].min() >= 254
    meta = json.loads((_call_dir(tmp_path) / "meta.json").read_text())
    means = meta["model_input_images"]["base_0_rgb"]["channel_mean_as_rgb"]
    assert means["b"] > means["r"]


def test_meta_records_shapes_and_policy(tmp_path):
    policy = DebugCapturePolicy(_TransformOnlyPolicy(), str(tmp_path))
    policy.infer(_obs())
    policy.flush()
    meta = json.loads((_call_dir(tmp_path) / "meta.json").read_text())
    assert meta["policy"] == "_TransformOnlyPolicy"
    assert meta["prompt"] == "pick up the cup"
    assert meta["wire_images"]["cam_high"]["shape"] == [3, 480, 640]
    assert meta["model_input_images"]["base_0_rgb"]["shape"] == [224, 224, 3]


def test_policies_without_openpi_transforms_still_get_wire_images(tmp_path):
    policy = DebugCapturePolicy(_PlainPolicy(), str(tmp_path))
    out = policy.infer(_obs())
    policy.flush()
    assert out["actions"].shape == (25, 14)
    assert (_call_dir(tmp_path) / "wire_cam_high.png").exists()
    assert not list(_call_dir(tmp_path).glob("model_input_*.png"))


class _UnusedTransformPolicy:
    """Like FalconVLAPolicy: builds an input transform but never calls it."""

    @property
    def metadata(self) -> dict:
        return {}

    def __init__(self) -> None:
        self._input_transform = lambda data: data

    def infer(self, obs: dict, **kwargs) -> dict:
        return {"actions": np.zeros((25, 7), dtype=np.float32)}


def test_a_policy_that_never_runs_its_transform_is_not_hooked(tmp_path, caplog):
    """No model_input_*.png is promised, or written, for FalconVLA-style policies."""
    with caplog.at_level("INFO"):
        policy = DebugCapturePolicy(_UnusedTransformPolicy(), str(tmp_path))
    assert "model_input" not in caplog.text
    policy.infer(_obs())
    policy.flush()
    assert (_call_dir(tmp_path) / "wire_cam_high.png").exists()
    assert not list(_call_dir(tmp_path).glob("model_input_*.png"))


def test_calls_are_numbered_and_actions_pass_through(tmp_path):
    inner = _TransformOnlyPolicy()
    policy = DebugCapturePolicy(inner, str(tmp_path))
    for _ in range(3):
        assert policy.infer(_obs())["actions"].shape == (50, 14)
    policy.flush()
    (run,) = tmp_path.iterdir()
    assert sorted(p.name for p in run.iterdir()) == ["call_00000", "call_00001", "call_00002"]


def test_metadata_is_forwarded(tmp_path):
    assert DebugCapturePolicy(_TransformOnlyPolicy(), str(tmp_path)).metadata == {"test": True}


@pytest.mark.parametrize("flag", ["--debug_dir", "--debug-dir"])
def test_serve_policy_accepts_the_flag(flag):
    import importlib.util  # noqa: PLC0415
    import pathlib  # noqa: PLC0415
    import sys  # noqa: PLC0415

    import tyro  # noqa: PLC0415

    path = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "serve_policy.py"
    spec = importlib.util.spec_from_file_location("serve_policy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # tyro reads the class source via inspect
    spec.loader.exec_module(module)
    assert tyro.cli(module.Args, args=[flag, "/tmp/x"]).debug_dir == "/tmp/x"
    assert tyro.cli(module.Args, args=[]).debug_dir is None
