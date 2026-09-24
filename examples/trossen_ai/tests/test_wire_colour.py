"""End-to-end colour tests: a red camera pixel must reach every model as red.

The bug these guard against: the client swapped the camera's RGB into BGR on the
wire, and each server was left to flip it back. pi0/pi05, FalconVLA, OpenVLA and
ACT never did, and OpenVLA-OFT flipped twice, so five of seven paths fed their
model red and blue swapped. The rule now is zero conversions end to end.

A pure-red frame is used because a channel swap turns it pure blue, which is
unambiguous. The openpi client policies for CogACT and OpenVLA-OFT are driven
through a fake HTTP layer, so what they would send to their servers is checked
without any server running.

Needs the client venv (``main`` imports lerobot).
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
POLICIES = HERE.parents[2] / "src" / "openpi" / "policies"

main = pytest.importorskip("main", reason="needs the client venv (lerobot)")

CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")
RED, GREEN, BLUE = 0, 1, 2  # RGB channel indices


def _red_frame(height: int = 480, width: int = 640) -> np.ndarray:
    """A pure-red frame in the order lerobot's OpenCVCamera returns it: RGB."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, RED] = 255
    return frame


def _assert_red_hwc(image: np.ndarray, where: str) -> None:
    image = np.asarray(image)
    assert image[..., RED].min() == 255, f"{where}: red channel lost — colours swapped"
    assert image[..., BLUE].max() == 0, f"{where}: red arrived as blue — colours swapped"
    assert image[..., GREEN].max() == 0, f"{where}: unexpected green"


def _wire_request() -> dict:
    bridge = main.TrossenOpenPIBridge.__new__(main.TrossenOpenPIBridge)
    bridge.starvla = False
    bridge.robot = type("FakeRobot", (), {})()
    bridge.robot._cameras_ft = {cam: (480, 640, 3) for cam in CAMERAS}
    raw = {f"joint_{i}.pos": 0.0 for i in range(14)}
    raw.update({cam: _red_frame() for cam in CAMERAS})
    return bridge._build_observation(raw, "pick up the red cup")


def _load_policy_module(name: str):
    """Load an openpi client policy by path, without importing the openpi package."""
    spec = importlib.util.spec_from_file_location(f"_under_test_{name}", POLICIES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Reply:
    def __init__(self, payload) -> None:
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._payload


# -- the wire ------------------------------------------------------------------


def test_the_wire_carries_rgb():
    """The camera's RGB goes out unchanged: red stays in channel 0."""
    request = _wire_request()
    for cam in CAMERAS:
        _assert_red_hwc(np.transpose(request["images"][cam], (1, 2, 0)), f"wire/{cam}")


def test_the_starvla_resize_path_also_keeps_rgb():
    bridge = main.TrossenOpenPIBridge.__new__(main.TrossenOpenPIBridge)
    bridge.starvla = True
    bridge.robot = type("FakeRobot", (), {})()
    bridge.robot._cameras_ft = {cam: (480, 640, 3) for cam in CAMERAS}
    raw = {f"joint_{i}.pos": 0.0 for i in range(14)}
    raw.update({cam: _red_frame() for cam in CAMERAS})
    request = bridge._build_observation(raw, "pick up the red cup")
    for cam in CAMERAS:
        _assert_red_hwc(np.transpose(request["images"][cam], (1, 2, 0)), f"wire-starvla/{cam}")


# -- CogACT: openpi client policy -> PNG to the CogACT server ------------------


def test_cogact_sends_the_server_rgb(monkeypatch):
    """CogACT's server opens the PNG with PIL and does not flip, so the PNG
    itself must be red. It used to flip once to undo the client's swap."""
    cogact = _load_policy_module("cogact_policy")
    sent = {}

    def fake_post(url, files=None, timeout=None, **kwargs):
        sent.update({name: part for name, part in files})
        return _Reply([[0.0] * 14])

    monkeypatch.setattr(cogact.requests, "post", fake_post)
    cogact.CogACTClientPolicy(server_url="http://fake/api/inference").infer(_wire_request())

    _, png_buffer, _ = sent["images"]
    png_buffer.seek(0)
    _assert_red_hwc(np.asarray(Image.open(io.BytesIO(png_buffer.read())).convert("RGB")), "cogact PNG")


# -- OpenVLA-OFT: openpi client policy -> JSON to the OFT server ---------------


def _load_oft(monkeypatch):
    # json_numpy is only used to serialise the request; capture the object.
    stub = types.ModuleType("json_numpy")
    stub.patch = lambda: None
    stub.dumps = lambda obj: obj
    stub.loads = json.loads
    monkeypatch.setitem(sys.modules, "json_numpy", stub)
    return _load_policy_module("openvlaoft_policy")


def test_openvla_oft_sends_the_server_rgb(monkeypatch):
    """The OFT server no longer flips either (its prepare_images_for_vla swap is
    removed in openvla-oft), so what this policy sends is what the model sees.
    Before, the policy flipped and the server flipped again: BGR in the model."""
    oft = _load_oft(monkeypatch)
    sent = {}

    def fake_post(url, data=None, headers=None, timeout=None, **kwargs):
        sent["payload"] = data
        return _Reply([[0.0] * 14])

    monkeypatch.setattr(oft.requests, "post", fake_post)
    oft.OpenVLAOFTClientPolicy(server_url="http://fake/act").infer(_wire_request())

    payload = sent["payload"]
    for key in ("full_image", "cam_left_wrist", "cam_right_wrist"):
        _assert_red_hwc(payload[key], f"oft/{key}")


# -- the shipped policies must not grow a flip back ----------------------------


@pytest.mark.parametrize("name", ["cogact_policy", "openvlaoft_policy", "openvla_policy"])
def test_client_policies_contain_no_channel_swap(name):
    """Two independent "fixes" made nine minutes apart is how OFT ended up
    flipping twice. Any new swap on these paths should be a deliberate change
    to this test, not a quiet edit."""
    source = (POLICIES / f"{name}.py").read_text()
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert "[..., ::-1]" not in code
    assert "COLOR_BGR2RGB" not in code
    assert "COLOR_RGB2BGR" not in code
