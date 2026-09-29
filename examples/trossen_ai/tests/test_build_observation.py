"""Offline tests for the observation the client sends to the policy server.

Images go out exactly as the camera produced them: native resolution, RGB,
untouched. The server resizes and must not flip. These tests pin what a server
depends on: the size is untouched, the pixels are not resampled, and the
channels arrive in the camera's RGB order. (The client used to swap them into
BGR, and five of seven server paths never flipped them back.)

The bridge is built with ``__new__`` so no robot, camera or server is involved.
Needs the client venv, since ``main`` imports lerobot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

main = pytest.importorskip("main", reason="needs the client venv (lerobot)")

CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")


def _bridge(height: int = 480, width: int = 640):
    bridge = main.TrossenOpenPIBridge.__new__(main.TrossenOpenPIBridge)
    bridge.robot = type("FakeRobot", (), {})()
    bridge.robot._cameras_ft = {cam: (height, width, 3) for cam in CAMERAS}
    return bridge


def _raw_observation(height: int = 480, width: int = 640, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    observation = {f"joint_{i}.pos": float(i) / 10 for i in range(14)}
    for cam in CAMERAS:
        observation[cam] = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return observation


def test_images_are_sent_at_native_resolution():
    """No client-side resize: the server resizes the way its model was trained."""
    request = _bridge()._build_observation(_raw_observation(), "pick up the cup")
    for cam in CAMERAS:
        assert request["images"][cam].shape == (3, 480, 640)
        assert request["images"][cam].dtype == np.uint8


def test_pixels_are_sent_untouched():
    """Every pixel survives exactly: only the channel axis moves (HWC -> CHW),
    nothing is resampled and nothing is swapped."""
    raw = _raw_observation()
    request = _bridge()._build_observation(raw, "pick up the cup")
    for cam in CAMERAS:
        np.testing.assert_array_equal(request["images"][cam], np.transpose(raw[cam], (2, 0, 1)))


def test_the_wire_is_rgb():
    """A red camera pixel arrives red: channel 0. Under the old client it
    arrived in channel 2, and most servers fed that to their model as blue."""
    raw = _raw_observation()
    for cam in CAMERAS:
        raw[cam][:] = 0
        raw[cam][:, :, 0] = 255  # pure red, as the camera (RGB) reports it
    request = _bridge()._build_observation(raw, "pick up the cup")
    for cam in CAMERAS:
        sent = request["images"][cam]
        assert sent[0].min() == 255  # red stays in channel 0: RGB on the wire
        assert sent[1].max() == 0
        assert sent[2].max() == 0


@pytest.mark.parametrize(("height", "width"), [(480, 640), (720, 1280), (224, 224), (300, 500)])
def test_any_camera_resolution_passes_through(height, width):
    """Nothing is hard-coded to a particular size any more."""
    bridge = _bridge(height, width)
    request = bridge._build_observation(_raw_observation(height, width), "pick up the cup")
    for cam in CAMERAS:
        assert request["images"][cam].shape == (3, height, width)


def test_camera_order_is_preserved():
    """msgpack keeps dict order and servers treat it as significant."""
    request = _bridge()._build_observation(_raw_observation(), "pick up the cup")
    assert tuple(request["images"]) == CAMERAS


def test_state_and_prompt_are_passed_through():
    request = _bridge()._build_observation(_raw_observation(), "pick up the cup")
    np.testing.assert_allclose(request["state"], np.arange(14) / 10)
    assert request["prompt"] == "pick up the cup"


def test_the_sent_arrays_are_contiguous_enough_to_serialize():
    """msgpack_numpy packs via tobytes(); a transposed view must still pack."""
    from openpi_client import msgpack_numpy  # noqa: PLC0415 - client venv only

    request = _bridge()._build_observation(_raw_observation(), "pick up the cup")
    decoded = msgpack_numpy.unpackb(msgpack_numpy.packb(request))
    for cam in CAMERAS:
        np.testing.assert_array_equal(decoded["images"][cam], request["images"][cam])
