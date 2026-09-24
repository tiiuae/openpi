"""Offline tests for the observation the client sends to the policy server.

Images now go out at the camera's native resolution and the server resizes.
These tests pin three things a server depends on: the size is untouched, the
pixels are not resampled, and the channel order is exactly what it was before
(servers built against CLIENT_SCHEMA.md flip BGR back to RGB, so changing it
would silently swap their colours).

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


def test_pixels_are_not_resampled():
    """Every pixel survives exactly; only the channel axis moves and swaps."""
    raw = _raw_observation()
    request = _bridge()._build_observation(raw, "pick up the cup")
    for cam in CAMERAS:
        sent = request["images"][cam]
        # Channel c on the wire is channel (2 - c) from the camera, pixel for pixel.
        for channel in range(3):
            np.testing.assert_array_equal(sent[channel], raw[cam][:, :, 2 - channel])


def test_channel_order_on_the_wire_is_unchanged():
    """Pinned to the previous client's order, minus only the resize. The camera
    returns RGB and the client's swap puts BGR on the wire; servers flip it
    back. A red camera pixel must still arrive with red in the last channel."""
    raw = _raw_observation()
    for cam in CAMERAS:
        raw[cam][:] = 0
        raw[cam][:, :, 0] = 255  # pure red, as the camera (RGB) reports it
    request = _bridge()._build_observation(raw, "pick up the cup")
    for cam in CAMERAS:
        sent = request["images"][cam]
        assert sent[2].min() == 255  # red lands in channel 2: BGR on the wire
        assert sent[0].max() == 0
        assert sent[1].max() == 0


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
