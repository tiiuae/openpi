"""Encode robot-observation camera frames to JPEG bytes.

Shared by the policy bridge and the teleop controller so the cv2.imencode
logic lives in exactly one place.
"""
from __future__ import annotations

import cv2


def encode_camera_jpegs(obs: dict, cam_keys) -> dict[str, bytes]:
    """Return {camera_name: jpeg_bytes} for each key in cam_keys present in obs.

    lerobot's OpenCVCamera yields RGB frames, but ``cv2.imencode`` expects BGR —
    encoding RGB directly swaps the red/blue channels (an orange object shows up
    blue in the browser). Convert RGB->BGR first so the JPEG decodes correctly.
    """
    out: dict[str, bytes] = {}
    for cam in cam_keys:
        if cam in obs:
            bgr = cv2.cvtColor(obs[cam], cv2.COLOR_RGB2BGR)
            out[cam] = cv2.imencode(".jpg", bgr)[1].tobytes()
    return out
