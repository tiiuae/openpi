"""Encode robot-observation camera frames to JPEG bytes.

Shared by the policy bridge and the teleop controller so the cv2.imencode
logic lives in exactly one place.
"""
from __future__ import annotations

import cv2


def encode_camera_jpegs(obs: dict, cam_keys) -> dict[str, bytes]:
    """Return {camera_name: jpeg_bytes} for each key in cam_keys present in obs."""
    out: dict[str, bytes] = {}
    for cam in cam_keys:
        if cam in obs:
            out[cam] = cv2.imencode(".jpg", obs[cam])[1].tobytes()
    return out
