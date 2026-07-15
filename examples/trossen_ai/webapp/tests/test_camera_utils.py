import numpy as np
from camera_utils import encode_camera_jpegs


def test_encodes_only_named_cameras_as_jpeg_bytes():
    obs = {
        "cam_high": np.zeros((4, 4, 3), dtype=np.uint8),
        "cam_low": np.full((4, 4, 3), 255, dtype=np.uint8),
        "left.pos": 0.1,  # non-camera key must be ignored
    }
    out = encode_camera_jpegs(obs, ["cam_high", "cam_low"])
    assert set(out) == {"cam_high", "cam_low"}
    # JPEG SOI marker
    assert out["cam_high"][:2] == b"\xff\xd8"
    assert isinstance(out["cam_low"], bytes)
