from typing import Any

import numpy as np


def ensure_hwc_uint8(image: Any) -> np.ndarray:
    """Convert image to uint8 HWC (H, W, C)."""
    img = np.asarray(image)

    # If channel-first (C, H, W), move to (H, W, C)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] != img.shape[-1]:
        img = np.moveaxis(img, 0, -1)

    # If float in [0, 1] or [0, 255], convert to uint8
    if np.issubdtype(img.dtype, np.floating):
        img = (255.0 * img).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img
