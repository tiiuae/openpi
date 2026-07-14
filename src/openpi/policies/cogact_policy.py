import io
import json
import time
from typing import Any, Dict, Optional, TypeAlias

import numpy as np
import requests
from PIL import Image

from openpi_client import base_policy as _base_policy

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _ensure_hwc_uint8(image: Any) -> np.ndarray:
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


class CogACTClientPolicy(BasePolicy):
    def __init__(
        self,
        server_url: str = "http://localhost:8777/api/inference",
        timeout: float = 10.0,
        unnorm_key: Optional[str] = None,
        metadata: Dict[str, Any] | None = None,
        default_prompt: str = "",
    ) -> None:
        """CogACT HTTP client policy.

        The CogACT deploy server exposes `POST /api/inference` and expects a
        multipart/form-data request with two files:
          - ``images``: a PNG/JPG image (single third-person view)
          - ``json``:  a JSON file containing ``{"task_description": str}``
        The response is a JSON array — either a single action (shape
        ``[action_dim]``) when ``action_ensemble`` is enabled server-side, or a
        chunked action sequence (shape ``[N, action_dim]``) when
        ``action_chunking`` is enabled.

        Args:
            server_url: URL of the CogACT REST server's inference endpoint.
            timeout: Request timeout in seconds.
            unnorm_key: Kept for API parity with the OpenVLA-OFT client.
                CogACT reads its un-normalization key from the server's CLI args
                rather than per request, so this value is currently unused.
            default_prompt: Fallback instruction when ``obs["prompt"]`` is empty.
        """
        self.server_url = server_url
        self.timeout = timeout
        self.unnorm_key = unnorm_key
        self.metadata = metadata or {}
        self.default_prompt = default_prompt

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Infer an action by querying the CogACT REST server."""
        image_data = obs["images"]
        instruction = obs.get("prompt") or self.default_prompt or ""

        # CogACT's deploy.py takes a single image; pick the front/overhead view.
        if isinstance(image_data, dict):
            cam_high = image_data.get("cam_high")
            if cam_high is None:
                raise ValueError("Expected 'cam_high' key in image data for CogACT policy.")
        else:
            cam_high = image_data

        # Match openvlaoft_policy: ensure HWC uint8 and swap BGR -> RGB.
        cam_high = _ensure_hwc_uint8(cam_high)
        cam_high = cam_high[..., ::-1]

        # Encode to PNG in-memory.
        img_buf = io.BytesIO()
        Image.fromarray(cam_high).save(img_buf, format="PNG")
        img_buf.seek(0)

        query_buf = io.BytesIO(json.dumps({"task_description": instruction}).encode("utf-8"))

        files = [
            ("images", ("image.png", img_buf, "image/png")),
            ("json", ("query.json", query_buf, "application/json")),
        ]

        infer_start = time.monotonic()
        try:
            response = requests.post(self.server_url, files=files, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"CogACT server request failed: {e}")
        infer_ms = (time.monotonic() - infer_start) * 1000

        action = np.asarray(response.json(), dtype=np.float32)
        # Normalize to (N, action_dim) so downstream consumers see a consistent shape.
        if action.ndim == 1:
            action = action[None, :]

        return {
            "state": obs.get("state"),
            "actions": action,
            "policy_timing": {"infer_ms": infer_ms},
        }
