from typing import Any, Dict, Optional, TypeAlias
from pprint import pprint
import requests
import numpy as np
import json_numpy
import numpy as np
from openpi_client import base_policy as _base_policy


BasePolicy: TypeAlias = _base_policy.BasePolicy
json_numpy.patch()


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

class OpenVLAOFTClientPolicy(BasePolicy):
    def __init__(
        self,
        server_url: str = "http://localhost:8777/act",
        timeout: float = 5.0,
        unnorm_key: Optional[str] = None,
        metadata: Dict[str, Any] | None = None,
        default_prompt: str = "",
        
    ) -> None:
        """OpenVLA-OFT HTTP client policy.

        Args:
            server_url: URL of the OpenVLA-OFT REST server (e.g. "http://localhost:8777/act").
            timeout: Request timeout in seconds.
            unnorm_key: Optional key passed to the server as `unnorm_key`.
                If provided, the server may use this to un-normalize actions
                based on dataset statistics.
    """
        self.server_url = server_url
        self.timeout = timeout
        self.unnorm_key = unnorm_key
        self.metadata = metadata or {}
        self.default_prompt = default_prompt

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Infer an action from an observation by querying the OpenVLA-OFT REST server."""
        image_data = obs["images"]
        proprio_data = obs.get("state", None)
        state_data = obs.get("state", None)
        instruction = obs.get("prompt") or self.default_prompt or ""

        # Extract cam_high from the observation
        if isinstance(image_data, dict):
            cam_high = image_data.get("cam_high")
            if cam_high is None:
                raise ValueError("Expected 'cam_high' key in image data for OpenVLA-OFT policy.")
            cam_left_wrist = image_data.get("cam_left_wrist")
            if cam_left_wrist is None:
                raise ValueError("Expected 'cam_left_wrist' key in image data for OpenVLA-OFT policy.")
            cam_right_wrist = image_data.get("cam_right_wrist")
            if cam_right_wrist is None:
                raise ValueError("Expected 'cam_right_wrist' key in image data for OpenVLA-OFT policy.")
            
            # Convert to HWC uint8
            cam_high = _ensure_hwc_uint8(cam_high) 
            cam_left_wrist = _ensure_hwc_uint8(cam_left_wrist) 
            cam_right_wrist = _ensure_hwc_uint8(cam_right_wrist) 
            

        ##### Build payload for openVLA-OFT server 
        observation = {
            "full_image": cam_high,  
            "cam_left_wrist": cam_left_wrist,  
            "cam_right_wrist": cam_right_wrist,  
            "instruction": instruction,  
            "state": state_data,  
            "proprio": proprio_data, 
            "unnorm_key": self.unnorm_key,
        }
        

        # Send the entire observation as the payload
        payload = observation

        ##### Send request to OpenVLA-OFT server
        import time
        infer_start = time.monotonic()
        try:
            response = requests.post(
                self.server_url,
                data=json_numpy.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"OpenVLA-OFT server request failed: {e}")
        infer_ms = (time.monotonic() - infer_start) * 1000

        ##### Parse response
        action = np.array(json_numpy.loads(response.text))
        out: Dict[str, Any] = {
            "state": obs.get("state"),
            "actions": action,
            "policy_timing": {"infer_ms": infer_ms},
        }

        return out
