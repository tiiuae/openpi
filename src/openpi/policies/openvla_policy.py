import base64
from typing import Any, Dict, Optional, TypeAlias
import requests
import json
import json_numpy
json_numpy.patch()
import numpy as np



from openpi_client import base_policy as _base_policy



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

BasePolicy: TypeAlias = _base_policy.BasePolicy

class OpenVLAClientPolicy(BasePolicy):
    def __init__(
        self,
        server_url: str = "http://localhost:8000/act",
        timeout: float = 5.0,
        unnorm_key: Optional[str] = None,
        metadata: Dict[str, Any] | None = None,
        default_prompt: str = "",
    ) -> None:
        """OpenVLA HTTP client policy.

        Args:
            server_url: URL of the OpenVLA REST server (e.g. "http://localhost:8000/act").
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
        """Infer actions from observations via the OpenVLA REST server."""
        # --- Extract image and instruction ---
        image_data = obs["images"]
        
        if isinstance(image_data, dict):
            preferred_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
            image_array = None
            for key in preferred_keys:
                if key in image_data:
                    image_array = image_data[key]
                    break
            if image_array is None:
                # Fall back to the first available camera
                image_array = next(iter(image_data.values()))
        else:
            image_array = image_data
        # Convert to HWC uint8
        image_array = _ensure_hwc_uint8(image_array)

        instruction = self.default_prompt

    
        # --- Build payload for OpenVLA server ---
        payload: Dict[str, Any] = {
            "image": image_array,  
            "instruction": instruction,  # str
        }

        if self.unnorm_key is not None:
            # This is what the server expects
            payload["unnorm_key"] = self.unnorm_key

        # --- Send request to OpenVLA server ---
        try:
            response = requests.post(
                self.server_url,
                data=json_numpy.dumps(payload),  # Use json.dumps to serialize the payload as JSON
                headers={"Content-Type": "application/json"},  # Set content type to JSON
                json=payload,           
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"OpenVLA server request failed: {e}") from e

        # --- Parse response ---
        action = json_numpy.loads(response.text)  # Use json.loads to parse the JSON response

        # Action chunking expects multiple timesteps - repeat action for chunk_size
        chunk_size = 25
        action = np.tile(action, (chunk_size, 1))  

        out: Dict[str, Any] = {"actions": action}

        return out
