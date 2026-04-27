import dataclasses
from typing import ClassVar
from collections.abc import Sequence
import logging
import time
from typing import Any, TypeAlias

import einops
import numpy as np
import jax
import jax.numpy as jnp
import torch
from typing_extensions import override

from openpi import transforms
from openpi import transforms as _transforms
from openpi.models import falconvla_config, model as _model
from openpi.shared import array_typing as at
from openpi_client import base_policy as _base_policy


BasePolicy: TypeAlias = _base_policy.BasePolicy


class FalconVLAPolicy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the FalconVLA Policy.

        Args:
            model: The FalconVLA model to use for action sampling.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._pytorch_device = pytorch_device

        if hasattr(self._model, 'eval'):
            self._model.eval()

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        from openpi.training.config import get_config
        start_time = time.monotonic()
        
        try:
            outputs = self._model.inference(obs)   # this is the flat (350,) array
        except Exception as e:
            logging.error(f"Error during model inference: {e}")
            raise e
        
        # Turn model output into actions
        actions = np.asarray(outputs)

        if actions.ndim == 1:
            model_config = self._model.config

            actions = actions.reshape(model_config.action_horizon, model_config.action_dim)  # reshape to (14, 25)

        outputs = {
            "actions": actions,
            "policy_timing": {
                "infer_ms": (time.monotonic() - start_time) * 1000,
            },
        }

        return outputs

        
    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata




@dataclasses.dataclass(frozen=True)
class FalconVLAInputs(transforms.DataTransformFn):
    """Inputs for the FalconVLA policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    - prompt: string
    
    This follows the same interface as AlohaInputs but is tailored for FalconVLA's requirements.
    """

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_falconvla(data)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists (cam_high for primary view).
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FalconVLAOutputs(transforms.DataTransformFn):
    """Outputs for the FalconVLA policy.
    
    This ensures the output format is compatible with Aloha expectations.
    """

    def __call__(self, data: dict) -> dict:
        # FalconVLA outputs actions directly, ensure it's in the right format
        actions = np.asarray(data["actions"])
        
        # If actions have more than 14 dims, only return the first 14
        if actions.shape[-1] > 14:
            actions = actions[:, :14]
        
        return {"actions": actions}


def _decode_falconvla(data: dict) -> dict:
    """Decode FalconVLA input data to the expected format.
    
    This converts images and state to the format expected by the model.
    """
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data
