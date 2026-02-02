import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_falconvla_example() -> dict:
    """Creates a random input example for the FalconVLA policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


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
