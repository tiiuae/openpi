from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Dict
from enum import Enum
import numpy as np


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on
ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio

def tokenize_proprio(
    x: float | np.ndarray, 
    num_bins: int = 256,
) -> list[str]:
    """
    Encode float(s) in [-1, 1] into bin indices at the *end* of the vocab.
    Indices will be in [vocab_size - num_bins, vocab_size - 1].

    Args:
        x (float): or array of floats in [-1, 1].
        num_bins (int): number of bins (default 256).

    Returns:
        list[str]: tokenized array of proprioceptive states ["<prop0>", "<prop11>",...]
    """
    if np.any((x < -1.0) | (x > 1.0)):
        raise ValueError("Input values must be in [-1, 1]")

    edges = np.linspace(-1, 1, num_bins)
    rel_idx = np.digitize(x, edges, right=False) - 1
    rel_idx = rel_idx.astype(np.int64).ravel()
    return [f"<prop{int(idx)}>" for idx in rel_idx]

IMAGE_BLOCK = "|<start_of_img>||<image>||<end_of_img>|"
TURN_PREFIX = "|<start_of_turn>|User: "
THINK_TAG = "<thinking>"

@dataclass
class ActionPromptBuilder:
    task_label: str
    main_image: Optional[Any] = None
    wrist_image: Optional[Any] = None
    secondary_image: Optional[Any] = None
    proprio_tokens: Optional[str] = None

    def add_main_image(self, image: Any) -> "ActionPromptBuilder":
        self.main_image = image
        return self

    def add_wrist_image(self, wrist_image: Any) -> "ActionPromptBuilder":
        self.wrist_image = wrist_image
        return self
    
    def add_secondary_image(self, secondary_image: Any) -> "ActionPromptBuilder":
        self.secondary_image = secondary_image
        return self

    def get_images(self) -> List[Any]:
        return [self.main_image, self.wrist_image, self.secondary_image]

    def add_proprio(self, proprio_tokens: str) -> "ActionPromptBuilder":
        """
        Add proprio tokens to the prompt.
        
        Args:
            proprio_tokens: String of proprio tokens (e.g., "<prop0><prop11>...")
        """
        self.proprio_tokens = proprio_tokens
        return self

    def build_text(self) -> str:
        """
        Render as many image blocks as we have images (main first, then wrist).
        Insert proprio tokens immediately before <thinking>.
        """
        images = self.get_images()
        image_block_count = sum(image is not None for image in images)
        image_section = IMAGE_BLOCK * image_block_count
        proprio_section = self.proprio_tokens or ""
        prompt = (
            f"{TURN_PREFIX}{image_section}"
            f"What action should the robot take to {self.task_label}?\n"
            f"Falcon:{proprio_section}{THINK_TAG}"
        )
        return prompt

    def build_inputs(self, processor) -> Tuple[dict, str]:
        """
        Returns (inputs_dict, text) so you can inspect the final prompt if needed.
        Keeps image order: [main, wrist] if both provided; omits None slots.
        """
        text = self.build_text()
        image_list = [image for image in self.get_images() if image is not None]
        inputs = processor(text=[text], images=image_list)
        return inputs, text
