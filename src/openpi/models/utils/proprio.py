# import Any
from typing import Any, Dict
import numpy as np
import jax.numpy as jnp
from openpi.models.utils.prompt_builder import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType


def fetch_proprio_stats(falconvla_model: Any, unnorm_key: str) -> dict:
    """Fetch the proprio stats from the model

    Args:
        falconvla_model (Any): FalconVLAModel Object
        unnorm_key (str): unnorm_key e.g. (libero, burger_270_episodes...)

    Returns:
        dict: A Dict containing the norm stats 
    """
    try:
        proprio_stats = falconvla_model.norm_stats[unnorm_key]['proprio']
        return proprio_stats
    except KeyError:
        raise ValueError(f"Could not find proprio stats for unnorm_key: {unnorm_key}\n\r"\
                         f"Available keys: {list(falconvla_model.norm_stats.keys())}")


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

    target_dim = proprio_low.shape[0]
    if proprio.shape[0] < target_dim:
        proprio = np.concatenate([proprio, np.zeros(target_dim - proprio.shape[0], dtype=proprio.dtype)])
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

def tokenize_proprio(x: float | np.ndarray, num_bins: int = 256,) -> list[str]:
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

def get_proprio_tokens(tokenized_prioprio_tokens: list[str]) -> str:
    """Get the tokenized proprio tokens as a single string

    Args:
        tokenized_prioprio_tokens (list[str]): tokenized proprio tokens

    Returns:
        str: A string of the tokenized proprio tokens
    """
    return "".join(tokenized_prioprio_tokens)