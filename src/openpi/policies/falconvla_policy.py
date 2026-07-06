import logging
import time
from typing import Any, TypeAlias

import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

BasePolicy: TypeAlias = _base_policy.BasePolicy


class FalconVLAPolicy(BasePolicy):
    """Websocket policy wrapper around the in-process FalconVLA model.

    Unlike the standard `Policy`, FalconVLA does **not** run openpi's transform pipeline:
    the wrapped HuggingFace checkpoint owns its own tokenization and (un)normalization, so
    `model.inference` consumes the raw robot observation (`{state, images:{cam:CHW}, prompt}`)
    and returns already-unnormalized actions. This wrapper only handles the default-prompt
    fallback, reshaping, and timing.
    """

    def __init__(
        self,
        model: Any,
        *,
        default_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the FalconVLA policy.

        Args:
            model: The FalconVLA model to run inference with.
            default_prompt: Fallback instruction injected when `obs["prompt"]` is absent/empty.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device the model runs on (informational; the model loads itself
                onto its configured device).
        """
        self._model = model
        self._default_prompt = default_prompt
        self._metadata = metadata or {}
        self._pytorch_device = pytorch_device

        if hasattr(self._model, "eval"):
            self._model.eval()

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        start_time = time.monotonic()

        if self._default_prompt and not obs.get("prompt"):
            obs = {**obs, "prompt": self._default_prompt}

        try:
            outputs = self._model.inference(obs)
        except Exception:
            logging.exception("Error during FalconVLA model inference")
            raise

        # `model.inference` already coerces its output to (action_horizon, action_dim).
        actions = np.asarray(outputs)

        return {
            "actions": actions,
            "policy_timing": {
                "infer_ms": (time.monotonic() - start_time) * 1000,
            },
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
