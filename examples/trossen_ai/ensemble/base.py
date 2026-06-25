"""The narrow action-ensemble interface.

Every ensemble strategy blends overlapping chunk predictions for a timestep. The
control loop and the async worker depend only on this abstraction, never on a
concrete strategy (Dependency Inversion).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ActionEnsemble(ABC):
    @abstractmethod
    def add_chunk(self, query_step: int, chunk: np.ndarray) -> None:
        """Register a new predicted action chunk.

        Args:
            query_step: The episode step at which this chunk was queried.
            chunk:      Array of shape (T, D) — T predicted actions of dim D.
        """

    @abstractmethod
    def get_action(self, current_step: int) -> np.ndarray | None:
        """Return the blended action for *current_step*.

        Returns None if no predictions are available for that step.
        """

    @abstractmethod
    def get_overlap_count(self, current_step: int) -> int:
        """Return how many overlapping predictions are available for *current_step*."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state. Call at the beginning of each episode."""

    @abstractmethod
    def buffer_size(self) -> int:
        """Number of buffered items (for telemetry / memory-bound checks)."""
