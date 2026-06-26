"""Hold-last-action fallback for steps with no ensemble prediction.

Commanding zeros on a missing prediction is unsafe (zero is a specific pose, not
"stay put"). This repeats the last real action instead; before any real action
exists it returns None so the caller can skip the step.
"""

from __future__ import annotations

import numpy as np


class HoldLastAction:
    def __init__(self) -> None:
        self._last: np.ndarray | None = None

    def resolve(self, a_t: np.ndarray | None) -> np.ndarray | None:
        if a_t is not None:
            self._last = np.asarray(a_t).copy()
            return a_t
        return self._last.copy() if self._last is not None else None

    def reset(self) -> None:
        self._last = None
