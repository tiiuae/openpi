"""Per-episode action-overlap logger.

Records how many overlapping predictions the ensemble used at each step and dumps
a JSON file at episode end (or on manual stop / Ctrl+C). Independent of any
ensemble strategy.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ActionLogger:
    """Records the number of overlapping predictions used at each episode step
    and saves a JSON file at episode end (or on manual stop / Ctrl+C).

    Call ``save()`` inside a ``finally`` block so it fires on normal
    completion, timeout, KeyboardInterrupt, and 'r'-restart alike.

    Usage::

        logger = ActionLogger(save_dir="./action_logs")
        logger.reset()                             # start of episode
        logger.log(step, ensemble.get_overlap_count(step))  # each step
        logger.save(tag="episode_0")               # end / interrupt

    Output JSON structure::

        {
            "tag": "episode_243steps",
            "total_steps": 243,
            "steps": [
                {"step": 0, "n_overlaps": 1},
                {"step": 1, "n_overlaps": 3},
                ...
            ]
        }
    """

    def __init__(self, save_dir: str = "./action_logs") -> None:
        self._save_dir = Path(save_dir)
        self._records: list[dict] = []

    def reset(self) -> None:
        self._records.clear()

    def log(self, step: int, n_overlaps: int) -> None:
        self._records.append({"step": step, "n_overlaps": n_overlaps})

    def save(self, tag: str = "episode") -> None:
        if not self._records:
            logger.info("ActionLogger: nothing to save.")
            return

        self._save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        path = self._save_dir / f"{tag}_{timestamp}.json"

        payload = {
            "tag": tag,
            "total_steps": len(self._records),
            "steps": self._records,
        }
        path.write_text(json.dumps(payload, indent=2))
        logger.info("ActionLogger: saved to %s", path)
