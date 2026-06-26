"""Read EE actions from a LeRobot v3.0 dataset, one episode at a time.

Pure data layer — no robot, no IK — so it is unit-testable off-hardware. The
replay entrypoint (``replay_ee_dataset.py``) feeds the returned ``(N, 16)``
absolute-EE chunk into the same IK decoder used by ``main_ee.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Absolute per-arm EE action columns (each 8-D: x,y,z,qw,qx,qy,qz,gripper),
# robot-base frame — the exact layout EEToJointsConverter.decode_chunk expects
# once the two arms are concatenated into a 16-D row.
EE_LEFT_COL = "action.ee_left"
EE_RIGHT_COL = "action.ee_right"
EE_DIM = 8


@dataclass(frozen=True)
class ReplayEpisode:
    """One episode's worth of EE action targets."""

    episode_index: int
    ee_chunk16: np.ndarray  # (N, 16) float32 — [left8 | right8], robot-base frame
    fps: int
    task: str | None

    @property
    def n_frames(self) -> int:
        return len(self.ee_chunk16)


class EpisodeReader:
    """Loads episodes from a LeRobot v3.0 dataset directory."""

    def __init__(self, dataset_dir: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        info_path = self.dataset_dir / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Not a LeRobot dataset (missing {info_path})")
        self._info = json.loads(info_path.read_text())
        self.fps = int(self._info["fps"])
        self.total_episodes = int(self._info.get("total_episodes", 0))

    def _data_files(self) -> list[Path]:
        files = sorted((self.dataset_dir / "data").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No data parquet files under {self.dataset_dir / 'data'}")
        return files

    def _task_for(self, df: pd.DataFrame) -> str | None:
        tasks_path = self.dataset_dir / "meta" / "tasks.parquet"
        if "task_index" not in df.columns or not tasks_path.is_file():
            return None
        tasks = pd.read_parquet(tasks_path)
        ti = int(df["task_index"].iloc[0])
        # tasks.parquet is indexed by task_index with a "task" column
        if "task" in tasks.columns:
            if ti in tasks.index:
                return str(tasks.loc[ti, "task"])
            if ti < len(tasks):
                return str(tasks.iloc[ti]["task"])
        return None

    def read_episode(self, episode_index: int) -> ReplayEpisode:
        """Return the absolute-EE action chunk for *episode_index*.

        Frames are ordered by ``frame_index``; the two per-arm 8-D EE columns are
        concatenated into a single ``(N, 16)`` chunk.
        """
        frames = []
        for f in self._data_files():
            df = pd.read_parquet(f)
            sub = df[df["episode_index"] == episode_index]
            if not sub.empty:
                frames.append(sub)
        if not frames:
            raise ValueError(f"Episode {episode_index} not found in dataset {self.dataset_dir}")

        df = pd.concat(frames).sort_values("frame_index")
        for col in (EE_LEFT_COL, EE_RIGHT_COL):
            if col not in df.columns:
                raise KeyError(f"Dataset is missing EE action column {col!r}")

        left = np.stack(df[EE_LEFT_COL].to_numpy())  # (N, 8)
        right = np.stack(df[EE_RIGHT_COL].to_numpy())  # (N, 8)
        if left.shape[1] != EE_DIM or right.shape[1] != EE_DIM:
            raise ValueError(f"Expected {EE_DIM}-D per-arm EE, got {left.shape[1]}/{right.shape[1]}")

        ee_chunk16 = np.hstack([left, right]).astype(np.float32)  # (N, 16)
        return ReplayEpisode(
            episode_index=episode_index,
            ee_chunk16=ee_chunk16,
            fps=self.fps,
            task=self._task_for(df),
        )
