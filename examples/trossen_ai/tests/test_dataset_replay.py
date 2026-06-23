"""EpisodeReader tests against the bundled converted_to_EE dataset (no hardware)."""
from pathlib import Path

import numpy as np
import pytest

from dataset_replay import EpisodeReader

# repo-root/converted_to_EE  (this file: repo/examples/trossen_ai/tests/)
DATASET_DIR = Path(__file__).resolve().parents[3] / "converted_to_EE"

pytestmark = pytest.mark.skipif(
    not (DATASET_DIR / "meta" / "info.json").is_file(),
    reason="converted_to_EE dataset not present",
)


def test_reader_reads_fps_and_episode_count():
    r = EpisodeReader(DATASET_DIR)
    assert r.fps == 50
    assert r.total_episodes == 250


def test_read_episode_shape_and_fps():
    ep = EpisodeReader(DATASET_DIR).read_episode(0)
    assert ep.ee_chunk16.ndim == 2
    assert ep.ee_chunk16.shape[1] == 16  # left8 | right8
    assert ep.n_frames == 676
    assert ep.fps == 50
    assert ep.ee_chunk16.dtype == np.float32


def test_episodes_are_distinct():
    r = EpisodeReader(DATASET_DIR)
    e0, e1 = r.read_episode(0), r.read_episode(1)
    assert not np.array_equal(e0.ee_chunk16, e1.ee_chunk16)


def test_unknown_episode_raises():
    with pytest.raises(ValueError):
        EpisodeReader(DATASET_DIR).read_episode(9999)
