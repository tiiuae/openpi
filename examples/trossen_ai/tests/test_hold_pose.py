"""Offline tests for what the control loop commands when it has no action.

``_hold_pose`` lives on the hardware-bound bridge class, which is why it went
untested and shipped referencing a constant that did not exist: the first
starved step raised ``NameError`` instead of holding position. The bridge is
built here with ``__new__`` and only the attributes ``_hold_pose`` reads, so no
robot, camera or policy server is involved.

Needs the client venv (``main`` imports lerobot): run from examples/trossen_ai
with ``.venv/bin/python -m pytest tests/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

main = pytest.importorskip("main", reason="needs the client venv (lerobot)")

DIM = 14
CONTROL_HZ = 25


class Recorder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


class FakeEnsemble:
    def __init__(self) -> None:
        self.reset = Recorder()


class FakeWorker:
    def __init__(self) -> None:
        self.flush = Recorder()


class FakeRobot:
    def __init__(self, pose: np.ndarray) -> None:
        self.pose = pose

    def get_observation(self) -> dict:
        return {f"joint_{i}.pos": value for i, value in enumerate(self.pose)}


def _bridge(last_action=None, measured=None, *, async_inference=True):
    bridge = main.TrossenOpenPIBridge.__new__(main.TrossenOpenPIBridge)
    bridge.control_frequency = CONTROL_HZ
    bridge.dt = 1.0 / CONTROL_HZ
    bridge.async_inference = async_inference
    bridge.ensemble = FakeEnsemble()
    bridge._policy_worker = FakeWorker()
    bridge._starved_steps = 0
    bridge._last_action = None if last_action is None else np.asarray(last_action, dtype=float)
    bridge.robot = FakeRobot(np.full(DIM, 0.7) if measured is None else np.asarray(measured, dtype=float))
    return bridge


BUDGET_STEPS = int(main.STARVED_HOLD_BUDGET_S * CONTROL_HZ)


def test_a_starved_step_holds_the_last_commanded_pose_not_zero():
    """The bug this whole branch exists for: zero is the folded rest pose with
    both grippers shut, not "do nothing"."""
    last = np.linspace(-1.0, 1.0, DIM)
    bridge = _bridge(last)
    pose, paused = bridge._hold_pose(paused=False)
    np.testing.assert_array_equal(pose, last)
    assert not np.allclose(pose, 0.0)
    assert paused is False


def test_with_no_commanded_pose_yet_it_holds_the_measured_pose():
    bridge = _bridge(last_action=None, measured=np.full(DIM, 0.3))
    pose, _ = bridge._hold_pose(paused=False)
    np.testing.assert_allclose(pose, np.full(DIM, 0.3))


def test_it_keeps_holding_inside_the_budget():
    bridge = _bridge(np.ones(DIM))
    for _ in range(BUDGET_STEPS - 1):
        _, paused = bridge._hold_pose(paused=False)
        assert paused is False


def test_starvation_past_the_budget_pauses_the_loop():
    """Regression: this is the call that raised NameError on
    STARVED_HOLD_BUDGET_S the first time the ensemble ran dry for a moment."""
    bridge = _bridge(np.ones(DIM))
    paused = False
    for _ in range(BUDGET_STEPS):
        pose, paused = bridge._hold_pose(paused=paused)
    assert paused is True
    np.testing.assert_array_equal(pose, np.ones(DIM))


def test_pausing_throws_away_predictions_made_for_the_stale_pose():
    bridge = _bridge(np.ones(DIM))
    for _ in range(BUDGET_STEPS):
        bridge._hold_pose(paused=False)
    assert bridge.ensemble.reset.calls == 1
    assert bridge._policy_worker.flush.calls == 1


def test_sync_mode_pauses_without_touching_a_worker_it_does_not_have():
    bridge = _bridge(np.ones(DIM), async_inference=False)
    bridge._policy_worker = None
    for _ in range(BUDGET_STEPS):
        _, paused = bridge._hold_pose(paused=False)
    assert paused is True


def test_the_budget_constant_is_defined_and_sensible():
    assert 0.0 < main.STARVED_HOLD_BUDGET_S <= 2.0
