"""Offline tests for interrupting a blocking motion with a typed stop.

Covers the latch in the prompt listener and the cancel checks in the motion
loops. numpy and pytest only — no terminal, no robot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripted_motions as sm  # noqa: E402
import terminal_ui  # noqa: E402

DIM = sm.BIMANUAL_DIM
CONTROL_HZ = 25


def _is_stop(text: str) -> bool:
    command = sm.parse_command(text)
    return command is not None and command.name in ("hold", "quit")


class FakeClock:
    """Time advances only when the motion loop sleeps, so a 4 s move is instant."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch):
    monkeypatch.setattr(sm, "time", FakeClock())


class Listener(terminal_ui.BasePromptListener):
    """The base listener's latch logic without a terminal attached."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def type(self, line: str) -> None:
        self._submit(line)


@pytest.mark.parametrize("line", ["stop", "hold", "freeze", "wait", "quit", "exit"])
def test_a_stop_line_latches_the_moment_it_is_typed(line):
    """It must not wait for the control loop to poll: the loop only polls
    between steps, and a blocking ramp is thousands of steps long."""
    listener = Listener("do the task", is_stop=_is_stop)
    listener.type(line)
    assert listener.stop_requested.is_set()


@pytest.mark.parametrize("line", ["pick up the blue cup", "home", "open", "close the drawer", ""])
def test_ordinary_lines_do_not_latch_a_stop(line):
    listener = Listener("do the task", is_stop=_is_stop)
    listener.type(line)
    assert not listener.stop_requested.is_set()


def test_the_line_is_still_delivered_to_the_loop():
    listener = Listener("do the task", is_stop=_is_stop)
    listener.type("stop")
    assert listener.poll() == "stop"


def test_the_latch_survives_until_it_is_cleared():
    """Latched, not a flag the reader resets: a stop must outlive however long
    the loop takes to come back and look."""
    listener = Listener("do the task", is_stop=_is_stop)
    listener.type("stop")
    listener.poll()
    assert listener.stop_requested.is_set()
    listener.clear_stop()
    assert not listener.stop_requested.is_set()


def test_a_listener_without_a_stop_predicate_still_works():
    listener = Listener("do the task")
    listener.type("stop")
    assert listener.poll() == "stop"
    assert not listener.stop_requested.is_set()


# -- the motion loops ----------------------------------------------------------


class FakeArm:
    def __init__(self):
        self.pose = np.zeros(DIM)
        self.sent: list[np.ndarray] = []

    def get_pose(self):
        return self.pose.copy()

    def send_pose(self, pose):
        self.sent.append(np.asarray(pose, dtype=float).copy())
        self.pose = self.sent[-1]


def _motions(arm, should_cancel=None):
    return sm.ScriptedMotions(
        get_pose=arm.get_pose,
        send_pose=arm.send_pose,
        control_frequency=CONTROL_HZ,
        joint_limits=np.tile(np.array([-4.0, 4.0]), (DIM, 1)),
        should_cancel=should_cancel,
    )


def test_a_move_runs_to_completion_when_nothing_cancels():
    arm = FakeArm()
    _motions(arm, lambda: False).home(sm.ARMS)
    np.testing.assert_allclose(arm.sent[-1][: sm.JOINTS_PER_ARM], sm.HOME_ARM_POSE)


def test_a_cancel_stops_a_move_part_way():
    arm = FakeArm()
    cancel_after = 10

    def should_cancel():
        return len(arm.sent) >= cancel_after

    _motions(arm, should_cancel).home(sm.ARMS)
    assert len(arm.sent) == cancel_after
    # Abandoned part way is exactly "stop here" for an open-loop move.
    assert not np.allclose(arm.sent[-1][: sm.JOINTS_PER_ARM], sm.HOME_ARM_POSE)


def test_a_cancel_before_the_first_tick_moves_nothing():
    arm = FakeArm()
    _motions(arm, lambda: True).home(sm.ARMS)
    assert arm.sent == []


def test_a_cancelled_sleep_does_not_start_its_second_leg():
    """sleep() is home-then-fold; a stop during the first leg must not be
    followed by a ramp down to the folded pose."""
    arm = FakeArm()
    _motions(arm, lambda: len(arm.sent) >= 5).sleep(sm.ARMS)
    assert len(arm.sent) == 5
    assert not np.allclose(arm.sent[-1], np.zeros(DIM))


def test_an_oscillation_can_be_cancelled():
    arm = FakeArm()
    _motions(arm, lambda: len(arm.sent) >= 8).wave(sm.ARMS)
    assert len(arm.sent) == 8


def test_the_latch_and_the_motion_loop_work_together():
    """End to end: the operator types stop mid-move and the arm stops within a
    tick, instead of after the whole move has played out."""
    listener = Listener("do the task", is_stop=_is_stop)
    arm = FakeArm()
    typed_at = 12

    def should_cancel():
        if len(arm.sent) == typed_at:
            listener.type("stop")
        return listener.stop_requested.is_set()

    _motions(arm, should_cancel).home(sm.ARMS)
    assert len(arm.sent) == typed_at
