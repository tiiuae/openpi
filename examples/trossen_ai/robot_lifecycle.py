"""Connecting to and releasing the arms without commanding a trajectory.

The lerobot Trossen driver folds motion into both ends of the lifecycle:

* ``WidowXAIFollower.connect()`` finishes with ``configure()``, which sets
  position mode and drives to ``staged_positions`` (blocking). So ``--mode
  test`` — documented as "no movement" — moves both arms before a single test
  action is logged, and ``capture_request.py`` does the same.
* ``WidowXAIFollower.disconnect()`` commands the staged pose and then all
  joints to zero before ``driver.cleanup()`` releases anything. Since every
  error path in the client ends at ``cleanup()``, a camera timeout, a malformed
  reply, a driver fault or Ctrl+C is answered with a *new* open-loop motion,
  from whatever pose the failure happened in, possibly holding something.

Neither is what the caller wanted in those moments, and both are avoidable:
``driver.cleanup()`` is the actual release, and the parking moves before it are
just two ``set_all_positions`` calls. This module separates the three things
that call conflated:

    connect_without_motion(robot)    powered and braked, no setpoints issued
    release_without_parking(robot)   let go, leaving the arm where it stands
    park_and_release(robot)          the old behaviour, now only when asked for

``Mode.idle`` is "all joints are braked" (trossen_arm SDK), not limp, so an
observation-only session holds its pose instead of dropping.

Releases are per resource and independently guarded: the bimanual driver
disconnects left, then right, then cameras, so one arm raising used to leave
the other arm and every camera held open, and the next run found the devices
busy.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import logging

import trossen_arm

logger = logging.getLogger(__name__)


def _arms(robot) -> list:
    """The individual arm objects of a bimanual robot, or the robot itself."""
    arms = [getattr(robot, name, None) for name in ("left_arm", "right_arm")]
    present = [arm for arm in arms if arm is not None]
    return present or [robot]


def _cameras(robot) -> dict:
    """Every camera the robot owns, including any held by its arms."""
    cameras = dict(getattr(robot, "cameras", {}) or {})
    for index, arm in enumerate(_arms(robot)):
        if arm is robot:
            continue
        for name, camera in (getattr(arm, "cameras", {}) or {}).items():
            cameras.setdefault(f"arm{index}:{name}", camera)
    return cameras


@contextmanager
def _braked_instead_of_staged(arm) -> Iterator[None]:
    """Replace one arm's ``configure()`` for the duration of a connect.

    ``configure()`` is the only part of ``connect()`` that moves the arm: it
    sets position mode and drives to the staged pose. Swapping just that method
    keeps the rest of the driver's connect sequence (SDK handshake, error
    clearing, camera setup) exactly as the vendor wrote it, instead of
    reimplementing it here and drifting from it.
    """
    original = arm.configure

    def brake_in_place() -> None:
        arm.driver.set_all_modes(trossen_arm.Mode.idle)

    arm.configure = brake_in_place
    try:
        yield
    finally:
        arm.configure = original


def connect_without_motion(robot) -> None:
    """Connect for observation only: powered, braked, never commanded to move.

    The arm can be read from (joint states and cameras) but is in idle mode, so
    it holds its current pose and no position goal is ever issued. Use for
    ``--mode test`` and for capturing observations.
    """
    arms = [arm for arm in _arms(robot) if arm is not robot]
    if not arms:
        raise RuntimeError("motion-free connect needs a Trossen arm robot with left_arm/right_arm")

    with _braked_instead_of_staged(arms[0]):
        if len(arms) == 1:
            robot.connect()
        else:
            with _braked_instead_of_staged(arms[1]):
                robot.connect()
    logger.info("Connected for observation only — arms are braked in place and will not be commanded")


def release_without_parking(robot) -> None:
    """Let go of the hardware, leaving the arms exactly where they are.

    For fault and interrupt paths: the reason the client is shutting down is
    usually a reason not to start a new open-loop motion. Every resource is
    released independently, so one failure cannot strand the others.
    """
    failures: list[str] = []

    for index, arm in enumerate(_arms(robot)):
        driver = getattr(arm, "driver", None)
        if driver is None:
            continue
        try:
            driver.cleanup()
        except Exception:
            failures.append(f"arm {index}")
            logger.warning("Could not release arm %d", index, exc_info=True)

    for name, camera in _cameras(robot).items():
        try:
            camera.disconnect()
        except Exception:
            failures.append(f"camera {name}")
            logger.warning("Could not release camera %s", name, exc_info=True)

    if failures:
        logger.error("Released with failures: %s — check the devices before the next run", ", ".join(failures))
    else:
        logger.info("Released the arms and cameras without commanding any motion")


def park_and_release(robot) -> None:
    """Drive both arms to the staged pose, then the folded pose, then release.

    This is the driver's own ``disconnect()`` and the right thing to do when the
    operator asked to finish — never as a reaction to a failure. Falls back to
    releasing without parking if the parking motion itself fails, so a fault
    during parking cannot leave the devices held open.
    """
    try:
        robot.disconnect()
    except Exception:
        logger.exception("Parking failed — releasing the hardware without it")
        release_without_parking(robot)
    else:
        logger.info("Arms parked and released")
