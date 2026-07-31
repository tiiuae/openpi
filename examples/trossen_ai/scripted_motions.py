"""Keyboard-triggered scripted motions for the bimanual WidowX AI.

These are canned, open-loop moves the operator can trigger by typing a keyword
while an episode is running — homing the arms, working the grippers, and small
"I'm alive" wiggles. They exist so the arm can be parked or demonstrated without
killing and relaunching the client.

Typed text is matched against an *exact* phrase table (after light
normalisation), never by substring. That matters: task instructions are natural
language and often contain command-ish words — ``"close the drawer"`` and
``"move the arm to the left"`` must reach the policy, not the gripper. Anything
that is not an exact match is treated as a task instruction.

Joint layout per arm (from ``WidowXAIFollowerConfig.joint_names``)::

    [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, left_carriage_joint]
       yaw                                          wrist    gripper, in METRES

A bimanual pose is the left arm's 7 values followed by the right arm's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)

JOINTS_PER_ARM = 7
BIMANUAL_DIM = 2 * JOINTS_PER_ARM

BASE_YAW_IDX = 0  # joint_0
WRIST_ROTATE_IDX = 5  # joint_5
GRIPPER_IDX = 6  # left_carriage_joint, in metres

ARMS = ("left", "right")
ARM_OFFSETS = {"left": 0, "right": JOINTS_PER_ARM}

# Trossen's own "staged" pose (WidowXAIFollowerConfig.staged_positions) — the pose
# the driver drives to on connect and just before sleeping, so it is a known-safe
# resting configuration. Radians, gripper in metres.
HOME_ARM_POSE = np.array([0.0, np.pi / 3, np.pi / 6, np.pi / 5, 0.0, 0.0, 0.0])

# Folded-down rest pose, all joints at zero — what the driver itself commands last
# on disconnect, and what sleep.py parks the arms at.
SLEEP_ARM_POSE = np.zeros(JOINTS_PER_ARM)

# Used only if the driver refuses to report its joint limits. The WidowX AI
# gripper is a linear carriage measured in metres, closed at 0.
FALLBACK_GRIPPER_RANGE = (0.0, 0.03)

HOME_DURATION_S = 4.0
SLEEP_DURATION_S = 4.0
GRIPPER_DURATION_S = 1.5
TWIST_AMPLITUDE_RAD = 0.5
TWIST_PERIOD_S = 2.0
TWIST_CYCLES = 2
WAVE_AMPLITUDE_RAD = 0.2
WAVE_PERIOD_S = 1.5
WAVE_CYCLES = 2


@dataclass(frozen=True)
class Command:
    """A recognised keyword command and the arms it applies to."""

    name: str
    arms: tuple[str, ...]

    @property
    def moves_arm(self) -> bool:
        """False for commands the control loop handles itself rather than the arm."""
        return self.name not in ("help", "quit")


def _build_phrase_table() -> dict[str, Command]:
    table: dict[str, Command] = {}

    def add(command: Command, *phrases: str) -> None:
        for phrase in phrases:
            table[phrase] = command

    add(Command("home", ARMS), "home", "go home", "return home", "move home")
    add(Command("sleep", ARMS), "sleep", "go sleep", "rest", "park")
    add(Command("quit", ()), "quit", "exit", "shutdown", "shut down", "disconnect")
    add(Command("open_gripper", ARMS), "open", "open gripper", "open grippers")
    add(Command("close_gripper", ARMS), "close", "close gripper", "close grippers")
    # "twist wrist left and right" normalises to "twist wrist left right" — it
    # describes the motion, not which arm, so it maps to both arms.
    add(
        Command("twist_wrist", ARMS),
        "twist",
        "twist wrist",
        "twist wrists",
        "twist wrist left right",
        "twist wrist right left",
    )
    add(Command("wave", ARMS), "wave", "wiggle", "shake")
    add(Command("help", ()), "help", "?", "commands")

    for arm in ARMS:
        add(Command("home", (arm,)), f"home {arm}", f"{arm} home", f"go home {arm}")
        add(Command("sleep", (arm,)), f"sleep {arm}", f"{arm} sleep", f"go sleep {arm}")
        add(Command("open_gripper", (arm,)), f"open {arm}", f"open {arm} gripper")
        add(Command("close_gripper", (arm,)), f"close {arm}", f"close {arm} gripper")
        add(Command("twist_wrist", (arm,)), f"twist {arm}", f"twist {arm} wrist")
        # Deliberately NOT "move {arm}": the default task prompt is
        # "move the arm to the left", which would normalise straight onto it.
        add(Command("wave", (arm,)), f"wave {arm}", f"wiggle {arm}", f"shake {arm}")

    return table


# Dropped before matching so "return to the home position" == "home". None of
# these carry meaning for a command, and dropping them cannot swallow a task
# instruction — matching is exact, so an unrecognised phrase still goes to the policy.
_FILLER_WORDS = frozenset(
    {"the", "a", "an", "to", "and", "please", "your", "its", "it", "arm", "arms", "position", "positions", "pos"}
)

_PHRASE_TABLE = _build_phrase_table()

HELP_ROWS = (
    ("home | go home", "both arms to the staged/home pose"),
    ("home left | home right", "one arm home"),
    ("sleep | rest | park", "home first, then fold both arms down to the zero pose"),
    ("sleep left | sleep right", "one arm to the sleep pose"),
    ("open | close", "both grippers"),
    ("open left | close right", "one gripper"),
    ("twist | twist wrist", "rotate both wrists left and right"),
    ("twist left | twist right", "one wrist"),
    ("wave | wiggle right | shake left", "small 'I'm alive' base movement"),
    ("quit | exit | disconnect", "park the arms, close the cameras and exit"),
    ("help | ?", "this list"),
)

HELP_TEXT = "\n".join(
    ["Keyword commands (anything else is sent to the policy as the task instruction):"]
    + [f"  {keys:34} — {effect}" for keys, effect in HELP_ROWS]
    + [
        "Any keyword command pauses the policy; type an instruction (or Enter alone",
        "for the default) to hand control back to it.",
    ]
)


def parse_command(text: str) -> Command | None:
    """Return the Command *text* names, or None if it is a task instruction."""
    words = "".join(c if c.isalnum() or c == "?" else " " for c in text.lower()).split()
    tokens = [word for word in words if word not in _FILLER_WORDS]
    return _PHRASE_TABLE.get(" ".join(tokens))


class ScriptedMotions:
    """Executes the canned motions, one blocking call per command.

    The caller must have stopped feeding the arm policy actions first — these
    moves stream their own targets at *control_frequency* and would otherwise
    fight the control loop.

    Args:
        get_pose:  Returns the current 14-dim measured joint pose.
        send_pose: Sends a 14-dim goal pose to the arm (honours test mode).
        control_frequency: Rate at which interpolated targets are streamed, Hz.
        joint_limits: (14, 2) array of [min, max] per joint from the driver, or
            None to fall back to conservative gripper values and no clamping.
        enabled_arms: Arms the operator allowed to move (``--use_left_arm_only``
            and friends). Commands for a disabled arm are refused, not silently
            re-targeted — the flag usually means that arm must not move at all.
    """

    def __init__(
        self,
        get_pose: Callable[[], np.ndarray],
        send_pose: Callable[[np.ndarray], None],
        control_frequency: int,
        joint_limits: np.ndarray | None = None,
        enabled_arms: tuple[str, ...] = ARMS,
    ) -> None:
        self._get_pose = get_pose
        self._send_pose = send_pose
        self._dt = 1.0 / control_frequency
        self._joint_limits = joint_limits
        self._enabled_arms = tuple(enabled_arms)

    # -- dispatch ----------------------------------------------------------

    def run(self, command: Command) -> bool:
        """Execute *command*. Returns True if the arm was actually moved."""
        if command.name == "help":
            logger.info(HELP_TEXT)
            return False

        arms = tuple(arm for arm in command.arms if arm in self._enabled_arms)
        if not arms:
            logger.warning(
                "Command '%s' targets %s, but only %s is enabled — ignoring",
                command.name,
                "/".join(command.arms),
                "/".join(self._enabled_arms) or "no arm",
            )
            return False

        logger.info("Running scripted motion '%s' on %s arm(s)", command.name, "+".join(arms))
        if command.name == "home":
            self.home(arms)
        elif command.name == "sleep":
            self.sleep(arms)
        elif command.name == "open_gripper":
            self.set_gripper(arms, opened=True)
        elif command.name == "close_gripper":
            self.set_gripper(arms, opened=False)
        elif command.name == "twist_wrist":
            self.twist_wrist(arms)
        elif command.name == "wave":
            self.wave(arms)
        else:
            logger.error("Unknown scripted motion '%s'", command.name)
            return False
        logger.info("Scripted motion '%s' done", command.name)
        return True

    # -- motions -----------------------------------------------------------

    def home(self, arms: tuple[str, ...]) -> None:
        goal = self._get_pose().copy()
        for arm in arms:
            offset = ARM_OFFSETS[arm]
            goal[offset : offset + JOINTS_PER_ARM] = HOME_ARM_POSE
        self._goto(goal, HOME_DURATION_S)

    def sleep(self, arms: tuple[str, ...]) -> None:
        """Park the arm folded down, going through home on the way.

        Same two-step sequence the driver runs on disconnect (staged, then all
        joints to zero): from an arbitrary pose a direct ramp to zero can drag
        the elbow through the workspace, while home is a known-safe waypoint.
        """
        self.home(arms)
        goal = self._get_pose().copy()
        for arm in arms:
            offset = ARM_OFFSETS[arm]
            goal[offset : offset + JOINTS_PER_ARM] = SLEEP_ARM_POSE
        self._goto(goal, SLEEP_DURATION_S)

    def set_gripper(self, arms: tuple[str, ...], *, opened: bool) -> None:
        goal = self._get_pose().copy()
        for arm in arms:
            index = ARM_OFFSETS[arm] + GRIPPER_IDX
            low, high = self._limits_for(index, FALLBACK_GRIPPER_RANGE)
            goal[index] = high if opened else low
        self._goto(goal, GRIPPER_DURATION_S)

    def twist_wrist(self, arms: tuple[str, ...]) -> None:
        self._oscillate(arms, WRIST_ROTATE_IDX, TWIST_AMPLITUDE_RAD, TWIST_PERIOD_S, TWIST_CYCLES)

    def wave(self, arms: tuple[str, ...]) -> None:
        self._oscillate(arms, BASE_YAW_IDX, WAVE_AMPLITUDE_RAD, WAVE_PERIOD_S, WAVE_CYCLES)

    # -- primitives --------------------------------------------------------

    def _goto(self, goal: np.ndarray, duration: float) -> None:
        """Stream a smooth ramp from the current pose to *goal*.

        PCHIP over [current, goal] — same scheme as move_to_start_position() and
        sleep.py — so the arm never sees a step change large enough to trip its
        velocity limits.
        """
        start = self._get_pose()
        goal = self._clamp(goal)
        interpolator = PchipInterpolator(np.array([0.0, duration]), np.array([start, goal]), axis=0)

        loop_start = time.perf_counter()
        elapsed = 0.0
        while elapsed < duration:
            step_start = time.perf_counter()
            self._send_pose(interpolator(elapsed))
            self._sleep_remaining(step_start)
            elapsed = time.perf_counter() - loop_start
        self._send_pose(goal)

    def _oscillate(self, arms: tuple[str, ...], joint_idx: int, amplitude: float, period: float, cycles: int) -> None:
        """Swing one joint per arm back and forth, ending exactly where it started."""
        start = self._get_pose()
        indices = [ARM_OFFSETS[arm] + joint_idx for arm in arms]
        duration = period * cycles

        loop_start = time.perf_counter()
        elapsed = 0.0
        while elapsed < duration:
            step_start = time.perf_counter()
            pose = start.copy()
            delta = amplitude * np.sin(2.0 * np.pi * elapsed / period)
            for index in indices:
                pose[index] = start[index] + delta
            self._send_pose(self._clamp(pose))
            self._sleep_remaining(step_start)
            elapsed = time.perf_counter() - loop_start
        self._send_pose(start)

    def _sleep_remaining(self, step_start: float) -> None:
        remaining = self._dt - (time.perf_counter() - step_start)
        if remaining > 0:
            time.sleep(remaining)

    def _limits_for(self, index: int, fallback: tuple[float, float]) -> tuple[float, float]:
        if self._joint_limits is None:
            return fallback
        return float(self._joint_limits[index, 0]), float(self._joint_limits[index, 1])

    def _clamp(self, pose: np.ndarray) -> np.ndarray:
        if self._joint_limits is None:
            return pose
        return np.clip(pose, self._joint_limits[:, 0], self._joint_limits[:, 1])
