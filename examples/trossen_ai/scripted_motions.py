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

import motion_limits
import numpy as np

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

# Floors, not fixed durations. How long a move actually takes is derived from how
# far it has to travel and the driver's velocity limit for the slowest joint
# involved (see motion_limits.ramp_duration): a canned "4 seconds to home" is only
# safe for the move it happened to be tuned on, and is over the limit from a far
# enough starting pose.
MIN_HOME_DURATION_S = 4.0
MIN_SLEEP_DURATION_S = 4.0
MIN_GRIPPER_DURATION_S = 1.5
# A derived duration is never capped (that would put back the over-speed it
# exists to prevent), but past this it is worth telling the operator why the arm
# is going to crawl for so long.
LONG_MOVE_WARNING_S = 15.0
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
        return self.name not in ("help", "quit", "record", "save", "reject")


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
    # Freeze in place: takes the pause path (policy stopped, ensemble flushed)
    # but performs no motion, so the arm keeps its current pose instead of
    # driving home. "stop" belongs here so a spoken stop can never reach the
    # policy as a task instruction.
    add(Command("hold", ARMS), "hold", "hold on", "hang on", "freeze", "pause", "stop", "wait")
    # Episode recording (episode_recorder.py): control-loop commands like
    # help/quit — the loop handles them, the arm never moves.
    add(Command("record", ()), "record", "start record", "start recording", "record episode")
    add(Command("save", ()), "save", "save episode", "keep", "keep episode")
    add(Command("reject", ()), "reject", "reject episode", "discard", "discard episode", "drop episode")
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
    ("hold | freeze | stop | wait", "pause the policy; the arm freezes in place (ends a recording take)"),
    ("record", "start recording an episode (needs --record_dir)"),
    ("save | keep", "save the recorded take into the dataset"),
    ("reject | discard", "throw the recorded take away"),
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
        velocity_limits: (14,) array of max joint speeds from the driver, used to
            size each move's duration. None falls back to the fixed floors, which
            is only as safe as the move they were tuned for.
        enabled_arms: Arms the operator allowed to move (``--use_left_arm_only``
            and friends). Commands for a disabled arm are refused, not silently
            re-targeted — the flag usually means that arm must not move at all.
        should_cancel: Checked before every streamed target. These moves block
            the control loop for seconds at a time, so without it a typed stop
            cannot be seen until the move has finished running.
    """

    def __init__(
        self,
        get_pose: Callable[[], np.ndarray],
        send_pose: Callable[[np.ndarray], None],
        control_frequency: int,
        joint_limits: np.ndarray | None = None,
        velocity_limits: np.ndarray | None = None,
        enabled_arms: tuple[str, ...] = ARMS,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        self._get_pose = get_pose
        self._send_pose = send_pose
        self._dt = 1.0 / control_frequency
        self._joint_limits = joint_limits
        self._velocity_limits = velocity_limits
        self._enabled_arms = tuple(enabled_arms)
        self._should_cancel = should_cancel

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
        elif command.name == "hold":
            # No motion on purpose: the caller has already paused the policy and
            # flushed the ensemble, and the arm holds its last commanded pose.
            pass
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
        self._goto(goal, MIN_HOME_DURATION_S)

    def sleep(self, arms: tuple[str, ...]) -> None:
        """Park the arm folded down, going through home on the way.

        Same two-step sequence the driver runs on disconnect (staged, then all
        joints to zero): from an arbitrary pose a direct ramp to zero can drag
        the elbow through the workspace, while home is a known-safe waypoint.
        """
        self.home(arms)
        if self._cancelled():
            # Cancelled during the first leg: do not start the second one.
            return
        goal = self._get_pose().copy()
        for arm in arms:
            offset = ARM_OFFSETS[arm]
            goal[offset : offset + JOINTS_PER_ARM] = SLEEP_ARM_POSE
        self._goto(goal, MIN_SLEEP_DURATION_S)

    def set_gripper(self, arms: tuple[str, ...], *, opened: bool) -> None:
        goal = self._get_pose().copy()
        for arm in arms:
            index = ARM_OFFSETS[arm] + GRIPPER_IDX
            low, high = self._limits_for(index, FALLBACK_GRIPPER_RANGE)
            goal[index] = high if opened else low
        self._goto(goal, MIN_GRIPPER_DURATION_S)

    def twist_wrist(self, arms: tuple[str, ...]) -> None:
        self._oscillate(arms, WRIST_ROTATE_IDX, TWIST_AMPLITUDE_RAD, TWIST_PERIOD_S, TWIST_CYCLES)

    def wave(self, arms: tuple[str, ...]) -> None:
        self._oscillate(arms, BASE_YAW_IDX, WAVE_AMPLITUDE_RAD, WAVE_PERIOD_S, WAVE_CYCLES)

    def _cancelled(self) -> bool:
        """True once the operator has asked the arm to stop.

        The arm holds wherever the move had reached: a canned motion is
        open-loop, so abandoning it part way is exactly "stop here".
        """
        if self._should_cancel is None or not self._should_cancel():
            return False
        logger.info("Scripted motion cancelled — holding position")
        return True

    # -- primitives --------------------------------------------------------

    def _goto(self, goal: np.ndarray, min_duration: float) -> None:
        """Stream a smooth ramp from the current pose to *goal*.

        The duration is derived from the distance and the driver's velocity
        limits, with *min_duration* as a floor — a "home" from an extended pose
        takes longer than one from a pose already near home, instead of both
        being crammed into the same fixed number of seconds.

        The profile is minimum-jerk rather than the two-point PCHIP this used to
        use: scipy special-cases PCHIP over two points to a straight line, which
        commands the arm to jump from rest to full speed and back again. Bounded
        velocity, unbounded acceleration.
        """
        start = self._get_pose()
        goal = self._clamp(goal)
        duration = motion_limits.ramp_duration(start, goal, self._velocity_limits, minimum=min_duration)
        if duration >= LONG_MOVE_WARNING_S:
            logger.warning(
                "This move needs %.0fs to stay inside the driver's velocity limits — the arm will move slowly",
                duration,
            )
        elif duration > min_duration + 1e-6:
            logger.info("Move sized to %.1fs by the driver's velocity limits (floor %.1fs)", duration, min_duration)

        loop_start = time.perf_counter()
        elapsed = 0.0
        while elapsed < duration:
            if self._cancelled():
                return
            step_start = time.perf_counter()
            self._send_pose(motion_limits.minimum_jerk_pose(start, goal, elapsed, duration))
            self._sleep_remaining(step_start)
            elapsed = time.perf_counter() - loop_start
        self._send_pose(goal)

    def _oscillate(self, arms: tuple[str, ...], joint_idx: int, amplitude: float, period: float, cycles: int) -> None:
        """Swing one joint per arm back and forth, ending exactly where it started.

        A*sin(2*pi*t/P) peaks at A*2*pi/P, so the period is stretched if that
        would be too fast for the slowest joint involved.
        """
        start = self._get_pose()
        indices = [ARM_OFFSETS[arm] + joint_idx for arm in arms]
        period = motion_limits.safe_oscillation_period(amplitude, period, indices, self._velocity_limits)
        duration = period * cycles

        loop_start = time.perf_counter()
        elapsed = 0.0
        while elapsed < duration:
            if self._cancelled():
                return
            step_start = time.perf_counter()
            pose = start.copy()
            delta = amplitude * np.sin(2.0 * np.pi * elapsed / period)
            for index in indices:
                pose[index] = start[index] + delta
            self._send_pose(self._clamp(pose))
            self._sleep_remaining(step_start)
            elapsed = time.perf_counter() - loop_start
        # Return to the exact starting pose. Clamped like every other sample:
        # the measured pose it came from can sit a hair outside the limits.
        self._send_pose(self._clamp(start))

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
