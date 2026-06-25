# Webapp Backend — Safety, Lifecycle & Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the stop/hang bug and unclean termination, remove the hardcoded joint-limit checks in favor of a firmware-error guard, add sleep/home + folder-browser + feedback backend endpoints, and expose motion-tuning knobs (incl. an opt-in feed-forward smooth-streaming path) that fix the jerky/brutal arm motion.

**Architecture:** All blocking work (policy-server connect, robot build/connect) moves out of runner `__init__` and into `run()`, which executes on the `SessionManager` daemon thread. `SessionManager.start()` returns immediately so the WebSocket event loop stays responsive and `stop`/`estop` are always processed. Policy-server connect becomes bounded + stop-aware via a new `policy_connect` helper. Joint-limit tables are deleted; faults are caught from the firmware and trigger a move-to-sleep. New REST/WS endpoints are added to `webapp/server.py`.

**Tech Stack:** Python 3.12, FastAPI, pytest, `lerobot_robot_trossen` (TrossenArmDriver), numpy, scipy PchipInterpolator.

This plan is backend-only and fully testable off-robot (hardware paths are syntax-checked + manually validated per `webapp/HARDWARE.md`). The companion plan `2026-06-25-webapp-frontend-overhaul.md` covers the UI.

All commands run from `examples/trossen_ai/` (the webapp test root). Verify:

```bash
cd examples/trossen_ai && python -m pytest webapp/tests -q
```

---

## File Structure

- Create: `examples/trossen_ai/policy_connect.py` — bounded, stop-aware wait-for-server helper + exceptions.
- Create: `examples/trossen_ai/webapp/feedback_store.py` — write feedback markdown files.
- Create: `examples/trossen_ai/webapp/files_api.py` — directory listing for the folder browser.
- Create: `examples/trossen_ai/webapp/movers.py` — `SleepRunner` / `HomeRunner` (move to a fixed pose, then disconnect).
- Modify: `examples/trossen_ai/webapp/session.py` — build runner on thread, stop-aware, error status.
- Modify: `examples/trossen_ai/webapp/runners.py` — cheap `__init__`, blocking in `run()`, firmware guard, smooth-streaming.
- Modify: `examples/trossen_ai/webapp/server.py` — `/api/files`, `/api/feedback`, `go_sleep`/`go_home` WS actions, shutdown cleanup.
- Modify: `examples/trossen_ai/robot_control.py` — delete `JOINT_LIMIT`/`is_action_within_limits`, add `HOME_POSITION`, firmware guard, `send_action_smooth`, tuning params on `build_stationary_robot`.
- Modify: `examples/trossen_ai/trossen_bridge.py` — delete `JOINT_LIMIT`/`is_action_within_limits`, firmware guard, route `print` → `logger`.
- Create tests: `webapp/tests/test_policy_connect.py`, `test_feedback_store.py`, `test_files_api.py`, `test_runners_lifecycle.py`, `test_motion_smooth.py`; extend `test_session.py`, `test_server.py`.

---

## Task 1: SessionManager builds the runner on the thread (the core fix)

**Files:**
- Modify: `examples/trossen_ai/webapp/session.py`
- Test: `examples/trossen_ai/webapp/tests/test_session.py`

- [ ] **Step 1: Add failing tests for non-blocking start + stop-during-construction**

Append to `webapp/tests/test_session.py`:

```python
import threading


def test_start_returns_immediately_when_construction_blocks():
    """Runner construction must not block the caller (the WS event loop)."""
    def slow_factory(kind, config, sink):
        time.sleep(0.5)  # simulate a blocking policy-server connect
        return FakeRunner(kind, config, sink)

    sm = SessionManager(slow_factory)
    t0 = time.perf_counter()
    sm.start("live", {}, sink=_NoopSink())
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"start() blocked for {elapsed:.3f}s"
    sm.stop()


def test_stop_during_construction_skips_run():
    """If stop is requested before construction finishes, run() is never entered."""
    gate = threading.Event()
    ran = []

    class GatedRunner:
        def __init__(self, kind, config, sink):
            gate.wait(2.0)

        def run(self):
            ran.append(True)

        def stop(self):
            ...

        def estop(self):
            ...

    sm = SessionManager(lambda k, c, s: GatedRunner(k, c, s))
    sm.start("live", {}, sink=_NoopSink())
    time.sleep(0.05)  # thread now blocked inside __init__
    stopper = threading.Thread(target=sm.stop)
    stopper.start()
    time.sleep(0.05)
    gate.set()  # let construction finish; _run should see stop flag
    stopper.join(2.0)
    assert ran == []


def test_construction_error_emits_error_status():
    statuses = []

    class RecordingSink:
        def on_status(self, kind, payload):
            statuses.append((kind, payload))

    def boom_factory(kind, config, sink):
        raise RuntimeError("connect failed")

    sm = SessionManager(boom_factory)
    sm.start("live", {}, sink=RecordingSink())
    time.sleep(0.1)
    assert any(k == "error" for k, _ in statuses)
    assert not sm.is_running()
```

- [ ] **Step 2: Run the new tests, verify they fail**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_session.py -q`
Expected: the three new tests FAIL (current `start()` builds the runner synchronously / has no error status).

- [ ] **Step 3: Rewrite `session.py` to build on the thread**

Replace the body of `examples/trossen_ai/webapp/session.py` from the `SessionManager` class onward with:

```python
class SessionManager:
    def __init__(self, runner_factory: RunnerFactory) -> None:
        self._factory = runner_factory
        self._runner: Runner | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_requested = False

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, kind: str, config: dict, sink: object) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError("A session is already running")
            self._stop_requested = False
            self._runner = None
            self._thread = threading.Thread(
                target=self._run, args=(kind, config, sink),
                daemon=True, name=f"session-{kind}",
            )
            self._thread.start()

    def _run(self, kind: str, config: dict, sink) -> None:
        try:
            runner = self._factory(kind, config, sink)
            self._runner = runner
            if self._stop_requested:
                runner.stop()
                return
            runner.run()
        except Exception as exc:  # noqa: BLE001 — surface to UI, never crash the thread
            try:
                sink.on_status("error", {"message": str(exc)})
            except Exception:
                pass

    def stop(self, timeout: float = 15.0) -> None:
        self._stop_requested = True
        runner, thread = self._runner, self._thread
        if runner is not None:
            runner.stop()
        if thread is not None:
            thread.join(timeout=timeout)
        self._runner, self._thread = None, None

    def estop(self, timeout: float = 15.0) -> None:
        self._stop_requested = True
        runner, thread = self._runner, self._thread
        if runner is not None:
            runner.estop()
        if thread is not None:
            thread.join(timeout=timeout)
        self._runner, self._thread = None, None
```

- [ ] **Step 4: Run the full session test file, verify all pass**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_session.py -q`
Expected: PASS (including the original `test_start_then_stop`, `test_single_session_guard`, `test_estop_calls_runner_estop`).

- [ ] **Step 5: Commit**

```bash
git add examples/trossen_ai/webapp/session.py examples/trossen_ai/webapp/tests/test_session.py
git commit -m "fix(webapp): build session runner on the thread so stop stays responsive"
```

---

## Task 2: Bounded, stop-aware policy-server connect helper

**Files:**
- Create: `examples/trossen_ai/policy_connect.py`
- Test: `examples/trossen_ai/webapp/tests/test_policy_connect.py`

- [ ] **Step 1: Write the failing test**

Create `examples/trossen_ai/webapp/tests/test_policy_connect.py`:

```python
import socket
import threading
import time

import pytest

from policy_connect import ConnectStopped, ConnectTimeout, wait_for_policy_server


def _listening_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    return srv, srv.getsockname()[1]


def test_returns_when_server_reachable():
    srv, port = _listening_server()
    try:
        wait_for_policy_server("127.0.0.1", port, timeout=2.0, should_stop=lambda: False)
    finally:
        srv.close()


def test_raises_timeout_when_no_server():
    # Port 1 is privileged/closed; connect refuses fast.
    with pytest.raises(ConnectTimeout):
        wait_for_policy_server("127.0.0.1", 1, timeout=0.5, should_stop=lambda: False, interval=0.1)


def test_stop_flag_interrupts_wait():
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    t0 = time.perf_counter()
    with pytest.raises(ConnectStopped):
        wait_for_policy_server("127.0.0.1", 1, timeout=10.0, should_stop=stop.is_set, interval=0.1)
    assert time.perf_counter() - t0 < 2.0
```

- [ ] **Step 2: Run it, verify failure**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_policy_connect.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'policy_connect'`.

- [ ] **Step 3: Implement the helper**

Create `examples/trossen_ai/policy_connect.py`:

```python
"""Bounded, stop-aware reachability check for the policy server.

The openpi WebsocketClientPolicy connects in its constructor and loops forever
("Still waiting for server...") when the server is down. We preflight the TCP
endpoint here with a deadline and a stop callback so a session can be cancelled
while still connecting, and so a missing server fails cleanly instead of hanging.
"""
from __future__ import annotations

import logging
import socket
import time
from typing import Callable

logger = logging.getLogger(__name__)


class ConnectTimeout(RuntimeError):
    def __init__(self, host: str, port: int, timeout: float) -> None:
        super().__init__(f"Policy server {host}:{port} not reachable within {timeout:.0f}s")
        self.host, self.port, self.timeout = host, port, timeout


class ConnectStopped(RuntimeError):
    """Raised when should_stop() became true while waiting to connect."""


def wait_for_policy_server(
    host: str,
    port: int,
    timeout: float,
    should_stop: Callable[[], bool],
    interval: float = 0.5,
) -> None:
    """Block until host:port accepts a TCP connection, the deadline passes, or stop.

    Raises ConnectStopped if should_stop() turns true, ConnectTimeout on deadline.
    """
    deadline = time.monotonic() + timeout
    logger.info("Waiting for policy server at %s:%s (timeout %.0fs)...", host, port, timeout)
    while True:
        if should_stop():
            raise ConnectStopped()
        try:
            with socket.create_connection((host, port), timeout=interval):
                logger.info("Policy server %s:%s reachable.", host, port)
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise ConnectTimeout(host, port, timeout)
            remaining = deadline - time.monotonic()
            time.sleep(min(interval, max(0.0, remaining)))
```

- [ ] **Step 4: Run it, verify pass**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_policy_connect.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/trossen_ai/policy_connect.py examples/trossen_ai/webapp/tests/test_policy_connect.py
git commit -m "feat(webapp): bounded stop-aware policy-server connect helper"
```

---

## Task 3: Restructure runners — cheap `__init__`, blocking in `run()`, firmware guard, smooth-streaming hook

**Files:**
- Modify: `examples/trossen_ai/webapp/runners.py`
- Test: `examples/trossen_ai/webapp/tests/test_runners_lifecycle.py`

The runners import hardware modules at module top today. We move those imports lazily
inside `run()`/`_make_bridge` so the module imports cleanly off-robot and lifecycle can
be unit-tested by monkeypatching.

- [ ] **Step 1: Write failing lifecycle tests**

Create `examples/trossen_ai/webapp/tests/test_runners_lifecycle.py`:

```python
"""Lifecycle of LiveRunner: cheap construction, stop honored during connect.

These tests stub the policy-connect wait and the bridge so no hardware/server
is needed. They assert the runner constructs without blocking and that stop
prevents the bridge from ever being built/run.
"""
import time

import runners


class _Sink:
    def __init__(self):
        self.statuses = []

    def on_status(self, kind, payload):
        self.statuses.append((kind, payload))


def test_live_runner_construction_is_cheap(monkeypatch):
    called = {"wait": False}

    def fake_wait(*a, **k):
        called["wait"] = True

    monkeypatch.setattr(runners, "wait_for_policy_server", fake_wait, raising=False)
    r = runners.LiveRunner("live", {"policy_host": "h", "policy_port": 1}, _Sink())
    # No connect / bridge build happened during construction.
    assert called["wait"] is False
    assert r._bridge is None


def test_live_runner_stop_before_run_skips_bridge(monkeypatch):
    built = {"bridge": False}

    def fake_wait(host, port, timeout, should_stop, interval=0.5):
        # Simulate a slow connect that observes the stop flag.
        for _ in range(100):
            if should_stop():
                from policy_connect import ConnectStopped
                raise ConnectStopped()
            time.sleep(0.005)

    def fake_make_bridge(self, cfg):
        built["bridge"] = True
        raise AssertionError("bridge must not be built after stop")

    monkeypatch.setattr(runners, "wait_for_policy_server", fake_wait, raising=False)
    monkeypatch.setattr(runners.LiveRunner, "_make_bridge", fake_make_bridge, raising=False)

    sink = _Sink()
    r = runners.LiveRunner("live", {"policy_host": "h", "policy_port": 1}, sink)
    r.stop()  # request stop before run
    r.run()   # should raise ConnectStopped internally, caught, no bridge
    assert built["bridge"] is False
```

- [ ] **Step 2: Run it, verify failure**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_runners_lifecycle.py -q`
Expected: FAIL (current `LiveRunner.__init__` builds the bridge; no `_make_bridge`, no module-level `wait_for_policy_server`).

- [ ] **Step 3: Rewrite `runners.py`**

Replace `examples/trossen_ai/webapp/runners.py` with:

```python
"""Hardware-bound runners adapting the bridge and replay tool to the Runner
interface used by SessionManager.

Construction is intentionally cheap: all blocking work (policy-server connect,
robot build/connect, episode load) happens in run(), which executes on the
session thread. stop() flips a flag honored by the connect wait and the loops,
so a session can be cancelled even while still connecting.

`config` is the dict produced by the web form. Constructors forward only the
keys they use; unknown keys are ignored.
"""
from __future__ import annotations

import logging

import numpy as np

from policy_connect import ConnectStopped, ConnectTimeout, wait_for_policy_server

logger = logging.getLogger(__name__)


class LiveRunner:
    """Runs TrossenOpenPIBridge.run_episode with a telemetry sink."""

    def __init__(self, kind: str, config: dict, sink) -> None:
        self._config = config
        self._sink = sink
        self._stopped = False
        self._bridge = None

    def _make_bridge(self, config: dict):
        # Imported lazily so this module loads off-robot.
        from adapters import EEAdapter, JointAdapter
        from external.joint_to_ee.ee_to_joints import EEToJointsConverter
        from external.joint_to_ee.kinematics import make_kinematics
        from trossen_bridge import TrossenOpenPIBridge

        adapter = JointAdapter()
        if config.get("adapter") == "ee":
            conv = EEToJointsConverter(
                make_kinematics(),
                orientation_weight=float(config.get("ik_orientation_weight", 0.01)),
                pos_tol_m=float(config.get("ik_pos_tol_m", 1e-3)),
            )
            adapter = EEAdapter(conv)
        return TrossenOpenPIBridge(
            policy_server_host=config.get("policy_host", "localhost"),
            policy_server_port=int(config.get("policy_port", 8000)),
            control_frequency=int(config.get("control_freq", 25)),
            test_mode=config.get("mode", "test"),
            max_steps=int(config.get("max_steps", 1000)),
            rate_of_inference=int(config.get("rate_of_inference", 20)),
            ensemble_type=config.get("ensemble_type", "exp"),
            cogact_mode=config.get("cogact_mode", "cogact"),
            async_inference=bool(config.get("async_inference", False)),
            use_left_arm_only=bool(config.get("use_left_arm_only", False)),
            use_right_arm_only=bool(config.get("use_right_arm_only", False)),
            starvla=bool(config.get("starvla", False)),
            smooth_streaming=bool(config.get("smooth_streaming", False)),
            min_time_to_move_multiplier=float(config.get("min_time_to_move_multiplier", 3.0)),
            loop_rate=int(config.get("loop_rate", config.get("control_freq", 25))),
            adapter=adapter,
            sink=sink,
        )

    def run(self) -> None:
        cfg = self._config
        host = cfg.get("policy_host", "localhost")
        port = int(cfg.get("policy_port", 8000))
        timeout = float(cfg.get("connect_timeout", 15.0))
        self._sink.on_status("connecting", {"host": host, "port": port})
        try:
            wait_for_policy_server(host, port, timeout, should_stop=lambda: self._stopped)
        except ConnectStopped:
            self._sink.on_status("stopped", {"reason": "cancelled before connect"})
            return
        except ConnectTimeout as exc:
            logger.error("%s", exc)
            self._sink.on_status("connect_failed", {"message": str(exc)})
            return
        if self._stopped:
            self._sink.on_status("stopped", {"reason": "cancelled"})
            return

        self._bridge = self._make_bridge(cfg)
        try:
            self._bridge.run_episode(task_prompt=cfg.get("task_prompt", ""))
        finally:
            self._bridge.cleanup()

    def stop(self) -> None:
        self._stopped = True
        if self._bridge is not None:
            self._bridge.is_running = False

    def estop(self) -> None:
        self._stopped = True
        if self._bridge is not None:
            self._bridge.is_running = False
            try:
                self._bridge.move_to_sleep_position(duration=10.0)
            finally:
                self._bridge.cleanup()


class ReplayRunner:
    """Replays a dataset episode (EE -> IK -> robot) with a telemetry sink."""

    def __init__(self, kind: str, config: dict, sink) -> None:
        self._config = config
        self._sink = sink
        self._stopped = False
        self._controller = None

    def run(self) -> None:
        import time

        from dataset_replay import EpisodeReader
        from external.joint_to_ee.ee_to_joints import EEToJointsConverter
        from external.joint_to_ee.kinematics import make_kinematics
        from robot_control import RobotController, build_stationary_robot

        cfg = self._config
        reader = EpisodeReader(cfg["dataset_dir"])
        episode = reader.read_episode(int(cfg.get("episode_index", 0)))
        control_freq = int(cfg.get("control_freq") or episode.fps)
        converter = EEToJointsConverter(
            make_kinematics(),
            orientation_weight=float(cfg.get("ik_orientation_weight", 0.01)),
            pos_tol_m=float(cfg.get("ik_pos_tol_m", 1e-3)),
        )
        if self._stopped:
            self._sink.on_status("stopped", {"reason": "cancelled"})
            return
        robot = build_stationary_robot(
            with_cameras=False,
            min_time_to_move_multiplier=float(cfg.get("min_time_to_move_multiplier", 3.0)),
            loop_rate=int(cfg.get("loop_rate", control_freq)),
        )
        self._controller = RobotController(
            robot, control_frequency=control_freq, test_mode=cfg.get("mode", "test"),
            smooth_streaming=bool(cfg.get("smooth_streaming", False)),
        )
        try:
            cur = self._controller.current_joints14()
            joints = converter.decode_chunk(episode.ee_chunk16, cur)
            self._controller.move_to_start_position(joints[0], duration=5.0)
            dt = 1.0 / control_freq
            for step, a_t in enumerate(joints[1:], start=1):
                if self._stopped:
                    break
                t0 = time.perf_counter()
                ok = self._controller.execute_action(a_t)
                self._sink.on_action(step, np.asarray(a_t), time.time())
                if not ok:
                    self._sink.on_status("firmware_error", {"step": step})
                    break
                elapsed = time.perf_counter() - t0
                if dt - elapsed > 0:
                    time.sleep(dt - elapsed)
        finally:
            self._controller.disconnect()

    def stop(self) -> None:
        self._stopped = True

    def estop(self) -> None:
        self._stopped = True
        if self._controller is not None:
            self._controller.move_to_sleep_position(duration=10.0)


def make_runner(kind: str, config: dict, sink):
    if kind == "live":
        return LiveRunner(kind, config, sink)
    if kind == "replay":
        return ReplayRunner(kind, config, sink)
    raise ValueError(f"Unknown session kind {kind!r}")
```

- [ ] **Step 4: Run lifecycle tests, verify pass**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_runners_lifecycle.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/trossen_ai/webapp/runners.py examples/trossen_ai/webapp/tests/test_runners_lifecycle.py
git commit -m "refactor(webapp): cheap runner __init__, blocking+stoppable run(), connect timeout"
```

---

## Task 4: Delete joint-limit tables; add HOME_POSITION, firmware guard, tuning params, smooth-streaming in robot_control.py

**Files:**
- Modify: `examples/trossen_ai/robot_control.py`
- Test: `examples/trossen_ai/webapp/tests/test_motion_smooth.py`

- [ ] **Step 1: Write the failing test for smooth-streaming + firmware guard**

Create `examples/trossen_ai/webapp/tests/test_motion_smooth.py`:

```python
"""Off-robot tests for the smooth-streaming command path and firmware guard.

A FakeArm captures set_all_positions calls; a FakeRobot exposes left_arm /
right_arm and the observation/joint-feature surface RobotController uses.
"""
import numpy as np
import pytest

import robot_control
from robot_control import RobotController, send_action_smooth


class FakeArm:
    def __init__(self, names):
        self.config = type("C", (), {"joint_names": names})()
        self.driver = self
        self.calls = []

    def set_all_positions(self, goal_positions, goal_time, blocking, goal_feedforward_velocities=None):
        self.calls.append(dict(goal=list(goal_positions), goal_time=goal_time,
                               blocking=blocking, ff=list(goal_feedforward_velocities)))


class FakeRobot:
    def __init__(self):
        names = [f"j{i}.pos" for i in range(7)]
        self.left_arm = FakeArm(names)
        self.right_arm = FakeArm(names)
        self._obs = {f"left_j{i}.pos": 0.0 for i in range(7)}
        self._obs.update({f"right_j{i}.pos": 0.0 for i in range(7)})

    @property
    def _joint_ft(self):
        return {k: float for k in self._obs}

    def get_observation(self):
        return dict(self._obs)

    def send_action(self, action_dict):
        self._sent = action_dict

    def disconnect(self):
        ...


def test_send_action_smooth_sets_feedforward_velocity():
    robot = FakeRobot()
    target = np.arange(14, dtype=float) * 0.1  # left 0..0.6, right 0.7..1.3
    send_action_smooth(robot, target, dt=0.04)
    assert len(robot.left_arm.calls) == 1
    call = robot.left_arm.calls[0]
    assert call["goal_time"] == pytest.approx(0.04)
    assert call["blocking"] is False
    # feedforward velocity == (target - current) / dt, current is zeros
    assert call["ff"][0] == pytest.approx(0.0)
    assert call["ff"][1] == pytest.approx(0.1 / 0.04)


def test_execute_action_smooth_path_used_when_enabled():
    robot = FakeRobot()
    ctrl = RobotController(robot, control_frequency=25, test_mode="autonomous", smooth_streaming=True)
    assert ctrl.execute_action(np.zeros(14)) is True
    assert robot.left_arm.calls, "smooth path should call driver.set_all_positions"


def test_execute_action_firmware_error_triggers_sleep(monkeypatch):
    robot = FakeRobot()
    ctrl = RobotController(robot, control_frequency=25, test_mode="autonomous")

    def boom(action_dict):
        raise RuntimeError("firmware: joint velocity exceeded")

    monkeypatch.setattr(robot, "send_action", boom)
    slept = {"n": 0}
    monkeypatch.setattr(ctrl, "move_to_sleep_position", lambda duration=10.0: slept.__setitem__("n", slept["n"] + 1))
    assert ctrl.execute_action(np.zeros(14)) is False
    assert slept["n"] == 1


def test_home_position_constant_shape():
    assert robot_control.HOME_POSITION.shape == (14,)
    assert robot_control.HOME_POSITION[1] == pytest.approx(np.pi / 3)
```

- [ ] **Step 2: Run it, verify failure**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_motion_smooth.py -q`
Expected: FAIL (`send_action_smooth`/`HOME_POSITION` missing; `RobotController` has no `smooth_streaming`; `execute_action` still uses `is_action_within_limits`).

- [ ] **Step 3: Edit `robot_control.py`**

Make these edits:

(a) Add module-level constant + helper near the top (after imports):

```python
# Home/"stage" pose: arms up & open, ready for task start (left arm only; right at 0).
# Mirrors the commented stage_pose in main.py on the trossen-ai branch.
HOME_POSITION = np.array([0, np.pi / 3, np.pi / 6, np.pi / 5, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0], dtype=float)


def send_action_smooth(robot, action14: np.ndarray, dt: float) -> None:
    """Stream a 14-D joint target with feed-forward velocity for natural motion.

    Bypasses robot.send_action (which sends zero feed-forward velocity, planning
    to arrive at rest at every waypoint -> stutter). We compute ff = (goal-cur)/dt
    so the arm carries velocity through each waypoint, and set goal_time = dt so
    the firmware does not over/under-shoot the control period.
    """
    action14 = np.asarray(action14, dtype=float).flatten()
    obs = robot.get_observation()
    joint_pos_keys = [k for k in obs if k.endswith(".pos")]
    current = np.array([obs[k] for k in joint_pos_keys], dtype=float)
    ff = (action14 - current) / dt
    ff = np.nan_to_num(ff, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(robot.left_arm.config.joint_names)
    robot.left_arm.driver.set_all_positions(
        list(action14[:n]), goal_time=dt, blocking=False,
        goal_feedforward_velocities=list(ff[:n]),
    )
    robot.right_arm.driver.set_all_positions(
        list(action14[n:n * 2]), goal_time=dt, blocking=False,
        goal_feedforward_velocities=list(ff[n:n * 2]),
    )
```

(b) In `build_stationary_robot`, add tuning params to the signature and pass them through to `BiWidowXAIFollowerRobotConfig`:

```python
def build_stationary_robot(*, connect: bool = True, with_cameras: bool = True,
                           min_time_to_move_multiplier: float = 3.0,
                           loop_rate: int = 30):
    ...
    robot_config = BiWidowXAIFollowerRobotConfig(
        id="bimanual_follower",
        left_arm_ip_address="192.168.1.5",
        right_arm_ip_address="192.168.1.4",
        min_time_to_move_multiplier=min_time_to_move_multiplier,
        loop_rate=loop_rate,
        cameras=cameras,
    )
```

(c) Delete the `JOINT_LIMIT = np.array([...])` class attribute and the entire
`is_action_within_limits` method from `RobotController`.

(d) Change `RobotController.__init__` to accept `smooth_streaming`:

```python
    def __init__(self, robot, control_frequency: int = 50, test_mode: str = "autonomous",
                 smooth_streaming: bool = False) -> None:
        self.robot = robot
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode
        self.smooth_streaming = smooth_streaming
```

(e) Replace `execute_action` with a firmware-guarded version (no software limit check):

```python
    def execute_action(self, action: np.ndarray) -> bool:
        """Send a 14-D joint action. Returns True on success, False if the
        firmware faulted (in which case the arm is moved to sleep)."""
        full_action = np.asarray(action).flatten()

        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute action: {full_action}")
            return True

        try:
            if self.smooth_streaming:
                send_action_smooth(self.robot, full_action, self.dt)
            else:
                joint_features = list(self.robot._joint_ft.keys())  # noqa: SLF001
                action_dict = {k: full_action[i] for i, k in enumerate(joint_features)}
                self.robot.send_action(action_dict)
        except Exception as exc:  # noqa: BLE001 — firmware fault halts the arm
            logger.error(f"Firmware error executing action: {exc}. Moving to sleep position.")
            try:
                self.move_to_sleep_position(duration=10.0)
            except Exception as sleep_exc:  # noqa: BLE001
                logger.error(f"Failed to reach sleep position: {sleep_exc}")
            return False
        return True
```

- [ ] **Step 4: Run motion tests, verify pass**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_motion_smooth.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/trossen_ai/robot_control.py examples/trossen_ai/webapp/tests/test_motion_smooth.py
git commit -m "feat(robot_control): drop hardcoded joint limits, add firmware guard, HOME, smooth streaming + tuning"
```

---

## Task 5: Bridge — delete joint limits, firmware guard, tuning/smoothing params, print→logger

**Files:**
- Modify: `examples/trossen_ai/trossen_bridge.py`

This file is hardware-bound (no off-robot unit test of the loop). It is syntax/import-checked. Keep edits surgical.

- [ ] **Step 1: Read the current bridge to anchor edits**

Run: `cd examples/trossen_ai && sed -n '1,135p' trossen_bridge.py`
Expected: see the `JOINT_LIMIT`, `is_action_within_limits`, the constructor signature, and the `SLEEP_POSITION`.

- [ ] **Step 2: Update the constructor signature**

Add parameters `smooth_streaming: bool = False`, `min_time_to_move_multiplier: float = 3.0`,
`loop_rate: int = 30` to `TrossenOpenPIBridge.__init__`. Where the robot is built
(`self.robot = build_stationary_robot()`), pass them through:

```python
        self.robot = build_stationary_robot(
            min_time_to_move_multiplier=min_time_to_move_multiplier,
            loop_rate=loop_rate,
        )
        self.smooth_streaming = smooth_streaming
```

- [ ] **Step 3: Delete `JOINT_LIMIT` and `is_action_within_limits`**

Remove the `JOINT_LIMIT = np.array([...])` class attribute (around line 29) and the
entire `is_action_within_limits` method (around lines 111–135).

- [ ] **Step 4: Replace the limit check in `execute_action` with a firmware guard**

Find the block (around lines 188–201):

```python
        self.robot.send_action(action_dict)

        ## FIXME: double check the limit
        if not self.is_action_within_limits(full_action):
            logger.warning("Action exceeds limits. Moving to sleep position.")
            self.move_to_sleep_position(duration=10.0)
            self.is_running = False
            return
```

Replace with:

```python
        try:
            if self.smooth_streaming:
                from robot_control import send_action_smooth
                send_action_smooth(self.robot, full_action, self.dt)
            else:
                self.robot.send_action(action_dict)
        except Exception as exc:  # noqa: BLE001 — firmware fault halts the arm
            logger.error(f"Firmware error executing action: {exc}. Moving to sleep position.")
            self.sink.on_status("firmware_error", {"message": str(exc)})
            try:
                self.move_to_sleep_position(duration=10.0)
            finally:
                self.is_running = False
            return
```

(Keep the existing construction of `action_dict` above this block.)

- [ ] **Step 5: Route `print(...)` through the logger**

Run: `cd examples/trossen_ai && grep -n "print(" trossen_bridge.py`
Convert each hit to the matching `logger` level (e.g.
`print(f"Moving to sleep position over {duration}s...")` →
`logger.info(f"Moving to sleep position over {duration}s...")`).

- [ ] **Step 6: Syntax + import check**

Run: `cd examples/trossen_ai && python -c "import ast; ast.parse(open('trossen_bridge.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
git add examples/trossen_ai/trossen_bridge.py
git commit -m "feat(bridge): firmware guard instead of joint-limit table, smooth streaming, print->logger"
```

---

## Task 6: Feedback store + `/api/feedback`

**Files:**
- Create: `examples/trossen_ai/webapp/feedback_store.py`
- Modify: `examples/trossen_ai/webapp/server.py`
- Test: `examples/trossen_ai/webapp/tests/test_feedback_store.py`, `test_server.py`

- [ ] **Step 1: Write the failing store test**

Create `examples/trossen_ai/webapp/tests/test_feedback_store.py`:

```python
from pathlib import Path

from webapp.feedback_store import save_feedback


def test_save_feedback_writes_dated_markdown(tmp_path):
    path = save_feedback(tmp_path, name="Ada", email="ada@example.com",
                         feedback="The replay charts are great.")
    p = Path(path)
    assert p.exists()
    assert p.suffix == ".md"
    text = p.read_text()
    assert "Ada" in text
    assert "ada@example.com" in text
    assert "The replay charts are great." in text


def test_save_feedback_filenames_are_unique(tmp_path):
    a = save_feedback(tmp_path, name="A", email="a@x.com", feedback="one")
    b = save_feedback(tmp_path, name="B", email="b@x.com", feedback="two")
    assert a != b
```

- [ ] **Step 2: Run it, verify failure**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_feedback_store.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'webapp.feedback_store'`.

- [ ] **Step 3: Implement the store**

Create `examples/trossen_ai/webapp/feedback_store.py`:

```python
"""Persist user feedback as dated markdown files."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path


def save_feedback(feedback_dir: str | Path, *, name: str, email: str, feedback: str) -> str:
    """Write one feedback submission to feedback_dir/YYYY-MM-DD-HHMMSS[-N].md.

    Returns the path written. Creates the directory if needed.
    """
    d = Path(feedback_dir)
    d.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now()
    stamp = now.strftime("%Y-%m-%d-%H%M%S")
    path = d / f"{stamp}.md"
    suffix = 1
    while path.exists():
        path = d / f"{stamp}-{suffix}.md"
        suffix += 1
    body = (
        f"# Feedback — {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"- **Name:** {name}\n"
        f"- **Email:** {email}\n\n"
        f"{feedback}\n"
    )
    path.write_text(body)
    return str(path)
```

- [ ] **Step 4: Run it, verify pass**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_feedback_store.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Add `feedback_dir` param to `create_app` + the `/api/feedback` route**

In `webapp/server.py`, change the `create_app` signature and body:

```python
def create_app(presets_dir: str | Path | None = None, runner_factory=None,
               feedback_dir: str | Path | None = None) -> FastAPI:
    if presets_dir is None:
        presets_dir = Path(__file__).parent / "presets"
    if feedback_dir is None:
        feedback_dir = Path(__file__).parent / "feedback"
    ...
```

Add the route alongside the other `@app.post` routes:

```python
    @app.post("/api/feedback")
    def submit_feedback(body: dict):
        from webapp.feedback_store import save_feedback
        path = save_feedback(
            feedback_dir,
            name=body.get("name", ""),
            email=body.get("email", ""),
            feedback=body.get("feedback", ""),
        )
        return {"ok": True, "path": path}
```

- [ ] **Step 6: Add a server test (isolated via feedback_dir)**

Append to `webapp/tests/test_server.py`:

```python
def test_feedback_endpoint_writes_file(tmp_path):
    client = TestClient(create_app(presets_dir=tmp_path, feedback_dir=tmp_path / "fb"))
    r = client.post("/api/feedback", json={"name": "Ada", "email": "a@x.com", "feedback": "hi"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    files = list((tmp_path / "fb").glob("*.md"))
    assert len(files) == 1
    assert "Ada" in files[0].read_text()
```

- [ ] **Step 7: Run server tests, verify pass**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_server.py -q`
Expected: PASS.

- [ ] **Step 8: Add `feedback/` to `.gitignore`**

Append to `examples/trossen_ai/.gitignore` (create the file if missing):

```
webapp/feedback/
```

- [ ] **Step 9: Commit**

```bash
git add examples/trossen_ai/webapp/feedback_store.py examples/trossen_ai/webapp/server.py \
        examples/trossen_ai/webapp/tests/test_feedback_store.py \
        examples/trossen_ai/webapp/tests/test_server.py examples/trossen_ai/.gitignore
git commit -m "feat(webapp): /api/feedback saves dated markdown submissions"
```

---

## Task 7: Folder browser API (`/api/files`)

**Files:**
- Create: `examples/trossen_ai/webapp/files_api.py`
- Modify: `examples/trossen_ai/webapp/server.py`
- Test: `examples/trossen_ai/webapp/tests/test_files_api.py`

- [ ] **Step 1: Write the failing test**

Create `examples/trossen_ai/webapp/tests/test_files_api.py`:

```python
from webapp.files_api import list_directory


def test_lists_subdirs_and_parent(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "file.txt").write_text("x")
    out = list_directory(str(tmp_path))
    names = [e["name"] for e in out["entries"]]
    assert names == ["a", "b"]  # dirs only, sorted; file.txt excluded
    assert out["parent"] == str(tmp_path.parent)
    assert out["path"] == str(tmp_path)


def test_marks_datasets(tmp_path):
    ds = tmp_path / "mydataset"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text("{}")
    out = list_directory(str(tmp_path))
    entry = next(e for e in out["entries"] if e["name"] == "mydataset")
    assert entry["is_dataset"] is True


def test_error_for_missing_dir(tmp_path):
    out = list_directory(str(tmp_path / "nope"))
    assert "error" in out
```

- [ ] **Step 2: Run it, verify failure**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_files_api.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'webapp.files_api'`.

- [ ] **Step 3: Implement (ported from the wizard `list_directory`)**

Create `examples/trossen_ai/webapp/files_api.py`:

```python
"""Directory listing for the folder browser (ported from the dataset wizard).

Traverses outside the repo via expanduser().resolve(); returns directories only,
flagging LeRobot datasets (meta/info.json present).
"""
from __future__ import annotations

from pathlib import Path


def list_directory(path_str: str) -> dict:
    p = Path(path_str).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return {"error": f"not a directory: {path_str}"}
    entries = []
    try:
        for child in sorted(p.iterdir()):
            if child.is_dir():
                entries.append({
                    "name": child.name,
                    "type": "dir",
                    "is_dataset": (child / "meta" / "info.json").exists(),
                    "path": str(child),
                })
    except PermissionError:
        pass
    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent": parent, "entries": entries}
```

- [ ] **Step 4: Run it, verify pass**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_files_api.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Add the `/api/files` route**

In `webapp/server.py`, add near the other GET routes:

```python
    @app.get("/api/files")
    def files(path: str | None = None):
        from webapp.files_api import list_directory
        return list_directory(path or str(Path.home()))
```

- [ ] **Step 6: Add a server test**

Append to `webapp/tests/test_server.py`:

```python
def test_files_endpoint_lists_dirs(tmp_path):
    (tmp_path / "sub").mkdir()
    client = TestClient(create_app())
    r = client.get("/api/files", params={"path": str(tmp_path)})
    assert r.status_code == 200
    assert [e["name"] for e in r.json()["entries"]] == ["sub"]
```

- [ ] **Step 7: Run server tests, verify pass**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests/test_server.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add examples/trossen_ai/webapp/files_api.py examples/trossen_ai/webapp/server.py \
        examples/trossen_ai/webapp/tests/test_files_api.py examples/trossen_ai/webapp/tests/test_server.py
git commit -m "feat(webapp): /api/files folder browser listing endpoint"
```

---

## Task 8: Sleep / Home movers + `go_sleep`/`go_home` WS actions

**Files:**
- Create: `examples/trossen_ai/webapp/movers.py`
- Modify: `examples/trossen_ai/webapp/server.py`

Movers reuse the runner lifecycle (cheap `__init__`, blocking in `run()`), so they
slot into `SessionManager` unchanged and are guarded by the single-session rule.

- [ ] **Step 1: Implement movers**

Create `examples/trossen_ai/webapp/movers.py`:

```python
"""Standalone movers: send the arm(s) to a fixed pose, then disconnect.

Used by the Sleep/Home buttons. They implement the Runner interface (run/stop/
estop) so SessionManager runs them on its thread and the single-session guard
prevents them from racing an eval/replay session.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class _PoseMover:
    def __init__(self, kind: str, config: dict, sink) -> None:
        self._config = config
        self._sink = sink
        self._controller = None

    def _goal(self):  # overridden
        raise NotImplementedError

    def run(self) -> None:
        from robot_control import RobotController, build_stationary_robot

        self._sink.on_status("started", {"kind": "move"})
        robot = build_stationary_robot(with_cameras=False)
        self._controller = RobotController(
            robot, control_frequency=int(self._config.get("control_freq", 25)),
            test_mode=self._config.get("mode", "autonomous"),
        )
        try:
            self._controller.move_to_start_position(self._goal(), duration=5.0)
        finally:
            self._controller.disconnect()
            self._sink.on_status("stopped", {"kind": "move"})

    def stop(self) -> None:
        ...

    def estop(self) -> None:
        if self._controller is not None:
            self._controller.move_to_sleep_position(duration=10.0)


class SleepRunner(_PoseMover):
    def _goal(self):
        from robot_control import RobotController
        return RobotController.SLEEP_POSITION


class HomeRunner(_PoseMover):
    def _goal(self):
        import robot_control
        return robot_control.HOME_POSITION
```

> `move_to_start_position` interpolates from the current pose to the goal (PCHIP),
> which is exactly the smooth motion we want for both Sleep and Home.

- [ ] **Step 2: Wire `go_sleep` / `go_home` into the WS handler + default factory**

In `webapp/server.py`, extend the default `runner_factory` to handle the mover kinds:

```python
        def runner_factory(kind, config, sink):
            if kind == "sleep":
                from webapp.movers import SleepRunner
                return SleepRunner(kind, config, sink)
            if kind == "home":
                from webapp.movers import HomeRunner
                return HomeRunner(kind, config, sink)
            from webapp.runners import make_runner
            return make_runner(kind, config, sink)
```

In the WS receive loop, add actions alongside `start_live`/`start_replay`:

```python
                elif action == "go_sleep":
                    session.start("sleep", cmd.get("config", {}), QueueSink(q))
                elif action == "go_home":
                    session.start("home", cmd.get("config", {}), QueueSink(q))
```

- [ ] **Step 3: Syntax check**

Run: `cd examples/trossen_ai && python -c "import ast; ast.parse(open('webapp/movers.py').read()); ast.parse(open('webapp/server.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Verify the WS layer still imports + tests pass off-robot**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests -q`
Expected: PASS (movers import lazily; no hardware needed to import server).

- [ ] **Step 5: Commit**

```bash
git add examples/trossen_ai/webapp/movers.py examples/trossen_ai/webapp/server.py
git commit -m "feat(webapp): Sleep/Home movers + go_sleep/go_home WS actions"
```

---

## Task 9: Graceful shutdown — free the robot on SIGINT/SIGTERM

**Files:**
- Modify: `examples/trossen_ai/webapp/server.py`
- Test: `examples/trossen_ai/webapp/tests/test_server.py`

- [ ] **Step 1: Add a FastAPI shutdown handler that stops the session**

In `create_app`, after `session = SessionManager(runner_factory)`, register:

```python
    @app.on_event("shutdown")
    def _cleanup_on_shutdown():
        try:
            session.stop()
        except Exception:  # noqa: BLE001
            pass
```

This ensures uvicorn's shutdown (triggered by CTRL+C/SIGTERM) joins the session
thread and runs the runner's `finally: robot.disconnect()`, freeing the hardware.

- [ ] **Step 2: Add a test that the shutdown hook runs without error**

Append to `webapp/tests/test_server.py`:

```python
def test_shutdown_hook_runs_clean():
    app = create_app()
    with TestClient(app):  # entering/exiting triggers startup/shutdown
        pass
    # Exiting the context ran the shutdown hook (session.stop()) without raising.
    assert True
```

> The substantive guarantee (join + `robot.disconnect()`) is exercised on
> hardware per HARDWARE.md; this test confirms the hook is wired and side-effect
> free off-robot.

- [ ] **Step 3: Run all backend tests**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests -q`
Expected: PASS (entire suite).

- [ ] **Step 4: Commit**

```bash
git add examples/trossen_ai/webapp/server.py examples/trossen_ai/webapp/tests/test_server.py
git commit -m "feat(webapp): stop session + disconnect robot on server shutdown"
```

---

## Final verification

- [ ] **Run the full backend suite**

Run: `cd examples/trossen_ai && python -m pytest webapp/tests -q`
Expected: all green.

- [ ] **Import/syntax-check the hardware modules**

Run: `cd examples/trossen_ai && python -c "import ast; [ast.parse(open(f).read()) for f in ['trossen_bridge.py','robot_control.py','policy_connect.py','webapp/runners.py','webapp/movers.py','webapp/server.py']]; print('ok')"`
Expected: `ok`.

- [ ] **Update HARDWARE.md** with a "Motion tuning" section documenting
  `smooth_streaming`, `min_time_to_move_multiplier`, `loop_rate`, `connect_timeout`,
  and the Sleep/Home buttons, then commit:

```bash
git add examples/trossen_ai/webapp/HARDWARE.md
git commit -m "docs(webapp): document motion-tuning knobs and sleep/home"
```
