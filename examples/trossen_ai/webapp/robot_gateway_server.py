"""Loopback-only robot gateway intended to run on the ALOHA laptop.

Starting this service never constructs a runner and therefore never connects to
the cameras or arms.  Hardware construction remains inside the existing lazy
``SessionManager`` path and happens only after an explicit, acknowledged command.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
import dataclasses
import logging
import os
from pathlib import Path
import queue
import sys
import time
from typing import Any
import uuid

if __package__ in {None, ""}:
    # Allow `python webapp/robot_gateway_server.py` from examples/trossen_ai.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from webapp.gateway_protocol import PROTOCOL_VERSION
from webapp.runner_factory import make_web_runner
from webapp.session import SessionManager
from webapp.telemetry import QueueSink

logger = logging.getLogger(__name__)

_START_ACTIONS = {"start_live": "live", "go_home": "home", "go_sleep": "sleep"}
_UPDATE_ACTIONS = {"update_prompt"}
_STOP_ACTIONS = {"stop", "estop"}
_MAX_TASK_PROMPT_LENGTH = 2000
_MAX_RECENT_COMMANDS = 256


@dataclasses.dataclass(frozen=True)
class RobotGatewaySettings:
    policy_host: str = "192.168.50.174"
    policy_port: int = 8800
    heartbeat_timeout: float = 5.0
    state_interval: float = 1.0

    @classmethod
    def from_environment(cls) -> RobotGatewaySettings:
        return cls(
            policy_host=os.environ.get("OPENPI_GATEWAY_POLICY_HOST", "192.168.50.174"),
            policy_port=int(os.environ.get("OPENPI_GATEWAY_POLICY_PORT", "8800")),
            heartbeat_timeout=float(os.environ.get("OPENPI_GATEWAY_HEARTBEAT_TIMEOUT", "5")),
            state_interval=float(os.environ.get("OPENPI_GATEWAY_STATE_INTERVAL", "1")),
        )

    def validate(self) -> None:
        if self.policy_host not in {"192.168.50.174", "127.0.0.1", "localhost", "::1"}:
            raise ValueError("Invalid fixed policy-server host")
        if not 1 <= self.policy_port <= 65535:
            raise ValueError("Invalid gateway policy port")
        if self.heartbeat_timeout <= 0 or self.state_interval <= 0:
            raise ValueError("Gateway timeouts must be positive")


class RobotGatewayController:
    """Own one lazy robot session and dispatch idempotent lifecycle commands."""

    def __init__(self, session: SessionManager, settings: RobotGatewaySettings) -> None:
        settings.validate()
        self._session = session
        self._settings = settings
        self._boot_id = str(uuid.uuid4())
        self._kind: str | None = None
        self._interactive = False
        self._stopping = False
        self._closing = False
        self._command_lock = asyncio.Lock()
        self._stop_task: asyncio.Task[None] | None = None
        self._recent_acks: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def snapshot(self) -> dict[str, Any]:
        running = self._session.is_running()
        if not running and not self._stopping:
            self._kind = None
            self._interactive = False
        state = "stopping" if self._stopping else ("busy" if running else "idle")
        return {
            "gateway": "closing" if self._closing else "ready",
            "session": state,
            "kind": self._kind,
            "boot_id": self._boot_id,
        }

    def hello(self) -> dict[str, Any]:
        return {"type": "hello", "protocol": PROTOCOL_VERSION, "state": self.snapshot()}

    async def dispatch(self, message: Any, sink: QueueSink) -> dict[str, Any]:
        command_id, action = self._validate_command(message)
        cached = self._recent_acks.get(command_id)
        if cached is not None:
            return dict(cached)

        async with self._command_lock:
            cached = self._recent_acks.get(command_id)
            if cached is not None:
                return dict(cached)

            if self._closing:
                ack = self._rejected(command_id, action, "shutting_down", "Robot gateway is shutting down")
            elif action in _START_ACTIONS:
                ack = self._start(command_id, action, message.get("config"), sink)
            elif action in _UPDATE_ACTIONS:
                ack = self._update_prompt(command_id, action, message.get("config"))
            else:
                ack = self._begin_stop(command_id, action)

            self._remember_ack(command_id, ack)
            return dict(ack)

    def _start(self, command_id: str, action: str, config: Any, sink: QueueSink) -> dict[str, Any]:
        state = self.snapshot()["session"]
        if state != "idle":
            return self._rejected(command_id, action, "busy", f"Robot session is {state}")
        if config is None:
            config = {}
        if not isinstance(config, dict):
            return self._rejected(command_id, action, "invalid", "config must be an object")

        runner_config = dict(config)
        if action == "start_live":
            interactive = runner_config.get("interactive", False)
            if not isinstance(interactive, bool):
                return self._rejected(command_id, action, "invalid", "interactive must be true or false")
            # The browser cannot redirect the robot to an arbitrary policy server.
            # On Laptop B these values always name the SSH reverse tunnel.
            runner_config["policy_host"] = self._settings.policy_host
            runner_config["policy_port"] = self._settings.policy_port

        kind = _START_ACTIONS[action]
        try:
            self._session.start(kind, runner_config, sink)
        except Exception as exc:
            logger.warning("Robot gateway rejected %s: %s", action, exc)
            return self._rejected(command_id, action, "start_failed", str(exc))

        self._kind = kind
        self._interactive = bool(runner_config.get("interactive", False))
        return self._accepted(command_id, action)

    def _update_prompt(self, command_id: str, action: str, config: Any) -> dict[str, Any]:
        if self.snapshot()["session"] != "busy" or self._kind != "live":
            return self._rejected(command_id, action, "not_running", "No live episode is running")
        if not self._interactive:
            return self._rejected(
                command_id,
                action,
                "not_interactive",
                "Prompt updates require an Interactive episode",
            )
        if not isinstance(config, dict):
            return self._rejected(command_id, action, "invalid", "config must be an object")
        task_prompt = config.get("task_prompt")
        if not isinstance(task_prompt, str):
            return self._rejected(command_id, action, "invalid", "task_prompt must be text")
        task_prompt = task_prompt.strip()
        if not task_prompt or len(task_prompt) > _MAX_TASK_PROMPT_LENGTH:
            return self._rejected(command_id, action, "invalid", "task_prompt must be 1 to 2000 characters")
        try:
            self._session.update_prompt(task_prompt)
        except RuntimeError as exc:
            return self._rejected(command_id, action, "update_failed", str(exc))
        return self._accepted(command_id, action)

    def _begin_stop(self, command_id: str, action: str) -> dict[str, Any]:
        if self._stop_task is None or self._stop_task.done():
            self._stopping = self._session.is_running()
            self._stop_task = asyncio.create_task(self._stop_worker(estop=action == "estop"))
        return self._accepted(command_id, action)

    async def _stop_worker(self, *, estop: bool) -> None:
        try:
            operation = self._session.estop if estop else self._session.stop
            await asyncio.to_thread(operation)
        except Exception:
            logger.exception("Robot gateway failed to stop the local session")
        finally:
            self._stopping = False
            if not self._session.is_running():
                self._kind = None
                self._interactive = False

    async def stop_for_disconnect(self) -> None:
        """Stop locally when Machine A loses its controller lease."""

        if self._stop_task is None or self._stop_task.done():
            self._stopping = self._session.is_running()
            self._stop_task = asyncio.create_task(self._stop_worker(estop=False))
        # A WebSocket disconnect cancels its request scope. Shield the local
        # stop task so cancellation cannot leave the robot session running.
        await asyncio.shield(self._stop_task)

    async def shutdown(self) -> None:
        self._closing = True
        await self.stop_for_disconnect()

    @staticmethod
    def _validate_command(message: Any) -> tuple[str, str]:
        if not isinstance(message, dict) or message.get("type") != "command":
            raise ValueError("Expected a command message")
        if message.get("protocol") != PROTOCOL_VERSION:
            raise ValueError(f"Unsupported gateway protocol; expected {PROTOCOL_VERSION}")
        command_id = message.get("command_id")
        if not isinstance(command_id, str) or len(command_id) > 128:
            raise ValueError("command_id must be a UUID string")
        try:
            uuid.UUID(command_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("command_id must be a UUID string") from exc
        action = message.get("action")
        if action not in set(_START_ACTIONS) | _UPDATE_ACTIONS | _STOP_ACTIONS:
            raise ValueError(f"Unsupported robot action: {action!r}")
        return command_id, action

    def _accepted(self, command_id: str, action: str) -> dict[str, Any]:
        return {
            "type": "ack",
            "command_id": command_id,
            "action": action,
            "accepted": True,
            "state": self.snapshot(),
        }

    def _rejected(self, command_id: str, action: str, code: str, message: str) -> dict[str, Any]:
        return {
            "type": "ack",
            "command_id": command_id,
            "action": action,
            "accepted": False,
            "code": code,
            "message": message,
            "state": self.snapshot(),
        }

    def _remember_ack(self, command_id: str, ack: dict[str, Any]) -> None:
        self._recent_acks[command_id] = dict(ack)
        self._recent_acks.move_to_end(command_id)
        while len(self._recent_acks) > _MAX_RECENT_COMMANDS:
            self._recent_acks.popitem(last=False)


def create_gateway_app(
    *,
    runner_factory=None,
    settings: RobotGatewaySettings | None = None,
) -> FastAPI:
    """Create the Laptop B gateway without touching robot hardware."""

    if runner_factory is None:
        runner_factory = make_web_runner
    if settings is None:
        settings = RobotGatewaySettings.from_environment()
    settings.validate()

    session = SessionManager(runner_factory)
    controller = RobotGatewayController(session, settings)
    connection_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await controller.shutdown()

    app = FastAPI(title="OpenPI ALOHA Robot Gateway", lifespan=lifespan)
    app.state.session_manager = session
    app.state.gateway_controller = controller

    @app.get("/healthz")
    def healthz():
        # This reads in-memory state only.  It must never probe/connect the robot.
        return {"ok": True, "protocol": PROTOCOL_VERSION, "state": controller.snapshot()}

    @app.websocket("/ws/robot")
    async def robot_ws(ws: WebSocket):
        await ws.accept()
        if connection_lock.locked():
            await ws.send_json({"type": "error", "code": "controller_busy", "message": "A controller is connected"})
            await ws.close(code=1013)
            return

        await connection_lock.acquire()
        event_queue: queue.Queue = queue.Queue(maxsize=1000)
        sink = QueueSink(event_queue)
        send_lock = asyncio.Lock()
        last_seen = time.monotonic()

        async def send_json(message: dict[str, Any]) -> None:
            async with send_lock:
                await ws.send_json(message)

        async def pump_telemetry() -> None:
            while True:
                try:
                    event = await asyncio.to_thread(event_queue.get, block=True, timeout=0.5)
                except queue.Empty:
                    continue
                await send_json(event)

        async def publish_state() -> None:
            previous: dict[str, Any] | None = None
            while True:
                await asyncio.sleep(settings.state_interval)
                state = controller.snapshot()
                if state != previous:
                    await send_json({"type": "gateway_state", "state": state})
                    previous = state

        async def watchdog() -> None:
            nonlocal last_seen
            interval = min(1.0, settings.heartbeat_timeout / 2.0)
            while True:
                await asyncio.sleep(interval)
                if time.monotonic() - last_seen <= settings.heartbeat_timeout:
                    continue
                logger.error("Machine A heartbeat expired; stopping the robot session locally")
                await controller.stop_for_disconnect()
                await ws.close(code=1011, reason="controller heartbeat expired")
                return

        telemetry_task = asyncio.create_task(pump_telemetry())
        state_task = asyncio.create_task(publish_state())
        watchdog_task = asyncio.create_task(watchdog())
        try:
            await send_json(controller.hello())
            while True:
                message = await ws.receive_json()
                last_seen = time.monotonic()
                if isinstance(message, dict) and message.get("type") == "heartbeat":
                    await send_json(
                        {
                            "type": "heartbeat_ack",
                            "protocol": PROTOCOL_VERSION,
                            "seq": message.get("seq"),
                            "state": controller.snapshot(),
                        }
                    )
                    continue
                try:
                    ack = await controller.dispatch(message, sink)
                except ValueError as exc:
                    await send_json({"type": "error", "code": "invalid_message", "message": str(exc)})
                    continue
                await send_json(ack)
                await send_json({"type": "gateway_state", "state": controller.snapshot()})
        except WebSocketDisconnect:
            logger.warning("Machine A disconnected from the robot gateway")
        except Exception:
            logger.exception("Robot gateway controller connection failed")
        finally:
            for task in (telemetry_task, state_task, watchdog_task):
                task.cancel()
            await asyncio.gather(telemetry_task, state_task, watchdog_task, return_exceptions=True)
            await controller.stop_for_disconnect()
            connection_lock.release()

    return app


app = create_gateway_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("OPENPI_GATEWAY_HOST", "192.168.50.219"),
        port=int(os.environ.get("OPENPI_GATEWAY_PORT", "8001")),
        lifespan="on",
    )
