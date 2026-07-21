"""Persistent Machine A client for the SSH-tunneled ALOHA robot gateway."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import dataclasses
import json
import logging
import os
from typing import Any
import uuid

import websockets.asyncio.client

from webapp.gateway_protocol import PROTOCOL_VERSION

logger = logging.getLogger(__name__)


class RobotGatewayError(RuntimeError):
    """Base error for remote robot gateway operations."""


class RobotGatewayUnavailableError(RobotGatewayError):
    """No live SSH-tunneled gateway connection exists."""


class RobotCommandRejectedError(RobotGatewayError):
    """Laptop B explicitly rejected a lifecycle command."""


class RobotCommandOutcomeUnknownError(RobotGatewayError):
    """A command was sent but its acknowledgement was lost."""


@dataclasses.dataclass(frozen=True)
class RobotGatewayClientSettings:
    url: str
    heartbeat_interval: float = 1.0
    command_timeout: float = 5.0
    reconnect_interval: float = 1.0
    open_timeout: float = 5.0
    max_message_bytes: int = 32 * 1024 * 1024

    @classmethod
    def from_environment(cls) -> RobotGatewayClientSettings | None:
        url = os.environ.get("OPENPI_ROBOT_GATEWAY_URL")
        if not url:
            return None
        return cls(
            url=url,
            heartbeat_interval=float(os.environ.get("OPENPI_GATEWAY_HEARTBEAT_INTERVAL", "1")),
            command_timeout=float(os.environ.get("OPENPI_GATEWAY_COMMAND_TIMEOUT", "5")),
            reconnect_interval=float(os.environ.get("OPENPI_GATEWAY_RECONNECT_INTERVAL", "1")),
            open_timeout=float(os.environ.get("OPENPI_GATEWAY_OPEN_TIMEOUT", "5")),
            max_message_bytes=int(os.environ.get("OPENPI_GATEWAY_MAX_MESSAGE_BYTES", str(32 * 1024 * 1024))),
        )

    def validate(self) -> None:
        if not self.url.startswith(("ws://127.0.0.1", "ws://localhost", "ws://[::1]")):
            raise ValueError("Robot gateway URL must use a loopback WebSocket carried by SSH")
        if min(self.heartbeat_interval, self.command_timeout, self.reconnect_interval, self.open_timeout) <= 0:
            raise ValueError("Robot gateway client timeouts must be positive")
        if self.max_message_bytes < 1024:
            raise ValueError("Robot gateway maximum message size is too small")


class RobotGatewayClient:
    """Reconnect transport, correlate ACKs, and broadcast telemetry.

    Only the transport reconnects.  Lifecycle commands are never queued and are
    never replayed after a reconnect.
    """

    def __init__(self, settings: RobotGatewayClientSettings) -> None:
        settings.validate()
        self._settings = settings
        self._ws = None
        self._run_task: asyncio.Task[None] | None = None
        self._closing = False
        self._connected = False
        self._state_known = False
        self._possible_active_session = False
        self._state: dict[str, Any] = {
            "gateway": "unavailable",
            "session": "unknown",
            "kind": None,
            "boot_id": None,
        }
        self._send_lock = asyncio.Lock()
        self._command_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._subscribers: set[asyncio.Queue] = set()
        self._heartbeat_seq = 0

    @classmethod
    def from_environment(cls) -> RobotGatewayClient | None:
        settings = RobotGatewayClientSettings.from_environment()
        return cls(settings) if settings is not None else None

    async def start(self) -> None:
        if self._run_task is None:
            self._closing = False
            self._run_task = asyncio.create_task(self._run_forever(), name="robot-gateway-client")

    async def close(self) -> None:
        self._closing = True
        if self._connected and self.blocks_model_change():
            try:
                await self.send_command("stop", timeout=min(2.0, self._settings.command_timeout))
            except RobotGatewayError:
                logger.warning("Could not acknowledge remote robot stop during shutdown")
        ws = self._ws
        if ws is not None:
            await ws.close()
        task, self._run_task = self._run_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._mark_disconnected("Machine A gateway client stopped")

    async def stop_and_drop_controller(self) -> None:
        """Best-effort Stop, then withdraw the lease even if its ACK is lost."""

        if self._connected:
            try:
                await self.send_command("stop", timeout=min(2.0, self._settings.command_timeout))
            except RobotGatewayError:
                logger.warning("Remote Stop was not acknowledged; dropping controller lease")
        ws = self._ws
        if ws is not None:
            await ws.close()

    def subscribe(self, maxsize: int = 1000) -> asyncio.Queue:
        event_queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(event_queue)
        event_queue.put_nowait({"type": "gateway", **self.snapshot()})
        return event_queue

    def unsubscribe(self, event_queue: asyncio.Queue) -> None:
        self._subscribers.discard(event_queue)

    def snapshot(self) -> dict[str, Any]:
        return {
            "connected": self._connected,
            "state_known": self._state_known,
            "possible_active_session": self._possible_active_session,
            "url": self._settings.url,
            "state": dict(self._state),
        }

    def is_running(self) -> bool:
        return self.blocks_model_change()

    def blocks_model_change(self) -> bool:
        state = self._state.get("session")
        return self._possible_active_session or state in {"busy", "stopping"}

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def state_known(self) -> bool:
        return self._state_known

    async def send_command(
        self,
        action: str,
        *,
        config: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        async with self._command_lock:
            if not self._connected or self._ws is None:
                raise RobotGatewayUnavailableError("ALOHA robot gateway is not connected")

            command_id = str(uuid.uuid4())
            message: dict[str, Any] = {
                "type": "command",
                "protocol": PROTOCOL_VERSION,
                "command_id": command_id,
                "action": action,
            }
            if config is not None:
                message["config"] = config

            loop = asyncio.get_running_loop()
            acknowledgement = loop.create_future()
            self._pending[command_id] = acknowledgement
            try:
                await self._send(message)
                try:
                    ack = await asyncio.wait_for(
                        asyncio.shield(acknowledgement),
                        timeout=timeout or self._settings.command_timeout,
                    )
                except TimeoutError as exc:
                    raise RobotCommandOutcomeUnknownError(
                        f"{action} was sent but Laptop B did not acknowledge it; it will not be retried"
                    ) from exc
            finally:
                self._pending.pop(command_id, None)

            if not ack.get("accepted"):
                raise RobotCommandRejectedError(ack.get("message") or f"Laptop B rejected {action}")
            if action in {"start_live", "go_home", "go_sleep"}:
                self._possible_active_session = True
            return ack

    async def _run_forever(self) -> None:
        while not self._closing:
            try:
                async with websockets.asyncio.client.connect(
                    self._settings.url,
                    open_timeout=self._settings.open_timeout,
                    max_size=self._settings.max_message_bytes,
                    compression=None,
                    proxy=None,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._state_known = False
                    await self._broadcast({"type": "gateway", **self.snapshot()})
                    receiver = asyncio.create_task(self._receive_loop(ws))
                    heartbeat = asyncio.create_task(self._heartbeat_loop(ws))
                    done, pending = await asyncio.wait(
                        {receiver, heartbeat}, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for task in done:
                        task.result()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._closing:
                    logger.warning("Robot gateway unavailable: %s", exc)
            finally:
                self._mark_disconnected("Connection to Laptop B was lost")

            if not self._closing:
                await asyncio.sleep(self._settings.reconnect_interval)

    async def _receive_loop(self, ws) -> None:
        async for raw_message in ws:
            if not isinstance(raw_message, str):
                logger.warning("Ignoring non-JSON robot gateway message")
                continue
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.warning("Ignoring invalid robot gateway JSON")
                continue
            if not isinstance(message, dict):
                continue

            message_type = message.get("type")
            state = message.get("state")
            if message_type in {"hello", "gateway_state", "heartbeat_ack", "ack"} and isinstance(state, dict):
                self._update_state(state)

            if message_type == "ack":
                future = self._pending.get(message.get("command_id"))
                if future is not None and not future.done():
                    future.set_result(message)
                continue
            if message_type in {"hello", "heartbeat_ack"}:
                continue
            await self._broadcast(message)

    async def _heartbeat_loop(self, ws) -> None:
        while ws is self._ws and not self._closing:
            self._heartbeat_seq += 1
            await self._send(
                {
                    "type": "heartbeat",
                    "protocol": PROTOCOL_VERSION,
                    "seq": self._heartbeat_seq,
                }
            )
            await asyncio.sleep(self._settings.heartbeat_interval)

    async def _send(self, message: dict[str, Any]) -> None:
        async with self._send_lock:
            ws = self._ws
            if not self._connected or ws is None:
                raise RobotGatewayUnavailableError("ALOHA robot gateway disconnected before send")
            await ws.send(json.dumps(message))

    def _update_state(self, state: dict[str, Any]) -> None:
        self._state = {
            "gateway": state.get("gateway", "unknown"),
            "session": state.get("session", "unknown"),
            "kind": state.get("kind"),
            "boot_id": state.get("boot_id"),
        }
        self._state_known = True
        if self._state["session"] in {"busy", "stopping"}:
            self._possible_active_session = True
        elif self._state["session"] == "idle":
            self._possible_active_session = False

    def _mark_disconnected(self, reason: str) -> None:
        was_connected = self._connected
        self._connected = False
        self._state_known = False
        self._ws = None
        if self._possible_active_session:
            self._state["session"] = "unknown"
        self._state["gateway"] = "unavailable"
        error = RobotCommandOutcomeUnknownError(reason)
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        if was_connected:
            self._broadcast_nowait(
                {
                    "type": "status",
                    "kind": "error",
                    "payload": {"message": reason},
                }
            )
        self._broadcast_nowait({"type": "gateway", **self.snapshot()})

    async def _broadcast(self, message: dict[str, Any]) -> None:
        self._broadcast_nowait(message)

    def _broadcast_nowait(self, message: dict[str, Any]) -> None:
        for event_queue in tuple(self._subscribers):
            try:
                event_queue.put_nowait(dict(message))
            except asyncio.QueueFull:
                with suppress(asyncio.QueueEmpty):
                    event_queue.get_nowait()
                with suppress(asyncio.QueueFull):
                    event_queue.put_nowait(dict(message))
