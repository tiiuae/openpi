import asyncio
import json
import queue
import threading
import time
import uuid

from fastapi.testclient import TestClient
import pytest

from webapp.gateway_protocol import PROTOCOL_VERSION
from webapp.robot_gateway_client import RobotGatewayClient
from webapp.robot_gateway_client import RobotGatewayClientSettings
from webapp.robot_gateway_client import RobotGatewayUnavailableError
from webapp.robot_gateway_server import create_gateway_app
from webapp.robot_gateway_server import RobotGatewayController
from webapp.robot_gateway_server import RobotGatewaySettings
from webapp.server import create_app
from webapp.session import SessionManager
from webapp.telemetry import QueueSink


class _BlockingRunner:
    def __init__(self, kind, config, sink, created, release):
        self.kind = kind
        self.config = config
        self.sink = sink
        self.release = release
        created.append(self)

    def run(self):
        self.sink.on_status("started", {"kind": self.kind})
        self.release.wait(timeout=5)
        self.sink.on_status("stopped", {"reason": "finished"})

    def stop(self):
        self.release.set()

    def estop(self):
        self.release.set()


def _command(action, *, command_id=None, config=None):
    message = {
        "type": "command",
        "protocol": PROTOCOL_VERSION,
        "command_id": command_id or str(uuid.uuid4()),
        "action": action,
    }
    if config is not None:
        message["config"] = config
    return message


def test_gateway_startup_and_health_never_construct_runner():
    created = []
    app = create_gateway_app(runner_factory=lambda *args: created.append(args))

    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["state"]["session"] == "idle"
        with client.websocket_connect("/ws/robot") as ws:
            assert ws.receive_json()["type"] == "hello"

    assert created == []


def test_controller_start_is_lazy_idempotent_and_forces_tunnel_policy():
    created = []
    release = threading.Event()

    def factory(kind, config, sink):
        return _BlockingRunner(kind, config, sink, created, release)

    async def scenario():
        session = SessionManager(factory)
        controller = RobotGatewayController(
            session,
            RobotGatewaySettings(policy_host="127.0.0.1", policy_port=18800),
        )
        sink = QueueSink(queue.Queue())
        command_id = str(uuid.uuid4())
        message = _command(
            "start_live",
            command_id=command_id,
            config={"policy_host": "untrusted.example", "policy_port": 9999},
        )

        first = await controller.dispatch(message, sink)
        duplicate = await controller.dispatch(message, sink)
        deadline = time.monotonic() + 2
        while not created and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

        assert first["accepted"] is True
        assert duplicate == first
        assert len(created) == 1
        assert created[0].config["policy_host"] == "127.0.0.1"
        assert created[0].config["policy_port"] == 18800

        await controller.stop_for_disconnect()
        assert not session.is_running()

    asyncio.run(scenario())


def test_gateway_websocket_disconnect_stops_local_session():
    created = []
    release = threading.Event()

    def factory(kind, config, sink):
        return _BlockingRunner(kind, config, sink, created, release)

    app = create_gateway_app(
        runner_factory=factory,
        settings=RobotGatewaySettings(heartbeat_timeout=30),
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws/robot") as ws:
            assert ws.receive_json()["type"] == "hello"
            command = _command("start_live", config={})
            ws.send_json(command)
            for _ in range(10):
                message = ws.receive_json()
                if message.get("type") == "ack":
                    assert message["accepted"] is True
                    break
            else:
                pytest.fail("gateway did not acknowledge start_live")

        deadline = time.monotonic() + 2
        while app.state.session_manager.is_running() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not app.state.session_manager.is_running()
        assert release.is_set()


def test_gateway_client_does_not_queue_commands_while_offline():
    async def scenario():
        client = RobotGatewayClient(RobotGatewayClientSettings(url="ws://127.0.0.1:8001/ws/robot"))
        with pytest.raises(RobotGatewayUnavailableError):
            await client.send_command("start_live", config={})

    asyncio.run(scenario())


def test_gateway_client_correlates_ack_without_retry():
    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, value):
            self.sent.append(json.loads(value))

    async def scenario():
        client = RobotGatewayClient(RobotGatewayClientSettings(url="ws://127.0.0.1:8001/ws/robot"))
        websocket = FakeWebSocket()
        client._connected = True
        client._ws = websocket

        task = asyncio.create_task(client.send_command("start_live", config={"x": 1}))
        while not websocket.sent:
            await asyncio.sleep(0)
        sent = websocket.sent[0]
        assert sent["action"] == "start_live"
        assert len(websocket.sent) == 1
        client._pending[sent["command_id"]].set_result(
            {
                "type": "ack",
                "command_id": sent["command_id"],
                "action": "start_live",
                "accepted": True,
                "state": {"gateway": "ready", "session": "busy", "kind": "live", "boot_id": "b"},
            }
        )
        acknowledgement = await task
        assert acknowledgement["accepted"] is True
        assert len(websocket.sent) == 1

    asyncio.run(scenario())


def test_machine_a_remote_mode_forwards_start_without_local_runner(tmp_path):
    class FakeModelManager:
        def status(self):
            return {"state": "stopped"}

        def close(self):
            pass

    class FakeGateway:
        def __init__(self):
            self.commands = []
            self.event_queue = None
            self.started = False
            self.closed = False
            self.dropped = 0

        async def start(self):
            self.started = True

        async def close(self):
            self.closed = True

        def subscribe(self, maxsize=1000):
            self.event_queue = asyncio.Queue(maxsize=maxsize)
            self.event_queue.put_nowait({"type": "gateway", **self.snapshot()})
            return self.event_queue

        def unsubscribe(self, event_queue):
            assert event_queue is self.event_queue

        async def send_command(self, action, *, config=None, timeout=None):
            self.commands.append((action, config))
            if action == "start_live":
                self.event_queue.put_nowait(
                    {"type": "status", "kind": "connecting", "payload": {"host": "127.0.0.1", "port": 8800}}
                )
                return {
                    "accepted": True,
                    "state": {"gateway": "ready", "session": "busy", "kind": "live", "boot_id": "boot"},
                }
            return {
                "accepted": True,
                "state": {"gateway": "ready", "session": "idle", "kind": None, "boot_id": "boot"},
            }

        async def stop_and_drop_controller(self):
            self.dropped += 1

        def snapshot(self):
            return {
                "connected": True,
                "state_known": True,
                "possible_active_session": False,
                "url": "ws://127.0.0.1:8001/ws/robot",
                "state": {"gateway": "ready", "session": "idle", "kind": None, "boot_id": "boot"},
            }

        def is_running(self):
            return False

        def blocks_model_change(self):
            return False

    gateway = FakeGateway()

    def forbidden_runner(*_args):
        raise AssertionError("Machine A must not construct a local robot runner in remote mode")

    app = create_app(
        runs_dir=tmp_path,
        runner_factory=forbidden_runner,
        model_manager_factory=FakeModelManager,
        robot_gateway_factory=lambda: gateway,
    )
    with TestClient(app) as client:
        assert client.get("/api/health").json()["robot_gateway"]["connected"] is True
        with client.websocket_connect("/ws/telemetry") as ws:
            ws.send_json({"action": "start_live", "config": {"task_prompt": "test"}})
            saw_run = False
            for _ in range(10):
                event = ws.receive_json()
                if event.get("type") == "status" and event.get("kind") == "run_started":
                    saw_run = True
                    break
            assert saw_run

    assert gateway.started is True
    assert gateway.closed is True
    assert gateway.commands == [("start_live", {"task_prompt": "test"})]
    assert gateway.dropped == 1
