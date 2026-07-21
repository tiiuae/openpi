"""FastAPI app: static UI, preset REST, health, and a telemetry WebSocket.

The hardware-bound runner factory is imported lazily inside create_app so the
REST/WebSocket layer is testable off-robot (where lerobot_robot_trossen is
absent) by injecting a fake factory.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import queue
import threading

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp import model_api
from webapp.config_store import ConfigStore
from webapp.metrics import Metrics
from webapp.model_process_manager import ModelProcessManager
from webapp.recording import RecordingSink
from webapp.recording import TeeSink
from webapp.robot_gateway_client import RobotCommandRejectedError
from webapp.robot_gateway_client import RobotGatewayClient
from webapp.robot_gateway_client import RobotGatewayError
from webapp.run_store import RunStore
from webapp.runner_factory import make_web_runner
from webapp.session import SessionManager
from webapp.telemetry import QueueSink

STATIC_DIR = Path(__file__).parent / "static"
URDF_PKG_DIR = Path(__file__).parent.parent / "external" / "joint_to_ee" / "trossen_arm_description"
URDF_FILE = URDF_PKG_DIR / "urdf" / "generated" / "mobile_ai.urdf"

logger = logging.getLogger(__name__)
ModelManagerFactory = Callable[[], ModelProcessManager]
RobotGatewayFactory = Callable[[], RobotGatewayClient | None]


def create_app(
    presets_dir: str | Path | None = None,
    runner_factory=None,
    feedback_dir: str | Path | None = None,
    runs_dir: str | Path | None = None,
    model_manager_factory: ModelManagerFactory | None = None,
    robot_gateway_factory: RobotGatewayFactory | None = None,
) -> FastAPI:
    if presets_dir is None:
        presets_dir = Path(__file__).parent / "presets"
    if feedback_dir is None:
        feedback_dir = Path(__file__).parent / "feedback"
    if runs_dir is None:
        runs_dir = Path(__file__).parent / "runs"
    if runner_factory is None:
        runner_factory = make_web_runner

    store = ConfigStore(presets_dir)
    session = SessionManager(runner_factory)
    lifecycle_lock = threading.RLock()
    runs_dir = Path(runs_dir)
    run_store = RunStore(runs_dir)
    if model_manager_factory is None:
        model_manager_factory = ModelProcessManager.from_environment
    if robot_gateway_factory is None:
        robot_gateway_factory = RobotGatewayClient.from_environment

    def _cleanup_on_shutdown(manager: ModelProcessManager) -> None:
        session_stopped = False
        with lifecycle_lock:
            try:
                session_stopped = session.stop()
            except Exception:
                logger.exception("Robot session cleanup failed during shutdown")

            try:
                manager.close()
            except Exception:
                logger.exception("Inference child cleanup failed during shutdown")
            finally:
                # Stopping inference can release a live session blocked in a
                # policy request, so give that session one final bounded join.
                if not session_stopped and session.is_running():
                    try:
                        if not session.stop(timeout=5.0):
                            logger.error("Robot session is still running after shutdown cleanup")
                    except Exception:
                        logger.exception("Final robot session cleanup failed")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = model_manager_factory()
        gateway = robot_gateway_factory()
        app.state.model_manager = manager
        app.state.robot_gateway = gateway
        if gateway is not None:
            await gateway.start()
        try:
            yield
        finally:
            # Closing the remote controller socket makes Laptop B stop locally;
            # do this before stopping inference so the robot is not left waiting
            # on a policy child that Machine A has already removed.
            if gateway is not None:
                await gateway.close()
            await asyncio.to_thread(_cleanup_on_shutdown, manager)
            app.state.model_manager = None
            app.state.robot_gateway = None

    app = FastAPI(title="Trossen Control", lifespan=lifespan)
    app.state.session_manager = session
    app.state.lifecycle_lock = lifecycle_lock
    app.state.model_manager = None
    app.state.robot_gateway = None
    app.state.browser_controller_lock = asyncio.Lock()
    app.include_router(model_api.router)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/pkg/trossen_arm_description", StaticFiles(directory=URDF_PKG_DIR), name="urdf_pkg")

    @app.get("/robot.urdf")
    def robot_urdf():
        return FileResponse(URDF_FILE, media_type="application/xml")

    @app.get("/favicon.ico")
    def favicon():
        # Browsers implicitly request /favicon.ico; serve the robot-arm icon so
        # it no longer 404s (the <link rel="icon"> tags also point here).
        return FileResponse(STATIC_DIR / "robot-arm.png", media_type="image/png")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/replay")
    def replay():
        return FileResponse(STATIC_DIR / "replay.html")

    @app.get("/runs")
    def runs_page():
        return FileResponse(STATIC_DIR / "runs.html")

    @app.get("/teleop")
    def teleop_page():
        return FileResponse(STATIC_DIR / "teleop.html")

    @app.get("/api/health")
    def health():
        manager = app.state.model_manager
        gateway = app.state.robot_gateway
        return {
            "session_running": gateway.is_running() if gateway is not None else session.is_running(),
            "robot_gateway": gateway.snapshot() if gateway is not None else {"mode": "local"},
            "model": manager.status() if manager is not None else {"state": "unavailable"},
        }

    @app.get("/api/presets")
    def list_presets():
        return store.list_names()

    @app.get("/api/presets/{name}")
    def get_preset(name: str):
        return store.load(name)

    @app.post("/api/presets")
    def save_preset(body: dict):
        store.save(body["name"], body["config"])
        return {"ok": True}

    @app.delete("/api/presets/{name}")
    def delete_preset(name: str):
        store.delete(name)
        return {"ok": True}

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

    @app.get("/api/runs")
    def api_list_runs():
        return run_store.list()

    @app.get("/api/runs/{run_id}")
    def api_get_run(run_id: str, events: int = 0):
        try:
            return run_store.get(run_id, with_events=bool(events))
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="run not found")

    @app.patch("/api/runs/{run_id}/rating")
    def api_rate_run(run_id: str, rating: dict):
        try:
            run_store.set_rating(run_id, rating)
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="run not found")
        return {"ok": True}

    @app.delete("/api/runs/{run_id}")
    def api_delete_run(run_id: str):
        try:
            run_store.delete(run_id)
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="run not found")
        return {"ok": True}

    @app.get("/api/files")
    def files(path: str | None = None):
        from webapp.files_api import list_directory

        return list_directory(path or str(Path.home()))

    @app.get("/api/episodes")
    def episodes(dataset_dir: str):
        from dataset_replay import EpisodeReader  # lazy

        r = EpisodeReader(dataset_dir)
        return {"fps": r.fps, "total_episodes": r.total_episodes}

    @app.get("/api/episode_trajectory")
    def episode_trajectory(
        dataset_dir: str,
        episode_index: int = 0,
        control_freq: int = 0,
        max_joint_speed: float = 3.0,
        ik_orientation_weight: float = 0.01,
        ik_pos_tol_m: float = 1e-3,
        ik_max_joint_jump_deg: float = 0.0,
    ):
        from fastapi import HTTPException

        from webapp import episode_preview  # lazy: pulls IK/dataset deps on demand

        # Pure off-robot preview: the trajectory is computed by IK only. We never
        # connect the arm here (connecting wakes/homes it). Frame-0 IK is seeded
        # from the home pose, exactly like the test-mode runner.
        try:
            return episode_preview.build_trajectory(
                dataset_dir=dataset_dir,
                episode_index=int(episode_index),
                control_freq=int(control_freq),
                max_joint_speed=float(max_joint_speed),
                seed_joints=None,
                ik_orientation_weight=float(ik_orientation_weight),
                ik_pos_tol_m=float(ik_pos_tol_m),
                max_joint_jump_deg=(float(ik_max_joint_jump_deg) if float(ik_max_joint_jump_deg) > 0 else None),
            )
        except Exception as exc:
            # e.g. a dataset without EE action columns, an unreadable episode, or
            # an IK/placo failure. Surface it as a 400 with the message so the
            # Preview page can show why instead of silently dropping the chart.
            logger.warning("episode_trajectory failed: %s", exc)
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}")

    @asynccontextmanager
    async def remote_lifecycle_operation():
        # model_api uses this same threading lock from a worker thread.  Polling
        # non-blockingly keeps the FastAPI event loop responsive while a model
        # start/stop operation finishes.
        while not lifecycle_lock.acquire(blocking=False):
            await asyncio.sleep(0.05)
        try:
            yield
        finally:
            lifecycle_lock.release()

    async def remote_telemetry_ws(ws: WebSocket, gateway: RobotGatewayClient) -> None:
        """Proxy one browser controller to the persistent Laptop B gateway."""

        await ws.accept()
        controller_lock = app.state.browser_controller_lock
        if controller_lock.locked():
            await ws.send_json(
                {
                    "type": "status",
                    "kind": "error",
                    "payload": {"message": "Another browser is controlling the robot"},
                }
            )
            await ws.close(code=1013)
            return

        await controller_lock.acquire()
        events = gateway.subscribe()
        metrics = Metrics()
        send_lock = asyncio.Lock()
        recorder: RecordingSink | None = None

        async def send_json(event: dict) -> None:
            async with send_lock:
                await ws.send_json(event)

        def new_run_id() -> str:
            import datetime as _dt

            stamp = _dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            run_id, number = stamp, 1
            while (runs_dir / run_id).exists():
                run_id = f"{stamp}-{number}"
                number += 1
            return run_id

        def close_recorder() -> None:
            nonlocal recorder
            if recorder is not None:
                recorder.close()
                recorder = None

        async def pump_gateway_events() -> None:
            nonlocal recorder
            while True:
                event = await events.get()
                if event.get("type") == "inference":
                    metrics.add_rtt(event["rtt_ms"])
                elif event.get("type") == "action":
                    metrics.add_action(event["ts"], event["action"])
                if recorder is not None:
                    recorder.ingest_event(event)
                if event.get("type") == "status" and event.get("kind") in {
                    "stopped",
                    "error",
                    "connect_failed",
                }:
                    close_recorder()
                await send_json(event)

        async def metrics_tick() -> None:
            while True:
                await asyncio.sleep(0.5)
                await send_json({"type": "metrics", **metrics.snapshot()})

        event_task = asyncio.create_task(pump_gateway_events())
        metrics_task = asyncio.create_task(metrics_tick())
        try:
            while True:
                command = await ws.receive_json()
                action = command.get("action")
                logger.info("Remote robot command: %s", action)
                if action not in {"start_live", "go_home", "go_sleep", "stop", "estop"}:
                    await send_json(
                        {
                            "type": "status",
                            "kind": "error",
                            "payload": {
                                "message": f"{action!r} is not supported by the remote robot gateway"
                            },
                        }
                    )
                    continue

                try:
                    async with remote_lifecycle_operation():
                        ack = await gateway.send_command(
                            action,
                            config=command.get("config") if action in {"start_live", "go_home", "go_sleep"} else None,
                        )
                except RobotCommandRejectedError as exc:
                    await send_json(
                        {"type": "status", "kind": "error", "payload": {"message": str(exc)}}
                    )
                    continue
                except RobotGatewayError as exc:
                    # The command is never retried.  Dropping the controller lease
                    # makes Laptop B stop locally if an ACK was lost after execution.
                    await send_json(
                        {"type": "status", "kind": "error", "payload": {"message": str(exc)}}
                    )
                    await gateway.stop_and_drop_controller()
                    continue

                if action == "start_live" and ack.get("state", {}).get("session") != "idle":
                    close_recorder()
                    run_id = new_run_id()
                    recorder = RecordingSink(runs_dir / run_id, run_id=run_id, config=command.get("config", {}))
                    run_started = {"type": "status", "kind": "run_started", "payload": {"run_id": run_id}}
                    recorder.ingest_event(run_started)
                    await send_json(run_started)
        except WebSocketDisconnect:
            logger.warning("Browser controller disconnected; stopping remote robot session")
        finally:
            event_task.cancel()
            metrics_task.cancel()
            await asyncio.gather(event_task, metrics_task, return_exceptions=True)
            gateway.unsubscribe(events)
            close_recorder()
            await gateway.stop_and_drop_controller()
            controller_lock.release()

    @app.websocket("/ws/telemetry")
    async def telemetry_ws(ws: WebSocket):
        gateway = app.state.robot_gateway
        if gateway is not None:
            await remote_telemetry_ws(ws, gateway)
            return

        await ws.accept()
        # Bounded: QueueSink drops the oldest event when full, so a fast control
        # loop can't build a backlog that lags the live chart behind the robot.
        q: queue.Queue = queue.Queue(maxsize=1000)
        metrics = Metrics()
        loop = asyncio.get_event_loop()

        async def pump():
            while True:
                try:
                    evt = await loop.run_in_executor(None, q.get, True, 0.5)
                except queue.Empty:
                    continue
                if evt["type"] == "inference":
                    metrics.add_rtt(evt["rtt_ms"])
                elif evt["type"] == "action":
                    metrics.add_action(evt["ts"], evt["action"])
                await ws.send_json(evt)

        async def metrics_tick():
            while True:
                await asyncio.sleep(0.5)
                await ws.send_json({"type": "metrics", **metrics.snapshot()})

        import datetime as _dt

        def _new_run_id() -> str:
            stamp = _dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            run_id, n = stamp, 1
            while (runs_dir / run_id).exists():
                run_id = f"{stamp}-{n}"
                n += 1
            return run_id

        def start(kind: str, config: dict):
            # Surface "a session is already running" (and any factory error) to
            # the UI as a status event instead of killing the WS handler.
            try:
                with lifecycle_lock:
                    sink = QueueSink(q)
                    if kind == "live":
                        run_id = _new_run_id()
                        recorder = RecordingSink(runs_dir / run_id, run_id=run_id, config=config)
                        sink = TeeSink([sink, recorder])
                        # Tell the browser which run this is (for the rating prompt).
                        q.put({"type": "status", "kind": "run_started", "payload": {"run_id": run_id}})
                    session.start(kind, config, sink)
            except Exception as exc:
                logger.warning("Cannot start %s session: %s", kind, exc)
                q.put({"type": "status", "kind": "error", "payload": {"message": str(exc)}})

        pump_task = asyncio.ensure_future(pump())
        metrics_task = asyncio.ensure_future(metrics_tick())
        try:
            while True:
                cmd = await ws.receive_json()
                action = cmd.get("action")
                logger.info("WS command: %s", action)
                if action == "start_live":
                    await asyncio.to_thread(start, "live", cmd["config"])
                elif action == "start_replay":
                    await asyncio.to_thread(start, "replay", cmd["config"])
                elif action == "go_sleep":
                    await asyncio.to_thread(start, "sleep", cmd.get("config", {}))
                elif action == "go_home":
                    await asyncio.to_thread(start, "home", cmd.get("config", {}))
                elif action == "stop":
                    await loop.run_in_executor(None, session.stop)
                elif action == "estop":
                    await loop.run_in_executor(None, session.estop)
                elif action == "start_teleop":
                    await asyncio.to_thread(start, "teleop", cmd["config"])
                elif action in ("teleop_input", "switch_arm"):
                    runner = getattr(session, "_runner", None)
                    if runner is not None and hasattr(runner, "input"):
                        payload = {"switch_arm": True} if action == "switch_arm" else cmd.get("payload", {})
                        runner.input.update(payload)
        except WebSocketDisconnect:
            logger.info("WS disconnected; stopping session")
            await loop.run_in_executor(None, session.stop)
        finally:
            pump_task.cancel()
            metrics_task.cancel()

    return app


app = create_app()  # for `uvicorn webapp.server:app`; safe off-robot — runners import is deferred to session start
