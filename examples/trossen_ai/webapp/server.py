"""FastAPI app: static UI, preset REST, health, and a telemetry WebSocket.

The hardware-bound runner factory is imported lazily inside create_app so the
REST/WebSocket layer is testable off-robot (where lerobot_robot_trossen is
absent) by injecting a fake factory.
"""
from __future__ import annotations

import asyncio
import logging
import queue
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.config_store import ConfigStore
from webapp.metrics import Metrics
from webapp.session import SessionManager
from webapp.telemetry import QueueSink

STATIC_DIR = Path(__file__).parent / "static"
URDF_PKG_DIR = (Path(__file__).parent.parent / "external" / "joint_to_ee"
                / "trossen_arm_description")
URDF_FILE = URDF_PKG_DIR / "urdf" / "generated" / "mobile_ai.urdf"

logger = logging.getLogger(__name__)


def create_app(presets_dir: str | Path | None = None, runner_factory=None,
               feedback_dir: str | Path | None = None) -> FastAPI:
    if presets_dir is None:
        presets_dir = Path(__file__).parent / "presets"
    if feedback_dir is None:
        feedback_dir = Path(__file__).parent / "feedback"
    if runner_factory is None:
        # Defer the robot-only import until a session actually starts, so the
        # REST/WebSocket layer (and the tests) work off-robot where
        # lerobot_robot_trossen is absent.
        def runner_factory(kind, config, sink):
            if kind == "sleep":
                from webapp.movers import SleepRunner
                return SleepRunner(kind, config, sink)
            if kind == "home":
                from webapp.movers import HomeRunner
                return HomeRunner(kind, config, sink)
            if kind == "teleop":
                from webapp.teleop_runner import TeleopRunner
                return TeleopRunner(kind, config, sink)
            from webapp.runners import make_runner
            return make_runner(kind, config, sink)

    app = FastAPI(title="Trossen Control")
    store = ConfigStore(presets_dir)
    session = SessionManager(runner_factory)

    @app.on_event("shutdown")
    def _cleanup_on_shutdown():
        try:
            session.stop()
        except Exception:  # noqa: BLE001
            pass

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/pkg/trossen_arm_description",
              StaticFiles(directory=URDF_PKG_DIR), name="urdf_pkg")

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

    @app.get("/teleop")
    def teleop_page():
        return FileResponse(STATIC_DIR / "teleop.html")

    @app.get("/api/health")
    def health():
        return {"session_running": session.is_running()}

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
    def episode_trajectory(dataset_dir: str, episode_index: int = 0,
                           control_freq: int = 0, max_joint_speed: float = 3.0,
                           ik_orientation_weight: float = 0.01, ik_pos_tol_m: float = 1e-3,
                           ik_max_joint_jump_deg: float = 0.0):
        from fastapi import HTTPException
        from webapp import episode_preview  # lazy: pulls IK/dataset deps on demand
        # Pure off-robot preview: the trajectory is computed by IK only. We never
        # connect the arm here (connecting wakes/homes it). Frame-0 IK is seeded
        # from the home pose, exactly like the test-mode runner.
        try:
            return episode_preview.build_trajectory(
                dataset_dir=dataset_dir, episode_index=int(episode_index),
                control_freq=int(control_freq), max_joint_speed=float(max_joint_speed),
                seed_joints=None,
                ik_orientation_weight=float(ik_orientation_weight),
                ik_pos_tol_m=float(ik_pos_tol_m),
                max_joint_jump_deg=(float(ik_max_joint_jump_deg)
                                    if float(ik_max_joint_jump_deg) > 0 else None),
            )
        except Exception as exc:  # noqa: BLE001 — report the reason to the UI
            # e.g. a dataset without EE action columns, an unreadable episode, or
            # an IK/placo failure. Surface it as a 400 with the message so the
            # Preview page can show why instead of silently dropping the chart.
            logger.warning("episode_trajectory failed: %s", exc)
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}")

    @app.websocket("/ws/telemetry")
    async def telemetry_ws(ws: WebSocket):
        await ws.accept()
        q: queue.Queue = queue.Queue()
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

        def start(kind: str, config: dict):
            # Surface "a session is already running" (and any factory error) to
            # the UI as a status event instead of killing the WS handler.
            try:
                session.start(kind, config, QueueSink(q))
            except Exception as exc:  # noqa: BLE001
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
                    start("live", cmd["config"])
                elif action == "start_replay":
                    start("replay", cmd["config"])
                elif action == "go_sleep":
                    start("sleep", cmd.get("config", {}))
                elif action == "go_home":
                    start("home", cmd.get("config", {}))
                elif action == "stop":
                    await loop.run_in_executor(None, session.stop)
                elif action == "estop":
                    await loop.run_in_executor(None, session.estop)
                elif action == "start_teleop":
                    start("teleop", cmd["config"])
                elif action in ("teleop_input", "switch_arm"):
                    runner = getattr(session, "_runner", None)
                    if runner is not None and hasattr(runner, "input"):
                        payload = ({"switch_arm": True} if action == "switch_arm"
                                   else cmd.get("payload", {}))
                        runner.input.update(payload)
        except WebSocketDisconnect:
            logger.info("WS disconnected; stopping session")
            await loop.run_in_executor(None, session.stop)
        finally:
            pump_task.cancel()
            metrics_task.cancel()

    return app


app = create_app()  # for `uvicorn webapp.server:app`; safe off-robot — runners import is deferred to session start
