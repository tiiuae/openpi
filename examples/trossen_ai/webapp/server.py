"""FastAPI app: static UI, preset REST, health, and a telemetry WebSocket.

The hardware-bound runner factory is imported lazily inside create_app so the
REST/WebSocket layer is testable off-robot (where lerobot_robot_trossen is
absent) by injecting a fake factory.
"""
from __future__ import annotations

import asyncio
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

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

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

        pump_task = asyncio.ensure_future(pump())
        metrics_task = asyncio.ensure_future(metrics_tick())
        try:
            while True:
                cmd = await ws.receive_json()
                action = cmd.get("action")
                if action == "start_live":
                    session.start("live", cmd["config"], QueueSink(q))
                elif action == "start_replay":
                    session.start("replay", cmd["config"], QueueSink(q))
                elif action == "go_sleep":
                    session.start("sleep", cmd.get("config", {}), QueueSink(q))
                elif action == "go_home":
                    session.start("home", cmd.get("config", {}), QueueSink(q))
                elif action == "stop":
                    await loop.run_in_executor(None, session.stop)
                elif action == "estop":
                    await loop.run_in_executor(None, session.estop)
        except WebSocketDisconnect:
            await loop.run_in_executor(None, session.stop)
        finally:
            pump_task.cancel()
            metrics_task.cancel()

    return app


app = create_app()  # for `uvicorn webapp.server:app`; safe off-robot — runners import is deferred to session start
