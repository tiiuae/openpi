"""REST routes for the web application's checkpoint process manager."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request
from pydantic import BaseModel
from pydantic import Field

from webapp.model_process_manager import CheckpointNotFoundError
from webapp.model_process_manager import InferencePortInUseError
from webapp.model_process_manager import InvalidCheckpointError
from webapp.model_process_manager import ManagerClosedError
from webapp.model_process_manager import ModelConflictError
from webapp.model_process_manager import ModelManagerError
from webapp.model_process_manager import ModelProcessManager
from webapp.model_process_manager import ModelStartError
from webapp.model_process_manager import ModelStopError

router = APIRouter(prefix="/api/model", tags=["model serving"])


class StartModelRequest(BaseModel):
    checkpoint_id: str = Field(min_length=1, max_length=1024)
    unnorm_key: str | None = Field(default=None, min_length=1, max_length=256)
    use_proprio: bool | None = None
    proprio_mode: Literal["tokens", "film"] | None = None


def _manager(request: Request) -> ModelProcessManager:
    manager = getattr(request.app.state, "model_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Model manager is not running")
    return manager


def _require_idle_robot_session(request: Request) -> None:
    gateway = getattr(request.app.state, "robot_gateway", None)
    if gateway is not None and gateway.blocks_model_change():
        raise HTTPException(
            status_code=409,
            detail="Stop the remote robot session before starting or stopping a checkpoint",
        )
    session = getattr(request.app.state, "session_manager", None)
    if gateway is None and session is not None and session.is_running():
        raise HTTPException(
            status_code=409,
            detail="Stop the active robot session before starting or stopping a checkpoint",
        )


def _lifecycle_lock(request: Request):
    lock = getattr(request.app.state, "lifecycle_lock", None)
    if lock is None:
        raise HTTPException(status_code=503, detail="Lifecycle interlock is not running")
    return lock


@router.get("/checkpoints")
def list_checkpoints(request: Request):
    manager = _manager(request)
    try:
        checkpoints = manager.list_checkpoints()
    except ManagerClosedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "checkpoints": checkpoints,
        "discovery": manager.discovery_status(),
    }


@router.get("/status")
def model_status(request: Request):
    return _manager(request).status()


@router.post("/start", status_code=202)
def start_model(body: StartModelRequest, request: Request):
    with _lifecycle_lock(request):
        _require_idle_robot_session(request)
        try:
            return _manager(request).start(
                body.checkpoint_id,
                unnorm_key=body.unnorm_key,
                use_proprio=body.use_proprio,
                proprio_mode=body.proprio_mode,
            )
        except CheckpointNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidCheckpointError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (ModelConflictError, InferencePortInUseError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ManagerClosedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ModelStartError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ModelManagerError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/stop")
def stop_model(request: Request):
    with _lifecycle_lock(request):
        _require_idle_robot_session(request)
        try:
            return _manager(request).stop()
        except ManagerClosedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ModelStopError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ModelManagerError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
