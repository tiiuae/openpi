"""Manage one local OpenPI checkpoint server as a child process.

This module intentionally does not import :mod:`openpi`.  The web application
stays lightweight and launches ``scripts/serve_policy.py`` with the repository's
model environment only when a checkpoint is requested.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import dataclasses
import datetime as dt
import enum
import errno
import fcntl
import json
import logging
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import signal
import socket
import subprocess
import tempfile
import threading
import time
from typing import Any, Literal
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_ROOT = Path("/home")
_PROCESSOR_MARKERS = {
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
}
_WEIGHT_MARKERS = {
    "model.safetensors",
    "pytorch_model.bin",
}
_WEIGHT_INDEXES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}
_INCOMPLETE_SUFFIXES = (".incomplete", ".partial", ".tmp")


class ModelState(str, enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


class ModelManagerError(RuntimeError):
    """Base error raised by :class:`ModelProcessManager`."""


class ManagerAlreadyRunningError(ModelManagerError):
    """Another web process already owns the exclusive manager lock."""


class ManagerClosedError(ModelManagerError):
    """The manager was used after application shutdown began."""


class CheckpointNotFoundError(ModelManagerError):
    """A requested checkpoint ID is not selectable under the trusted root."""


class InvalidCheckpointError(ModelManagerError):
    """A selected directory stopped being a valid FalconVLA checkpoint."""


class ModelConflictError(ModelManagerError):
    """A different checkpoint is already starting or running."""


class InferencePortInUseError(ModelManagerError):
    """The inference port is occupied by a process this manager does not own."""


class ModelStartError(ModelManagerError):
    """The inference child could not be created."""


class ModelStopError(ModelManagerError):
    """The inference process group could not be stopped and reaped."""


GpuSampler = Callable[[int], dict[str, Any]]
ProprioMode = Literal["tokens", "film"]


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _as_finite_number(value: Any) -> int | float | None:
    """Convert a metric to a finite JSON number, or ``None`` when unavailable."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _safe_number(call: Callable[[], Any]) -> int | float | None:
    try:
        return _as_finite_number(call())
    except Exception:  # NVML wrappers can raise driver-specific exceptions.
        return None


def _safe_text(call: Callable[[], Any]) -> str | None:
    try:
        return str(call())
    except Exception:
        return None


def sample_gpu_for_pid(pid: int) -> dict[str, Any]:
    """Return a JSON-safe nvitop snapshot without making monitoring fatal."""

    sampled_at = _utc_now()
    try:
        from nvitop import Device
    except (ImportError, OSError) as exc:
        return {
            "available": False,
            "sampled_at": sampled_at,
            "error": f"nvitop unavailable: {exc}",
        }

    try:
        devices = Device.all()
    except Exception as exc:  # NVML/driver exception types vary by release.
        return {
            "available": False,
            "sampled_at": sampled_at,
            "error": f"GPU query failed: {exc}",
        }

    if not devices:
        return {
            "available": False,
            "sampled_at": sampled_at,
            "error": "No NVIDIA GPU is visible",
        }

    rows: list[dict[str, Any]] = []
    child_visible = False
    for device in devices:
        row: dict[str, Any] = {
            "index": str(device.index),
            "name": _safe_text(device.name),
            "gpu_utilization_percent": _safe_number(device.gpu_utilization),
            "memory_used_bytes": _safe_number(device.memory_used),
            "memory_total_bytes": _safe_number(device.memory_total),
            "temperature_c": _safe_number(device.temperature),
            "power_usage_mw": _safe_number(device.power_usage),
            "child_memory_bytes": None,
        }
        try:
            child_process = device.processes().get(pid)
            child_visible = child_visible or child_process is not None
            if child_process is not None:
                row["child_memory_bytes"] = _safe_number(child_process.gpu_memory)
        except Exception as exc:
            row["process_query_error"] = str(exc)
        rows.append(row)

    return {
        "available": True,
        "sampled_at": sampled_at,
        "child_pid": pid,
        "child_visible_on_gpu": child_visible,
        "devices": rows,
    }


@dataclasses.dataclass(frozen=True)
class ModelManagerSettings:
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    inference_host: str = "127.0.0.1"
    health_host: str = "127.0.0.1"
    inference_port: int = 8800
    startup_timeout: float = 900.0
    shutdown_timeout: float = 30.0
    monitor_interval: float = 2.0
    health_timeout: float = 1.0
    discovery_max_depth: int = 6
    discovery_max_directories: int = 5000
    discovery_max_entries: int = 20000
    lock_path: Path | None = None
    repo_root: Path | None = None
    python_executable: Path | None = None

    @classmethod
    def from_environment(cls) -> ModelManagerSettings:
        default_repo_root = Path(__file__).resolve().parents[3]
        repo_root = Path(os.environ.get("OPENPI_REPO_ROOT", str(default_repo_root))).expanduser()
        lock_value = os.environ.get("OPENPI_MANAGER_LOCK")
        return cls(
            checkpoint_root=Path(os.environ.get("OPENPI_CHECKPOINT_ROOT", str(DEFAULT_CHECKPOINT_ROOT))).expanduser(),
            inference_host=os.environ.get("OPENPI_INFERENCE_HOST", "127.0.0.1"),
            health_host=os.environ.get("OPENPI_HEALTH_HOST", "127.0.0.1"),
            inference_port=int(os.environ.get("OPENPI_INFERENCE_PORT", "8800")),
            startup_timeout=float(os.environ.get("OPENPI_STARTUP_TIMEOUT", "900")),
            shutdown_timeout=float(os.environ.get("OPENPI_SHUTDOWN_TIMEOUT", "30")),
            monitor_interval=float(os.environ.get("OPENPI_MONITOR_INTERVAL", "2")),
            health_timeout=float(os.environ.get("OPENPI_HEALTH_TIMEOUT", "1")),
            discovery_max_depth=int(os.environ.get("OPENPI_DISCOVERY_MAX_DEPTH", "6")),
            discovery_max_directories=int(os.environ.get("OPENPI_DISCOVERY_MAX_DIRECTORIES", "5000")),
            discovery_max_entries=int(os.environ.get("OPENPI_DISCOVERY_MAX_ENTRIES", "20000")),
            lock_path=Path(lock_value).expanduser() if lock_value else None,
            repo_root=repo_root,
            python_executable=Path(
                os.environ.get("OPENPI_INFERENCE_PYTHON", str(repo_root / ".venv" / "bin" / "python"))
            ).expanduser(),
        )


class ModelProcessManager:
    """Own exactly one ``serve_policy.py`` process for the app lifetime."""

    def __init__(
        self,
        settings: ModelManagerSettings,
        *,
        gpu_sampler: GpuSampler = sample_gpu_for_pid,
    ) -> None:
        checkpoint_root = settings.checkpoint_root.expanduser().resolve(strict=False)
        if checkpoint_root.exists() and not checkpoint_root.is_dir():
            raise NotADirectoryError(f"Checkpoint root is not a directory: {checkpoint_root}")
        if not 1 <= settings.inference_port <= 65535:
            raise ValueError(f"Invalid inference port: {settings.inference_port}")
        if (
            settings.startup_timeout <= 0
            or settings.shutdown_timeout <= 0
            or settings.monitor_interval <= 0
            or settings.health_timeout <= 0
        ):
            raise ValueError("Timeouts and monitor interval must be positive")
        if (
            settings.discovery_max_depth < 0
            or settings.discovery_max_directories < 1
            or settings.discovery_max_entries < 1
        ):
            raise ValueError("Checkpoint discovery limits must be positive")

        repo_root = (
            settings.repo_root.expanduser().resolve(strict=True)
            if settings.repo_root is not None
            else Path(__file__).resolve().parents[3]
        )
        serve_script = repo_root / "scripts" / "serve_policy.py"
        if not serve_script.is_file():
            raise FileNotFoundError(f"serve_policy.py not found: {serve_script}")
        python_executable = (
            settings.python_executable.expanduser()
            if settings.python_executable is not None
            else repo_root / ".venv" / "bin" / "python"
        )
        self.checkpoint_root = checkpoint_root
        self.inference_host = settings.inference_host
        self.health_host = settings.health_host
        self.inference_port = settings.inference_port
        self.startup_timeout = settings.startup_timeout
        self.shutdown_timeout = settings.shutdown_timeout
        self.monitor_interval = settings.monitor_interval
        self.health_timeout = settings.health_timeout
        self.discovery_max_depth = settings.discovery_max_depth
        self.discovery_max_directories = settings.discovery_max_directories
        self.discovery_max_entries = settings.discovery_max_entries
        self.repo_root = repo_root
        self.serve_script = serve_script
        self.python_executable = str(python_executable)
        self._gpu_sampler = gpu_sampler

        lock_path = settings.lock_path
        if lock_path is None:
            uid = getattr(os, "getuid", lambda: 0)()
            lock_path = Path(tempfile.gettempdir()) / (f"openpi-model-manager-{uid}-{self.inference_port}.lock")
        self.lock_path = lock_path.expanduser()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            lock_flags |= os.O_NOFOLLOW
        lock_fd = os.open(self.lock_path, lock_flags, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(lock_fd)
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise ManagerAlreadyRunningError(
                f"Another model manager owns {self.lock_path}; run the web app with one worker"
            ) from exc
        self._manager_lock_fd = lock_fd

        self._operation_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._discovery_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._state = ModelState.STOPPED
        self._checkpoint_id: str | None = None
        self._start_options: dict[str, Any] = {}
        self._process: subprocess.Popen[bytes] | None = None
        self._last_pid: int | None = None
        self._exit_code: int | None = None
        self._error: str | None = None
        self._health_ok = False
        self._health_checked_at: str | None = None
        self._gpu: dict[str, Any] = {
            "available": False,
            "error": "No inference child is running",
        }
        self._started_at: str | None = None
        self._ready_at: str | None = None
        self._generation = 0
        self._expected_stop = False
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop: threading.Event | None = None
        self._last_discovery: dict[str, Any] = {
            "checked_at": None,
            "directories_scanned": 0,
            "truncated": False,
        }
        self._health_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def from_environment(cls) -> ModelProcessManager:
        return cls(ModelManagerSettings.from_environment())

    def __enter__(self) -> ModelProcessManager:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def list_checkpoints(self) -> list[dict[str, str]]:
        """Return bounded-recursive, safe relative checkpoint IDs."""

        with self._operation_lock:
            self._ensure_open()
            with self._discovery_lock:
                checkpoints = self._discover_checkpoints_locked()
        return [{"id": checkpoint_id, "name": path.name} for checkpoint_id, path in sorted(checkpoints.items())]

    def discovery_status(self) -> dict[str, Any]:
        with self._discovery_lock:
            return dict(self._last_discovery)

    def resolve_checkpoint(self, checkpoint_id: str) -> Path:
        """Resolve an ID only when it appears in a fresh discovery allowlist."""

        if not self._is_safe_checkpoint_id(checkpoint_id):
            raise CheckpointNotFoundError(f"Invalid checkpoint ID: {checkpoint_id!r}")
        with self._discovery_lock:
            selectable = self._discover_checkpoints_locked()
        checkpoint = selectable.get(checkpoint_id)
        if checkpoint is None:
            raise CheckpointNotFoundError(f"Checkpoint is not selectable under the configured root: {checkpoint_id}")
        try:
            resolved = checkpoint.resolve(strict=True)
            resolved.relative_to(self.checkpoint_root)
        except (FileNotFoundError, ValueError) as exc:
            raise CheckpointNotFoundError(f"Checkpoint disappeared from the configured root: {checkpoint_id}") from exc
        if not resolved.is_dir() or not self._has_checkpoint_files(resolved):
            raise InvalidCheckpointError(f"{checkpoint_id!r} is not a self-describing FalconVLA checkpoint")
        return resolved

    def start(
        self,
        checkpoint_id: str,
        *,
        unnorm_key: str | None = None,
        use_proprio: bool | None = None,
        proprio_mode: ProprioMode | None = None,
    ) -> dict[str, Any]:
        """Create the inference child and return while the model is loading."""

        with self._operation_lock:
            self._ensure_open()
            if unnorm_key is not None and not unnorm_key:
                raise InvalidCheckpointError("unnorm_key must not be empty")
            if proprio_mode not in {None, "tokens", "film"}:
                raise InvalidCheckpointError("proprio_mode must be 'tokens' or 'film'")
            if (use_proprio is None) != (proprio_mode is None):
                raise InvalidCheckpointError("use_proprio and proprio_mode must be provided together")
            start_options = {
                key: value
                for key, value in {
                    "unnorm_key": unnorm_key,
                    "use_proprio": use_proprio,
                    "proprio_mode": proprio_mode,
                }.items()
                if value is not None
            }
            checkpoint = self.resolve_checkpoint(checkpoint_id)

            with self._state_lock:
                process = self._process
                if process is not None and process.poll() is None:
                    if checkpoint_id == self._checkpoint_id and start_options == self._start_options:
                        return self._status_locked()
                    raise ModelConflictError(
                        f"Checkpoint {self._checkpoint_id!r} is already {self._state.value}; "
                        "stop it before changing the checkpoint or its overrides"
                    )

            if not Path(self.python_executable).is_file():
                error = f"Inference Python interpreter not found: {self.python_executable}"
                self._record_start_failure(checkpoint_id, start_options, error)
                raise ModelStartError(error)

            if self._port_is_in_use():
                error = f"Inference port {self.inference_port} is already in use; refusing to kill an unrelated process"
                self._record_start_failure(checkpoint_id, start_options, error)
                raise InferencePortInUseError(error)

            command = [
                self.python_executable,
                str(self.serve_script),
                "--host",
                self.inference_host,
                "--port",
                str(self.inference_port),
                "policy:auto-checkpoint",
                f"--policy.dir={checkpoint}",
            ]
            if unnorm_key is not None:
                command.append(f"--policy.unnorm-key={unnorm_key}")
            if use_proprio is not None:
                command.append(f"--policy.use-proprio={use_proprio}")
            if proprio_mode is not None:
                command.append(f"--policy.proprio-mode={proprio_mode}")
            logger.info("Starting checkpoint %s on port %d", checkpoint_id, self.inference_port)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=self.repo_root,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as exc:
                error = f"Could not start inference child: {exc}"
                self._record_start_failure(checkpoint_id, start_options, error)
                raise ModelStartError(error) from exc

            monitor_stop = threading.Event()
            with self._state_lock:
                self._generation += 1
                generation = self._generation
                self._state = ModelState.STARTING
                self._checkpoint_id = checkpoint_id
                self._start_options = start_options
                self._process = process
                self._last_pid = process.pid
                self._exit_code = None
                self._error = None
                self._health_ok = False
                self._health_checked_at = None
                self._gpu = {
                    "available": False,
                    "error": "Waiting for the model to allocate the GPU",
                }
                self._started_at = _utc_now()
                self._ready_at = None
                self._expected_stop = False
                self._monitor_stop = monitor_stop

            try:
                monitor = threading.Thread(
                    target=self._monitor_child,
                    args=(process, generation, monitor_stop),
                    name=f"openpi-model-monitor-{process.pid}",
                    daemon=True,
                )
                with self._state_lock:
                    self._monitor_thread = monitor
                monitor.start()
            except Exception as exc:
                monitor_stop.set()
                cleanup_error: ModelStopError | None = None
                exit_code: int | None = None
                try:
                    exit_code = self._terminate_process_group(process)
                except ModelStopError as stop_exc:
                    cleanup_error = stop_exc
                error = f"Could not start inference monitor: {exc}"
                if cleanup_error is not None:
                    error = f"{error}; child cleanup failed: {cleanup_error}"
                with self._state_lock:
                    self._generation += 1
                    self._state = ModelState.FAILED
                    self._process = process if cleanup_error is not None else None
                    self._exit_code = exit_code
                    self._error = error
                    self._health_ok = False
                    self._health_checked_at = _utc_now()
                    self._gpu = {
                        "available": False,
                        "error": "Inference monitor failed to start",
                    }
                    self._ready_at = None
                    self._monitor_thread = None
                    self._monitor_stop = None
                raise ModelStartError(error) from exc
            return self.status()

    def stop(self) -> dict[str, Any]:
        """Stop the owned process group and wait until its leader is reaped."""

        monitor: threading.Thread | None = None
        stop_error: ModelStopError | None = None
        result: dict[str, Any]
        with self._operation_lock:
            self._ensure_open(allow_closing=True)
            with self._state_lock:
                process = self._process
                monitor = self._monitor_thread
                monitor_stop = self._monitor_stop
                if process is None:
                    self._state = ModelState.STOPPED
                    self._health_ok = False
                    self._error = None
                    self._gpu = {
                        "available": False,
                        "error": "No inference child is running",
                    }
                    result = self._status_locked()
                else:
                    self._state = ModelState.STOPPING
                    self._expected_stop = True
                    self._generation += 1
                    if monitor_stop is not None:
                        monitor_stop.set()

            if process is not None:
                try:
                    exit_code = (
                        process.returncode if process.poll() is not None else self._terminate_process_group(process)
                    )
                except ModelStopError as exc:
                    stop_error = exc
                    with self._state_lock:
                        self._state = ModelState.FAILED
                        self._error = str(exc)
                        self._health_ok = False
                        self._health_checked_at = _utc_now()
                        result = self._status_locked()
                else:
                    with self._state_lock:
                        if self._process is process:
                            self._process = None
                        self._state = ModelState.STOPPED
                        self._exit_code = exit_code
                        self._health_ok = False
                        self._health_checked_at = _utc_now()
                        self._error = None
                        self._gpu = {
                            "available": False,
                            "error": "No inference child is running",
                        }
                        self._monitor_thread = None
                        self._monitor_stop = None
                        result = self._status_locked()

        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=self.monitor_interval + self.health_timeout + 1.0)
            if monitor.is_alive():
                logger.warning("Inference monitor thread did not stop promptly; it will exit with the web process")
        if stop_error is not None:
            raise stop_error
        return result

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return self._status_locked()

    def close(self) -> None:
        """Stop the child before releasing the lifetime manager lock."""

        with self._operation_lock:
            if self._closed or self._closing:
                return
            self._closing = True
        try:
            self.stop()
        except Exception:
            with self._operation_lock:
                self._closing = False
            raise
        with self._operation_lock:
            self._closed = True
            try:
                fcntl.flock(self._manager_lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._manager_lock_fd)

    def _status_locked(self) -> dict[str, Any]:
        process = self._process
        if process is not None:
            exit_code = process.poll()
            if exit_code is not None:
                if self._monitor_stop is not None:
                    self._monitor_stop.set()
                self._generation += 1
                self._process = None
                self._exit_code = exit_code
                self._health_ok = False
                self._health_checked_at = _utc_now()
                self._gpu = {
                    "available": False,
                    "error": "No inference child is running",
                }
                if self._expected_stop or self._state == ModelState.STOPPING:
                    self._state = ModelState.STOPPED
                    self._error = None
                else:
                    self._state = ModelState.FAILED
                    self._error = f"Inference child exited with status {exit_code}"
                self._monitor_thread = None
                self._monitor_stop = None
        live_pid = self._process.pid if self._process is not None and self._process.poll() is None else None
        return {
            "state": self._state.value,
            "checkpoint_id": self._checkpoint_id,
            "start_options": dict(self._start_options),
            "pid": live_pid,
            "last_pid": self._last_pid,
            "exit_code": self._exit_code,
            "error": self._error,
            "healthz": {"ok": self._health_ok, "checked_at": self._health_checked_at},
            "gpu": self._gpu,
            "started_at": self._started_at,
            "ready_at": self._ready_at,
            "inference": {"host": self.inference_host, "port": self.inference_port},
        }

    def _record_start_failure(
        self,
        checkpoint_id: str,
        start_options: dict[str, Any],
        error: str,
    ) -> None:
        """Record a failed launch without retaining state from an older child."""

        with self._state_lock:
            if self._monitor_stop is not None:
                self._monitor_stop.set()
            self._generation += 1
            self._state = ModelState.FAILED
            self._checkpoint_id = checkpoint_id
            self._start_options = dict(start_options)
            self._process = None
            self._last_pid = None
            self._exit_code = None
            self._error = error
            self._health_ok = False
            self._health_checked_at = _utc_now()
            self._gpu = {
                "available": False,
                "error": "No inference child is running",
            }
            self._started_at = None
            self._ready_at = None
            self._expected_stop = False
            self._monitor_thread = None
            self._monitor_stop = None

    def _ensure_open(self, *, allow_closing: bool = False) -> None:
        if self._closed or (self._closing and not allow_closing):
            raise ManagerClosedError("Model manager is closed")

    def _discover_checkpoints_locked(self) -> dict[str, Path]:
        if not self.checkpoint_root.is_dir():
            self._last_discovery = {
                "checked_at": _utc_now(),
                "directories_scanned": 0,
                "truncated": False,
                "max_depth": self.discovery_max_depth,
                "max_directories": self.discovery_max_directories,
                "max_entries": self.discovery_max_entries,
                "error": f"Checkpoint root is unavailable: {self.checkpoint_root}",
            }
            return {}

        checkpoints: dict[str, Path] = {}
        pending: deque[tuple[Path, int]] = deque([(self.checkpoint_root, 0)])
        seen = {self.checkpoint_root}
        scanned = 0
        entries_scanned = 0
        truncated = False

        while pending and not truncated:
            if scanned >= self.discovery_max_directories:
                truncated = True
                break
            candidate, depth = pending.popleft()
            scanned += 1

            if self._has_checkpoint_files(candidate):
                relative = candidate.relative_to(self.checkpoint_root)
                checkpoint_id = "." if relative == Path(".") else relative.as_posix()
                if self._is_safe_checkpoint_id(checkpoint_id):
                    checkpoints[checkpoint_id] = candidate
                continue

            if depth >= self.discovery_max_depth:
                continue
            try:
                with os.scandir(candidate) as children:
                    for child_entry in children:
                        if entries_scanned >= self.discovery_max_entries:
                            truncated = True
                            break
                        entries_scanned += 1
                        if self._skip_discovery_name(child_entry.name) or child_entry.is_symlink():
                            continue
                        try:
                            if not child_entry.is_dir(follow_symlinks=False):
                                continue
                            child = Path(child_entry.path)
                            resolved = child.resolve(strict=True)
                            resolved.relative_to(self.checkpoint_root)
                        except (FileNotFoundError, OSError, ValueError):
                            continue
                        if resolved in seen:
                            continue
                        if len(seen) >= self.discovery_max_directories:
                            truncated = True
                            break
                        seen.add(resolved)
                        pending.append((resolved, depth + 1))
            except OSError:
                continue

        self._last_discovery = {
            "checked_at": _utc_now(),
            "directories_scanned": scanned,
            "entries_scanned": entries_scanned,
            "truncated": truncated,
            "max_depth": self.discovery_max_depth,
            "max_directories": self.discovery_max_directories,
            "max_entries": self.discovery_max_entries,
        }
        if truncated:
            logger.warning(
                "Checkpoint discovery stopped after %d directories under %s",
                scanned,
                self.checkpoint_root,
            )
        return checkpoints

    @staticmethod
    def _skip_discovery_name(name: str) -> bool:
        return name.startswith(".") or name.endswith(_INCOMPLETE_SUFFIXES)

    @classmethod
    def _is_safe_checkpoint_id(cls, checkpoint_id: str) -> bool:
        if checkpoint_id == ".":
            return True
        if not checkpoint_id or "\\" in checkpoint_id or "\x00" in checkpoint_id:
            return False
        path = PurePosixPath(checkpoint_id)
        return (
            not path.is_absolute()
            and path.as_posix() == checkpoint_id
            and all(part not in {"", ".", ".."} and not cls._skip_discovery_name(part) for part in path.parts)
        )

    @staticmethod
    def _has_checkpoint_files(candidate: Path) -> bool:
        try:
            return (
                (candidate / "config.json").is_file()
                and (candidate / "norm_stats.json").is_file()
                and any((candidate / marker).is_file() for marker in _PROCESSOR_MARKERS)
                and ModelProcessManager._has_complete_weights(candidate)
            )
        except OSError:
            # A broad root such as /home can contain user directories that the
            # web service is not allowed to inspect. Treat those directories as
            # non-checkpoints and let discovery continue into readable paths.
            return False

    @staticmethod
    def _has_complete_weights(candidate: Path) -> bool:
        for marker in _WEIGHT_MARKERS:
            weight_file = candidate / marker
            try:
                if weight_file.is_file() and weight_file.stat().st_size > 0:
                    return True
            except OSError:
                continue

        for index_name in _WEIGHT_INDEXES:
            index_file = candidate / index_name
            try:
                index_data = json.loads(index_file.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            weight_map = index_data.get("weight_map") if isinstance(index_data, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                continue
            shard_values = list(weight_map.values())
            if any(not isinstance(name, str) for name in shard_values):
                continue
            shard_names = set(shard_values)

            complete = True
            for shard_name in shard_names:
                shard_path = PurePosixPath(shard_name)
                if shard_path.is_absolute() or any(part in {"", ".", ".."} for part in shard_path.parts):
                    complete = False
                    break
                try:
                    shard = (candidate / Path(*shard_path.parts)).resolve(strict=True)
                    shard.relative_to(candidate)
                    if not shard.is_file() or shard.stat().st_size == 0:
                        complete = False
                        break
                except (FileNotFoundError, OSError, ValueError):
                    complete = False
                    break
            if complete:
                return True
        return False

    def _port_is_in_use(self) -> bool:
        try:
            with socket.create_connection((self.health_host, self.inference_port), timeout=0.25):
                return True
        except OSError:
            return False

    def _health_check(self) -> bool:
        try:
            request = urllib.request.Request(
                f"http://{self.health_host}:{self.inference_port}/healthz",
                method="GET",
            )
            with self._health_opener.open(request, timeout=self.health_timeout) as response:
                return response.status == 200
        except (OSError, ValueError, urllib.error.URLError, TimeoutError):
            return False

    def _monitor_child(
        self,
        process: subprocess.Popen[bytes],
        generation: int,
        stop_event: threading.Event,
    ) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while not stop_event.is_set():
            exit_code = process.poll()
            if exit_code is not None:
                with self._state_lock:
                    if generation != self._generation:
                        return
                    self._process = None
                    self._exit_code = exit_code
                    self._health_ok = False
                    self._health_checked_at = _utc_now()
                    self._gpu = {
                        "available": False,
                        "error": "No inference child is running",
                    }
                    if self._expected_stop or self._state == ModelState.STOPPING:
                        self._state = ModelState.STOPPED
                        self._error = None
                    else:
                        self._state = ModelState.FAILED
                        self._error = f"Inference child exited with status {exit_code}"
                    self._monitor_thread = None
                    self._monitor_stop = None
                return

            health_ok = self._health_check()
            try:
                gpu = self._gpu_sampler(process.pid)
            except Exception as exc:
                gpu = {
                    "available": False,
                    "sampled_at": _utc_now(),
                    "error": f"GPU query failed: {exc}",
                }

            with self._state_lock:
                if generation != self._generation:
                    return
                self._health_ok = health_ok
                self._health_checked_at = _utc_now()
                self._gpu = gpu
                is_starting = self._state == ModelState.STARTING
                if is_starting and health_ok:
                    self._state = ModelState.READY
                    self._ready_at = _utc_now()
                    logger.info("Checkpoint %s is ready", self._checkpoint_id)

            if is_starting and not health_ok and time.monotonic() >= deadline:
                self._handle_startup_timeout(process, generation, stop_event)
                return

            stop_event.wait(self.monitor_interval)

    def _handle_startup_timeout(
        self,
        process: subprocess.Popen[bytes],
        generation: int,
        stop_event: threading.Event,
    ) -> None:
        error = f"Checkpoint did not become healthy within {self.startup_timeout:.0f} seconds"
        logger.error(error)
        with self._operation_lock:
            with self._state_lock:
                if generation != self._generation or self._state != ModelState.STARTING or stop_event.is_set():
                    return
            try:
                exit_code = self._terminate_process_group(process)
            except ModelStopError as exc:
                with self._state_lock:
                    if generation == self._generation:
                        self._state = ModelState.FAILED
                        self._health_ok = False
                        self._health_checked_at = _utc_now()
                        self._error = f"{error}; {exc}"
                return
            with self._state_lock:
                if generation == self._generation:
                    self._process = None
                    self._state = ModelState.FAILED
                    self._exit_code = exit_code
                    self._health_ok = False
                    self._health_checked_at = _utc_now()
                    self._gpu = {
                        "available": False,
                        "error": "No inference child is running",
                    }
                    self._error = error
                    self._monitor_thread = None
                    self._monitor_stop = None

    def _terminate_process_group(self, process: subprocess.Popen[bytes]) -> int:
        if process.poll() is not None:
            assert process.returncode is not None
            return process.returncode
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            try:
                return process.wait(timeout=1.0)
            except subprocess.TimeoutExpired as exc:
                raise ModelStopError(f"Inference child {process.pid} disappeared but was not reaped") from exc
        except OSError:
            process.terminate()

        try:
            return process.wait(timeout=self.shutdown_timeout)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Inference child did not stop after %.1fs; sending SIGKILL",
                self.shutdown_timeout,
            )
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                process.kill()
            try:
                return process.wait(timeout=5.0)
            except subprocess.TimeoutExpired as exc:
                raise ModelStopError(f"Inference process group {process.pid} did not exit after SIGKILL") from exc
