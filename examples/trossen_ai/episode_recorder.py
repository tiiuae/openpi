"""Interactive LeRobot-format episode recording for the eval client.

Records (observation, executed action) pairs from the control loop into one
of two LeRobot v3.0 datasets — "pass" or "fail" — so successful and failed
takes land in separate, immediately-usable datasets instead of one pile that
has to be sorted by hand afterwards. The operator drives the lifecycle from
the terminal:

    record                 -> start()        begin buffering frames
    hold / stop             -> end()          close the take, decision pending
    pass / fail              -> commit(label) queue the take for the matching dataset
    reject / discard        -> reject()      drop the buffered take

A take is buffered as plain Python objects (raw observation/action arrays)
while it is running — nothing touches disk yet, so the destination doesn't
need to be known until the operator decides. Only commit() hands the
buffered frames to the winning dataset: it writes images through lerobot's
own episode buffer (temporary PNGs via a background image-writer thread
pool) and queues the take for a single shared background saver thread, so
the expensive ffmpeg video encoding never blocks the operator — the next
take can start recording immediately, even while the previous one is still
being written and encoded. The saver reports per-camera progress through
save_cb so the terminal status line can show it. reject() just drops the
buffer: a discarded take never costs any disk I/O. close() waits for queued
saves before finalizing both datasets.

Multiple episodes recorded in one run (and across runs — an existing "pass"
or "fail" dataset at the same root is appended to) land in that dataset.
"""

from __future__ import annotations

from collections.abc import Callable
import io
import logging
import os
from pathlib import Path
import queue
import threading
from typing import Any

# The SVT-AV1 encoder prints a ~20-line config banner per encoded video
# directly to stderr, bypassing the libav log level lerobot already sets.
# It reads this env var at encoder init, so it must be set before encoding.
os.environ.setdefault("SVT_LOG", "1")  # 1 = errors only

import av
import datasets
import lerobot.datasets.lerobot_dataset as _lerobot_dataset_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.video_utils import encode_video_frames
from lerobot.utils.constants import ACTION
from lerobot.utils.constants import OBS_STR
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Same default as lerobot-record: PNG writer threads per camera.
_IMAGE_WRITER_THREADS_PER_CAMERA = 4

IDLE = "idle"
RECORDING = "recording"
PENDING = "pending"

LABELS = ("pass", "fail")

# NVENC options mirroring lerobot's CPU defaults where they exist: g=2 keeps a
# keyframe every 2 frames for fast random access during training, which NVENC
# only accepts with B-frames off. cq 30 lands near libsvtav1 crf=30 file sizes.
_NVENC_OPTIONS = {"g": "2", "bf": "0", "preset": "p4", "tune": "hq", "rc": "vbr", "cq": "30"}


def _nvenc_smoke_test() -> bool:
    """Whether h264_nvenc encoding actually works here — the codec being listed
    is not enough (libnvidia-encode and a working driver are also needed), so
    run one tiny in-memory encode with the exact options used in production.
    NVENC refuses frames below 145x49, so probe just above that."""
    try:
        buffer = io.BytesIO()
        with av.open(buffer, "w", format="mp4") as output:
            stream = output.add_stream("h264_nvenc", 25, options=dict(_NVENC_OPTIONS))
            stream.width, stream.height = 160, 64
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 160, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    except Exception:
        return False
    return True


def _encode_video_frames_nvenc(imgs_dir: Path | str, video_path: Path | str, fps: int) -> None:
    """h264_nvenc counterpart of lerobot's encode_video_frames: same frame
    globbing and muxing, GPU encoder. Leaves the PNGs in place (the caller,
    _encode_temporary_episode_video, deletes them)."""
    input_list = sorted(Path(imgs_dir).glob("frame-" + "[0-9]" * 6 + ".png"))
    if not input_list:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    with Image.open(input_list[0]) as first_image:
        width, height = first_image.size

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path), "w") as output:
        stream = output.add_stream("h264_nvenc", fps, options=dict(_NVENC_OPTIONS))
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height
        for input_path in input_list:
            with Image.open(input_path) as input_image:
                frame = av.VideoFrame.from_image(input_image.convert("RGB"))
            packet = stream.encode(frame)
            if packet:
                output.mux(packet)
        packet = stream.encode()
        if packet:
            output.mux(packet)

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


class _DatasetSink:
    """One persistent LeRobot dataset — either the "pass" set or the "fail"
    set. Owns its own episode-index counter and image writer; encoding and
    saving is driven by EpisodeRecorder's single shared background saver
    thread, never by this class directly."""

    def __init__(
        self,
        label: str,
        robot,
        features: dict,
        fps: int,
        root: Path,
        repo_id: str,
        report_progress: Callable[[str], None],
    ):
        self.label = label
        num_cameras = sum(isinstance(shape, tuple) for shape in robot.observation_features.values())
        threads = _IMAGE_WRITER_THREADS_PER_CAMERA * max(num_cameras, 1)

        if (root / "meta" / "info.json").exists():
            self.dataset = LeRobotDataset(repo_id, root=root)
            if self.dataset.fps != fps:
                raise ValueError(
                    f"Existing '{label}' dataset at {root} was recorded at {self.dataset.fps} Hz, "
                    f"but the control loop runs at {fps} Hz — pick another --record_dir"
                )
            missing = set(features) - set(self.dataset.features)
            if missing:
                raise ValueError(
                    f"Existing '{label}' dataset at {root} lacks features {sorted(missing)} — pick another --record_dir"
                )
            self.dataset.start_image_writer(num_processes=0, num_threads=threads)
            logger.info(
                "'%s' recording resumes dataset %s at %s (%d episodes so far)",
                label,
                repo_id,
                root,
                self.dataset.meta.total_episodes,
            )
        else:
            self.dataset = LeRobotDataset.create(
                repo_id,
                fps,
                features=features,
                root=root,
                robot_type=getattr(robot, "name", None),
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=threads,
            )
            logger.info("'%s' recording to new dataset %s at %s", label, repo_id, root)

        # By default lerobot batches many episodes into one shared parquet/video
        # file (100MB/500MB thresholds) and only writes that file's footer when
        # the writer is closed — which normally only happens at process exit or
        # once the threshold is hit. If the process is killed before that (e.g.
        # the arm driver faults and the operator has to force-kill a hung
        # process), every episode still sitting in that open file is left
        # without a footer — not just the one being recorded when it died. Worse,
        # video episodes sharing a file are merged in by rewriting the *entire*
        # shared file each time (see _save_episode_video's concatenate_video_files
        # call), so a kill mid-rewrite can damage already-saved episodes too.
        # Forcing near-zero thresholds makes (almost) every episode round-trip
        # through its own file via a plain move/close instead of an in-place
        # rewrite, so a crash can only ever cost the one episode being written
        # at that instant — never episodes already on disk. Paired with the
        # per-episode dataset.finalize() call in EpisodeRecorder._save_worker,
        # which closes that one episode's files immediately instead of leaving
        # them open until the next episode (or process exit) closes them.
        self.dataset.meta.update_chunk_settings(data_files_size_in_mb=0.001, video_files_size_in_mb=1)

        # GPU (NVENC) video encoding when possible. lerobot only exposes its
        # CPU codecs, so encode_video_frames is swapped out module-wide by
        # EpisodeRecorder (shared by both sinks — see _dispatching_encode).
        self.vcodec = self._choose_video_codec()
        self.next_episode_index = self.dataset.meta.total_episodes

        # Report per-camera encoding progress: _save_episode_video is called by
        # the saver thread once per camera, so wrapping it is the one hook that
        # sees "which camera, which episode" without reimplementing save_episode.
        video_keys = list(self.dataset.meta.video_keys)
        original_save_video = self.dataset._save_episode_video

        def _save_video_with_progress(video_key: str, episode_index: int):
            position = video_keys.index(video_key) + 1
            report_progress(f"{label} ep {episode_index}: encoding {video_key} ({position}/{len(video_keys)})")
            return original_save_video(video_key, episode_index)

        self.dataset._save_episode_video = _save_video_with_progress

    def _choose_video_codec(self) -> str | None:
        """Pick "h264_nvenc" (GPU), "h264" (CPU), or None for lerobot's default
        (libsvtav1, CPU).

        Episodes are appended into shared video files by concatenation, so the
        codec must stay consistent within a dataset: a resumed dataset keeps
        whatever codec it already uses, and only an h264 one can go to NVENC.
        """
        existing = None
        for key in self.dataset.meta.video_keys:
            info = self.dataset.meta.info["features"][key].get("info") or {}
            if info.get("video.codec"):
                existing = info["video.codec"]
                break

        if existing is not None and existing != "h264":
            logger.info("'%s' video encoding: CPU (%s — keeping the resumed dataset's codec)", self.label, existing)
            return None
        if _nvenc_smoke_test():
            logger.info("'%s' video encoding: GPU (h264_nvenc)", self.label)
            return "h264_nvenc"
        if existing == "h264":
            logger.info(
                "'%s' video encoding: CPU (h264 — NVENC unavailable, keeping the resumed dataset's codec)", self.label
            )
            return "h264"
        logger.info("'%s' video encoding: CPU (libsvtav1 — NVENC unavailable)", self.label)
        return None

    def close(self) -> None:
        self.dataset.stop_image_writer()
        self.dataset.finalize()
        total = self.dataset.meta.total_episodes
        if total:
            logger.info("'%s' dataset finalized: %d episode(s) at %s", self.label, total, self.dataset.root)


class EpisodeRecorder:
    """Buffers one take at a time in memory and routes it to a "pass" or
    "fail" LeRobot dataset once the operator decides.

    State machine: IDLE --start()--> RECORDING --end()--> PENDING, then
    commit(label)/reject() back to IDLE. commit()/reject() also accept the
    RECORDING state directly (they close the take first), so the operator
    can type "pass" without typing "hold" beforehand.

    While RECORDING, add() appends raw (observation, action, task) frames to
    a plain in-memory list — no dataset is touched, so nothing is written to
    disk until the operator commits to a label. This also makes reject()
    free: there are no temporary images to clean up.

    Episode indices are assigned per-dataset by this class (not by lerobot's
    default of meta.total_episodes) because a new take can begin while
    earlier takes for the same label are still queued for saving — i.e.
    before that dataset's episode count has caught up. A single background
    saver thread (shared across both labels) processes queued takes in FIFO
    order, which is what lerobot's save-time validation (buffer index ==
    total episodes) needs for each dataset independently.
    """

    def __init__(self, robot, fps: int, root: str | Path, repo_id: str):
        self._joint_names = list(robot.action_features.keys())
        self._features = {
            **hw_to_dataset_features(robot.observation_features, OBS_STR),
            **hw_to_dataset_features(robot.action_features, ACTION),
        }
        root = Path(root)

        self._sinks = {
            label: _DatasetSink(
                label, robot, self._features, fps, root / label, f"{repo_id}_{label}", self._set_save_status
            )
            for label in LABELS
        }

        # save_episode maps the buffer through an HF dataset, whose "Map: 100%|…"
        # progress bars would scribble over the pinned input line. Loading an
        # existing dataset above re-enables them, so disable *after* both sinks load.
        datasets.disable_progress_bars()

        if any(sink.vcodec is not None for sink in self._sinks.values()):
            _lerobot_dataset_module.encode_video_frames = self._dispatching_encode

        self._state = IDLE
        self._pending_frames: list[dict] = []
        self._save_queue: queue.Queue[tuple[str, int, list[dict], int] | None] = queue.Queue()
        self._saver: threading.Thread | None = None
        # Set by the (single) saver thread right before it calls save_episode,
        # so _dispatching_encode knows which sink's codec preference applies —
        # safe because only one save runs at a time across both labels.
        self._saving_sink: _DatasetSink | None = None
        # Called with (state, steps) on every transition and every added frame,
        # so the terminal UI can show a live "● REC n steps" indicator.
        self.status_cb: Callable[[str, int], None] | None = None
        # Called with a short progress string while a save is running in the
        # background (per-camera encoding progress), and None when done.
        self.save_cb: Callable[[str | None], None] | None = None

    # -- video codec ---------------------------------------------------------

    def _dispatching_encode(self, imgs_dir, video_path, fps, **kwargs) -> None:
        """Stands in for encode_video_frames inside lerobot's save path,
        shared by both sinks. Tries NVENC when the currently-saving sink
        wants it, and on failure falls back to CPU — staying on h264 so
        already encoded episodes can still be concatenated with."""
        sink = self._saving_sink
        if sink.vcodec == "h264_nvenc":
            try:
                _encode_video_frames_nvenc(imgs_dir, video_path, fps)
                return
            except Exception:
                logger.warning("NVENC encoding failed — falling back to CPU h264", exc_info=True)
                sink.vcodec = "h264"
                Path(video_path).unlink(missing_ok=True)  # drop any partial output
        kwargs.setdefault("overwrite", True)
        encode_video_frames(imgs_dir, video_path, fps, vcodec="h264", **kwargs)

    # -- state -------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_recording(self) -> bool:
        return self._state == RECORDING

    @property
    def is_pending(self) -> bool:
        return self._state == PENDING

    @property
    def steps(self) -> int:
        return len(self._pending_frames)

    def _notify(self) -> None:
        if self.status_cb is not None:
            self.status_cb(self._state, self.steps)

    def _set_save_status(self, text: str | None) -> None:
        if text is not None and (queued := self._save_queue.qsize()):
            text += f" · +{queued} queued"
        if self.save_cb is not None:
            self.save_cb(text)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._state != IDLE:
            raise RuntimeError(f"cannot start recording from state {self._state!r}")
        self._state = RECORDING
        self._pending_frames = []
        logger.info("● Recording — 'hold'/'stop' ends the take, then 'pass', 'fail', or 'reject'")
        self._notify()

    def add(self, observation: dict[str, Any], action: np.ndarray, task: str) -> None:
        """Buffer one control step. No-op unless a take is running, so callers
        can invoke it unconditionally from every point that drives the arm.
        Frames are kept as plain Python objects — nothing touches disk here."""
        if self._state != RECORDING:
            return
        frame = {
            **build_dataset_frame(self._features, observation, prefix=OBS_STR),
            **build_dataset_frame(self._features, dict(zip(self._joint_names, action, strict=True)), prefix=ACTION),
            "task": str(task),
        }
        self._pending_frames.append(frame)
        self._notify()

    def end(self) -> None:
        """Close the running take; it stays buffered until commit()/reject()."""
        if self._state != RECORDING:
            return
        if self.steps == 0:
            self._state = IDLE
            logger.info("Take had no frames — nothing to save")
        else:
            self._state = PENDING
            logger.info("■ Take ended after %d steps — type 'pass', 'fail', or 'reject'", self.steps)
        self._notify()

    def commit(self, label: str) -> None:
        """Hand the buffered take to the *label* ("pass"/"fail") dataset and
        queue it for background saving. Returns immediately — the next take
        can start while this one is written and encoded."""
        if label not in self._sinks:
            raise ValueError(f"unknown label {label!r} — expected one of {LABELS}")
        self.end()
        if self._state != PENDING:
            return
        sink = self._sinks[label]
        frames = self._pending_frames
        self._pending_frames = []
        steps = len(frames)
        episode_index = sink.next_episode_index
        sink.next_episode_index += 1
        self._state = IDLE
        if self._saver is None:
            self._saver = threading.Thread(target=self._save_worker, daemon=True, name="episode-saver")
            self._saver.start()
        self._save_queue.put((label, episode_index, frames, steps))
        logger.info(
            "Episode queued as '%s' #%d (%d steps) — encoding in the background, 'record' works right away",
            label,
            episode_index,
            steps,
        )
        self._notify()

    def reject(self) -> None:
        """Drop the buffered take. Nothing was ever written to disk, so this
        is just dropping a list — no cleanup needed."""
        self.end()
        if self._state != PENDING:
            return
        steps = self.steps
        self._pending_frames = []
        self._state = IDLE
        logger.info("Rejected take (%d steps discarded)", steps)
        self._notify()

    def close(self) -> None:
        """Discard any unsaved take (quitting is a rejection), wait for queued
        saves, and finalize both datasets so their parquet files are readable."""
        if self._state in (RECORDING, PENDING) and self.steps > 0:
            logger.warning("Quitting with an unsaved take — discarding %d steps", self.steps)
        if self._state in (RECORDING, PENDING):
            self._state = PENDING
            self.reject()
        if self._saver is not None:
            if not self._save_queue.empty():
                logger.info("Waiting for background episode saves to finish...")
            self._save_queue.put(None)
            self._saver.join()
            self._saver = None
        for sink in self._sinks.values():
            sink.close()
        _lerobot_dataset_module.encode_video_frames = encode_video_frames

    # -- background saver ----------------------------------------------------

    def _save_worker(self) -> None:
        """Saves queued takes strictly in FIFO order, one at a time across
        both labels. The main thread only touches _pending_frames (a plain
        list, not any dataset), so building the episode buffer and calling
        save_episode here never races the operator starting a new take."""
        while True:
            job = self._save_queue.get()
            if job is None:
                return
            label, episode_index, frames, steps = job
            sink = self._sinks[label]
            self._saving_sink = sink
            try:
                sink.dataset.episode_buffer = sink.dataset.create_episode_buffer(episode_index)
                self._set_save_status(f"{label} ep {episode_index}: writing frames ({steps} steps)")
                for frame in frames:
                    sink.dataset.add_frame(frame)
                buffer = sink.dataset.episode_buffer
                self._set_save_status(f"{label} ep {episode_index}: writing data ({steps} steps)")
                sink.dataset.save_episode(episode_data=buffer)
                # Close this episode's parquet writers the moment it's saved,
                # instead of leaving them open until the next episode rolls
                # over (or process exit) closes them — see the matching
                # update_chunk_settings() call in _DatasetSink.__init__ for why
                # this only ever costs the in-flight episode on a hard kill.
                sink.dataset.finalize()
                logger.info("Saved '%s' episode %d (%d steps)", label, episode_index, steps)
            except Exception:
                # Episode indices are consecutive, so a failed save also breaks
                # the takes queued behind it for this label — say so instead of
                # failing quietly.
                logger.exception(
                    "Saving '%s' episode %d FAILED — its frames are lost and queued '%s' saves may fail too",
                    label,
                    episode_index,
                    label,
                )
            finally:
                self._saving_sink = None
                self._set_save_status(None)
                self._save_queue.task_done()
