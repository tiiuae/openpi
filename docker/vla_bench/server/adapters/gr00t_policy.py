"""Adapter for NVIDIA GR00T N1.7 (Isaac-GR00T), the one benchmark model that is not LeRobot-native.

GR00T keeps its own observation contract (`gr00t.policy.gr00t_policy.Gr00tPolicy`), so this adapter translates
the benchmark's four inputs into it and translates the answer back:

    benchmark obs                      GR00T obs
    ----------------------------       --------------------------------------------------------------
    primary  HxWx3 uint8 RGB      ->    observation["video"]["front"]  (1, 1, H, W, 3) uint8, channels LAST
    wrist    HxWx3 uint8 RGB      ->    observation["video"]["wrist"]  (1, 1, H, W, 3) uint8
    state    float32[7]           ->    observation["state"]["single_arm"] (1,1,6) + ["gripper"] (1,1,1) float32
    task     str                  ->    observation["language"]["annotation.human.task_description"] [[task]]

    GR00T action dict {"single_arm": (1, T, 6), "gripper": (1, T, 1)}  ->  float32[T, 7] absolute joint targets.

Two failure modes seen during Phase-2 bring-up are handled here explicitly:

  * `KeyError: 'q01'` — the N1.7 processor normalises with percentiles (`use_percentiles=True`) but LeRobot
    statistics only carry min/max/mean/std. A fine-tuned checkpoint already ships the percentiles inside its own
    `statistics.json` (verified for this run: new_embodiment/{state,action}/{single_arm,gripper} all have q01/q99),
    so the fast path needs no recomputation. The `--gr00t-recompute-stats` fallback (`dataset_dir=` kwarg) mirrors
    `models/gr00t/sanity_check.py::_add_percentiles` and computes q01/q99 from the parquet files for checkpoints
    that predate that fix.
  * channels-last vs channels-first video — this build's torchcodec returns (T, H, W, 3), other builds return
    (T, 3, H, W). `_as_hwc()` normalises whatever it is handed; GR00T's own `check_observation` requires
    (B, T, H, W, 3) uint8 and will reject anything else.

Style follows adapters/openvla_oft.py: heavy imports happen in warmup(), not at module import, so `eval.py` can
import every adapter module in an environment that only has numpy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .base import PolicyAdapter

DEFAULT_STATE_SLICES = {"single_arm": (0, 6), "gripper": (6, 7)}
DEFAULT_VIDEO_MAP = {"primary": "front", "wrist": "wrist"}


def _as_hwc(img: np.ndarray) -> np.ndarray:
    """Return an HxWx3 uint8 array from either HWC or CHW input."""
    a = np.asarray(img)
    if a.ndim != 3:
        raise ValueError(f"expected a single HxWx3 image, got shape {a.shape}")
    if a.shape[-1] != 3 and a.shape[0] == 3:      # CHW -> HWC
        a = np.transpose(a, (1, 2, 0))
    if a.shape[-1] != 3:
        raise ValueError(f"expected 3 colour channels, got shape {a.shape}")
    if a.dtype != np.uint8:                        # accept float [0,1] or [0,255] defensively
        a = np.asarray(a, np.float32)
        a = a * 255.0 if float(np.nanmax(a)) <= 1.0 else a
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


class Gr00tAdapter(PolicyAdapter):
    name = "gr00t_n1.7"

    def __init__(self, checkpoint: str, repo_dir: str, embodiment_tag: str = "new_embodiment",
                 device: str = "cuda:0", chunk_len: int = 30, exec_len: int | None = None,
                 dataset_dir: str | None = None, modality_config: str | None = None,
                 backbone: str | None = None, video_map: dict | None = None,
                 state_slices: dict | None = None):
        """
        checkpoint      : the HF-Trainer checkpoint dir (must hold config.json, the model safetensors and either
                          processor_config.json or a processor/ subdir).
        repo_dir        : $WS/models/gr00t/repo — the Isaac-GR00T source tree (`import gr00t`).
        dataset_dir     : optional LeRobot v2.1 derivative used for training. Only needed by the fallback path
                          (checkpoint without percentiles in its statistics, or without the embodiment tag).
        modality_config : optional right7_config.py — only needed by the same fallback path.
        backbone        : optional local dir overriding the config's `model_name` (nvidia/Cosmos-Reason2-2B).
        """
        self.ckpt = Path(checkpoint)
        self.repo_dir = str(repo_dir)
        self.embodiment_tag_str = embodiment_tag
        self.device = device
        self.chunk_len = int(chunk_len)
        self.exec_len = int(exec_len) if exec_len else None
        self.dataset_dir = dataset_dir
        self.modality_config = modality_config
        self.backbone = backbone
        self.video_map = dict(video_map or DEFAULT_VIDEO_MAP)
        self.state_slices = {k: tuple(v) for k, v in (state_slices or DEFAULT_STATE_SLICES).items()}
        self._policy = None
        self._loaded_via = None
        if not (self.ckpt / "config.json").is_file():
            raise FileNotFoundError(f"{self.ckpt}/config.json missing — is this a GR00T checkpoint directory?")
        if not (self.ckpt / "processor_config.json").is_file() and not (self.ckpt / "processor").is_dir():
            raise FileNotFoundError(
                f"{self.ckpt} has neither processor_config.json nor processor/. GR00T's normalisation statistics "
                f"and modality config live there; without them the policy cannot un-normalise its output.")
        if dataset_dir is not None:
            mj = Path(dataset_dir) / "meta" / "modality.json"
            if mj.is_file():
                m = json.loads(mj.read_text())
                self.state_slices = {k: (v["start"], v["end"]) for k, v in m["state"].items()}

    # ---------------------------------------------------------------- loading
    def warmup(self) -> None:
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        import torch
        import gr00t.model  # noqa: F401  (registers Gr00tN1d7 with AutoModel/AutoProcessor)
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self._torch = torch
        self._tag = EmbodimentTag.resolve(self.embodiment_tag_str)
        try:
            self._policy = Gr00tPolicy(embodiment_tag=self._tag, model_path=str(self.ckpt),
                                       device=self.device, strict=True)
            self._loaded_via = "Gr00tPolicy.__init__"
        except (ValueError, KeyError) as e:
            # Base checkpoints have no NEW_EMBODIMENT entry, and pre-fix statistics have no q01/q99.
            if self.dataset_dir is None or self.modality_config is None:
                raise RuntimeError(
                    f"Gr00tPolicy could not load {self.ckpt} ({type(e).__name__}: {e}). Pass dataset_dir= and "
                    f"modality_config= to use the statistics-injection fallback.") from e
            self._policy = self._load_with_injected_stats()
            self._loaded_via = "manual assembly + injected statistics"

        horizon = len(self._policy.modality_configs["action"].delta_indices)
        if horizon != self.chunk_len:
            raise RuntimeError(
                f"checkpoint action horizon is {horizon} but the benchmark asked for chunk_len={self.chunk_len}. "
                f"The horizon is baked into the saved modality config; do not silently truncate it.")
        self._action_keys = list(self._policy.modality_configs["action"].modality_keys)
        self._language_key = self._policy.language_key
        # one throwaway call so the reported latency excludes lazy CUDA / kernel init.
        # peak_vram_gb() is deliberately NOT reset afterwards: every adapter in this benchmark reports
        # torch's peak allocation since process start (weights + activations), so the numbers stay comparable.
        self.predict({"primary": np.zeros((480, 640, 3), np.uint8),
                      "wrist": np.zeros((480, 640, 3), np.uint8),
                      "state": np.zeros(7, np.float32), "task": "warmup"})
        torch.cuda.synchronize()

    def _load_with_injected_stats(self):
        """Mirror models/gr00t/sanity_check.py: build the processor from our modality config + dataset statistics
        (with percentiles computed from the parquet files) and assemble Gr00tPolicy without its __init__."""
        import torch
        from transformers import AutoModel, AutoProcessor
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.policy.policy import BasePolicy

        cfg_path = Path(self.modality_config).resolve()
        if str(cfg_path.parent) not in sys.path:
            sys.path.append(str(cfg_path.parent))
        __import__(cfg_path.stem)                       # registers the modality config
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        modality_cfg = MODALITY_CONFIGS[self._tag.value]

        stats = LeRobotEpisodeLoader(self.dataset_dir, modality_cfg).get_dataset_statistics()
        stats = _add_percentiles(stats, self.dataset_dir)

        model_kwargs = {"model_name": self.backbone} if self.backbone else {}
        model = AutoModel.from_pretrained(str(self.ckpt), **model_kwargs)
        model.eval().to(device=self.device, dtype=torch.bfloat16)
        proc_kwargs = {"modality_configs": {self._tag.value: modality_cfg}}
        if self.backbone:
            proc_kwargs["model_name"] = self.backbone
        processor = AutoProcessor.from_pretrained(str(self.ckpt), **proc_kwargs)
        processor.set_statistics({self._tag.value: stats}, override=True)
        processor.eval()

        policy = Gr00tPolicy.__new__(Gr00tPolicy)
        BasePolicy.__init__(policy, strict=True)
        policy.model = model
        policy.processor = processor
        policy.embodiment_tag = self._tag
        policy.modality_configs = {k: v for k, v in modality_cfg.items() if k != "rl_info"}
        policy.collate_fn = processor.collator
        policy.language_key = modality_cfg["language"].modality_keys[0]
        return policy

    # ------------------------------------------------------------- inference
    def predict(self, obs: dict) -> np.ndarray:
        if self._policy is None:
            raise RuntimeError("warmup() must run before predict(); it is what loads the model.")
        state = np.asarray(obs["state"], np.float32).reshape(-1)
        observation = {
            "video": {self.video_map[role]: _as_hwc(obs[role])[None, None]
                      for role in ("primary", "wrist")},
            "state": {k: state[a:b][None, None].astype(np.float32)
                      for k, (a, b) in self.state_slices.items()},
            "language": {self._language_key: [[obs["task"]]]},
        }
        action, _ = self._policy.get_action(observation)
        chunk = np.concatenate([np.asarray(action[k]) for k in self._action_keys], axis=-1)
        if chunk.ndim == 3:
            chunk = chunk[0]
        if chunk.shape != (self.chunk_len, 7):
            raise RuntimeError(
                f"GR00T returned {chunk.shape}, expected ({self.chunk_len}, 7). Action keys {self._action_keys} "
                f"with per-key shapes {[np.asarray(action[k]).shape for k in self._action_keys]}.")
        return np.ascontiguousarray(chunk, dtype=np.float32)

    def info(self) -> dict:
        d = super().info()
        d.update({"name": self.name, "checkpoint": str(self.ckpt), "embodiment_tag": self.embodiment_tag_str,
                  "loaded_via": self._loaded_via, "video_map": self.video_map,
                  "state_slices": {k: list(v) for k, v in self.state_slices.items()},
                  "action_keys": getattr(self, "_action_keys", None)})
        return d


def _add_percentiles(stats: dict, dataset_dir: str) -> dict:
    """q01/q99 per state/action group, computed over every frame (float64). Copied in behaviour from
    models/gr00t/sanity_check.py::_add_percentiles; slices come from meta/modality.json so the groups match
    the training config."""
    import glob
    import pyarrow.parquet as pq

    need = [(m, g) for m in ("state", "action") for g in stats.get(m, {}) if "q01" not in stats[m][g]]
    if not need:
        return stats
    mj = json.loads((Path(dataset_dir) / "meta" / "modality.json").read_text())
    files = sorted(glob.glob(str(Path(dataset_dir) / "data" / "*" / "*.parquet")))
    arrs = {}
    for modality, col in (("state", "observation.state"), ("action", "action")):
        chunks = [np.stack(pq.read_table(f, columns=[col])[col].to_numpy(zero_copy_only=False)).astype(np.float64)
                  for f in files]
        arrs[modality] = np.concatenate(chunks, axis=0)
    for modality, group in need:
        sl = mj[modality][group]
        a = arrs[modality][:, sl["start"]:sl["end"]]
        stats[modality][group]["q01"] = np.percentile(a, 1, axis=0).tolist()
        stats[modality][group]["q99"] = np.percentile(a, 99, axis=0).tolist()
    return stats
