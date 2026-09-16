"""GigaBrain-0.7 — NOT LeRobot. Wraps the author's own deployment path.

`models/gigabrain07/repo/scripts/inference/inference_agilex_server_unified.py::get_policy` is the
reference PaliGemma2 flow-matching inference server the authors ship. It resolves the checkpoint,
reads the `inference_config.json` sidecar the trainer wrote next to `model_ema/`, rebuilds the
inference-mode ImageTransform / PromptTokenizerTransform / Normalize / Unnormalize / AbsoluteActions
pipeline, and binds a `policy.inference(request)` method. This adapter calls exactly that, so the
code path is the author's deployment path and not a re-implementation.

Request contract of `policy.inference` (see `_canonicalize_inference_request`):
    {"observation.images.cam_high": HxWx3|3xHxW uint8 RGB,
     "observation.images.cam_right_wrist": ...,
     "observation.state": float[7],
     "task": str}
and it returns a torch tensor [T, 7] of ABSOLUTE actions (joints rad, gripper carriage m): the
q01/q99 un-normalisation and the de-delta against the reference state (delta mask
[T,T,T,T,T,T,F], joints delta / gripper absolute) are already applied inside.

Checkpoint: the trainer's `model_ema/` directory (config.json + diffusion_pytorch_model.bin +
inference_config.json) — the EMA weights the authors deploy, not the ~14 GB FSDP training blob
`pytorch_model_fsdp.bin` in the parent directory.

embodiment_id 8 is the slot this benchmark's right-arm 7-D embodiment was trained into
(num_embodiments 9; rows 0-7 are the pretrained GigaBrain-0.7 embodiments).
"""
from __future__ import annotations
import importlib.util, os, sys
import numpy as np
from .base import PolicyAdapter

CAM_HIGH = "observation.images.cam_high"
CAM_WRIST = "observation.images.cam_right_wrist"


def _patch_image_transform() -> list:
    """The pinned giga-models build predates two kwargs the author's newer inference script passes.

    `inference_agilex_server_unified.py` calls `ImageTransform(..., high_res_cam=image_cfg.get('high_res_cam'))`,
    but the installed `giga_models` ImageTransform has no such parameter. Our `inference_config.json` does not
    set it, so the value is None and dropping it changes nothing. Drop it -- but ONLY when it is None/False, so
    a build mismatch that would actually change behaviour still raises instead of being silently ignored.
    Returns the names dropped, which the adapter records in info().
    """
    import inspect
    from giga_models.pipelines.vla.giga_brain_0 import giga_brain_0_utils as U
    real = U.ImageTransform
    if getattr(real, "_deploy_eval_compat", False):
        return list(getattr(real, "_dropped", []))
    accepted = set(inspect.signature(real.__init__).parameters)
    dropped_names: list = []

    class _Compat(real):  # type: ignore[misc,valid-type]
        _deploy_eval_compat = True
        _dropped = dropped_names

        def __init__(self, *a, **kw):
            extra = {k: v for k, v in kw.items() if k not in accepted}
            bad = {k: v for k, v in extra.items() if v not in (None, False)}
            if bad:
                raise TypeError(
                    f"giga_models ImageTransform in this environment does not accept {sorted(bad)} "
                    f"and the values are not empty ({bad}); refusing to drop them silently"
                )
            for k in extra:
                kw.pop(k)
                if k not in dropped_names:
                    dropped_names.append(k)
            super().__init__(*a, **kw)

    U.ImageTransform = _Compat
    return dropped_names


def _load_author_module(repo_dir: str, model_dir: str):
    for p in (model_dir, repo_dir):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    path = os.path.join(repo_dir, "scripts", "inference", "inference_agilex_server_unified.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"author inference server not found: {path}")
    spec = importlib.util.spec_from_file_location("gb07_inference_server", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gb07_inference_server"] = mod
    spec.loader.exec_module(mod)
    return mod


class GigaBrain07Adapter(PolicyAdapter):
    def __init__(self, checkpoint: str, repo_dir: str, model_dir: str, tokenizer_path: str,
                 fast_tokenizer_path: str, norm_stats_path: str, embodiment_id: int = 8,
                 robot_type: str = "trossen_ai_mobile", original_action_dim: int = 7,
                 chunk_len: int = 30, exec_len: int = 30, dtype: str = "bfloat16",
                 device: str = "cuda:0", seed: int | None = 1000):
        self.name = f"gigabrain07:{checkpoint}"
        self.chunk_len, self.exec_len = int(chunk_len), int(exec_len)
        self.checkpoint, self.device, self.dtype, self.seed = checkpoint, device, dtype, seed
        self.embodiment_id, self.robot_type = int(embodiment_id), robot_type
        self.original_action_dim = int(original_action_dim)
        # The author's server hard-codes .to('cuda'); pin the visible device instead of fighting it.
        if device.startswith("cuda:") and device != "cuda:0":
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
        mod = _load_author_module(repo_dir, model_dir)
        self._mod = mod
        self.compat_dropped_kwargs = _patch_image_transform()
        self.policy = mod.get_policy(
            model_path=checkpoint,
            pretrained_path=tokenizer_path,
            fast_tokenizer_path=fast_tokenizer_path,
            embodiment_id=self.embodiment_id,
            norm_stats_path=norm_stats_path,
            use_bf16=(dtype == "bfloat16"),
            robot_type=robot_type,
            original_action_dim=self.original_action_dim,
            expected_state_dim=self.original_action_dim,
        )
        self.server_info = dict(getattr(self.policy, "server_info", {}))
        n_steps = int(getattr(self.policy, "n_action_steps", self.chunk_len))
        if n_steps != self.chunk_len:
            raise ValueError(f"checkpoint predicts {n_steps} steps, adapter configured for {self.chunk_len}")

    def warmup(self) -> None:
        import torch
        if self.seed is not None:
            torch.manual_seed(int(self.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.seed))

    def predict(self, obs: dict) -> np.ndarray:
        req = {
            CAM_HIGH: np.ascontiguousarray(obs["primary"]),
            CAM_WRIST: np.ascontiguousarray(obs["wrist"]),
            "observation.state": np.asarray(obs["state"], dtype=np.float32).reshape(-1)[: self.original_action_dim],
            "task": obs["task"],
        }
        a = self.policy.inference(req)
        a = a.detach().float().cpu().numpy()
        if a.ndim == 3:
            a = a[0]
        a = a[: self.chunk_len, : 7]
        assert a.shape == (self.chunk_len, 7), f"expected ({self.chunk_len}, 7), got {a.shape}"
        return a.astype(np.float32)

    def info(self) -> dict:
        d = super().info()
        d.update({"policy_type": "gigabrain07", "checkpoint": self.checkpoint,
                  "embodiment_id": self.embodiment_id, "robot_type": self.robot_type,
                  "dtype": self.dtype, "seed": self.seed,
                  "cameras": {"primary": CAM_HIGH, "wrist": CAM_WRIST},
                  "server_info": self.server_info,
                  "compat_dropped_image_transform_kwargs": self.compat_dropped_kwargs})
        return d
