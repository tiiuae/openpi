"""LingBot-VLA 2.0 — NOT LeRobot. Wraps the authors' own deployment path.

`models/lingbot_v2/repo/deploy/lingbot_vla_v2_policy.py::LingbotVLAv2Server` IS the authors'
deployment entry point: its `main()` wraps it in their websocket policy server, and their own
open-loop evaluator `scripts/open_loop_eval.py` drives it through exactly the two calls this
adapter makes --

    server = LingbotVLAv2Server(path_to_pi_model=<hf_ckpt>, use_length=..., chunk_ret=True,
                                use_bf16=..., use_fp32=..., use_compile=...)
    server.reset(robo_name)                       # builds the FeatureTransform for our embodiment
    out = server.infer({<raw camera keys>, 'observation.state': (7,), 'task': str})

Nothing about the model, the flow sampler, the image processor, the tokeniser, the 55-D canonical
padding or the meanstd normalisation is re-implemented here. `server.load_vla()` reads the *training*
config `lingbotvla_cli.yaml` from `Path(ckpt).parent.parent.parent`, so the evaluated checkpoint
directory has to keep the run's directory shape:

    <root>/lingbotvla_cli.yaml
    <root>/checkpoints/global_step_16000/hf_ckpt/      <- the path passed as `checkpoint`

`server.reset()` and the robot config's `norm_stats:` entry are both **cwd-relative**
(`configs/robot_configs/<robo_name>.yaml`, `assets/norm_stats/right7_2view.json`), so this adapter
chdir()s into `repo_dir` -- the same thing `models/lingbot_v2/sanity_check.py` does and the same
thing the authors' scripts assume.

CAMERA ORDER IS LOAD-BEARING. Both distillation teachers read camera index 0 only
(`lingbotvla/models/vla/vision_models/module_utils.py:364,429`: `pil_images[:, :1]`), and the
declared order comes from `data.cameras` in the training config, not from the order this adapter
happens to insert keys. Index 0 must be the fixed scene view. The adapter therefore ASSERTS, against
the live FeatureTransform built from the run's own config, that

    feature_config.images == ['observation.images.camera_top', 'observation.images.camera_wrist_right']
    key_mapping['observation.images.camera_top']['origin_keys']          == 'observation.images.cam_high'
    key_mapping['observation.images.camera_wrist_right']['origin_keys']  == 'observation.images.cam_right_wrist'

and refuses to serve if any of that is not true, rather than trusting the config it was pointed at.

Two further checks run at load time and raise rather than warn:
  * `chunk_size` / `n_action_steps` on the loaded config must equal the configured chunk length (30);
  * the un-mapped chunk returned by the authors' own `unapply` must be exactly (chunk_len, 7).

Weights are loaded by the authors' `load_model_weights(..., strict=True)`, which raises on any
missing or unexpected key -- the same strictness `post_training: true` enforces during training.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np

from .base import PolicyAdapter

CAM_HIGH = "observation.images.cam_high"
CAM_WRIST = "observation.images.cam_right_wrist"
CANON_TOP = "observation.images.camera_top"
CANON_WRIST = "observation.images.camera_wrist_right"
STATE_KEY = "observation.state"


class LingBotV2Adapter(PolicyAdapter):
    """(30, 7) absolute joint targets: dims 0-5 arm radians, dim 6 gripper carriage metres."""

    def __init__(self, checkpoint: str, repo_dir: str,
                 robo_name: str = "right7_2view",
                 norm_stats_path: str | None = None,
                 chunk_len: int = 30, exec_len: int = 30,
                 dtype: str = "bfloat16", use_compile: bool = False,
                 device: str = "cuda:0", seed: int | None = 1000,
                 qwen3vl_path: str | None = None):
        self.name = f"lingbot_v2:{checkpoint}"
        self.chunk_len, self.exec_len = int(chunk_len), int(exec_len)
        self.checkpoint, self.dtype, self.seed = str(checkpoint), str(dtype), seed
        self.use_compile = bool(use_compile)
        self.robo_name = robo_name
        self.repo_dir = str(Path(repo_dir).resolve())
        if dtype not in ("bfloat16", "float32"):
            raise ValueError(f"dtype must be bfloat16 or float32, got {dtype!r} "
                             f"(the authors' deploy path has no other option)")
        # The authors' server hard-codes .cuda(); pin the visible device instead of fighting it.
        if device.startswith("cuda:") and device != "cuda:0":
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
        if qwen3vl_path:
            # load_vla() honours QWEN3VL_PATH over the training config's tokenizer_path. Only set it
            # when asked; by default the config's own (absolute) path is used, which is what trained.
            os.environ["QWEN3VL_PATH"] = str(qwen3vl_path)

        ckpt = Path(checkpoint)
        cli = ckpt.parent.parent.parent / "lingbotvla_cli.yaml"
        if not cli.is_file():
            raise FileNotFoundError(
                f"the authors' loader reads the training config at <ckpt>/../../../lingbotvla_cli.yaml; "
                f"{cli} does not exist. The checkpoint must keep the run's directory shape "
                f"(<root>/lingbotvla_cli.yaml + <root>/checkpoints/<step>/hf_ckpt)."
            )
        self.train_config_path = str(cli)

        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        os.chdir(self.repo_dir)          # robot config + norm_stats in the configs are repo-relative

        import torch
        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server   # the authors' deploy module

        t0 = time.time()
        self.server = LingbotVLAv2Server(
            path_to_pi_model=self.checkpoint,
            robot_norm_path=norm_stats_path,          # None -> the robot config's own `norm_stats:`
            use_length=self.chunk_len,
            chunk_ret=True,                           # return the whole chunk, as the robot consumes it
            use_bf16=(dtype == "bfloat16"),
            use_fp32=(dtype == "float32"),
            use_compile=self.use_compile,
        )
        self.server.reset(self.robo_name)
        self.load_seconds = time.time() - t0

        ft = self.server.vla.feature_transform
        cams = list(ft.feature_config.images)
        if cams != [CANON_TOP, CANON_WRIST]:
            raise ValueError(
                f"camera order is load-bearing (the depth/DINO teachers read camera index 0 only); "
                f"expected {[CANON_TOP, CANON_WRIST]}, the training config declares {cams}"
            )
        mapped = {k: v.get("origin_keys") for k, v in ft.key_mapping.items() if k in (CANON_TOP, CANON_WRIST)}
        if mapped.get(CANON_TOP) != CAM_HIGH or mapped.get(CANON_WRIST) != CAM_WRIST:
            raise ValueError(f"robot config camera mapping is not the trained one: {mapped}")
        self.cameras = {"index0_primary": f"{CANON_TOP} <- {CAM_HIGH}",
                        "index1_wrist": f"{CANON_WRIST} <- {CAM_WRIST}"}

        cfg = self.server.config
        n_steps = int(getattr(cfg, "n_action_steps", getattr(cfg, "chunk_size", -1)))
        if n_steps != self.chunk_len:
            raise ValueError(f"checkpoint predicts {n_steps} steps, adapter configured for {self.chunk_len}")
        self.num_denoise_steps = int(getattr(cfg, "num_steps", -1))
        self.attention_implementation = getattr(cfg, "attention_implementation", "?")
        self.max_action_dim = int(getattr(cfg, "max_action_dim", -1))
        self.norm_stats_file = str(self.server.robot_norm_path)
        self.action_keys = list(ft.org_features["actions"])
        self.torch_dtype = str(next(self.server.vla.parameters()).dtype)
        self.weights_gib = float(torch.cuda.memory_allocated() / 2 ** 30) if torch.cuda.is_available() else None

    # -----------------------------------------------------------------------------------------
    def warmup(self) -> None:
        import torch
        if self.seed is not None:
            torch.manual_seed(int(self.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.seed))

    def predict(self, obs: dict) -> np.ndarray:
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)[:7]
        traj = {
            # RAW LeRobot camera keys: the robot config maps them onto the canonical slots, and the
            # canonical order (scene camera first) is fixed by the training config, asserted above.
            CAM_HIGH: np.ascontiguousarray(obs["primary"]),
            CAM_WRIST: np.ascontiguousarray(obs["wrist"]),
            STATE_KEY: state,
            "task": str(obs["task"]),
        }
        out = self.server.infer(traj)
        a = out["action"]
        if hasattr(a, "detach"):
            a = a.detach().float().cpu().numpy()
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 3:
            a = a[0]
        if a.shape != (self.chunk_len, 7):
            raise ValueError(f"expected ({self.chunk_len}, 7) absolute joint targets, got {a.shape}")
        return a.astype(np.float32)

    def info(self) -> dict:
        d = super().info()
        d.update({
            "policy_type": "lingbot_vla_v2",
            "checkpoint": self.checkpoint,
            "training_config": self.train_config_path,
            "robo_name": self.robo_name,
            "norm_stats": self.norm_stats_file,
            "dtype": self.dtype,
            "torch_param_dtype": self.torch_dtype,
            "use_compile": self.use_compile,
            "attention_implementation": self.attention_implementation,
            "denoise_steps": self.num_denoise_steps,
            "canonical_action_dim": self.max_action_dim,
            "action_keys": self.action_keys,
            "cameras": self.cameras,
            "load_seconds": round(self.load_seconds, 1),
            "weights_gib": None if self.weights_gib is None else round(self.weights_gib, 2),
            "seed": self.seed,
        })
        return d
