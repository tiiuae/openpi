"""Adapter for any LeRobot-integrated policy (smolvla, xvla, vla_jepa, molmoact2, groot, wall_x, ...).

Maps the benchmark's fixed observation dict to whatever input keys the checkpoint declares, so one class serves
every LeRobot policy type. Camera mapping is resolved in this order:
  1. explicit `rename_map` argument           {"primary": "observation.images.camera1", "wrist": "...camera2"}
  2. the checkpoint's own input_features, matched by name heuristics (front/exterior/primary/base/camera1 -> primary;
     wrist/hand/camera2 -> wrist)
  3. positional fallback: first image feature <- primary, second <- wrist
State/action dimension padding is handled by the policy's own processors (LeRobot pads to max_state_dim/max_action_dim
and trims the output), so we assert the trimmed output is [T, 7].
"""
from __future__ import annotations
import numpy as np, torch
from .base import PolicyAdapter

PRIMARY_HINTS = ("primary", "front", "exterior", "base", "high", "scene", "top", "camera1", "cam_high", "image")
WRIST_HINTS = ("wrist", "hand", "camera2", "cam_right", "gripper")

class LeRobotAdapter(PolicyAdapter):
    def __init__(self, checkpoint: str, device: str = "cuda", dtype: str = "bfloat16", chunk_len: int | None = None,
                 exec_len: int | None = None, rename_map: dict | None = None, overrides: dict | None = None,
                 image_uint8: bool = False, drop_postprocessor_steps: list | None = None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        self.name = f"lerobot:{checkpoint}"
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = device
        for k, v in (overrides or {}).items():
            setattr(cfg, k, v)
        if chunk_len is not None:
            cfg.chunk_size = chunk_len
            if hasattr(cfg, "n_action_steps"):
                cfg.n_action_steps = min(getattr(cfg, "n_action_steps", chunk_len), chunk_len)
        self.cfg = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint)
        # A checkpoint's SAVED post-processor can carry steps its own config disables: `lerobot-train` re-loads the
        # pipeline from --policy.path and re-saves it verbatim, so a flag flipped on the command line (e.g. VLA-JEPA's
        # binarize_gripper_action=false) never reaches the saved JSON. Drop those steps by class name instead of
        # rebuilding the pipeline from the config, which would also throw away the baked-in camera rename and the
        # normalisation statistics. Verified equal to the from-config pipeline for every model that uses this.
        self.dropped_steps = []
        if drop_postprocessor_steps:
            names = [type(s).__name__ for s in self.post.steps]
            missing = [n for n in drop_postprocessor_steps if n not in names]
            if missing:
                raise ValueError(f"drop_postprocessor_steps: {missing} not found in post-processor steps {names}")
            keep = []
            for s in self.post.steps:
                if type(s).__name__ in drop_postprocessor_steps:
                    self.dropped_steps.append(type(s).__name__)
                else:
                    keep.append(s)
            self.post.steps = keep
        self.device, self.torch_dtype, self.image_uint8 = device, getattr(torch, dtype), image_uint8
        self.chunk_len = int(chunk_len or getattr(cfg, "chunk_size", 1))
        self.exec_len = int(exec_len or getattr(cfg, "n_action_steps", self.chunk_len))
        self.img_keys = self._resolve_cameras(rename_map)
        self.state_key = next((k for k in cfg.input_features if "state" in k), "observation.state")

    def _resolve_cameras(self, rename_map):
        if rename_map:
            return dict(rename_map)
        feats = [k for k, v in self.cfg.input_features.items() if getattr(v.type, "name", str(v.type)).upper().startswith("VISUAL") or "image" in k]
        feats = sorted(feats)
        out = {}
        for role, hints in (("primary", PRIMARY_HINTS), ("wrist", WRIST_HINTS)):
            for k in feats:
                low = k.lower()
                if any(h in low for h in hints) and k not in out.values():
                    out[role] = k
                    break
        for role, idx in (("primary", 0), ("wrist", 1)):
            if role not in out and len(feats) > idx:
                out[role] = feats[idx]
        return out

    def _img(self, arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)  # HWC uint8 -> CHW
        if self.image_uint8:
            return t.unsqueeze(0).to(self.device)
        return (t.float() / 255.0).unsqueeze(0).to(self.device)

    def warmup(self):
        pass

    @torch.no_grad()
    def predict(self, obs: dict) -> np.ndarray:
        batch = {self.img_keys[r]: self._img(obs[r]) for r in ("primary", "wrist") if r in self.img_keys}
        batch[self.state_key] = torch.from_numpy(np.asarray(obs["state"], np.float32)).unsqueeze(0).to(self.device)
        batch["task"] = [obs["task"]]
        with torch.autocast(device_type="cuda", dtype=self.torch_dtype, enabled=self.device.startswith("cuda")):
            chunk = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        a = chunk["action"] if isinstance(chunk, dict) else chunk
        a = a.detach().float().cpu().numpy()
        if a.ndim == 3:
            a = a[0]
        a = a[: self.chunk_len, :7]
        assert a.shape == (self.chunk_len, 7), f"expected ({self.chunk_len}, 7), got {a.shape}"
        return a.astype(np.float32)

    def info(self):
        d = super().info()
        d.update({"policy_type": getattr(self.cfg, "type", "?"), "cameras": self.img_keys, "state_key": self.state_key,
                  "postprocessor_steps": [type(s).__name__ for s in self.post.steps],
                  "dropped_postprocessor_steps": self.dropped_steps})
        return d
