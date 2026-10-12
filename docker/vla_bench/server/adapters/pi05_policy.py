r"""Adapter for pi0.5 (LeRobot's `pi05` policy, lerobot main @ bad0260a4), model id `pi05`.

The inference path is the shared LeRobotAdapter's, unchanged (`predict` is inherited):
native 640x480 RGB frames -> CHW float in [0, 1] under the checkpoint's own input keys
(cam_high, cam_right_wrist) -> the checkpoint's saved pre-processor (rename to base_0_rgb /
left_wrist_0_rgb, quantile normalisation, the 256-bin state written into the prompt
"Task: ..., State: ...;\nAction: ", PaliGemma tokenisation) -> PI05Policy.predict_action_chunk
(resize with pad to 224x224 inside the model, the missing right_wrist_0_rgb slot masked,
10 flow-matching steps) under bf16 autocast, as lerobot-train ran its forward -> the saved
post-processor (quantile un-normalisation). Every step is the trainer's own code, so the model
sees what it saw in training.

What this subclass adds are checks that make a mis-assembled deployment fail at load instead of
driving the arm with a wrong model:

* `tokenizer_dir`: the PaliGemma tokenizer is loaded from a local directory instead of by Hub
  name. The upload set ships it (`<checkpoint>/paligemma-3b-pt-224/`, the gated
  `google/paligemma-3b-pt-224` at revision 35e4f464, the revision training used), so the server
  needs no Hugging Face cache, no token and no network. Every file is checked against its sha256
  at that revision before it is used.
* Weights: `PI05Policy.from_pretrained` catches every load error, prints it and returns the model
  with RANDOM weights. Here every tensor of `model.safetensors` (after the policy's own key fixes)
  is compared with the loaded model, and a missing, extra or different tensor is a hard error.
* Camera slots: primary must land in the first image slot (base_0_rgb) and wrist in the second,
  the training rename (BLOCKERS D-PI2); the quantile statistics must be present and finite.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .lerobot_policy import LeRobotAdapter

TOKENIZER_REPO = "google/paligemma-3b-pt-224"
TOKENIZER_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"  # gitleaks:allow (a git commit id)
# sha256 of every file of the tokenizer snapshot at TOKENIZER_REVISION (what training loaded).
# Public file hashes, not secrets.
TOKENIZER_SHA256 = {
    "tokenizer.model":  # gitleaks:allow
        "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6",  # gitleaks:allow
    "tokenizer.json":  # gitleaks:allow
        "ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563",  # gitleaks:allow
    "tokenizer_config.json":  # gitleaks:allow
        "3259402b1d1802e02417d7bff75a889ec61d359d15be6050a957b307c48edbbe",  # gitleaks:allow
    "special_tokens_map.json":  # gitleaks:allow
        "5ef37093ae4236587b6e8266acb815b46e2db8ce656c66552bfa574d32880405",  # gitleaks:allow
    "added_tokens.json":  # gitleaks:allow
        "7d0bad90030d638a4bf89a82e91206e25f1fbe0012f14bca21666a64b643bc49",  # gitleaks:allow
    "config.json":
        "e00c72cdff16296bf1229c3267b99f5247d8c740cae84336866338d64fa7918f",  # gitleaks:allow
}


def _sha256(path: Path) -> str:
    """Hex sha256 of a file, read in 8 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def check_tokenizer_dir(tokenizer_dir: str) -> str:
    """Return the resolved directory if every pinned tokenizer file matches its sha256, else raise.

    Args:
        tokenizer_dir: directory holding the six files of TOKENIZER_SHA256.

    Returns:
        The directory as an absolute path.

    Raises:
        RuntimeError: a file is missing or its sha256 differs from the pinned one.
    """
    d = Path(tokenizer_dir)
    bad = []
    for name, want in TOKENIZER_SHA256.items():
        f = d / name
        if not f.is_file():
            bad.append(f"{name}: missing")
        elif (got := _sha256(f)) != want:
            bad.append(f"{name}: sha256 {got} != pinned {want}")
    if bad:
        raise RuntimeError(f"tokenizer_dir {d} is not {TOKENIZER_REPO}@{TOKENIZER_REVISION}:\n  "
                           + "\n  ".join(bad))
    return str(d.resolve())


class Pi05Adapter(LeRobotAdapter):
    """LeRobot pi05 checkpoint, served like the other LeRobot policies, with load-time checks."""

    def __init__(self, checkpoint: str, tokenizer_dir: str, *, device: str = "cuda",
                 dtype: str = "bfloat16", chunk_len: int = 30, exec_len: int = 30,
                 rename_map: dict | None = None, verify_weights: bool = True) -> None:
        """Load with LeRobot's own loaders, then check tokenizer, cameras, statistics, weights.

        Args:
            checkpoint: the LeRobot `pretrained_model/` directory.
            tokenizer_dir: the PaliGemma tokenizer files (checked by `check_tokenizer_dir`).
            device: torch device for the policy.
            dtype: autocast dtype of the forward pass (`bfloat16`, as in training).
            chunk_len: rows returned per call.
            exec_len: rows the client executes per call.
            rename_map: {"primary": <input key>, "wrist": <input key>} fed to the checkpoint.
            verify_weights: compare every weight tensor with `model.safetensors` (keep True).
        """
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.name = f"pi05:{checkpoint}"
        self.tokenizer_dir = check_tokenizer_dir(tokenizer_dir)
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        if cfg.type != "pi05":
            raise ValueError(f"{checkpoint} is a '{cfg.type}' checkpoint, not pi05")
        cfg.device = device
        cfg.chunk_size = chunk_len
        cfg.n_action_steps = min(cfg.n_action_steps, chunk_len)
        self.cfg = cfg
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(checkpoint, config=cfg).to(device).eval()
        overrides = {"tokenizer_processor": {"tokenizer_name": self.tokenizer_dir}}
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint,
                                                       preprocessor_overrides=overrides)
        self.dropped_steps = []
        self.device, self.torch_dtype, self.image_uint8 = device, getattr(torch, dtype), False
        self.chunk_len, self.exec_len = int(chunk_len), int(exec_len)
        self.img_keys = self._resolve_cameras(rename_map)
        self.state_key = "observation.state"
        self._check_camera_slots()
        self._check_quantiles()
        self.weights_checked = None
        if verify_weights:
            self.weights_checked = self._verify_weights(Path(checkpoint) / "model.safetensors")

    def _step(self, cls_name: str) -> object:
        """The pre-processor step of that class (exactly one must exist)."""
        steps = [s for s in self.pre.steps if type(s).__name__ == cls_name]
        if len(steps) != 1:
            raise RuntimeError(f"expected one {cls_name} in the pre-processor, found {len(steps)}")
        return steps[0]

    def _check_camera_slots(self) -> None:
        """Check primary -> image slot 0, wrist -> slot 1, through the checkpoint's own rename."""
        slots = list(self.cfg.image_features)
        rename = self._step("RenameObservationsProcessorStep").rename_map
        got = {role: rename.get(key, key) for role, key in self.img_keys.items()}
        want = {"primary": slots[0], "wrist": slots[1]}
        if got != want:
            raise RuntimeError(f"camera slots {got} != expected {want} "
                               f"(image features {slots}, rename {rename})")
        self.camera_slots = got

    def _check_quantiles(self) -> None:
        """Check the normaliser carries finite 7-D q01 < q99 for state and action."""
        stats = self._step("NormalizerProcessorStep").stats
        for key in ("observation.state", "action"):
            q01 = np.asarray(stats[key]["q01"], np.float64).reshape(-1)
            q99 = np.asarray(stats[key]["q99"], np.float64).reshape(-1)
            finite = np.isfinite(q01).all() and np.isfinite(q99).all()
            if q01.shape != (7,) or not (finite and (q99 > q01).all()):
                raise RuntimeError(f"bad quantile statistics for {key}: q01={q01} q99={q99}")

    @torch.no_grad()
    def _verify_weights(self, path: Path) -> int:
        """Compare every tensor of the file, after PI05Policy's key fixes, with the loaded model.

        Returns:
            The number of state-dict entries checked.
        """
        from safetensors.torch import load_file

        fixed = self.policy._fix_pytorch_state_dict_keys(load_file(str(path)), self.policy.config)
        expected = {k if k.startswith("model.") else f"model.{k}": v for k, v in fixed.items()}
        loaded = self.policy.state_dict()
        missing, extra = sorted(set(loaded) - set(expected)), sorted(set(expected) - set(loaded))
        differ = [k for k in expected if k in loaded
                  and not torch.equal(loaded[k], expected[k].to(loaded[k].device, loaded[k].dtype))]
        if missing or extra or differ:
            raise RuntimeError(f"weights not loaded from {path}: "
                               f"{len(missing)} missing {missing[:3]}, "
                               f"{len(extra)} unexpected {extra[:3]}, "
                               f"{len(differ)} different {differ[:3]}")
        del fixed, expected
        return len(loaded)

    def info(self) -> dict:
        """The LeRobot adapter's info plus what was checked at load."""
        d = super().info()
        d.update({"tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION,
                                "dir": self.tokenizer_dir},
                  "camera_slots": self.camera_slots, "weights_checked": self.weights_checked})
        return d
