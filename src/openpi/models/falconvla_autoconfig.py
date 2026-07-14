"""Auto-detect a `FalconVLAConfig` from a checkpoint directory.

FalconVLA checkpoints are self-describing: `config.json` carries the training-time
`action_dim` / `time_horizon` / `n_action_bins`, `norm_stats.json` carries the dataset's
`unnorm_key` as its (usually only) top-level key, and the tokenizer reveals whether the
checkpoint was trained with discrete `<prop*>` proprio tokens. This module reads those files
directly (no `transformers`/`trust_remote_code` import, no weights touched) so a `FalconVLAConfig`
can be built without hand-writing a new `TrainConfig` entry per checkpoint.
"""

import json
import pathlib
from typing import Any

from openpi.models.falconvla_config import FalconVLAConfig

# Mirrors the marker set `falconvla.load_falcon_model`'s path resolution checks for, so this
# module inspects the exact same directory the model will actually be loaded from.
_PROCESSOR_MARKER_FILES = {"processor_config.json", "preprocessor_config.json", "tokenizer_config.json"}


def detect_falconvla_config(checkpoint_dir: str | pathlib.Path, **overrides: Any) -> FalconVLAConfig:
    """Build a `FalconVLAConfig` by inspecting a FalconVLA checkpoint directory.

    Detected fields: `action_dim`, `action_horizon`, `num_bins`, `unnorm_key`, `use_proprio`,
    `proprio_mode`. Any keyword in `overrides` takes precedence over the detected value (e.g. pass
    `unnorm_key=...` to resolve an ambiguous `norm_stats.json`, or `use_wrist=False` to force a
    field detection doesn't touch at all).

    Raises:
        FileNotFoundError: `checkpoint_dir` doesn't exist, or has no `config.json`.
        ValueError: `config.json` is missing `action_dim`/`time_horizon`, or `unnorm_key` is
            ambiguous (multiple `norm_stats.json` keys) and wasn't resolved by an override.
    """
    resolved_dir = _resolve_checkpoint_dir(pathlib.Path(checkpoint_dir))

    config_json = _load_json(resolved_dir / "config.json")
    if config_json is None:
        raise FileNotFoundError(f"No config.json found under {resolved_dir}.")

    action_head = config_json.get("action_head_configs") or {}
    action_dim = config_json.get("action_dim", action_head.get("action_dim"))
    action_horizon = config_json.get("time_horizon", action_head.get("action_horizon"))
    if action_dim is None or action_horizon is None:
        raise ValueError(
            f"{resolved_dir / 'config.json'} has no action_dim/time_horizon -- "
            "pass action_dim=/action_horizon= explicitly."
        )

    detected: dict[str, Any] = {
        "action_dim": int(action_dim),
        "action_horizon": int(action_horizon),
        "num_bins": int(config_json.get("n_action_bins", 256)),
    }

    detected["unnorm_key"] = overrides.get("unnorm_key") or _detect_unnorm_key(resolved_dir, config_json)

    if overrides.get("use_proprio") is None and overrides.get("proprio_mode") is None:
        detected["use_proprio"], detected["proprio_mode"] = _detect_proprio(resolved_dir, config_json)

    merged = {**detected, **{k: v for k, v in overrides.items() if v is not None}}
    return FalconVLAConfig(**merged)


def _detect_unnorm_key(resolved_dir: pathlib.Path, config_json: dict[str, Any]) -> str:
    norm_stats = _load_json(resolved_dir / "norm_stats.json")
    if not norm_stats:
        raise ValueError(f"No norm_stats.json found under {resolved_dir} -- pass unnorm_key= explicitly.")

    keys = list(norm_stats.keys())
    if len(keys) == 1:
        return keys[0]

    default_key = config_json.get("default_unnorm_key")
    if default_key in norm_stats:
        return default_key

    raise ValueError(
        f"{resolved_dir / 'norm_stats.json'} has {len(keys)} dataset keys {keys}; "
        "pass unnorm_key=<one of these> explicitly."
    )


def _detect_proprio(resolved_dir: pathlib.Path, config_json: dict[str, Any]) -> tuple[bool, str]:
    action_head = config_json.get("action_head_configs") or {}
    use_film = bool(config_json.get("use_film_proprio", action_head.get("use_film_proprio", False)))
    if config_json.get("proprio_mode") == "film" or use_film:
        return True, "film"
    if _has_proprio_tokens(resolved_dir):
        return True, "tokens"
    return False, "tokens"


def _has_proprio_tokens(resolved_dir: pathlib.Path) -> bool:
    for filename in ("tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        path = resolved_dir / filename
        if path.exists() and "<prop0>" in path.read_text():
            return True
    return False


def _resolve_checkpoint_dir(model_dir: pathlib.Path) -> pathlib.Path:
    """If `model_dir` has no processor files but has a single subdirectory that does, use that
    instead. Mirrors `falconvla.load_falcon_model`'s own path resolution."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"FalconVLA checkpoint directory not found: {model_dir}")

    entries = list(model_dir.iterdir())
    if not _PROCESSOR_MARKER_FILES & {p.name for p in entries}:
        subdirs = [p for p in entries if p.is_dir()]
        if len(subdirs) == 1 and _PROCESSOR_MARKER_FILES & {p.name for p in subdirs[0].iterdir()}:
            return subdirs[0]
    return model_dir


def _load_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())
