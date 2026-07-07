import json
import pathlib

import pytest

from openpi.models import falconvla_autoconfig

_BASE_CONFIG = {
    "action_dim": 14,
    "action_head_configs": {"action_dim": 14, "action_horizon": 25},
    "time_horizon": 25,
    "n_action_bins": 256,
}


def _write_checkpoint(
    root: pathlib.Path,
    *,
    config: dict | None = None,
    norm_stats_keys: tuple[str, ...] = ("aidrc_cups_manipulation_14",),
    prop_tokens: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config if config is not None else _BASE_CONFIG))
    (root / "norm_stats.json").write_text(
        json.dumps({k: {"action": {"q01": [0.0], "q99": [1.0]}, "proprio": {"mean": [0.0]}} for k in norm_stats_keys})
    )
    tokenizer_config = {"added_tokens_decoder": {}}
    if prop_tokens:
        tokenizer_config["added_tokens_decoder"] = {str(i): f"<prop{i}>" for i in range(256)}
    (root / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (root / "preprocessor_config.json").write_text("{}")


def test_tokens_mode(tmp_path: pathlib.Path):
    _write_checkpoint(tmp_path, prop_tokens=True)
    config = falconvla_autoconfig.detect_falconvla_config(tmp_path)
    assert config.action_dim == 14
    assert config.action_horizon == 25
    assert config.num_bins == 256
    assert config.unnorm_key == "aidrc_cups_manipulation_14"
    assert config.use_proprio is True
    assert config.proprio_mode == "tokens"


def test_film_mode(tmp_path: pathlib.Path):
    config_json = {**_BASE_CONFIG, "proprio_mode": "film", "use_film_proprio": True, "proprio_dim": 19}
    _write_checkpoint(tmp_path, config=config_json, norm_stats_keys=("aloha_geometry_translated_rlds",))
    config = falconvla_autoconfig.detect_falconvla_config(tmp_path)
    assert config.use_proprio is True
    assert config.proprio_mode == "film"


def test_no_proprio(tmp_path: pathlib.Path):
    config_json = {**_BASE_CONFIG, "proprio_mode": "vlm_inject"}
    _write_checkpoint(tmp_path, config=config_json)
    config = falconvla_autoconfig.detect_falconvla_config(tmp_path)
    assert config.use_proprio is False


def test_ambiguous_unnorm_key_without_override_raises(tmp_path: pathlib.Path):
    _write_checkpoint(tmp_path, norm_stats_keys=("dataset_a", "dataset_b"))
    with pytest.raises(ValueError, match="dataset keys"):
        falconvla_autoconfig.detect_falconvla_config(tmp_path)


def test_ambiguous_unnorm_key_resolved_by_default_unnorm_key(tmp_path: pathlib.Path):
    config_json = {**_BASE_CONFIG, "default_unnorm_key": "dataset_b"}
    _write_checkpoint(tmp_path, config=config_json, norm_stats_keys=("dataset_a", "dataset_b"))
    config = falconvla_autoconfig.detect_falconvla_config(tmp_path)
    assert config.unnorm_key == "dataset_b"


def test_ambiguous_unnorm_key_resolved_by_override(tmp_path: pathlib.Path):
    _write_checkpoint(tmp_path, norm_stats_keys=("dataset_a", "dataset_b"))
    config = falconvla_autoconfig.detect_falconvla_config(tmp_path, unnorm_key="dataset_a")
    assert config.unnorm_key == "dataset_a"


def test_overrides_take_precedence(tmp_path: pathlib.Path):
    _write_checkpoint(tmp_path, prop_tokens=True)
    config = falconvla_autoconfig.detect_falconvla_config(tmp_path, use_proprio=False, use_wrist=False)
    assert config.use_proprio is False
    assert config.use_wrist is False


def test_resolves_single_nested_subdir(tmp_path: pathlib.Path):
    _write_checkpoint(tmp_path / "only_subdir", prop_tokens=True)
    config = falconvla_autoconfig.detect_falconvla_config(tmp_path)
    assert config.unnorm_key == "aidrc_cups_manipulation_14"


def test_missing_config_json_raises(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        falconvla_autoconfig.detect_falconvla_config(tmp_path)
