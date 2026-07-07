# Serving FalconVLA Checkpoints

FalconVLA is an in-process PyTorch VLA baseline served through openpi's own websocket policy
interface, alongside pi0/pi05. Unlike pi0/pi05, a FalconVLA checkpoint is a HuggingFace
`AutoModelForVision2Seq` directory (`config.json`, a `modeling_*.py`, `norm_stats.json`, tokenizer
files, safetensors weights) that owns its own tokenization and (un)normalization.

## Running the server without `uv run`

`uv run` re-checks (and, if needed, re-syncs) the project's dependencies against `pyproject.toml`
every time it starts, which adds a startup delay and requires resolving/network access. To skip
that check entirely, invoke the project's virtualenv interpreter directly:

```bash
.venv/bin/python scripts/serve_policy.py policy:checkpoint --policy.config=FalconVLA-AD14-H25 --policy.dir=/path/to/checkpoint
```

This is equivalent to `uv run scripts/serve_policy.py ...` once the environment is already synced
(e.g. after `uv sync`), just without the per-run sync check.

## Serving a checkpoint

A FalconVLA checkpoint can be served two ways.

### 1. Explicit config

A named entry in `_CONFIGS` (`src/openpi/training/config.py`) pins `action_dim`, `action_horizon`,
`unnorm_key`, and proprio settings by hand:

```bash
.venv/bin/python scripts/serve_policy.py policy:checkpoint --policy.config=FalconVLA-AD14-H25 --policy.dir=/path/to/checkpoint
```

### 2. Auto-detected config

Those same fields are read directly from the checkpoint directory's own `config.json` /
`norm_stats.json` / tokenizer files, so no `_CONFIGS` entry is needed at all:

```bash
.venv/bin/python scripts/serve_policy.py policy:auto-checkpoint --policy.dir=/path/to/checkpoint
# or: make serve-auto CKPT=/path/to/checkpoint
```

Detection rules (see `src/openpi/models/falconvla_autoconfig.py`):

| Field | Source |
| --- | --- |
| `action_dim` | `config.json["action_dim"]` |
| `action_horizon` | `config.json["time_horizon"]` |
| `num_bins` | `config.json["n_action_bins"]` (default 256) |
| `unnorm_key` | the sole top-level key of `norm_stats.json`, or `config.json["default_unnorm_key"]` if there are several |
| `use_proprio` / `proprio_mode` | `"film"` if `config.json["proprio_mode"] == "film"` (or `use_film_proprio: true`); else `"tokens"` if the tokenizer has `<prop0>`..`<propN>` special tokens; else no proprio |

If detection is ambiguous (e.g. a checkpoint's `norm_stats.json` has more than one dataset key),
pass an explicit override:

```bash
.venv/bin/python scripts/serve_policy.py policy:auto-checkpoint --policy.dir=/path/to/checkpoint --policy.unnorm-key=<key>
```

Also available: `--policy.use-proprio`, `--policy.proprio-mode`.

## Dependencies

FalconVLA requires the `transformers-internal` submodule (tiiuae's Falcon3-VL fork, pinned to the
`falcon3_vlm_integration` branch) instead of PyPI `transformers` -- this is set up in
`pyproject.toml` via `[tool.uv.sources]`, plus `timm` and `accelerate`. After
`git submodule update --init --recursive`, running `uv sync` (or `uv run`) installs `transformers`
as an editable package from `transformers-internal/` rather than pulling a release from PyPI.
