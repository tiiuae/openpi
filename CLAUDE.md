# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Before editing any file, read it first. Before modifying a function, grep for all callers. Research before you edit.

## What this repo is

This is the **tiiuae fork of openpi** (branch `trossen-ai`). Upstream openpi (Physical Intelligence) provides the π₀ / π₀-FAST / π₀.₅ vision-language-action models in both JAX (Flax) and PyTorch. This fork adds two things on top:

1. **Alternative VLA baselines** served through openpi's own websocket policy interface: FalconVLA (in-process PyTorch, tiiuae's model), plus CogACT / OpenVLA / OpenVLA-OFT / ACT / starVLA as **external REST servers** the openpi server forwards to.
2. **Trossen AI bimanual WidowX** robot support: training configs (`pi0_trossen_*`, `pi05_trossen_*`) and a real-robot client under `examples/trossen_ai/`.

The upstream `README.md` documents the base pi0/pi05 workflow and is accurate for that part. Fork-specific behavior is described below.

## Setup

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync            # GIT_LFS_SKIP_SMUDGE=1 needed to pull LeRobot
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
git submodule update --init --recursive  # external VLA repos live as submodules
```

Python 3.11, managed by `uv`. JAX (`jax[cuda12]==0.5.3`) and PyTorch (`torch==2.7.1`) both installed; `transformers==4.53.2` is pinned.

**PyTorch pi0/pi05 requires a transformers patch** before running (AdaRMS, activation precision, non-updating KV cache):
```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```
This mutates the uv transformers cache permanently (hardlink); undo with `uv cache clean transformers`.

## Common commands

```bash
# Lint / format (line-length 120, ruff config in pyproject.toml)
uv run ruff check .
uv run ruff format .
pre-commit run --all-files          # also runs uv-lock

# Tests (testpaths = src, scripts, packages)
uv run pytest                        # whole suite
uv run pytest src/openpi/policies/policy_test.py            # single file
uv run pytest src/openpi/policies/policy_test.py::test_name # single test
# tests marked `manual` are skipped by default (see pyproject markers)

# Norm stats — REQUIRED before training any config
uv run scripts/compute_norm_stats.py --config-name <config_name>

# Train (JAX). Set XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 to use 90% GPU mem.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<run> --overwrite

# Train (PyTorch pi0/pi05 only; requires transformers patch above)
uv run scripts/train_pytorch.py <config_name> --exp_name <run> [--resume]
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<N> scripts/train_pytorch.py <config_name> --exp_name <run>

# Serve a checkpoint policy over websocket (default port 8800; upstream examples use 8000)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<ckpt_dir>
# Serve a default policy for an environment
uv run scripts/serve_policy.py --env <ALOHA|DROID|LIBERO|FALCONVLA_ALOHA|OPENVLA|OPENVLA_OFT|COGACT>
```

## Architecture

### Config registry drives everything
`src/openpi/training/config.py` holds a list of named `TrainConfig` objects (registry at the bottom, looked up by `_config.get_config(name)`). A config bundles: the model config (`pi0.Pi0Config`, `FalconVLAConfig`, …), a data config (`LeRobotAlohaDataConfig`, `LeRobotLiberoDataConfig`, …) with repack/normalize transforms, weight loader, and hyperparameters. **Adding a robot/dataset = adding a `TrainConfig`**, not new plumbing. The config name is the single string threaded through `compute_norm_stats.py`, `train.py`, and `serve_policy.py`. Fork-specific configs: `pi0_trossen_*`, `pi05_trossen_*`, and the `FalconVLA-*` family.

### Policy = model + transforms, behind a websocket
`src/openpi/policies/policy.py` (`Policy`) wraps a model with input/output transform pipelines. `policy_config.create_trained_policy` builds one from a config + checkpoint dir (auto-detects JAX vs converted-PyTorch). `serving/websocket_policy_server.py` exposes `policy.infer(obs)` over a websocket; `openpi_client` (in `packages/`) is the matching client. Robot clients send `{state, images:{cam:CHW}, prompt}` and get back `{"actions": (horizon, action_dim)}`.

### Two policy families (the fork's central pattern)
`scripts/serve_policy.py` routes on an `EnvMode` enum before falling back to the checkpoint path:
- **In-process models** — pi0/pi05 (JAX/PyTorch) and **FalconVLA** (PyTorch, loaded via `AutoModelForVision2Seq` + `trust_remote_code`, see `models/falconvla.py` / `falconvla_config.py`, policy in `policies/falconvla_policy.py`). These run inside the openpi server process on the GPU.
- **REST client policies** — `cogact_policy.py`, `openvla_policy.py`, `openvlaoft_policy.py`. These do **not** run a model; `infer()` HTTP-POSTs the observation to a separate server (CogACT/OpenVLA run from their own submodule + Docker image on a different port) and reshapes the response into openpi's action contract. So the openpi websocket stays the single interface the robot talks to, regardless of backend.

### External VLA submodules
`act`, `CogACT`, `openvla`, `openvla-oft`, `starVLA-internal`, `transformers-internal` are git submodules (see `.gitmodules`), each with its **own environment and inference server**. They are launched via `scripts/docker/compose.{cogact,openvla-oft}.yml` and `compose.yml` (FalconVLA), which mount checkpoints from a host `VLA_MODELS` dir into `/models` and expose ports (e.g. CogACT on 8777). Don't try to import these into the openpi env — they're wired in over HTTP, not Python imports.

### `act_infer/` — standalone ACT server
A self-contained mini-server (own `.venv`, vendored `msgpack_numpy.py` from openpi) that serves an ACT checkpoint to the **unchanged** `examples/trossen_ai/main.py` client over websocket. ACT is visuomotor: it accepts `--task_prompt` but ignores it. See `act_infer/README.md` for the joint-order / camera-map / image-resolution checks that must be verified before `--mode autonomous`.

### Trossen AI client and the two-LeRobot-version split
`examples/trossen_ai/` is a **separate uv project** (its own `pyproject.toml` + `.venv`) because it needs LeRobot **v0.3.2** (Interbotix fork, `BiWidowXAIFollower` support) for the robot client, while the openpi training env uses LeRobot **v0.1.0** for data loading. Run **training from the repo root**, run the **robot client from `examples/trossen_ai/`**. Editing LeRobot means picking the right `.venv`. Client entry point: `uv run main.py --mode <test|autonomous> --task_prompt "..."` (`--mode test` logs actions without moving the arm — always run it first).

## Conventions

- ruff enforces single-line isort with `force-sort-within-sections`; `T201` (print) is allowed. `third_party/` and `models_pytorch/transformers_replace/` are excluded from lint.
- Checkpoints download from `gs://openpi-assets` and cache in `~/.cache/openpi` (override with `OPENPI_DATA_HOME`).
- `unnorm_key` (e.g. `aidrc_cups_manipulation_14`) selects dataset normalization stats on the external VLA servers; keep it matched to the training dataset or predictions are scrambled.
