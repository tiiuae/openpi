# gigabrain07

GigaBrain-0.7 (PaliGemma2 backbone, flow-matching action expert) on the right-arm 7-D, 2-camera
embodiment. Predicts and executes **30 actions per call**. Source: `open-gigaai/giga-brain-0`, cloned
at a pinned SHA; the model package itself (`giga-models`) is a pinned git dependency in
`requirements.lock.txt`.

## Build

```bash
docker build -f docker/vla_bench/models/gigabrain07/Dockerfile -t vla-bench-gigabrain07:latest .
```

## Run

```bash
WEIGHTS_DIR=/opt/vla_weights HF_CACHE_DIR=$HOME/.cache/huggingface \
  docker/vla_bench/run.sh gigabrain07 --probe
docker/vla_bench/run.sh gigabrain07 --checkpoint /models/gigabrain07/model_ema
```

The checkpoint is the trainer's **EMA export** (`config.json`, `diffusion_pytorch_model.bin`,
`inference_config.json`) — the diffusers-style directory the authors deploy. The sibling
`pytorch_model_fsdp.bin` is the 14 GB FSDP *training* blob and must not be used.

## Two things specific to this model

**Its sidecar config carries absolute host paths, and the image overrides them.**
`inference_config.json` records the tokenizer, FAST tokenizer and norm-stats locations as absolute
paths from the training machine, which do not exist in a container. The adapter passes all three to
`get_policy()` explicitly, so `VLA_BENCH_ADAPTER_KWARGS` is what actually decides them: the two
tokenizers resolve inside the `/hf` mount and **`norm_stats_path` must be present under the weights
mount at `/models/gigabrain07/norm_stats_right7_emb8.json`**. The benchmark's upload set
(`VLA-SOTA/repo/scripts/upload_checkpoints.sh`) ships it there, next to `model_ema/`. If it is missing, the load
stops with `FileNotFoundError` (measured 2026-09-25); it does not fall back to other statistics.

The tokenizer path is a snapshot directory, so `/hf` must hold `google/paligemma2-3b-pt-224` at exactly revision
`96eeb174da13ca1a2b247e4d0867436296c36420`, tokenizer/processor files only:
`hf download google/paligemma2-3b-pt-224 --revision 96eeb174da13ca1a2b247e4d0867436296c36420 --include "*.json"`
(35 MB). **The repo is gated** (Gemma licence, manual approval), so request access first. The FAST-tokenizer
path is only checked for being non-empty and is never opened while serving, so `physical-intelligence/fast` is
not needed. Deployment audit 2026-09-25: `--probe` passes offline with nothing but the upload set and that one
snapshot, and the benchmark replay through that setup reproduces the recorded reference exactly
(`mae_joints_rad` 0.0077496). The upload set also carries `pytorch_model_fsdp.bin` (14 GB) and the trainer's
RNG/scheduler files. None of them is read.

**embodiment_id 8 is not a default.** The pretrained model has nine embodiment slots; slot 8 is the one
this benchmark trained into, and the delta mask (`[T]*6 + [F]` — joints delta, gripper absolute) is
keyed on it. For this checkpoint any other id stops the load with
`KeyError: "No delta mask for embodiment_id=7; available keys: ['8']"` (measured).

**Known serving defect: the served prompt is not the trained prompt.** Training rendered
`Task: <task>, Control mode: joint, End effector: gripper, State: <|propri|>;` (the run's config sets
`end_effector_override="gripper"`, and `model_ema/inference_config.json` records it). The authors' server
ignores that key: it infers the end effector from the delta mask, gets `None` for this single-arm mask, and drops
`End effector: gripper` from every prompt. Nothing warns. Measured 2026-09-25 on the benchmark's five reference
episodes: the image as shipped reproduces the recorded reference exactly (`mae_joints_rad` 0.0077496), and with
the trained prompt forced (a process-local patch, `VLA-SOTA/openpi_integration/tools/audit_gigabrain07_ee_server.py`)
it is 0.0068355, **-11.8 %**, with every aggregate metric 8-15 % better. The benchmark's GigaBrain row and this
image both serve the degraded prompt. Fixing it is a one-line code change in the adapter or server (pass
`prompt_cfg['end_effector_override']` through), not a deployment setting.
