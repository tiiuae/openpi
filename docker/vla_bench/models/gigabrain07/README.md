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
`get_policy()` explicitly, so the adapter kwargs (image defaults in `VLA_BENCH_ADAPTER_KWARGS_DEFAULTS`,
overridable key by key through `VLA_BENCH_ADAPTER_KWARGS`) are what actually decide them: the two
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
(`mae_joints_rad` 0.0077496, the authors' prompt on the legacy 224x224 BGR wire; the new-wire numbers are in
`VLA-SOTA/results/deploy_eval/gigabrain07/`). The upload set also carries `pytorch_model_fsdp.bin` (14 GB) and the trainer's
RNG/scheduler files. None of them is read.

**embodiment_id 8 is not a default.** The pretrained model has nine embodiment slots; slot 8 is the one
this benchmark trained into, and the delta mask (`[T]*6 + [F]` — joints delta, gripper absolute) is
keyed on it. For this checkpoint any other id stops the load with
`KeyError: "No delta mask for embodiment_id=7; available keys: ['8']"` (measured).

**The served prompt is the trained prompt (fixed 2026-09-28).** Training rendered
`Task: <task>, Control mode: joint, End effector: gripper, State: <|propri|>;` (the run's config sets
`end_effector_override="gripper"`, and `model_ema/inference_config.json` records it). The authors' server
ignores that key: it infers the end effector from the delta mask, gets `None` for this single-arm mask, and drops
`End effector: gripper` from every prompt, with no warning. Until 2026-09-28 this image and the benchmark's
GigaBrain row both served that degraded prompt. The adapter now serves the value the checkpoint recorded
(`"end_effector": "checkpoint"`, the default) and logs `gigabrain07 prompt end effector: {... 'served': 'gripper' ...}`
at startup. `-e 'VLA_BENCH_ADAPTER_KWARGS={"end_effector":"author"}'` restores the authors' behaviour.

Measured on the benchmark's five reference episodes (170 queries, rate 20):

| | mae_joints_rad | rmse_joints_rad | mae_gripper_m | mae_normalised | traj_mae_joints_rad | worst step (rad) |
|---|---:|---:|---:|---:|---:|---:|
| authors' prompt (old benchmark row) | 0.0077496 | 0.0150149 | 0.0009684 | 0.0369241 | 0.0062118 | 0.2470 |
| trained prompt (this image, new row) | **0.0068355** | 0.0137422 | 0.0008489 | 0.0328690 | 0.0052968 | 0.2842 |
| change | **-11.80 %** | -8.48 % | -12.34 % | -10.98 % | -14.73 % | +15.06 % |

The benchmark row was re-run on 2026-09-28 (SLURM 483515) with the fixed adapter. This image, served from the upload
set alone on a network with no route off the host, reproduces it bit-exactly: 35,700/35,700 predicted values. Its
seeded probe output equals the audit's forced-prompt run exactly, and with `end_effector=author` it equals the old
served output exactly, so the fix changes the prompt and nothing else. The old row is kept as
`results/deploy_eval/gigabrain07/*_oldprompt.*`.
