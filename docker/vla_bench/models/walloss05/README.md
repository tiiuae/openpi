# walloss05

Wall-OSS-0.5 (`wall-x`, Qwen2.5-VL backbone + flow-matching action expert, 4.17 B params) on the
right-arm 7-D, 2-camera embodiment. Predicts and executes **30 actions per call**. Source:
`X-Square-Robot/wall-x`, cloned at a pinned SHA during the build.

## Build

```bash
docker build -f docker/vla_bench/models/walloss05/Dockerfile -t vla-bench-walloss05:latest .
```

## Run

```bash
WEIGHTS_DIR=/opt/vla_weights docker/vla_bench/run.sh walloss05 --probe \
  --checkpoint /models/walloss05/eval_ckpt/ep19
```

The checkpoint is the directory holding `model.safetensors`, `config.json`, `config.yml`,
`norm_stats.json`, `normalizer_{action,propri}.pth`, `preprocessor_config.json` and the tokenizer
files. The optimizer/scheduler/RNG state next to a raw training save is not read and does not have to
be copied.

**No weights are in this image.** Worth saying explicitly for this model: the wall-x *code* is
Apache-2.0, but the Wall-OSS-0.5 *weights* on the Hub carry no licence at all, so a fine-tuned
derivative checkpoint cannot be redistributed. This image contains source and pinned dependencies and
mounts the checkpoint at run time, exactly like every other image here — nothing changes that.

**`/hf` is required, for one file set.** The *model's* config and processor do come out of the
mounted checkpoint (`build_model_config` reads `<ckpt>/config.json`, and
`load_train_config_with_ckpt_overlay` repoints `processor_path` at `<ckpt>`). But the **training
collator** loads its own processor first, from the `model.processor_path` in the YAML, and then adds
`<|propri|>` and `<|action|>` to that base vocabulary itself — so `/hf` must hold
`Qwen/Qwen2.5-VL-3B-Instruct` at revision `66285546d2b821cf421d4f5eb2576359d3770cd3` (config,
tokenizer and preprocessor only; the multi-GB safetensors are not read). `run.sh` mounts the cache.
Substituting the checkpoint's merged tokenizer here is a different tokenizer and a different image
processor, which is why the YAML names the snapshot explicitly.

## Five things specific to this model

**Only CUDA-13 image, only one that compiles at build time.** `pip install -e .` runs a
`torch.utils.cpp_extension` CUDAExtension build of `wall_x/model/core/ops/csrc/*.cu`, and
`_cuda_ext.load()` raises at import if the `.so` is missing — so the base is
`pytorch/pytorch:2.10.0-cuda13.0-cudnn9-**devel**`, not a runtime image, and `vla-bench-base:torch210-cu130`
is a devel image for this model alone. The build compiles for **sm_80 only**
(`--build-arg CUDA_ARCH_LIST=8.0`); pass a longer list for other cards.

**One mandatory patch, two files.** `patches/0001-move-batch-to-device-accept-Mapping.patch` widens
`isinstance(batch, dict)` to `collections.abc.Mapping` in `wall_x/trainer/utils/data.py` and
`wall_x/trainer/adapters/vla_model_adapter.py`. wall-x 1.1.0's LeRobot collator returns a transformers
`BatchFeature` (a `UserDict`, not a `dict`), so both device-move helpers were silent no-ops and the
first `embed_tokens(input_ids)` died on a CPU tensor. Any fresh clone at this SHA needs it.

**The training YAML is part of the inference contract.** `assets/right7_2view_ep20.yml` is the run's
own config with six host paths rewritten (its header lists them, and nothing else changed). The
adapter rebuilds the **training** data pipeline from it — the prompt text, the camera name mapping,
the image resize chain, the q01/q99 normalisation, the 7 → 26 zero padding, `dof_mask` — because
neither shipped entry point fits this embodiment: `scripts/fake_inference.py` goes through the LIBERO
end-effector encoder (we have no EEF slice at all) and `scripts/run_serving.sh` only knows the
authors' own robots. Change the YAML and you change the input distribution.

**188 KB of LeRobot dataset `meta/` is baked in.** `LeRobotDatasetMetadata(...).camera_keys` decides
the ORDER the two views are fed in, so it is not optional — but only `meta/` is read, not the 2.4 GB
of parquet and video beside it. Point `data.lerobot_config.root` in the YAML at a mounted dataset if
you would rather supply it yourself. `repo_id` must stay `right7_2view_v1`: it doubles as the
normalizer `ParameterDict` key baked into the checkpoint, and the adapter refuses to run if
`build_normalizers` falls back to a different key rather than silently applying another dataset's scale.

**It is reproducible, unlike the LeRobot policies here.** Flow inference starts from `torch.randn`;
the adapter seeds torch once at warmup (seed 1000, the evaluation seed), so the same replay through
this image returns the same numbers. Expect ~580 ms per inference on an A100 (bf16, 10 Euler steps,
two 256 px views) and ~8 GB of weights — that is **1.7 Hz**, not the tech report's figure; quote the
measurement, not the paper, if a deployment gate depends on rate.

## Verification status — **verified, bit-exact** (2026-09-24)

`--probe` against the real mounted epoch-20 checkpoint (`eval_ckpt/ep19`, the plain-file copy of run
directory `19`): **ok**, `(30, 7)`, 108.2 s to load, 657 ms first call. The 2026-09-18 attempt had
stopped at `model.to(self.device)` with 44 MiB free on a card another job held; on a free A100 it loads
and serves with nothing changed in the image.

The 170-query replay of the five reference episodes (`39,102,120,163,167`, rate 20) through this image,
driven by the same `deploy_eval/client.py` that produced the benchmark's numbers, reproduces the recorded
reference **bit-exactly**: every aggregate metric matches to 17 significant digits, and element by element
**all 35,700 predicted values over the 170 queries are identical (max |delta| = 0.000e+00)**. The adapter
seeds torch at warmup (seed, one throwaway pass, re-seed), so the same query sequence draws the same
noise; bit-exact is the expected result, and it means the CUDA-13 image, the in-image nvcc build of the
wall-x operators, the baked-in `meta/` and the `/tmp` datasets cache all reproduce the validated venv.

| aggregate metric | reference | container | delta |
|---|---:|---:|---:|
| `mae_joints_rad` | 0.02970975193236513 | 0.02970975193236513 | 0.00 % |
| `rmse_joints_rad` | 0.0383539216046684 | 0.0383539216046684 | 0.00 % |
| `mae_gripper_m` | 0.0016492825877318026 | 0.0016492825877318026 | 0.00 % |
| `mae_normalised` | 0.12923127253742614 | 0.12923127253742614 | 0.00 % |
| `traj_mae_joints_rad` | 0.028363796192193952 | 0.028363796192193952 | 0.00 % |

Command (benchmark workspace):
`GPU=3 openpi_integration/tools/verify_container.sh walloss05 /models/walloss05/eval_ckpt/ep19 39,102,120,163,167 8852`
-- it mounts `/hf` read-only, and the image's own `HF_DATASETS_CACHE=/tmp/hf_datasets` keeps the
`datasets` FileLock off that mount.

Server-side latency during the replay: 668 ms p50 (mean 703 ms), against 625 ms p50 in the recorded
reference -- take the rate from the benchmark's deployment guide, not from a verification run on a shared
node.

Two walls recorded earlier are unchanged and are properties of wall-x, not of the image: there is **no CPU
path** (`rot_pos_emb` dispatches to the compiled CUDA operator unconditionally), and the build compiles
for **sm_80 only** unless `CUDA_ARCH_LIST` is widened.

Image `vla-bench-walloss05:latest`: **15.68 GB** (`docker image inspect`, 15,675,319,509 bytes), on the
13.36 GB `torch210-cu130` devel base.
