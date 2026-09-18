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

## Verification status — loads and builds its batch, **not** replay-verified

**Verified.** Builds, including the nvcc step (62 s), and `_cuda_ext.is_available()` passes inside the
image. `--probe` against the real mounted checkpoint then gets all the way through the load: the
training collator is rebuilt from the YAML, `LeRobotDatasetMetadata` reads the baked-in `meta/`, the
checkpoint is resolved and overlaid, the normalizers are built **under the `right7_2view_v1` key** (the
adapter raises rather than fall back to another dataset's scale, so this is a real check), the model
config is built from `<ckpt>/config.json`, the processors load, the token embeddings are resized and
the state dict loads with **no missing tensors**. Run with `device=cpu` it goes further still and
reaches the vision tower inside `generate_flow_action`, which means `_vision_preprocess`, the collator
batch build and the `<|action|>` / `<|propri|>` token-id assertions all pass too.

**Not verified.** Inference, and therefore the 170-query replay. Two independent walls:

* On GPU it dies at `model.to(self.device)` (`adapters/walloss05.py:201`) with **7.60 GiB in use and
  44 MiB free**. ENV.md measures the weights alone at 7.76 GiB bf16 and the peak at 7.96 GiB, and the
  build host had 7.83 GB free on every card for the whole session (another user's 8-GPU job held
  73.3 GB of each A100). **This one is within ~0.3 GB of running** — a card with 9 GB free is enough.
* On CPU it dies in `wall_x/model/core/ops/_cuda_wrappers.py::get_token_counts_kernel`: wall-x
  dispatches `rot_pos_emb` to its compiled CUDA operator unconditionally and ships no CPU path, so
  there is no CPU fallback for this model.

Finish it with
`openpi_integration/tools/verify_container.sh walloss05 /models/walloss05/eval_ckpt/ep19 39,102,120,163,167`
and compare against `results/deploy_eval/walloss05/metrics.json`. Because the adapter seeds torch at
warmup, this model should match its reference exactly rather than approximately.
