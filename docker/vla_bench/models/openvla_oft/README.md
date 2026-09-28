# openvla_oft

OpenVLA-OFT+ with FiLM, the L1 regression action head, proprio and a merged r=32 LoRA. Predicts and
executes **30 actions per call**. Source: openpi's own `openvla-oft` submodule — this is the one model
that is not cloned from upstream during the build.

## Build

```bash
git submodule update --init openvla-oft                      # from the openpi root
docker build -f docker/vla_bench/models/openvla_oft/Dockerfile -t vla-bench-openvla_oft:latest .
```

The build applies `patches/0001-right7-constants.patch`, which adds the benchmark's RIGHT7 embodiment
block (chunk 30, action 7, proprio 7) to `prismatic/vla/constants.py` and stops the fork's hardcoded
`constants = ALOHA_CONSTANTS` from overriding it. The build fails loudly if the constants do not come
out as (30, 7, 7); it does not silently fall back.

## Run

```bash
# WEIGHTS_DIR must hold a PRIVATE, real (not hardlinked) copy: the loader writes into it (see below)
WEIGHTS_DIR=/opt/vla_weights_private docker/vla_bench/run.sh openvla_oft --probe
WEIGHTS_DIR=/opt/vla_weights_private docker/vla_bench/run.sh openvla_oft
```

The checkpoint is a merged HF checkpoint directory (the step-50000 run, flat under `openvla_oft/` in the
benchmark's upload set). Since 2026-09-28 that is the image default, `/models/openvla_oft`; before, the image
set none and the server stopped with `TypeError: … missing 1 required positional argument: 'checkpoint_dir'`.
It must contain
`dataset_statistics.json`, which holds the action/proprio normalisation bounds; the adapter refuses to start
without it (`FileNotFoundError`). `unnorm_key` defaults to `right7_2view_v1`, the single key in that file.
`lora_adapter/` is never read (the LoRA is already merged into the shards).

**A read-only Hugging Face cache used to break it; the image now sets `HF_MODULES_CACHE=/tmp/hf_modules`.**
The checkpoint's `config.json` has an `auto_map` and the loader passes `trust_remote_code=True`, so
transformers copies `{modeling,configuration,processing}_prismatic.py` into
`$HF_MODULES_CACHE/transformers_modules/<checkpoint-dir-name>/`, by default under `$HF_HOME/modules`. On a
fresh cache that meant creating `/hf/modules` on the read-only mount: `OSError: [Errno 30] Read-only file
system: '/hf/modules'` (deployment audit, 2026-09-25); the cluster verification passed only because its cache
already held that module directory. With the module cache in the container's own `/tmp`, `run.sh`'s read-only
`/hf` mount is harmless (OFT resolves no Hub repo at all, so `/hf` may also be absent).

## Two things specific to this model

**It WRITES INTO THE CHECKPOINT DIRECTORY.** `openvla_utils.get_vla()` calls `update_auto_map()`, which
rewrites `config.json` **in place** (same inode; it also drops the trailing newline) and adds a
`config.json.back.<timestamp>` on every start. (`check_model_logic_mismatch()` would also overwrite the two
`.py` files, but it searches `./prismatic/` relative to the working directory, finds nothing under the image's
`/app`, and only prints a warning.) `run.sh` therefore mounts `/models` **read-write** for this model alone.
Give it a private copy of the checkpoint, never a shared snapshot and never the live training output, and make
it a **real** copy: a hardlink copy (`cp -al`, `rsync --link-dest`) shares the inode, so the in-place rewrite
goes straight through into the original.

**It is the odd environment out.** torch 2.2.0 / CUDA 12.1 / Python 3.10, a *devel* base, TensorFlow
2.15 alongside torch (the authors use TF for the lanczos resize and the centre crop — the adapter hides
the GPU from it), a `transformers` fork from git, and flash-attn 2.5.5. `tensorflow-metadata` must stay
at 1.13.1 and `peft` at 0.11.1; both are pinned in `requirements.lock.txt` and moving either breaks the
load. `center_crop=true` is not cosmetic — the run trained with `--image_aug`, so inference must
centre-crop at scale 0.9 to match the training distribution.
