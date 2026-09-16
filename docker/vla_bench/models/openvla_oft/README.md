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
WEIGHTS_DIR=/opt/vla_weights docker/vla_bench/run.sh openvla_oft --probe
WEIGHTS_DIR=/opt/vla_weights docker/vla_bench/run.sh openvla_oft --checkpoint /models/openvla_oft/step30000
```

The checkpoint is a merged HF checkpoint directory and must contain `dataset_statistics.json` — that is
where the action/proprio normalisation bounds live, and without it the predictions come out
un-normalised. `unnorm_key` defaults to `right7_2view_v1`, the single key in that file.

## Two things specific to this model

**It WRITES INTO THE CHECKPOINT DIRECTORY.** `openvla_utils.get_vla()` calls `update_auto_map()` and
`check_model_logic_mismatch()`, which rewrite `config.json` and overwrite `modeling_prismatic.py` and
`configuration_prismatic.py` **inside whatever directory you point it at**. `run.sh` therefore mounts
`/models` **read-write** for this model alone. Give it a private copy of the checkpoint, never a shared
snapshot and never the live training output — the first run changes the files for everyone else.

**It is the odd environment out.** torch 2.2.0 / CUDA 12.1 / Python 3.10, a *devel* base, TensorFlow
2.15 alongside torch (the authors use TF for the lanczos resize and the centre crop — the adapter hides
the GPU from it), a `transformers` fork from git, and flash-attn 2.5.5. `tensorflow-metadata` must stay
at 1.13.1 and `peft` at 0.11.1; both are pinned in `requirements.lock.txt` and moving either breaks the
load. `center_crop=true` is not cosmetic — the run trained with `--image_aug`, so inference must
centre-crop at scale 0.9 to match the training distribution.
