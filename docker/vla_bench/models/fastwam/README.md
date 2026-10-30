# fastwam

FastWAM world-action model: a frozen Wan2.2-TI2V-5B video expert plus a trained ActionDiT action
expert. Source: `yuantianyuan01/FastWAM`, cloned at a pinned SHA during the build.

**It predicts 32 steps and executes 30.** Its video tokenizer requires a horizon that is a multiple of
4, so it trained at the author-native 32; the deployment consumes the first 30, which keeps the replan
cadence at the same 1.00 s as every other model in the benchmark. Do not "fix" the 32.

## Build

```bash
docker build -f docker/vla_bench/models/fastwam/Dockerfile -t vla-bench-fastwam:latest .
```

## Run

```bash
WEIGHTS_DIR=/opt/vla_weights HF_CACHE_DIR=$HOME/.cache/huggingface \
  docker/vla_bench/run.sh fastwam --probe          # image default: /models/fastwam/checkpoints/weights/step_010920.pt
```

The checkpoint is a **single `.pt` file** (the consolidated `{mot, proprio_encoder, step}` blob), not a
directory. The sibling `checkpoints/state/` DeepSpeed tree (~85 GB) is training state and is not used. In the
benchmark's upload set (`VLA-SOTA/repo/scripts/upload_checkpoints.sh`) it is
`fastwam/checkpoints/weights/step_010920.pt`, which is the image default since 2026-09-28 (it was
`/models/fastwam/weights.pt`, which the upload set does not have), so no `--checkpoint` is needed.

## What this model needs beside the checkpoint — more than any other

Under the weights mount:

| path in the container | what it is |
|---|---|
| `/models/fastwam/run/config.yaml` | the run's fully-resolved Hydra config, as trained; the deployer repoints one path in it (below) |
| `/models/fastwam/run/dataset_stats.json` | min/max normalisation for state and action |
| `/models/fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | the 2 GB ActionDiT backbone the model is built from; generated, not downloadable |
| `/models/fastwam/text_embeds_cache/right7_2view/` | 4 precomputed T5 embeddings, ~4 MB |
| `/models/fastwam/configs/`, `/models/fastwam/config.yaml` | the run's yamls and its unmodified config -- provenance only, not read at inference |

`upload_checkpoints.sh` ships all of these (since 2026-09-25). Under the Hub-cache mount it needs
`/hf/diffsynth/Wan-AI/Wan2.2-TI2V-5B/` holding `Wan2.2_VAE.pth` and the three
`diffusion_pytorch_model-0000?-of-00003.safetensors` shards (22.8 GB). This is a **plain directory, not the
hub-cache layout**, so `hf download` must write it with `--local-dir`:

```bash
# one pattern per command: hf >= 1.0 takes one value per --include
for pat in "diffusion_pytorch_model-*.safetensors" "Wan2.2_VAE.pth"; do
  hf download Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
    --include "$pat" --local-dir "$HF_CACHE_DIR/diffsynth/Wan-AI/Wan2.2-TI2V-5B"
done
```

If a file is missing, the error is misleading: `ValueError: Cannot detect model type for wan_video_dit.
File: []` (`DIFFSYNTH_SKIP_DOWNLOAD` stops it from fetching). The umT5 text encoder (11.4 GB of that repo) is
**not** needed: the model is built `load_text_encoder=false`, exactly as it trained. The DiT shards and the
ActionDiT backbone are fully overwritten by the trained `mot` weights once loaded, but the constructor reads
them, so they must be present.

## Two things that will bite you

**The training `config.yaml` carries absolute paths from the training machine, and the deployer repoints
one of them.** `model.action_dit_pretrained_path` names the 2 GB ActionDiT backbone by its cluster path, and
`create_fastwam` raises `FileNotFoundError: action_dit_pretrained_path does not exist: /lustre1/…` when it is
absent (deployment audit, 2026-09-25; re-measured 2026-09-28 on the upload set as shipped). The upload set ships
`run/config.yaml` exactly as trained (the upload script's 2026-09-25 rewrite was reverted on 2026-09-28: configs
are never rewritten on the way out). **Before the first start, set `model.action_dit_pretrained_path` in
`run/config.yaml` to `/models/fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`**
(`VLA-SOTA/results/deployment/CONFIG_PATHS_TO_CHANGE.md`). Edit a real copy, not a hardlink. It is the only path
that is read: the adapter instantiates `cfg.model` alone, takes the T5 cache from its own `text_embed_cache` kwarg,
and reads only geometry from `data.train`. So `data.train.text_embedding_cache_dir`, `output_dir` and
`data.train.dataset_dirs` may keep their cluster values. With that one edit `--probe` passes with nothing but the
upload set and the `diffsynth` tree mounted, and its output is identical, value for value, to the verified
configuration (re-verified 2026-09-28).

**An uncached instruction is a hard failure, not a fallback.** The model is built with
`load_text_encoder=false`, so prompts are looked up in the T5 embedding cache by
`sha256(prompt_template)`. A task string that was not in the training set raises `FileNotFoundError`
rather than degrading. If you change the instruction set you must regenerate the cache. The shipped
cache covers exactly the four training instructions,
`pick up the {metal pot, green cup, blue cup, small mug} and place it in the basket`. Case and whitespace
must match, and `--probe` checks only the first one.
