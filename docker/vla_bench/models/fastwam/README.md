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
  docker/vla_bench/run.sh fastwam --probe --checkpoint /models/fastwam/weights.pt
```

The checkpoint is a **single `.pt` file** (the consolidated `{mot, proprio_encoder, step}` blob), not a
directory. The sibling `checkpoints/state/` DeepSpeed tree (~85 GB) is training state and is not used.

## What this model needs beside the checkpoint — more than any other

Under the weights mount:

| path in the container | what it is |
|---|---|
| `/models/fastwam/run/config.yaml` | the run's fully-resolved Hydra config |
| `/models/fastwam/run/dataset_stats.json` | min/max normalisation for state and action |
| `/models/fastwam/text_embeds_cache/right7_2view/` | 4 precomputed T5 embeddings, ~4 MB |
| `/models/fastwam/configs/` | the run's three task/model/data yamls |

and under the Hub-cache mount, `/hf/diffsynth/Wan-AI/Wan2.2-TI2V-5B/` with `Wan2.2_VAE.pth` and the
three DiT safetensors shards (~21 GB). The umT5 text encoder is **not** needed: the model is built
`load_text_encoder=false`, exactly as it trained.

## Two things that will bite you

**`config.yaml` carries absolute paths from the training machine.** `model.action_dit_pretrained_path`
points at the 2 GB ActionDiT checkpoint by absolute host path, and that path does not exist in a
container. Copy `config.yaml` into the mount and rewrite that one key to the container path (e.g.
`/models/fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`) before running. The other
absolute paths in the file are dataloader-side and are not read at inference.

**An uncached instruction is a hard failure, not a fallback.** The model is built with
`load_text_encoder=false`, so prompts are looked up in the T5 embedding cache by
`sha256(prompt_template)`. A task string that was not in the training set raises `FileNotFoundError`
rather than degrading. If you change the instruction set you must regenerate the cache.
