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
mount at `/models/gigabrain07/norm_stats_right7_emb8.json`** — copy that file next to the checkpoint.
If it is missing the policy loads but un-normalises with the wrong statistics.

**embodiment_id 8 is not a default.** The pretrained model has nine embodiment slots; slot 8 is the one
this benchmark trained into, and the delta mask (`[T]*6 + [F]` — joints delta, gripper absolute) is
keyed on it. Changing it silently produces plausible-looking but wrong actions.
