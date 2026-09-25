# smolvla

LeRobot-native SmolVLA fine-tuned on the right-arm 7-D, 2-camera embodiment. Predicts **30 actions
per call and executes all 30** (1.00 s at 30 fps). Source: `huggingface/lerobot` v0.6.1, cloned at a
pinned SHA during the build.

## Build

```bash
docker build -f docker/vla_bench/models/smolvla/Dockerfile -t vla-bench-smolvla:latest .   # from the openpi root
```

## Run

```bash
WEIGHTS_DIR=/opt/vla_weights HF_CACHE_DIR=$HOME/.cache/huggingface \
  docker/vla_bench/run.sh smolvla --probe                     # load + one inference, then exit
WEIGHTS_DIR=/opt/vla_weights docker/vla_bench/run.sh smolvla  # serve on :8800
```

The checkpoint is the LeRobot `pretrained_model/` directory (`config.json`, `model.safetensors`,
`policy_{pre,post}processor*`). The benchmark's upload set (`VLA-SOTA/repo/scripts/upload_checkpoints.sh`)
puts those files directly under `smolvla/`, so with `WEIGHTS_DIR` pointing at a download of that set the
image default `/models/smolvla` is the checkpoint and no `--checkpoint` is needed. For any other layout, point
at the directory that holds `config.json`:

```bash
docker/vla_bench/run.sh smolvla --checkpoint /models/smolvla/<dir-with-config.json>
```

## Two things specific to this model

**It needs the Hub cache, not just the checkpoint.** `config.json` sets
`vlm_model_name: HuggingFaceTB/SmolVLM2-500M-Video-Instruct` and `load_vlm_weights: true`, so the
policy rebuilds that backbone — and the preprocessor's tokenizer — from the Hub **by repo id** every
time it loads. `run.sh` mounts `HF_CACHE_DIR` at `/hf` for exactly this; without it the container
must reach huggingface.co. Pre-seed with
`hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --exclude "onnx/*"` (~1.9 GB).

**The camera names are pinned, not guessed.** Training renamed `cam_high -> camera1` and
`cam_right_wrist -> camera2`, and that rename is baked into `policy_preprocessor.json`. The image
therefore passes the *original* dataset key names in `VLA_BENCH_ADAPTER_KWARGS.rename_map` and lets
the checkpoint's own rename step do the mapping, instead of letting the adapter's name heuristics
pick. Override the kwargs only if you retrain with different names.
