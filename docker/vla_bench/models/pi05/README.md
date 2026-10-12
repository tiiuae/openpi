# pi05

pi0.5 (LeRobot's PyTorch port of Physical Intelligence's openpi pi0.5) fine-tuned from `lerobot/pi05_base` on the
right-arm 7-D, 2-camera embodiment. Predicts **30 actions per call and executes all 30** (1.00 s at 30 fps), with
10 flow-matching steps per call. Source: `huggingface/lerobot` main at `bad0260a4` (Python >= 3.12), cloned at
that pinned SHA during the build.

## Build

```bash
docker build -f docker/vla_bench/models/pi05/Dockerfile -t vla-bench-pi05:latest .   # from the openpi root
```

## Run

```bash
# fully offline: no network, no Hub token, and no Hugging Face cache needed
WEIGHTS_DIR=/opt/vla_weights docker/vla_bench/run.sh pi05 --probe   # load + one inference, then exit
WEIGHTS_DIR=/opt/vla_weights docker/vla_bench/run.sh pi05           # serve on :8800
```

Loading takes 2-5 minutes (132-300 s measured on an A100 node, warm and cold file cache): LeRobot first builds
the 4.1 B-parameter model on the CPU with random weights, then loads `model.safetensors` over it, and the adapter
then compares every tensor (below). The image sets `OMP_NUM_THREADS=8`, as the benchmark's host runs did: with
torch's default thread count one inference took 351 ms instead of 308 ms on a 16-core allocation.

The checkpoint is the LeRobot `pretrained_model/` directory (`config.json`, `model.safetensors`,
`policy_{pre,post}processor*`, `train_config.json`; 9.4 GB). The benchmark's upload set
(`VLA-SOTA/repo/scripts/upload_checkpoints.sh`) puts those files directly under `pi05/`, with the tokenizer in
`pi05/paligemma-3b-pt-224/`, so with `WEIGHTS_DIR` pointing at a download of that set the image default
`/models/pi05` is the checkpoint and no `--checkpoint` is needed.

## Three things specific to this model

**The tokenizer ships in the upload set, so it starts offline.** The pre-processor tokenises the prompt with
`google/paligemma-3b-pt-224`, a gated repo it would otherwise load **by name**. The upload set carries the six
tokenizer files at revision `35e4f46485b4d07967e7e9935bc3786aad50687c` (the one training loaded; 22 MB, no
weights) as `pi05/paligemma-3b-pt-224/`, and the image's adapter default is
`"tokenizer_dir": "/models/pi05/paligemma-3b-pt-224"`. The adapter checks every file's sha256 at that revision
before using it and stops otherwise. Nothing else is loaded from the Hub: the PaliGemma and action-expert
architectures are built from code, every weight comes from `model.safetensors`.

**A failed weight load is a hard error here.** LeRobot's `PI05Policy.from_pretrained` catches any error while
loading `model.safetensors`, prints it, and returns the model with its random initial weights. The adapter
compares every tensor of the file with the loaded model and refuses to start on any missing, extra or different
tensor, so a `--probe` that passes means every tensor matched.

**The camera slots are pinned, not guessed.** Training renamed `cam_high -> base_0_rgb` and
`cam_right_wrist -> left_wrist_0_rgb` (baked into `policy_preprocessor.json`); the third slot,
`right_wrist_0_rgb`, is absent and masked, exactly as in training. The image passes the *original* dataset key
names in `rename_map` and the adapter checks that `cam_high` lands in the first slot and `cam_right_wrist` in the
second. Do not send a third view and do not swap the cameras: the model would still answer.

Noise is drawn unseeded on every call (as in LeRobot's own inference), so two runs of the same request differ
slightly; set a seed only for comparisons.
