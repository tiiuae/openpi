# molmoact2

MolmoAct2 with a merged LoRA, on the right-arm 7-D, 2-camera embodiment. Predicts and executes **30
actions per call**. The policy code is LeRobot's `molmoact2` policy, so the pinned source is
`huggingface/lerobot` at the commit this run installed (0.6.2) — not the `allenai/molmoact2` author
clone, which the validated environment does not install either.

## Build

```bash
docker build -f docker/vla_bench/models/molmoact2/Dockerfile -t vla-bench-molmoact2:latest .
```

## Run

```bash
WEIGHTS_DIR=/opt/vla_weights HF_CACHE_DIR=$HOME/.cache/huggingface \
  docker/vla_bench/run.sh molmoact2 --probe
docker/vla_bench/run.sh molmoact2 --checkpoint /models/molmoact2/010000/pretrained_model
```

Expect a slow first load: ~8 minutes to READY, because the 12 GB LoRA-merged delta is loaded on top of
a 21 GB base snapshot. Inference is ~260 ms.

## Two things specific to this model

**The checkpoint is only half the weights, and it points at the other half by ABSOLUTE PATH.** Its
`config.json` carries `checkpoint_path` = the full filesystem path of the `allenai/MolmoAct2` base
snapshot on the machine that trained it (~21 GB), and `utils_molmoact2.resolve_checkpoint_location`
returns that string unchanged when it exists. Inside a container it does not exist, and the load fails.
Either mount the Hub cache at that same absolute path:

```bash
docker/vla_bench/run.sh molmoact2 ... -v /original/path/to/hf_cache:/original/path/to/hf_cache:ro
```

or re-save `config.json` with a container path before deploying. There is no environment variable that
overrides it. This is the single biggest deployment wart of this model.

**The camera names are already the dataset's own.** Training did no rename, so `input_features` uses
`observation.images.cam_high` / `cam_right_wrist` directly. The `rename_map` in the adapter kwargs is
the identity and is passed only to pin the camera-to-slot assignment instead of relying on the
adapter's name heuristics.
