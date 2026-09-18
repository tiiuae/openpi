# VLA-bench model image — Wall-OSS-0.5 (wall-x, Qwen2.5-VL backbone + flow action expert, chunk 30, 7-D right arm)
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/walloss05/Dockerfile -t vla-bench-walloss05:latest .
#
# This is the only CUDA-13 image in the set, the only one that COMPILES a CUDA extension during the
# build, and the only one whose base is therefore a *devel* image. Budget ~2 minutes for the nvcc step.
#
# Weights are never baked in — see docker/vla_bench/models/walloss05/README.md.
ARG BASE=vla-bench-base:torch210-cu130
FROM ${BASE}

# ---------------------------------------------------------------------------------------------
# 1. model source — wall-x at the pinned SHA, plus the one mandatory fix
# ---------------------------------------------------------------------------------------------
ARG SRC_REPO=https://github.com/X-Square-Robot/wall-x
ARG SRC_SHA=ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f
RUN git init -q /opt/model \
 && git -C /opt/model remote add origin "${SRC_REPO}" \
 && git -C /opt/model fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/model checkout -q FETCH_HEAD \
 && echo "wall-x @ $(git -C /opt/model rev-parse HEAD)"

# 0001 widens two `isinstance(batch, dict)` tests to `collections.abc.Mapping`, in
# wall_x/trainer/utils/data.py::move_batch_to_device and
# wall_x/trainer/adapters/vla_model_adapter.py::_move_batch. wall-x 1.1.0's LeRobot collator returns a
# transformers `BatchFeature`, which subclasses UserDict and is NOT a dict, so both helpers were silent
# no-ops: every tensor stayed on CPU and the first embed_tokens(input_ids) raised
# "Expected all tensors to be on the same device". It is one patch touching two files, and it is
# mandatory — any fresh clone of wall-x at this SHA needs it (VLA-SOTA BLOCKERS.md B10).
COPY docker/vla_bench/models/walloss05/patches /tmp/patches
RUN apt-get update && apt-get install -y --no-install-recommends patch && apt-get clean \
 && for p in /tmp/patches/*.patch; do echo "applying $p"; patch -p1 -d /opt/model < "$p"; done

# ---------------------------------------------------------------------------------------------
# 2. LeRobot v0.6.1. wall-x's own README installs LeRobot with --no-deps, and does not list it in its
#    requirements, so it is a separate checkout here as it is in the validated venv. The two packages
#    it then needs and does not get from --no-deps (draccus, gymnasium) are in the lock below.
# ---------------------------------------------------------------------------------------------
ARG LEROBOT_REPO=https://github.com/huggingface/lerobot
ARG LEROBOT_SHA=7e241bd630a3719a56157a497ce5d08f244784f1
RUN git init -q /opt/lerobot \
 && git -C /opt/lerobot remote add origin "${LEROBOT_REPO}" \
 && git -C /opt/lerobot fetch -q --depth 1 origin "${LEROBOT_SHA}" \
 && git -C /opt/lerobot checkout -q FETCH_HEAD \
 && echo "lerobot @ $(git -C /opt/lerobot rev-parse HEAD)"

# ---------------------------------------------------------------------------------------------
# 3. dependencies, pinned to the venv the reference numbers were measured in.
#    flash-attn comes from its GitHub release wheel (cu13 / torch2.10 / cp312 / cxx11abiTRUE) and is
#    load-bearing: the sanity run confirms flash_attention_2 is active for the model AND for
#    model.visual, and build_model_config hard-sets vision_config._attn_implementation there.
#    torchcodec is deliberately ABSENT: wall_x/data/backends/lerobot/loader.py hard-codes
#    LeRobotDataset(video_backend="pyav"), with no config knob, so PyAV is the video backend and
#    torchcodec would never be reached even if the server decoded video (it does not).
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/models/walloss05/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-deps -e /opt/lerobot \
 && python -c "import lerobot, transformers, flash_attn; print('lerobot', lerobot.__version__, \
'| transformers', transformers.__version__, '| flash_attn', flash_attn.__version__)"

# ---------------------------------------------------------------------------------------------
# 4. wall-x itself — the CUDA extension build.
#    --no-build-isolation because setup.py imports torch at build time (CUDAExtension), and an
#    isolated build env would not have it. TORCH_CUDA_ARCH_LIST=8.0 builds for A100/sm_80 only, which
#    is what the benchmark ran on and what keeps the compile at ~90 s instead of ~15 min; override the
#    build arg for other cards (e.g. --build-arg CUDA_ARCH_LIST="8.0;8.9;9.0").
#    The is_available() check is the point of doing it here: a missing .so does not fail the install,
#    it fails at the first inference, inside the robot session.
# ---------------------------------------------------------------------------------------------
ARG CUDA_ARCH_LIST=8.0
RUN cd /opt/model \
 && TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" MAX_JOBS=8 \
    pip install --no-cache-dir --no-build-isolation -e . \
 && python -c "from wall_x.model.core.ops import _cuda_ext; \
assert _cuda_ext.is_available(), 'wall-x CUDA operators did not build'; \
print('wall_x CUDA ext ok:', _cuda_ext.load().__file__)"

# ---------------------------------------------------------------------------------------------
# 5. the three non-upstream files this embodiment needs. None of them is a weight.
#      right7_2view_ep20.yml   the run's own training config, host paths rewritten. The adapter
#                              rebuilds the TRAINING collator from it, so the prompt, the camera
#                              mapping, the resize chain, the q01/q99 normalisation and the 7 -> 26
#                              padding all come from here. Read its header for the exact diff.
#      right7_2view_norm_stats.json  q01/q99/mean/std over the 139,662 training frames.
#      dataset_meta/           188 KB of LeRobot v3.0 meta/ for right7_2view_v1. Only
#                              LeRobotDatasetMetadata(...).camera_keys is read from it, and that
#                              decides the ORDER the two views are fed in, so it is not optional --
#                              but the 2.4 GB of parquet and video next to it is, hence just meta/.
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/models/walloss05/assets/right7_2view_ep20.yml /opt/wallx_assets/right7_2view_ep20.yml
COPY docker/vla_bench/models/walloss05/assets/right7_2view_norm_stats.json /opt/wallx_assets/right7_2view_norm_stats.json
COPY docker/vla_bench/models/walloss05/assets/dataset_meta /opt/dataset_meta/right7_2view_v1/meta

COPY packages/openpi-client /opt/openpi-client
COPY docker/vla_bench/server /app

# checkpoint_dir, not checkpoint: this adapter names the parameter differently from the rest.
# seed 1000 is the evaluation seed: flow inference starts from torch.randn and the adapter seeds once
# at warmup, so a replay through this image is reproducible end to end (unlike the LeRobot policies).
ENV VLA_BENCH_MODEL=walloss05 \
    VLA_BENCH_ADAPTER=walloss05:WallOSS05Adapter \
    VLA_BENCH_CHECKPOINT_KWARG=checkpoint_dir \
    VLA_BENCH_ADAPTER_KWARGS='{"repo_dir":"/opt/model","train_config":"/opt/wallx_assets/right7_2view_ep20.yml","chunk_len":30,"exec_len":30,"num_inference_timesteps":10,"device":"cuda","seed":1000}' \
    VLA_BENCH_CHECKPOINT=/models/walloss05/eval_ckpt/ep19 \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    PYTHONPATH=/opt/model \
    HF_HOME=/hf \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_CACHE=/tmp/hf_datasets

# HF_DATASETS_CACHE is moved OFF the /hf mount on purpose. LeRobotDatasetMetadata reads meta/episodes
# through `datasets.Dataset.from_parquet`, which takes a FileLock in the datasets cache; with /hf
# mounted read-only (as run.sh mounts it, correctly) that is an
# `OSError: [Errno 30] Read-only file system` during load. It is a scratch cache for parquet reading
# and holds nothing that reaches the model.
EXPOSE 8800
# 17 GB of safetensors off a network filesystem, then a bf16 cast: the start period is deliberately long.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
