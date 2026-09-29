# VLA-bench model image — Flex-pi (Wan2.2-TI2V-5B video expert + 1.02 B ActionDiT; co-generates future RGB, DINOv3
# features and DA3 pointmaps with the action chunk). Result label: "Flex-pi recipe from Wan2.2".
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/flexpi/Dockerfile -t vla-bench-flexpi:latest .
#
# Code only. The upload set (checkpoint, run config + statistics, umT5 embedding cache, camera intrinsics, the
# converted Wan2.2 VAE) is mounted at /models/flexpi; DINOv3 and Depth Anything 3 METRIC-LARGE come from the
# Hugging Face cache mounted at /hf (see README.md). Predicts 32 steps, returns the first 30.
ARG BASE=vla-bench-base:torch271-cu128
FROM ${BASE}

# 0. the base must carry exactly the torch stack the validated env ran (torch 2.7.1+cu128, cuDNN 9.7.1).
RUN python -c "import torch, torchvision, triton; \
assert torch.__version__ == '2.7.1+cu128', torch.__version__; \
assert torchvision.__version__ == '0.22.1+cu128', torchvision.__version__; \
assert triton.__version__ == '3.3.1', triton.__version__; \
assert torch.backends.cudnn.version() == 90701, torch.backends.cudnn.version(); print('base ok', torch.__version__)"

# 1. model source, pinned, unpatched (the benchmark ran the checkout with `git status` clean).
ARG SRC_REPO=https://github.com/geyan21/flex-pi
ARG SRC_SHA=20c1b2b71ea35a415d5d47c39b04443cfadad7a1
RUN git init -q /opt/model \
 && git -C /opt/model remote add origin "${SRC_REPO}" \
 && git -C /opt/model fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/model checkout -q FETCH_HEAD \
 && test "$(git -C /opt/model rev-parse HEAD)" = "${SRC_SHA}" \
 && echo "flex-pi @ $(git -C /opt/model rev-parse HEAD)"

# 2. Depth Anything 3, pinned (the revision that labelled the training depth). Source only, on the adapter's
#    sys.path (da3_src): its pyproject pulls export/GUI dependencies inference never imports.
ARG DA3_REPO=https://github.com/ByteDance-Seed/Depth-Anything-3
ARG DA3_SHA=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
RUN git init -q /opt/da3 \
 && git -C /opt/da3 remote add origin "${DA3_REPO}" \
 && git -C /opt/da3 fetch -q --depth 1 origin "${DA3_SHA}" \
 && git -C /opt/da3 checkout -q FETCH_HEAD \
 && test "$(git -C /opt/da3 rev-parse HEAD)" = "${DA3_SHA}" \
 && echo "Depth-Anything-3 @ $(git -C /opt/da3 rev-parse HEAD)"

# 3. dependencies: a freeze of the validated env, --no-deps (see requirements.lock.txt for what is left out and why).
COPY docker/vla_bench/models/flexpi/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-deps -e /opt/model \
 && python -c "import sys; sys.path.insert(0, '/opt/da3/src'); \
import flexpi.runtime, flexpi.models.flexpi, flexpi.datasets.lerobot.processors.flexpi_processor, timm, hydra, numpy; \
import depth_anything_3.cfg, depth_anything_3.registry, depth_anything_3.model.da3; \
print('flexpi', flexpi.__file__, '| timm', timm.__version__, '| numpy', numpy.__version__)"

COPY packages/openpi-client /opt/openpi-client
COPY docker/vla_bench/server /app

# 4. runtime.
#  * run_dir=/models/flexpi: the upload set IS the run directory (config.yaml, dataset_stats.json,
#    checkpoints/weights/step_002430.pt). config.yaml ships as trained; the training-host paths in it are not read.
#  * DIFFSYNTH_MODEL_BASE_PATH: the frozen Wan2.2 VAE is read from
#    /models/flexpi/diffsynth/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors (shipped: the
#    converted file is ModelScope-only); DIFFSYNTH_SKIP_DOWNLOAD keeps the loader off the network.
#  * VLA_BENCH_HUB_PINS: DINOv3 (timm, loaded by name) and DA3METRIC-LARGE at the revisions training used. The
#    adapter resolves da3_weights "hf:<repo>@<rev>" to that snapshot in the /hf cache.
#  * full joint (the authors' deployed mode), 4 Euler steps, bf16, eager, seed = the run's 1000. The authors' compiled
#    serve path is `-e 'VLA_BENCH_ADAPTER_KWARGS={"torch_compile":true}'` (minutes of compile at start-up; README).
ENV VLA_BENCH_MODEL=flexpi \
    VLA_BENCH_ADAPTER=flexpi:FlexPiAdapter \
    VLA_BENCH_ADAPTER_KWARGS_DEFAULTS='{"run_dir":"/models/flexpi","text_embed_cache":"/models/flexpi/text_embeds_cache/right7_2view","intrinsics_json":"/models/flexpi/camera_intrinsics.json","da3_weights":"hf:depth-anything/DA3METRIC-LARGE@4010e39f3634a45bc60553321fb49fb760bd594e","da3_src":"/opt/da3/src","device":"cuda:0","chunk_len":30,"exec_len":30,"regime":"full_joint","num_inference_steps":4,"torch_compile":false}' \
    VLA_BENCH_CHECKPOINT=/models/flexpi/checkpoints/weights/step_002430.pt \
    VLA_BENCH_HUB_PINS='{"timm/vit_base_patch16_dinov3.lvd1689m":{"revision":"c6a5fb7d12bbd3cf3b0079253141c3332aaed7da","allow_patterns":["model.safetensors"]},"depth-anything/DA3METRIC-LARGE":{"revision":"4010e39f3634a45bc60553321fb49fb760bd594e","allow_patterns":["model.safetensors","config.json"]}}' \
    VLA_BENCH_PROBE_TASK="pick up the metal pot and place it in the basket" \
    DIFFSYNTH_MODEL_BASE_PATH=/models/flexpi/diffsynth \
    DIFFSYNTH_SKIP_DOWNLOAD=true \
    HYDRA_FULL_ERROR=1 \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    OMP_NUM_THREADS=8 \
    HF_HOME=/hf

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
