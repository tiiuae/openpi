# VLA-bench model image — SmolVLA (LeRobot-native, chunk 30, 7-D right arm)
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/smolvla/Dockerfile -t vla-bench-smolvla:latest .
#
# The model source is NOT a submodule: it is cloned at a pinned commit during the build, so the image
# is reproducible without adding a 400 MB submodule to openpi. Override SRC_SHA to move the pin.
#
# Weights are never baked in — see docker/vla_bench/models/smolvla/README.md.
ARG BASE=vla-bench-base:torch211-cu128
FROM ${BASE}

# ---------------------------------------------------------------------------------------------
# 1. model source — LeRobot v0.6.1 (the tag SmolVLA was fine-tuned with)
# ---------------------------------------------------------------------------------------------
ARG SRC_REPO=https://github.com/huggingface/lerobot
ARG SRC_SHA=7e241bd630a3719a56157a497ce5d08f244784f1
RUN git init -q /opt/model \
 && git -C /opt/model remote add origin "${SRC_REPO}" \
 && git -C /opt/model fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/model checkout -q FETCH_HEAD \
 && echo "lerobot @ $(git -C /opt/model rev-parse HEAD)"

# ---------------------------------------------------------------------------------------------
# 2. dependencies, pinned to the venv the reference numbers were measured in
#    (torch/torchvision/nvidia-* deliberately absent: the base already has that exact build)
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/models/smolvla/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
# torchcodec is only reached by LeRobotDataset's video decoding, which a policy server never does.
# --no-deps keeps it from pulling the 3 GB cuda-toolkit wheel; a failure here is not fatal.
RUN pip install --no-cache-dir --no-deps torchcodec==0.11.1 \
 || echo "torchcodec unavailable — inference does not decode video, continuing"
RUN pip install --no-cache-dir --no-deps -e /opt/model \
 && python -c "import lerobot, transformers; print('lerobot', lerobot.__version__, '| transformers', transformers.__version__)"

# ---------------------------------------------------------------------------------------------
# 3. openpi's own wire codec, so the msgpack bytes match production exactly
# ---------------------------------------------------------------------------------------------
COPY packages/openpi-client /opt/openpi-client

# ---------------------------------------------------------------------------------------------
# 4. the shared server and the adapters
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/server /app

# Adapter kwargs are the ones the reference metrics were produced with
# (VLA-SOTA/repo/scripts/eval/jobs/smolvla.cfg). rename_map pins cam_high/cam_right_wrist to the
# checkpoint's camera1/camera2 slots instead of relying on name heuristics.
ENV VLA_BENCH_MODEL=smolvla \
    VLA_BENCH_ADAPTER=lerobot_policy:LeRobotAdapter \
    VLA_BENCH_ADAPTER_KWARGS='{"device":"cuda","dtype":"bfloat16","chunk_len":30,"exec_len":30,"rename_map":{"primary":"observation.images.cam_high","wrist":"observation.images.cam_right_wrist"}}' \
    VLA_BENCH_CHECKPOINT=/models/smolvla \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    HF_HOME=/hf \
    HF_HUB_OFFLINE=1

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
