# VLA-bench model image — MolmoAct2 (LoRA-merged), chunk 30, 7-D right arm
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/molmoact2/Dockerfile -t vla-bench-molmoact2:latest .
#
# The policy code is LeRobot's molmoact2 policy, so the source pinned here is huggingface/lerobot at
# the commit this run installed (0.6.2) — NOT the allenai/molmoact2 author clone, which is only a
# reference and is not installed in the validated venv either.
ARG BASE=vla-bench-base:torch211-cu128
FROM ${BASE}

ARG SRC_REPO=https://github.com/huggingface/lerobot
ARG SRC_SHA=c13d79e6e350c589babc5b7c5cf5ef05cb0fa58e
RUN git init -q /opt/model \
 && git -C /opt/model remote add origin "${SRC_REPO}" \
 && git -C /opt/model fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/model checkout -q FETCH_HEAD \
 && echo "lerobot @ $(git -C /opt/model rev-parse HEAD)"

COPY docker/vla_bench/models/molmoact2/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-deps torchcodec==0.11.1 \
 || echo "torchcodec unavailable — inference does not decode video, continuing"
RUN pip install --no-cache-dir --no-deps -e /opt/model \
 && python -c "import lerobot, transformers; print('lerobot', lerobot.__version__, '| transformers', transformers.__version__)"

COPY packages/openpi-client /opt/openpi-client
COPY docker/vla_bench/server /app

# MolmoAct2 trained with no camera rename, so its input_features already use the dataset's own
# cam_high / cam_right_wrist keys. The map below is the identity and is passed only to pin the
# camera->slot assignment rather than relying on the adapter's name heuristics.
ENV VLA_BENCH_MODEL=molmoact2 \
    VLA_BENCH_ADAPTER=lerobot_policy:LeRobotAdapter \
    VLA_BENCH_ADAPTER_KWARGS='{"device":"cuda","dtype":"bfloat16","chunk_len":30,"exec_len":30,"rename_map":{"primary":"observation.images.cam_high","wrist":"observation.images.cam_right_wrist"}}' \
    VLA_BENCH_CHECKPOINT=/models/molmoact2 \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    HF_HOME=/hf \
    HF_HUB_OFFLINE=1

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
