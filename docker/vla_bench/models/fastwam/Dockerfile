# VLA-bench model image — FastWAM (world-action model: Wan2.2-TI2V-5B video expert + ActionDiT)
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/fastwam/Dockerfile -t vla-bench-fastwam:latest .
#
# THE ONE HORIZON EXCEPTION IN THE BENCHMARK: its video tokenizer needs a horizon that is a multiple
# of 4, so it predicts 32 steps and the deployment executes the first 30. The replan cadence is
# therefore the same 1.00 s as every other model.
ARG BASE=vla-bench-base:torch271-cu128
FROM ${BASE}

ARG SRC_REPO=https://github.com/yuantianyuan01/FastWAM
ARG SRC_SHA=7faa71108368fbb3b6885649f112af607427a2d4
RUN git init -q /opt/model \
 && git -C /opt/model remote add origin "${SRC_REPO}" \
 && git -C /opt/model fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/model checkout -q FETCH_HEAD \
 && echo "FastWAM @ $(git -C /opt/model rev-parse HEAD)"

COPY docker/vla_bench/models/fastwam/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
# torchcodec 0.4.0 installs but cannot load without system FFmpeg shared libs, and the run's data
# config uses the pyav backend for exactly that reason. Inference decodes no video at all.
RUN pip install --no-cache-dir --no-deps "torchcodec==0.4.0" \
 || echo "torchcodec unavailable — the run uses the pyav backend and inference decodes no video"
RUN pip install --no-cache-dir --no-deps -e /opt/model \
 && python -c "import fastwam; print('fastwam ok')"

COPY packages/openpi-client /opt/openpi-client
COPY docker/vla_bench/server /app

# DIFFSYNTH_MODEL_BASE_PATH is mandatory: the Wan2.2 VAE and DiT are resolved as
# $DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.2-TI2V-5B/..., and DIFFSYNTH_SKIP_DOWNLOAD keeps the
# container off the network. Under the standard /hf mount that is /hf/diffsynth.
# LEROBOT_HOME must stay unset — fastwam/datasets/lerobot/constants.py raises if the deprecated
# name is present.
ENV FASTWAM_HOME=/models/fastwam \
    FASTWAM_CONFIGS=/models/fastwam/configs \
    DIFFSYNTH_MODEL_BASE_PATH=/hf/diffsynth \
    DIFFSYNTH_SKIP_DOWNLOAD=true \
    HYDRA_FULL_ERROR=1 \
    VLA_BENCH_PROBE_TASK="pick up the metal pot and place it in the basket" \
    VLA_BENCH_MODEL=fastwam \
    VLA_BENCH_ADAPTER=fastwam:FastWAMAdapter \
    VLA_BENCH_ADAPTER_KWARGS='{"run_dir":"/models/fastwam/run","text_embed_cache":"/models/fastwam/text_embeds_cache/right7_2view","device":"cuda:0","dtype":"bfloat16","chunk_len":32,"exec_len":30,"compile_action_infer":false}' \
    VLA_BENCH_CHECKPOINT=/models/fastwam/weights.pt \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    HF_HOME=/hf

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
