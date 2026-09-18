# VLA-bench model image — GigaBrain-0.7 (PaliGemma2 flow-matching), chunk 30, 7-D right arm
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/gigabrain07/Dockerfile -t vla-bench-gigabrain07:latest .
ARG BASE=vla-bench-base:torch271-cu128
FROM ${BASE}

# ---------------------------------------------------------------------------------------------
# 1. model source. Unlike the others this repo is NOT pip-installed: the author's deployment path is
#    a script exec'd from the tree and the package it imports (giga_models) is a separate wheel, so
#    the repo only has to be importable — hence PYTHONPATH rather than `pip install -e`.
# ---------------------------------------------------------------------------------------------
ARG SRC_REPO=https://github.com/open-gigaai/giga-brain-0
ARG SRC_SHA=5f450a9d3368876ddb6e4b5bd7134afba1801fb9
RUN git init -q /opt/model \
 && git -C /opt/model remote add origin "${SRC_REPO}" \
 && git -C /opt/model fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/model checkout -q FETCH_HEAD \
 && echo "giga-brain-0 @ $(git -C /opt/model rev-parse HEAD)"

# ---------------------------------------------------------------------------------------------
# 2. dependencies (giga-models comes from a pinned git URL inside the lock file)
# ---------------------------------------------------------------------------------------------
# evdev (a lerobot hardware dependency, pulled in transitively) publishes no wheel and needs a C
# compiler. It is not on the inference path, but it is in the pinned set, so give it a toolchain
# rather than silently dropping a pin.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && apt-get clean

COPY docker/vla_bench/models/gigabrain07/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-deps "torchcodec==0.5" \
 || echo "torchcodec unavailable — inference does not decode video, continuing"
RUN python -c "import giga_models, transformers; print('giga_models ok | transformers', transformers.__version__)"

COPY packages/openpi-client /opt/openpi-client
COPY docker/vla_bench/server /app

# The tokenizer paths are HF *snapshot directories*, not repo ids — the author's get_policy() takes
# them as arguments and they override the absolute host paths baked into the checkpoint's
# inference_config.json. They resolve under the /hf mount. norm_stats_path is a run artefact and has
# to be placed under the weights mount; see README.md.
ENV PYTHONPATH=/opt/model \
    VLA_BENCH_MODEL=gigabrain07 \
    VLA_BENCH_ADAPTER=gigabrain07:GigaBrain07Adapter \
    VLA_BENCH_ADAPTER_KWARGS='{"repo_dir":"/opt/model","model_dir":"/opt/model","tokenizer_path":"/hf/hub/models--google--paligemma2-3b-pt-224/snapshots/96eeb174da13ca1a2b247e4d0867436296c36420","fast_tokenizer_path":"/hf/hub/models--physical-intelligence--fast/snapshots/ec4d7aa71691cac0b8bed6942be45684db2110f4","norm_stats_path":"/models/gigabrain07/norm_stats_right7_emb8.json","embodiment_id":8,"robot_type":"trossen_ai_mobile","original_action_dim":7,"chunk_len":30,"exec_len":30,"dtype":"bfloat16","device":"cuda:0","seed":1000}' \
    VLA_BENCH_CHECKPOINT=/models/gigabrain07/model_ema \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    HF_HOME=/hf

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
