# VLA-bench model image — XR-1 (Xiaomi-Robotics-1-5B), chunk 30, 7-D right arm
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/xr1/Dockerfile -t vla-bench-xr1:latest .
ARG BASE=vla-bench-base:torch28-cu128
FROM ${BASE}

# ---------------------------------------------------------------------------------------------
# 1. model source. The installable package is the `xr1` SUB-directory of the repo (distribution
#    xiaomi-robotics-1, import name `mibot`), not the repo root.
# ---------------------------------------------------------------------------------------------
ARG SRC_REPO=https://github.com/XiaomiRobotics/Xiaomi-Robotics-1
ARG SRC_SHA=0dd7aef8dc87296246aae812a1f59ccb708e5546
RUN git init -q /opt/src \
 && git -C /opt/src remote add origin "${SRC_REPO}" \
 && git -C /opt/src fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/src checkout -q FETCH_HEAD \
 && echo "Xiaomi-Robotics-1 @ $(git -C /opt/src rev-parse HEAD)"

# 2. PROVENANCE ONLY. `xr1/assets/config.py` is where mibot.utils.cfg_utils dumps the fully resolved
#    Hydra config of a training run (mibot/utils/cfg_utils.py:59); upstream ships the authors' copy and
#    this replaces it with OURS, so the image records the run that produced the checkpoint. Nothing on
#    the inference path reads it — the adapter builds the model from explicit kwargs and takes the 60-D
#    mean/std/q01/q99 from the mounted `normalize.json` — so it is documentation, not a dependency.
#    Note the path: the installable package is the repo's `xr1/` SUB-directory, so assets/ sits at
#    /opt/src/xr1/assets, not /opt/src/xr1/xr1/assets.
COPY docker/vla_bench/models/xr1/assets_config.py /opt/src/xr1/assets/config.py

COPY docker/vla_bench/models/xr1/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-deps -e /opt/src/xr1 \
 && python -c "import mibot; print('mibot ok')"

# 3. the packed-60 -> absolute-7 inverse. The adapter imports unpack_action from this module; it is
#    benchmark code, not upstream, and it is what turns XR-1's packed action vector back into the
#    7-D absolute joint targets the robot client expects.
COPY docker/vla_bench/models/xr1/export_xr1_dataset.py /opt/xr1_tools/export_xr1_dataset.py

COPY packages/openpi-client /opt/openpi-client
COPY docker/vla_bench/server /app

ENV VLA_BENCH_MODEL=xr1 \
    VLA_BENCH_ADAPTER=xr1_policy:XR1Adapter \
    VLA_BENCH_ADAPTER_KWARGS='{"repo_dir":"/opt/src/xr1","model_dir":"/opt/xr1_tools","data_dir":"/models/xr1/data","device":"cuda:0","chunk_len":30,"exec_len":30,"seed":1000}' \
    VLA_BENCH_CHECKPOINT=/models/xr1/mp_rank_00_model_states.pt \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    HF_HOME=/hf

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
