# VLA-bench model image — LingBot-VLA 2.0 (Qwen3-VL-4B + 32-expert MoE + flow-matching action expert)
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/lingbot_v2/Dockerfile -t vla-bench-lingbot_v2:latest .
ARG BASE=vla-bench-base:torch28-cu128
FROM ${BASE}

# ---------------------------------------------------------------------------------------------
# 1. model source, pinned. The package IS pip-installed (`lingbotvla`, editable, --no-deps) because
#    the authors' deploy module imports it by name, and the two vendored depth packages
#    (`lingbot-depth` -> mdm, `MoGe`) are installed the same way, exactly as the authors'
#    tools/create_train_env.sh does. NO SOURCE PATCHES: the validated host checkout at this SHA has
#    no local modifications -- only the benchmark's two config assets, added as files in step 4.
# ---------------------------------------------------------------------------------------------
ARG SRC_REPO=https://github.com/Robbyant/lingbot-vla-v2.git
ARG SRC_SHA=bc643d74a0127fab8788da993b261d4d64101138
RUN git init -q /opt/src \
 && git -C /opt/src remote add origin "${SRC_REPO}" \
 && git -C /opt/src fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/src checkout -q FETCH_HEAD \
 && echo "lingbot-vla-v2 @ $(git -C /opt/src rev-parse HEAD)"

# ---------------------------------------------------------------------------------------------
# 2. dependencies, measured from the validated venv $WS/models/lingbot_v2/env.
#    flash-attn comes from its GitHub release wheel (cu12/torch2.8/cxx11abiTRUE/cp312) rather than
#    being compiled: the ViT runs flash_attention_2 (`vit_attn_implementation`), so it is required,
#    and building 2.8.3 from source takes hours.
# ---------------------------------------------------------------------------------------------
# crcmod 1.7 (via oss2) is sdist-only and needs a C compiler; give it a toolchain rather than dropping a pin.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && apt-get clean

COPY docker/vla_bench/models/lingbot_v2/requirements.lock.txt /tmp/requirements.lock.txt

# --no-deps FOR THE WHOLE LOCK, and that is deliberate. The lock is a complete `uv pip freeze` of the
# validated venv, so it is already a closed set -- but that venv contains FOUR pin pairs pip's
# resolver refuses to install in one pass, every one of them inherited from the authors' own
# tools/create_train_env.sh (which installs several packages `--no-deps` and then re-applies
# requirements.txt to restore the pinned core stack):
#
#     databricks-sdk 0.139.0  wants protobuf >=4.25.8   <->  requirements.txt pins protobuf 3.20.3
#     numpydantic 1.9.0       wants numpy >=2.0.0        <->  requirements.txt pins numpy 1.26.4
#     plyfile 1.1.4           wants numpy >=2.0          <->  the same numpy pin
#     mlflow 2.22.5           wants pyarrow <20          <->  requirements.txt pins pyarrow 21.0.0
#
# `pip check` in the venv reports exactly these four (plus LeRobot's and MoGe's deliberately absent
# optional deps), and ENV.md 1.1 records them. Letting pip resolve them would build a DIFFERENT
# environment from the one every measured number in this benchmark came from, so the freeze is
# installed verbatim instead. The import check and `--probe` are what prove the result actually works.
RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-deps "torchcodec==0.6.0"

# ---------------------------------------------------------------------------------------------
# 3. the three source installs, in the authors' own order, all --no-deps
# ---------------------------------------------------------------------------------------------
RUN pip install --no-cache-dir --no-deps -e /opt/src \
 && pip install --no-cache-dir --no-deps -e /opt/src/lingbotvla/models/vla/vision_models/lingbot-depth \
 && pip install --no-cache-dir --no-deps -e /opt/src/lingbotvla/models/vla/vision_models/MoGe \
 && python -c "import lingbotvla, mdm, moge, utils3d, flash_attn, transformers, numpy, lerobot; \
print('lingbotvla ok | flash_attn', flash_attn.__version__, '| transformers', transformers.__version__, \
      '| numpy', numpy.__version__)"

# The authors' env script ends with a non-fatal `pip check` for the same reason: the inconsistencies
# above are expected and are recorded here in the build log rather than hidden.
RUN pip check || echo "[expected] see the --no-deps comment above"

# ---------------------------------------------------------------------------------------------
# 4. the benchmark's embodiment mapping and its normalisation statistics. These are the only two
#    non-upstream files in the image and neither is a weight: 2 KB of YAML that maps our 7-D right
#    arm onto LingBot's 55-D canonical slots (and fixes the camera ORDER), and 4 KB of meanstd
#    statistics over the 139,662 training frames. `FeatureTransform` resolves both by
#    **cwd-relative** path, which is why the adapter chdir()s into /opt/src.
#    A different dataset needs a different norm_stats file; see README.md.
# ---------------------------------------------------------------------------------------------
COPY docker/vla_bench/models/lingbot_v2/assets/right7_2view.robot_config.yaml \
     /opt/src/configs/robot_configs/right7_2view.yaml
COPY docker/vla_bench/models/lingbot_v2/assets/right7_2view.norm_stats.json \
     /opt/src/assets/norm_stats/right7_2view.json

COPY packages/openpi-client /opt/openpi-client
COPY docker/vla_bench/server /app

# qwen3vl_path overrides the absolute host path the training config records for the Qwen3-VL
# config/tokenizer/processor (no VLM weights live there -- they come from the checkpoint). It is an
# HF *snapshot directory* under the /hf mount, revision-pinned, so nothing is resolved over the
# network. VLA_BENCH_CHECKPOINT points at the run's hf_ckpt; the authors' loader reads the training
# config from <ckpt>/../../../lingbotvla_cli.yaml, so the run's directory shape must be preserved
# under the weights mount.
ENV VLA_BENCH_MODEL=lingbot_v2 \
    VLA_BENCH_ADAPTER=lingbot_v2:LingBotV2Adapter \
    VLA_BENCH_ADAPTER_KWARGS='{"repo_dir":"/opt/src","robo_name":"right7_2view","chunk_len":30,"exec_len":30,"dtype":"bfloat16","use_compile":false,"device":"cuda:0","seed":1000,"qwen3vl_path":"/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"}' \
    VLA_BENCH_CHECKPOINT=/models/lingbot_v2/eval_ckpt/step16000/checkpoints/global_step_16000/hf_ckpt \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    HF_HOME=/hf

EXPOSE 8800
# 23.75 GiB of float32 safetensors off a network filesystem: the start period is deliberately long.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
