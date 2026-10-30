# VLA-bench model image — OpenWAM (world-action model: Wan2.2-TI2V-5B-shaped causal video expert + 2 B action
# expert, dual-expert MoT, program `joint`). Result label: "OpenWAM recipe from video pretraining".
#
# Build context is the OPENPI REPO ROOT:
#   docker build -f docker/vla_bench/models/openwam/Dockerfile -t vla-bench-openwam:latest .
#
# Code only. OpenWAM is AGPL-3.0: its source is NOT in this repository; the build clones it at the pinned commit.
# The upload set (model_state.pt, resolved_config.yaml, the dataset's action quantiles) is mounted at /models/openwam;
# the frozen Wan2.2 VAE and the UMT5-xxl text encoder + tokenizer come from the Hugging Face cache mounted at /hf
# (see README.md). Predicts 32 steps, returns the first 30.
ARG BASE=vla-bench-base:torch211-cu130
FROM ${BASE}

# 0. the base must carry exactly the torch build the validated venv ran (torch 2.11.0+cu130, Python 3.12).
RUN python -c "import sys, torch; \
assert torch.__version__ == '2.11.0+cu130', torch.__version__; \
assert sys.version_info[:2] == (3, 12), sys.version; print('base ok', torch.__version__, sys.version.split()[0])"

# 1. model source, pinned, unpatched (the benchmark ran the checkout with `git status` clean).
ARG SRC_REPO=https://github.com/OpenWAM/OpenWAM
ARG SRC_SHA=4b3814a82268f3523df5fcdadcbeb2c021ac7737
RUN git init -q /opt/model \
 && git -C /opt/model remote add origin "${SRC_REPO}" \
 && git -C /opt/model fetch -q --depth 1 origin "${SRC_SHA}" \
 && git -C /opt/model checkout -q FETCH_HEAD \
 && test "$(git -C /opt/model rev-parse HEAD)" = "${SRC_SHA}" \
 && echo "OpenWAM @ $(git -C /opt/model rev-parse HEAD)"

# 2. dependencies, pinned to the venv the reference numbers were measured in (requirements.lock.txt says what is
#    left out and why), then OpenWAM itself, editable from the pinned clone, --no-deps.
COPY docker/vla_bench/models/openwam/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --no-deps -e /opt/model \
 && python -c "import open_wam, torch, transformers, diffusers; \
from open_wam.evals.libero_policy_runtime import load_libero_policy_runtime; \
from open_wam.data.preparation.frames import fit_frames; \
print('open_wam', open_wam.__file__, '| transformers', transformers.__version__, '| diffusers', diffusers.__version__)"

# 3. openpi's own wire codec, so the msgpack bytes match production exactly
COPY packages/openpi-client /opt/openpi-client

# 4. the shared server and the adapters
COPY docker/vla_bench/server /app

# 5. runtime.
#  * checkpoint=/models/openwam: the upload set's model_state.pt (every tensor, fp32 masters) + resolved_config.yaml (ships
#    as trained; its training-host paths for the base assets are overridden by the adapter, nothing else is read).
#  * vae_dir / text_encoder_dir / tokenizer_dir: the revisions training loaded, from /hf through VLA_BENCH_HUB_PINS
#    (the adapter checks file sizes and the small files' sha256 before loading).
#  * norm_stats: the dataset's action q01/q99 in the upload set; the adapter refuses a checkpoint whose
#    de-normalisation quantiles differ from it.
#  * text_cache: the UMT5-xxl embeddings of the four training instructions and the blank prompt (upload set), made with
#    OpenWAM's own encode_text on the CPU in bf16 as training made them; other prompts are encoded online (~150 s each).
#  * the Wan2.2 VAE in fp32 on the GPU (the training latents were encoded in fp32), latents rounded through fp16 as
#    they were stored for training; the model's parameters in bf16, the training forward's precision; seed 1000 at
#    every request (each request is OpenWAM's episode-start transaction; nothing is carried between requests).
#  * OMP_NUM_THREADS=8, as the benchmark's host runs used (the flexpi, flux3_action and pi05 images set it too).
ENV VLA_BENCH_MODEL=openwam \
    VLA_BENCH_ADAPTER=openwam_policy:OpenWAMAdapter \
    VLA_BENCH_ADAPTER_KWARGS_DEFAULTS='{"vae_dir":"hf:Wan-AI/Wan2.2-TI2V-5B-Diffusers@b8fff7315c768468a5333511427288870b2e9635/vae","text_encoder_dir":"hf:robbyant/lingbot-va-base@68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c/text_encoder","tokenizer_dir":"hf:robbyant/lingbot-va-base@68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c/tokenizer","norm_stats":"/models/openwam/openwam_norm_stats.json","text_cache":"/models/openwam/text_embeds_cache/right7_2view_umt5_cpu_bf16.pt","device":"cuda:0","vae_dtype":"float32","latent_storage_dtype":"float16","param_dtype":"bfloat16","chunk_len":30,"exec_len":30,"seed":1000}' \
    VLA_BENCH_CHECKPOINT=/models/openwam \
    VLA_BENCH_HUB_PINS='{"Wan-AI/Wan2.2-TI2V-5B-Diffusers":{"revision":"b8fff7315c768468a5333511427288870b2e9635","allow_patterns":["vae/config.json","vae/diffusion_pytorch_model.safetensors"]},"robbyant/lingbot-va-base":{"revision":"68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c","allow_patterns":["text_encoder/*","tokenizer/*"]}}' \
    VLA_BENCH_PROBE_TASK="pick up the metal pot and place it in the basket" \
    VLA_BENCH_PORT=8800 \
    VLA_BENCH_PAD_TO=14 \
    OMP_NUM_THREADS=8 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/hf \
    HF_HUB_OFFLINE=1

EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s \
    CMD curl -fsS http://localhost:8800/healthz || exit 1
ENTRYPOINT ["python", "/app/vla_server.py"]
