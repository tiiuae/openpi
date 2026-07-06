from collections.abc import Mapping
import logging
import os
import random
import time
from typing import Any

from filelock import FileLock
import numpy as np
from PIL import Image as _PILImage
import torch
from transformers import AutoConfig
from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.falconvla_config import FalconVLAConfig
from openpi.models.utils.prompt_builder import ActionPromptBuilder
from openpi.models.utils.proprio import fetch_proprio_stats
from openpi.models.utils.proprio import normalize_proprio
from openpi.models.utils.proprio import tokenize_proprio
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")


def _resolve_model_path(model_name: str) -> str:
    """If model_name doesn't contain processor files but has a single subdirectory that does, use that instead."""
    processor_files = {"processor_config.json", "preprocessor_config.json", "tokenizer_config.json"}
    dir_files = set(os.listdir(model_name)) if os.path.isdir(model_name) else set()
    if not processor_files.intersection(dir_files):
        subdirs = [d for d in dir_files if os.path.isdir(os.path.join(model_name, d))]
        if len(subdirs) == 1:
            candidate = os.path.join(model_name, subdirs[0])
            candidate_files = set(os.listdir(candidate))
            if processor_files.intersection(candidate_files):
                return candidate
    return model_name


def _hf_modules_lock_path() -> str:
    modules_root = os.getenv("HF_MODULES_CACHE") or os.path.join(
        os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "modules"
    )
    os.makedirs(modules_root, exist_ok=True)
    return os.path.join(modules_root, ".falcon_load.lock")


def _action_head_dict(action_head_configs: Any) -> dict | None:
    """Return a mutable dict view of `config.action_head_configs` (dict, PretrainedConfig, or dataclass)."""
    if action_head_configs is None:
        return None
    if isinstance(action_head_configs, Mapping):
        return dict(action_head_configs)
    if hasattr(action_head_configs, "to_dict"):
        try:
            return dict(action_head_configs.to_dict())
        except Exception:
            pass
    if hasattr(action_head_configs, "__dict__"):
        return {k: v for k, v in vars(action_head_configs).items() if not k.startswith("_")}
    return None


def _patch_action_head_token_size(config: Any) -> None:
    """Self-heal `config.action_head_configs.token_size` from `text_config.hidden_size`.

    Older FalconVLA checkpoints saved `action_head_configs` without a `token_size` field; the
    action-head config then falls back to a default that doesn't match the trained DiT
    projection's in-features, producing a `size mismatch` error while loading weights. Mirrors
    the canonical loader's fix.
    """
    ahc_dict = _action_head_dict(getattr(config, "action_head_configs", None))
    if ahc_dict is None:
        return

    existing = ahc_dict.get("token_size")
    text_cfg = getattr(config, "text_config", None)
    llm_hidden = getattr(text_cfg, "hidden_size", None) if text_cfg is not None else None
    if llm_hidden is None:
        return

    if existing in (None, 0):
        ahc_dict["token_size"] = int(llm_hidden)
        config.action_head_configs = ahc_dict
        logger.info(
            "FalconVLA: injected action_head_configs.token_size=%d (checkpoint config.json was missing it)",
            llm_hidden,
        )
    elif int(existing) != int(llm_hidden):
        logger.warning(
            "FalconVLA: action_head_configs.token_size=%s disagrees with text_config.hidden_size=%s; "
            "using the stored value (set by training).",
            existing,
            llm_hidden,
        )


def load_falcon_model(
    model_name: str, device: str, hf_token: str | None
) -> tuple[AutoProcessor, AutoModelForVision2Seq]:
    model_name = _resolve_model_path(model_name)

    # Pre-load + self-heal the config before `from_pretrained` instantiates the model, guarded by
    # a file lock: concurrent first-time loads of the same checkpoint race on writing the HF
    # dynamic-module cache, which raises a transient AttributeError.
    lock_path = _hf_modules_lock_path()
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with FileLock(lock_path, timeout=600):
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
            break
        except AttributeError:
            if attempt == max_retries:
                raise
            wait = 2**attempt
            logger.warning(
                "FalconVLA: dynamic-module race on attempt %d/%d; retrying in %ds", attempt, max_retries, wait
            )
            time.sleep(wait)
    _patch_action_head_token_size(config)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=hf_token,
        low_cpu_mem_usage=True,
    ).to(device)

    # Patch config to ensure required token IDs are available
    # The HuggingFace model's forward pass expects these attributes to exist

    # Try to get image_token_id from processor's tokenizer
    image_token_id = 32000  # Default value
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        if hasattr(processor.tokenizer, "image_token_id"):
            image_token_id = processor.tokenizer.image_token_id
        elif hasattr(processor.tokenizer, "convert_tokens_to_ids"):
            # Try to find the image token by string
            for token_str in ["|<image>|", "<image>", "[IMG]"]:
                try:
                    token_id = processor.tokenizer.convert_tokens_to_ids(token_str)
                    if token_id and token_id not in [processor.tokenizer.unk_token_id, 0]:
                        image_token_id = token_id
                        break
                except Exception:
                    pass

    # Set the token IDs on the main config
    if not hasattr(model.config, "image_token_id"):
        model.config.image_token_id = image_token_id

    # Set video and vision_start token IDs with reasonable defaults
    if not hasattr(model.config, "video_token_id"):
        model.config.video_token_id = image_token_id + 1
    if not hasattr(model.config, "vision_start_token_id"):
        model.config.vision_start_token_id = image_token_id + 2

    # Also ensure vision_config has image_token_id in case the model code accesses it there
    if hasattr(model.config, "vision_config") and not hasattr(model.config.vision_config, "image_token_id"):
        model.config.vision_config.image_token_id = image_token_id

    # Inference-only: put the HF model in eval mode. `from_pretrained` returns it in train mode,
    # so without this dropout stays active at inference -> nondeterministic + degraded actions.
    # (The canonical FalconVLA server calls model.eval() for the same reason.)
    model.eval()

    return processor, model


def _to_pil(arr: np.ndarray) -> _PILImage.Image:
    """Convert a (CHW or HWC) uint8 image array to a PIL image.

    Mirrors the canonical FalconVLA server's `_chw_to_pil`: transpose CHW->HWC, drop a singleton
    channel, wrap in PIL. Color order is left as-is (RGB expected). Feeding raw CHW arrays to the
    HF processor lets it mis-read the channel axis, so the conversion must happen here.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return _PILImage.fromarray(np.ascontiguousarray(arr).astype(np.uint8))


def _seed_rngs(seed: int) -> None:
    """Seed the Python / numpy / torch RNGs the FM-DiT action head draws its noise from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pin_determinism(seed: int, *, strict: bool = True) -> None:
    """Pin RNGs so FalconVLA's flow-matching (FM-DiT) action head is reproducible.

    The action head integrates an ODE from random initial noise, so without a fixed seed each
    inference draws a different trajectory. Mirrors the canonical server's `pin_global_seed`.
    `strict` additionally forces deterministic cuDNN kernels and disables TF32 fast paths.
    """
    _seed_rngs(seed)
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(mode=True, warn_only=True)
        except (AttributeError, RuntimeError) as exc:
            logger.warning("FalconVLA: use_deterministic_algorithms failed (%s); continuing", exc)


def _warn_if_action_shape_mismatch(model_config: Any, action_dim: int, action_horizon: int) -> None:
    """Warn when the TrainConfig's action_dim/action_horizon disagree with the checkpoint's own config.json.

    FalconVLA checkpoints persist their training-time `action_dim` / `time_horizon` on
    `config.json`. openpi's TrainConfig sets these independently (they drive JAX shape specs
    before the checkpoint is loaded), so a typo'd config can silently produce garbled or
    truncated actions. Mirrors the canonical server, which sources these values from the
    checkpoint directly; here we can only warn, since openpi's shapes are fixed earlier.
    """
    ckpt_action_dim = getattr(model_config, "action_dim", None)
    ckpt_action_horizon = getattr(model_config, "time_horizon", None)
    if ckpt_action_dim is not None and int(ckpt_action_dim) != int(action_dim):
        logger.warning(
            "FalconVLA: configured action_dim=%d disagrees with checkpoint config.json action_dim=%s",
            action_dim,
            ckpt_action_dim,
        )
    if ckpt_action_horizon is not None and int(ckpt_action_horizon) != int(action_horizon):
        logger.warning(
            "FalconVLA: configured action_horizon=%d disagrees with checkpoint config.json time_horizon=%s",
            action_horizon,
            ckpt_action_horizon,
        )


def _coerce_actions(actions: np.ndarray, *, action_horizon: int, action_dim: int) -> np.ndarray:
    """Reshape/trim a raw model action output to the (action_horizon, action_dim) contract.

    Mirrors the canonical server's `_coerce_actions`: FalconVLA's `predict_action` can return a
    flat (H*A,) vector, a (1, H, A) batch-of-one, or (H, D) with D > action_dim when the
    checkpoint's action head emits extra columns (e.g. padding). Handle all three here so callers
    always see (action_horizon, action_dim).
    """
    if actions.ndim == 1:
        actions = actions.reshape(action_horizon, -1)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.shape[-1] > action_dim:
        actions = actions[..., :action_dim]
    return actions


class FalconVLA(_model.BaseModel):
    def __init__(self, config: FalconVLAConfig, checkpoint_dir: str = ""):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # Pin RNGs before loading so the flow-matching action head is reproducible.
        _pin_determinism(config.seed, strict=config.strict_determinism)

        # Load hf token from environment variable if available
        hf_token = os.getenv("HF_TOKEN", None)
        print("HF_TOKEN found in environment." if hf_token else "No HF_TOKEN found in environment.")

        # Initialize model parameters or load pretrained weights here
        self.processor, self.model = load_falcon_model(checkpoint_dir, config.device, hf_token)
        _warn_if_action_shape_mismatch(self.model.config, config.action_dim, config.action_horizon)

    @torch.no_grad()
    def inference(self, observation: _model.Observation, task_label: str | None = None) -> _model.Actions:
        """Run inference on observation.

        Args:
            observation: Model observation containing images, state, and optional prompt
            task_label: Optional task description. If None, will use prompt from observation

        Returns:
            Predicted actions
        """
        # Deterministic mode: reseed before each call so identical observations produce identical
        # actions. When disabled, the flow-matching head keeps advancing its RNG (stochastic).
        if self.config.deterministic_inference:
            _seed_rngs(self.config.seed)

        # The task label comes directly from the observation prompt.
        input_builder = ActionPromptBuilder(task_label=observation["prompt"])

        # Getting the images from the observation
        # Observation.images should be a dict with standard keys
        images = observation.get("images", None)

        # Map from standard image keys to camera names
        # base_0_rgb -> primary/main image (cam_high)
        # right_wrist_0_rgb -> wrist image (cam_right_wrist)
        # left_wrist_0_rgb -> secondary image (cam_left_wrist)

        primary_image = images.get("cam_high", None)
        if primary_image is None:
            raise ValueError("Missing required 'cam_high' image in observation")

        # Convert CHW uint8 -> HWC PIL before the processor (matches the canonical FalconVLA
        # server); feeding raw CHW lets the processor mis-read the channel axis.
        input_builder.add_main_image(image=_to_pil(primary_image))

        if self.config.use_wrist:
            wrist_image = images.get("cam_right_wrist", None)
            if wrist_image is not None:
                input_builder.add_wrist_image(_to_pil(wrist_image))

        if self.config.use_secondary:
            secondary_image = images.get("cam_left_wrist", None)
            if secondary_image is not None:
                input_builder.add_secondary_image(_to_pil(secondary_image))

        proprio_inputs = None
        if self.config.use_proprio:
            # Normalize the raw state using the checkpoint's own proprio stats; how it then
            # reaches the model depends on `proprio_mode` (see FalconVLAConfig).
            proprio_stats = fetch_proprio_stats(self.model, self.config.unnorm_key)

            raw_state = np.asarray(observation.get("state", None), dtype=np.float64).ravel()
            _q01 = np.asarray(proprio_stats["q01"], dtype=np.float64)
            _q99 = np.asarray(proprio_stats["q99"], dtype=np.float64)
            _mean = np.asarray(proprio_stats["mean"], dtype=np.float64)
            _span = _q99 - _q01

            # Pin "frozen" proprio channels (those with ~zero training span — e.g. the
            # static arm in a single-arm dataset) to the training mean before normalizing.
            # At deploy the physical arm sits at an arbitrary pose; normalizing that live
            # value on a near-degenerate [q01,q99] range saturates it to ±1 — a value the
            # model never saw in training, which corrupts the conditioning and makes the
            # policy emit a near-constant/jittery trajectory. Replacing those channels with
            # the training mean keeps the tokenized input in-distribution.
            frozen_span_thresh = 1e-3
            n = min(raw_state.shape[0], _span.shape[0])
            frozen_mask = np.zeros(raw_state.shape[0], dtype=bool)
            frozen_mask[:n] = _span[:n] < frozen_span_thresh
            pinned_state = raw_state.copy()
            pinned_state[:n][frozen_mask[:n]] = _mean[:n][frozen_mask[:n]]

            normalized_proprio = normalize_proprio(pinned_state, proprio_stats)

            # Debug: per-channel flags — PIN = frozen channel pinned to mean,
            # OOD = live value outside training bounds.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{'i':>2} {'raw':>12} {'pinned':>12} {'q01':>12} {'q99':>12} {'span':>10}  flag")
                for i in range(len(raw_state)):
                    ood = i < n and ((raw_state[i] < _q01[i] - 1e-6) or (raw_state[i] > _q99[i] + 1e-6))
                    flag = "PIN" if frozen_mask[i] else ("OOD" if ood else "")
                    _q01_i, _q99_i, _span_i = (_q01[i], _q99[i], _span[i]) if i < n else (float("nan"),) * 3
                    logger.debug(
                        f"{i:>2} {raw_state[i]:>12.5f} {pinned_state[i]:>12.5f} {_q01_i:>12.5f} {_q99_i:>12.5f} {_span_i:>10.5f}  {flag}"
                    )

            if self.config.proprio_mode == "tokens":
                # Discrete-token path: inject `<prop*>` text tokens into the prompt (the FM-DiT
                # `-p` checkpoints, e.g. aidrc_cups_manipulation_14 dualarm).
                proprio_tokens = tokenize_proprio(normalized_proprio, self.config.num_bins)
                input_builder.add_proprio("".join(proprio_tokens))
            else:
                # FiLM path: pass a continuous tensor as `proprio_inputs=` to `predict_action`
                # (the ResNet `-p-film` checkpoints).
                proprio_inputs = torch.as_tensor(normalized_proprio, device=self.config.device).float()
                # -> (B, T, proprio_dim); single-frame history => T=1
                if proprio_inputs.dim() == 1:
                    proprio_inputs = proprio_inputs.unsqueeze(0).unsqueeze(0)
                elif proprio_inputs.dim() == 2:
                    proprio_inputs = proprio_inputs.unsqueeze(1)

        inputs, _ = input_builder.build_inputs(processor=self.processor)
        inputs.pop("token_type_ids", None)
        inputs = {k: (v.to(self.config.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        predict_action_kwargs = {}
        if proprio_inputs is not None:
            predict_action_kwargs["proprio_inputs"] = proprio_inputs

        raw_actions = self.model.predict_action(
            inputs,
            unnorm_key=self.config.unnorm_key,
            horizon=self.config.action_horizon,
            do_sample=False,
            **predict_action_kwargs,
        )
        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.detach().cpu().float().numpy()
        return _coerce_actions(
            np.asarray(raw_actions), action_horizon=self.config.action_horizon, action_dim=self.config.action_dim
        )

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        raise NotImplementedError("FalconVLA is inference-only in openpi; training is not supported.")

    @override
    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation, **kwargs) -> _model.Actions:
        # FalconVLA runs synchronous PyTorch inference; delegate to `inference`.
        task_label = kwargs.get("task_label")
        return self.inference(observation, task_label=task_label)
