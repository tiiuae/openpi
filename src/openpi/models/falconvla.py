from typing import Optional, Tuple

from typing_extensions import override

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from openpi.models import model as _model
from openpi.models.falconvla_config import FalconVLAConfig
from openpi.models.utils.prompt_builder import ActionPromptBuilder
from openpi.models.utils.proprio import *
import openpi.shared.array_typing as at
import inspect
import os
import numpy as np


_DEBUG_PROPRIO = bool(os.getenv("FALCONVLA_DEBUG_PROPRIO"))


def _normalizes_internally(model) -> bool:
    """Whether the checkpoint normalizes proprio itself inside `predict_action`.

    FM-DiT (`proprio_mode="to_head"`) does, so it must be handed the RAW robot state; the ResNet
    heads have no such step and expect an already-normalized vector.
    """
    return hasattr(model, "_normalize_proprio")


def _proprio_is_kwarg(model) -> bool:
    """Whether `predict_action` declares `proprio_inputs`.

    ResNet takes it as a keyword argument; FM-DiT reads it out of the `inputs` dict and would
    silently swallow the keyword into `**kwargs`.
    """
    try:
        return "proprio_inputs" in inspect.signature(model.predict_action).parameters
    except (TypeError, ValueError):
        return False


def _text_token_proprio(model) -> bool:
    """Whether the checkpoint takes proprio as discretized tokens spliced into the VLM prefix."""
    return str(getattr(model.config, "proprio_mode", "")).lower() == "text_token"


def _proprio_token_ids(proprio_norm: np.ndarray, config) -> np.ndarray:
    """Discretize normalized proprio into the top-of-vocab token block, as training did.

    Training binned each scalar over `num_bins` edges spanning [bin_min, bin_max] and mapped bin
    `b` to `base + b`, ascending with the value. The bundled `predict_action` instead maps to
    `proprio_token_id_max - b`, which mirrors the encoding, so openpi builds the ids itself.
    """
    num_bins = int(getattr(config, "proprio_num_bins", 256))
    lo = float(getattr(config, "proprio_bin_min", -1.0))
    hi = float(getattr(config, "proprio_bin_max", 1.0))
    base = int(getattr(config, "proprio_token_id_max", 131071)) - num_bins + 1

    edges = np.linspace(lo, hi, num_bins)
    bins = np.digitize(np.clip(np.asarray(proprio_norm, dtype=np.float64).ravel(), lo, hi), edges) - 1
    return base + np.clip(bins, 0, num_bins - 1).astype(np.int64)


def _splice_before_thinking(inputs: dict, token_ids: np.ndarray, thinking_token_id: int) -> None:
    """Insert `token_ids` immediately before the first `<thinking>` token, in place.

    Training anchored the proprio tokens here. The bundled `predict_action` anchors on the last
    image patch instead, which lands them inside the final image block and displaces its
    `|<end_of_img>|` terminator.
    """
    ids = inputs["input_ids"]
    if ids.dim() != 2 or ids.shape[0] != 1:
        raise ValueError(f"expected a single-sample input_ids, got shape {tuple(ids.shape)}")
    found = (ids[0] == thinking_token_id).nonzero()
    if found.numel() == 0:
        raise ValueError("prompt contains no <thinking> token to anchor the proprio tokens to")
    at = int(found[0])

    extra = torch.as_tensor(token_ids, device=ids.device, dtype=ids.dtype).reshape(1, -1)
    inputs["input_ids"] = torch.cat([ids[:, :at], extra, ids[:, at:]], dim=1)
    mask = inputs.get("attention_mask")
    if mask is not None:
        ones = torch.ones((1, extra.shape[1]), device=mask.device, dtype=mask.dtype)
        inputs["attention_mask"] = torch.cat([mask[:, :at], ones, mask[:, at:]], dim=1)


def _resolve_model_path(model_name: str) -> str:
    """If model_name doesn't contain processor files but has a single subdirectory that does, use that instead."""
    import os
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


def load_falcon_model(model_name: str, device: str, hf_token: Optional[str]) -> Tuple[AutoProcessor, AutoModelForVision2Seq]:
    model_name = _resolve_model_path(model_name)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, token=hf_token, low_cpu_mem_usage=True
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
                except:
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
    if hasattr(model.config, "vision_config"):
        if not hasattr(model.config.vision_config, "image_token_id"):
            model.config.vision_config.image_token_id = image_token_id

    return processor, model


def _arm_offset(arm: str, robot_action_dim: int) -> int:
    """Index of the first channel belonging to `arm` in the robot's bimanual vector."""
    if arm not in ("both", "left", "right"):
        raise ValueError(f"arm must be 'both', 'left' or 'right', got {arm!r}")
    return robot_action_dim // 2 if arm == "right" else 0


def _fit(vec: np.ndarray, dim: int) -> np.ndarray:
    """Truncate or zero-extend `vec` to `dim` channels."""
    vec = np.asarray(vec, dtype=np.float64).ravel()
    if vec.shape[0] >= dim:
        return vec[:dim]
    return np.concatenate([vec, np.zeros(dim - vec.shape[0])])


def _build_proprio(
    state: np.ndarray,
    stats: dict,
    *,
    proprio_dim: int | None,
    offset: int,
    static_indices: tuple[int, ...] | None,
) -> np.ndarray:
    """Assemble the checkpoint's proprio vector from the live robot state.

    Starts from the training mean, so any channel the robot cannot supply -- dims beyond the
    state vector, or an arm that was static during collection -- keeps the constant value the
    model saw in training rather than a saturated out-of-range one.
    """
    mean = np.asarray(stats["mean"], dtype=np.float64)
    dim = mean.shape[0]
    if proprio_dim is not None and proprio_dim != dim:
        raise ValueError(f"config proprio_dim={proprio_dim} but checkpoint stats are {dim}-d")

    out = mean.copy()
    live = min(dim, max(state.shape[0] - offset, 0))
    out[:live] = state[offset : offset + live]
    pinned = [i for i in (static_indices or ()) if 0 <= i < dim]
    out[pinned] = mean[pinned]
    return out


def _to_robot_actions(
    actions: np.ndarray,
    *,
    action_horizon: int,
    action_dim: int,
    offset: int,
    robot_action_dim: int,
    hold: np.ndarray,
) -> np.ndarray:
    """Reshape the model output to (action_horizon, robot_action_dim).

    Heads wider than the robot (16-d with extra end-effector columns) are truncated; narrower
    single-arm heads are written at `offset` while the other arm holds its current pose.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(action_horizon, -1)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    actions = actions[:, :action_dim]

    if actions.shape[1] >= robot_action_dim:
        return actions[:, :robot_action_dim]
    out = np.tile(_fit(hold, robot_action_dim), (actions.shape[0], 1))
    out[:, offset : offset + actions.shape[1]] = actions
    return out


class FalconVLA(_model.BaseModel):
    def __init__(self, config: FalconVLAConfig, checkpoint_dir: str = ""):
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # Load hf token from environment variable if available
        hf_token = os.getenv("HF_TOKEN", None)
        print("HF_TOKEN found in environment." if hf_token else "No HF_TOKEN found in environment.")

        # Initialize model parameters or load pretrained weights here
        self.processor, self.model = load_falcon_model(checkpoint_dir, config.device, hf_token)

        self._thinking_token_id = None
        if _text_token_proprio(self.model):
            self._thinking_token_id = self.processor.tokenizer.encode("<thinking>", add_special_tokens=False)[0]
            print("FalconVLA: proprio_mode=text_token; splicing discretized proprio before <thinking>")

    @torch.no_grad()
    def inference(self, observation: _model.Observation, task_label: str = None) -> _model.Actions:
        """Run inference on observation.

        Args:
            observation: Model observation containing images, state, and optional prompt
            task_label: Optional task description. If None, will use prompt from observation

        Returns:
            Predicted actions
        """
        # Extract task label from observation if not provided
        # TODO: Fix this mess
        if task_label is None:
            if hasattr(observation, "tokenized_prompt") and observation.tokenized_prompt is not None:
                # If we have tokenized prompt, we need to decode it (simplified approach)
                task_label = "complete the task"  # Fallback
            else:
                task_label = "complete the task"  # Default fallback

        # Building the prompt for the model
        input_builder = ActionPromptBuilder(task_label=observation["prompt"])

        # Getting the images from the observation
        # Observation.images should be a dict with standard keys
        images = observation.get("images", None)

        # Map from standard image keys to camera names
        # base_0_rgb -> primary/main image (cam_high)
        # right_wrist_0_rgb -> wrist image (cam_right_wrist)
        # left_wrist_0_rgb -> secondary image (cam_left_wrist)

#! ### Debug - Saving images to disk just before they are consumed by
        # # Create a folder if doensn't exist for storing the images (for debugging)
        # # add timestamp to the folder name to avoid overwriting
        # import time
        # import cv2
        # import numpy as np
        
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        # debug_folder = f"debug/{timestamp}"
        # os.makedirs(debug_folder, exist_ok=True)
        # if images is not None:
        #     for key, img in images.items():
        #         try:
        #             img_path = f"{debug_folder}/{timestamp}_{key}.png"
        #             img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else np.array(img)

        #             # Convert CHW to HWC and RGB to BGR for cv2
        #             if img_np.shape[0] == 3:
        #                 img_np = np.transpose(img_np, (1, 2, 0))
        #                 # img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        #             cv2.imwrite(img_path, img_np)
        #             print(f"Saved {key} to {img_path}")
        #         except Exception as e:
        #             print(f"Failed to save {key}: {e}")
#! ####

        primary_image = images.get("cam_high", None)
        if primary_image is None:
            raise ValueError("Missing required 'cam_high' image in observation")

        # Adding the primary image to the prompt
        input_builder.add_main_image(image=primary_image)

        if self.config.use_wrist:
            wrist_image = images.get("cam_right_wrist", None)
            if wrist_image is not None:
                input_builder.add_wrist_image(wrist_image)

        if self.config.use_secondary:
            secondary_image = images.get("cam_left_wrist", None)
            if secondary_image is not None:
                input_builder.add_secondary_image(secondary_image)

        state = observation.get("state", None)
        raw_state = np.zeros(0) if state is None else np.asarray(state, dtype=np.float64).ravel()
        offset = _arm_offset(self.config.arm, self.config.robot_action_dim)

        proprio_inputs = None
        proprio_token_ids = None
        if self.config.use_proprio:
            proprio_stats = fetch_proprio_stats(self.model, self.config.unnorm_key)
            proprio = _build_proprio(
                raw_state,
                proprio_stats,
                proprio_dim=self.config.proprio_dim,
                offset=offset,
                static_indices=self.config.static_proprio_indices,
            )
            if _DEBUG_PROPRIO:
                _mn = np.asarray(proprio_stats["min"], dtype=np.float64)
                _mx = np.asarray(proprio_stats["max"], dtype=np.float64)
                ood = [i for i in range(len(proprio)) if not _mn[i] - 1e-6 <= proprio[i] <= _mx[i] + 1e-6]
                print(f"proprio arm={self.config.arm} dim={len(proprio)} ood={ood}")
                print(" ".join(f"{v:.4f}" for v in proprio))

            if self._thinking_token_id is not None:
                # Tokens are spliced into the prompt below; leaving `proprio_inputs` unset keeps
                # the checkpoint from also splicing its own (mirrored, mis-anchored) copy.
                proprio_token_ids = _proprio_token_ids(
                    normalize_proprio(proprio, proprio_stats), self.model.config
                )
            else:
                normalizes_internally = _normalizes_internally(self.model)
                if not normalizes_internally:
                    proprio = normalize_proprio(proprio, proprio_stats)

                # Checkpoints that normalize internally build their bounds on CPU and never move
                # them, so the raw vector has to stay on CPU; they relocate it to the model device
                # themselves once normalized.
                proprio_device = "cpu" if normalizes_internally else self.config.device
                proprio_inputs = torch.as_tensor(np.asarray(proprio), device=proprio_device).float()
                # -> (B, T, proprio_dim); single-frame history => T=1
                if proprio_inputs.dim() == 1:
                    proprio_inputs = proprio_inputs.unsqueeze(0).unsqueeze(0)
                elif proprio_inputs.dim() == 2:
                    proprio_inputs = proprio_inputs.unsqueeze(1)

        inputs, _ = input_builder.build_inputs(processor=self.processor)
        inputs.pop("token_type_ids", None)
        inputs = {k: (v.to(self.config.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        if proprio_token_ids is not None:
            _splice_before_thinking(inputs, proprio_token_ids, self._thinking_token_id)

        predict_kwargs = {}
        if proprio_inputs is not None:
            if _proprio_is_kwarg(self.model):
                predict_kwargs["proprio_inputs"] = proprio_inputs
            else:
                inputs["proprio_inputs"] = proprio_inputs

        actions = self.model.predict_action(
            inputs,
            unnorm_key=self.config.unnorm_key,
            horizon=self.config.action_horizon,
            do_sample=False,
            **predict_kwargs,
        )
        return _to_robot_actions(
            actions,
            action_horizon=self.config.action_horizon,
            action_dim=self.config.action_dim,
            offset=offset,
            robot_action_dim=self.config.robot_action_dim,
            hold=raw_state,
        )


    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @override
    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation, **kwargs) -> _model.Actions:
        # Extract task_label from kwargs if provided
        task_label = kwargs.get("task_label", None)
        return self.__call__(observation, task_label=task_label)
