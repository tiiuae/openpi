from typing import Optional, Tuple

from typing_extensions import override

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from openpi.models import model as _model
from openpi.models.falconvla_config import FalconVLAConfig
from openpi.models.utils.prompt_builder import ActionPromptBuilder
from openpi.models.utils.proprio import *
import openpi.shared.array_typing as at
import os
import numpy as np



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


class FalconVLA(_model.BaseModel):
    def __init__(self, config: FalconVLAConfig, checkpoint_dir: str = ""):
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # Load hf token from environment variable if available
        hf_token = os.getenv("HF_TOKEN", None)
        print("HF_TOKEN found in environment." if hf_token else "No HF_TOKEN found in environment.")

        # Initialize model parameters or load pretrained weights here
        self.processor, self.model = load_falcon_model(checkpoint_dir, config.device, hf_token)

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

        # if self.config.use_proprio:
        #     # Computing the proprio tokens
        #     proprio_stats = fetch_proprio_stats(self.model, self.config.unnorm_key)
        #     normalized_proprio = normalize_proprio(observation.get("state", None), proprio_stats)
        #     tokenized_proprio = tokenize_proprio(normalized_proprio, self.config.num_bins)
        #     proprio_tokens = get_proprio_tokens(tokenized_proprio)

        #     input_builder.add_proprio(proprio_tokens=proprio_tokens)

        # inputs, _ = input_builder.build_inputs(processor=self.processor)
        # inputs.pop("token_type_ids", None)
        # inputs = {k: (v.to(self.config.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # actions = self.model.predict_action(
        #     inputs, unnorm_key=self.config.unnorm_key, horizon=self.config.action_horizon, do_sample=False
        # )
        # return actions
        proprio_inputs = None
        if self.config.use_proprio:
            

            # FiLM proprio path: normalize the raw state and pass it straight into the
            # action head as `proprio_inputs`. Do NOT tokenize it into the prompt — the
            # FiLM checkpoint (proprio_mode="film") was not trained on <prop*> tokens.
            proprio_stats = fetch_proprio_stats(self.model, self.config.unnorm_key)

            raw_state = np.asarray(observation.get("state", None), dtype=np.float64).ravel()
            _mn   = np.asarray(proprio_stats["min"], dtype=np.float64)
            _mx   = np.asarray(proprio_stats["max"], dtype=np.float64)
            _mean = np.asarray(proprio_stats["mean"], dtype=np.float64)
            _span = _mx - _mn

            # Pin "frozen" proprio channels (those with ~zero training span — e.g. the
            # static arm in a single-arm dataset) to the training mean before normalizing.
            # At deploy the physical arm sits at an arbitrary pose; normalizing that live
            # value on a near-degenerate [min,max] range saturates it to ±1 — a value the
            # FiLM head never saw in training, which corrupts the conditioning and makes
            # the policy emit a near-constant/jittery trajectory. Replacing those channels
            # with the training mean keeps the FiLM input in-distribution.
            FROZEN_SPAN_THRESH = 1e-3
            n = min(raw_state.shape[0], _span.shape[0])
            frozen_mask = np.zeros(raw_state.shape[0], dtype=bool)
            frozen_mask[:n] = _span[:n] < FROZEN_SPAN_THRESH
            pinned_state = raw_state.copy()
            pinned_state[:n][frozen_mask[:n]] = _mean[:n][frozen_mask[:n]]

            normalized_proprio = normalize_proprio(pinned_state, proprio_stats)

            # Debug: per-channel flags — PIN = frozen channel pinned to mean,
            # OOD = live value outside training bounds.
            print(f"{'i':>2} {'raw':>12} {'pinned':>12} {'min':>12} {'max':>12} {'span':>10}  flag")
            for i in range(len(raw_state)):
                ood = i < n and ((raw_state[i] < _mn[i] - 1e-6) or (raw_state[i] > _mx[i] + 1e-6))
                flag = "PIN" if frozen_mask[i] else ("OOD" if ood else "")
                _mn_i, _mx_i, _span_i = (_mn[i], _mx[i], _span[i]) if i < n else (float("nan"),) * 3
                print(f"{i:>2} {raw_state[i]:>12.5f} {pinned_state[i]:>12.5f} {_mn_i:>12.5f} {_mx_i:>12.5f} {_span_i:>10.5f}  {flag}")

            proprio_inputs = torch.as_tensor(normalized_proprio, device=self.config.device).float()
            # -> (B, T, proprio_dim); single-frame history => T=1
            if proprio_inputs.dim() == 1:
                proprio_inputs = proprio_inputs.unsqueeze(0).unsqueeze(0)
            elif proprio_inputs.dim() == 2:
                proprio_inputs = proprio_inputs.unsqueeze(1)

        inputs, _ = input_builder.build_inputs(processor=self.processor)
        inputs.pop("token_type_ids", None)
        inputs = {k: (v.to(self.config.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        actions = self.model.predict_action(
            inputs,
            unnorm_key=self.config.unnorm_key,
            horizon=self.config.action_horizon,
            do_sample=False,
            proprio_inputs=proprio_inputs,
        )
        return actions


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
