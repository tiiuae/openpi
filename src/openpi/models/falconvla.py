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



def load_falcon_model(model_name: str, device: str, hf_token: Optional[str]) -> Tuple[AutoProcessor, AutoModelForVision2Seq]:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=hf_token
    ).to(device)
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
        if task_label is None:
            if hasattr(observation, 'tokenized_prompt') and observation.tokenized_prompt is not None:
                # If we have tokenized prompt, we need to decode it (simplified approach)
                task_label = "complete the task"  # Fallback
            else:
                task_label = "complete the task"  # Default fallback

        # Building the prompt for the model
        input_builder = ActionPromptBuilder(task_label=task_label)

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

        if self.config.use_proprio:
            # Computing the proprio tokens
            proprio_stats = fetch_proprio_stats(self.model, self.config.unnorm_key)
            normalized_proprio = normalize_proprio(observation.get("state",None), proprio_stats)
            tokenized_proprio = tokenize_proprio(normalized_proprio, self.config.num_bins)
            proprio_tokens = get_proprio_tokens(tokenized_proprio)

            input_builder.add_proprio(proprio_tokens=proprio_tokens)

        inputs, _ = input_builder.build_inputs(processor=self.processor)
        inputs.pop("token_type_ids", None)
        inputs = {k: (v.to(self.config.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        actions = self.model.predict_action(
            inputs,
            unnorm_key=self.config.unnorm_key,
            horizon=self.config.action_horizon,
            do_sample=False
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
        task_label = kwargs.get('task_label', None)
        return self.__call__(observation, task_label=task_label) 
