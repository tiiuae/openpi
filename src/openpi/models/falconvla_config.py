import dataclasses
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from typing_extensions import override
import tyro

from openpi.models import model as _model
import openpi.shared.array_typing as at

if TYPE_CHECKING:
    from openpi.models.falconvla import FalconVLA


@dataclasses.dataclass(frozen=True)
class FalconVLAConfig(_model.BaseModelConfig):
    """Configuration for FalconVLA.

    FalconVLA is served in-process but wraps a HuggingFace ``AutoModelForVision2Seq``
    checkpoint that owns its own tokenization and (un)normalization. openpi's transform
    pipeline is therefore intentionally minimal for this model (see
    ``LeRobotFalconVLADataConfig`` / the ``FALCONVLA`` branch of ``ModelTransformFactory``).
    """

    # Action space. Required by BaseModelConfig.
    action_dim: int = 14
    action_horizon: int = 25
    # Unused by FalconVLA (the HF processor handles tokenization); kept only to satisfy
    # BaseModelConfig, which requires a value.
    max_token_len: int = 180

    # Device to load the model on.
    device: str = "cuda"

    # Which camera views to feed the model.
    use_wrist: bool = True
    use_secondary: bool = True

    # Proprioception. When True, `inference` normalizes the raw state using the checkpoint's own
    # proprio stats (keyed by `unnorm_key`). How it reaches the model depends on `proprio_mode`
    # below — different FalconVLA checkpoint families were trained with different mechanisms.
    use_proprio: bool = True
    # "tokens": inject discrete `<prop0>`..`<propN-1>` text tokens into the prompt (the
    #     FM-DiT `-p` checkpoints, e.g. aidrc_cups_manipulation_14 dualarm).
    # "film": pass a continuous tensor as `proprio_inputs=` to `predict_action`, consumed via the
    #     checkpoint's own FiLM/vlm_inject conditioning (e.g. the ResNet `-p-film` checkpoints).
    # These are not interchangeable: a "film" checkpoint's tokenizer has no `<prop*>` tokens at
    # all, and a "tokens" checkpoint's `predict_action` doesn't accept `proprio_inputs`.
    proprio_mode: Literal["tokens", "film"] = "tokens"
    # Number of discrete bins proprio values are tokenized into (`<prop0>` .. `<propN-1>`) when
    # `proprio_mode == "tokens"`. Must match the checkpoint's `n_action_bins` (256 for all current
    # FalconVLA checkpoints).
    num_bins: int = 256

    # Selects the dataset normalization stats baked into the HF checkpoint. Must match the
    # training dataset or predictions are scrambled — there is no sane default, so it must be
    # set explicitly by each TrainConfig.
    unnorm_key: str = tyro.MISSING

    # Determinism. FalconVLA's flow-matching (FM-DiT) action head samples from random noise, so a
    # fixed seed is required for reproducible actions (mirrors the canonical FalconVLA server's
    # pin_global_seed). strict_determinism also forces deterministic cuDNN kernels + disables TF32.
    seed: int = 7
    strict_determinism: bool = False
    # When True, reseed to `seed` before every inference so identical observations always yield
    # identical actions (deterministic controller). When False, the action head advances its RNG
    # each call, so repeated inference on the same observation samples slightly different actions.
    deterministic_inference: bool = False

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.FALCONVLA

    @override
    def create(self) -> "FalconVLA":
        from openpi.models.falconvla import FalconVLA

        return FalconVLA(self)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images=dict.fromkeys(_model.IMAGE_KEYS, image_spec),
                image_masks=dict.fromkeys(_model.IMAGE_KEYS, image_mask_spec),
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec
