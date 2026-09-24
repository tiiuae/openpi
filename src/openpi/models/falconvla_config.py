import dataclasses
from typing import TYPE_CHECKING, Literal

from typing_extensions import override

from openpi.models import model as _model

if TYPE_CHECKING:
    from openpi.models.falconvla import FalconVLA

@dataclasses.dataclass(frozen=True)
class FalconVLAConfig(_model.BaseModelConfig):
    """Configuration for FalconVLA."""

    # Model name or path in HuggingFace model hub
    model_name: str = "tiiuae/FalconVLA-8B-ALOHA-FM-DiT-3V-NP"
    model_path: str = "falconvla"

    max_token_len: int = None  # type: ignore
    discrete_state_input: bool = None  # type: ignore

    # Device to load the model on
    device: str = "cuda"

    # Model-specific parameters
    dtype: str = "bfloat16"
    hidden_dim: int = 512

    # Required for policy serving
    action_dim: int = 14
    action_horizon: int = 25

    use_wrist: bool = True
    use_secondary: bool = True

    # Proprioception settings
    use_proprio: bool = True
    unnorm_key: str = "libero"
    num_bins: int = 256
    use_proprio_projector: bool = True
    proprio_history_window: int = 1

    # Which arm of the robot's bimanual vector this checkpoint drives: "left" is dims 0:7,
    # "right" is dims 7:14, "both" is the full vector. This selects the proprio slice read from
    # the live state and where predicted actions are written back.
    arm: Literal["both", "left", "right"] = "both"
    # Width of the command the robot expects back (bimanual ALOHA = 14).
    robot_action_dim: int = 14
    # Proprio width the checkpoint was trained with (its config.json `proprio_dim`). Channels the
    # robot cannot supply are filled with the training mean.
    proprio_dim: int | None = None
    # Channels that never moved during collection (e.g. the unused arm); pinned to the training
    # mean so the conditioning stays in-distribution.
    static_proprio_indices: tuple[int, ...] | None = None



    @property
    @override
    def model_type(self) -> _model.ModelType:
        # Add your model type to ModelType enum first
        return _model.ModelType.FALCONVLA

    @override
    def create(self) -> "FalconVLA":
        from openpi.models.falconvla import FalconVLA
        return FalconVLA(self)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        pass
