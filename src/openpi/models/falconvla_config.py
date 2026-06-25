import dataclasses
from typing import TYPE_CHECKING

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
    proprio_dim: int=14
    proprio_history_window: int=1



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
