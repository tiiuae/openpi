# Guide: Integrating a New Model for Serving with serve_policy.py

This guide explains how to integrate a new model (similar to pi0) that can be served using `scripts/serve_policy.py`.

## Overview

The OpenPI serving system has several key components:
1. **Model Definition** - The actual model architecture
2. **Model Config** - Configuration for the model
3. **Train Config** - Training configuration that includes data transforms and metadata
4. **Policy** - Wrapper that handles transforms and inference
5. **Serving** - WebSocket server that exposes the policy

## Step-by-Step Integration

### Step 1: Define Your Model Architecture

Create a new file in `src/openpi/models/` for your model (e.g., `my_model.py`):

```python
import flax.nnx as nnx
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at

class MyModel(nnx.Module):
    """Your model implementation."""
    
    def __init__(self, config: "MyModelConfig", rngs: nnx.Rngs):
        # Initialize your model layers
        self.config = config
        # ... your model initialization
    
    @at.typecheck
    def __call__(
        self,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> _model.ModelOutput:
        """Forward pass of the model."""
        # Your model forward pass
        # Return ModelOutput with predicted actions
        pass
    
    @at.typecheck
    def sample_actions(
        self,
        observation: _model.Observation,
        *,
        rng: at.KeyArrayLike,
    ) -> at.Float[at.Array, "b ah ad"]:
        """Sample actions from the model during inference."""
        # Your sampling logic
        pass
```

### Step 2: Create Model Config

Create a config file `src/openpi/models/my_model_config.py`:

```python
import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.my_model import MyModel


@dataclasses.dataclass(frozen=True)
class MyModelConfig(_model.BaseModelConfig):
    """Configuration for MyModel."""
    
    # Model-specific parameters
    dtype: str = "bfloat16"
    hidden_dim: int = 512
    
    # Required for policy serving
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    
    @property
    @override
    def model_type(self) -> _model.ModelType:
        # Add your model type to ModelType enum first
        return _model.ModelType.MY_MODEL
    
    @override
    def create(self, rng: at.KeyArrayLike) -> "MyModel":
        from openpi.models.my_model import MyModel
        return MyModel(self, rngs=nnx.Rngs(rng))
    
    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """Define the expected input shapes."""
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        
        return observation_spec, action_spec
```

### Step 3: Add Model Type to Enum

Edit `src/openpi/models/model.py` to add your model type:

```python
class ModelType(enum.Enum):
    """Supported model types."""
    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"
    MY_MODEL = "my_model"  # Add your model type
```

### Step 4: Add Model Transforms

Edit `src/openpi/training/config.py` to add transforms for your model in the `ModelTransformFactory.__call__` method:

```python
def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
    match model_config.model_type:
        case _model.ModelType.PI0:
            # ... existing code
        case _model.ModelType.MY_MODEL:
            return _transforms.Group(
                inputs=[
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    ),
                    _transforms.PadStatesAndActions(model_config.action_dim),
                ],
            )
```

### Step 5: Create Train Config

Add a training configuration in `src/openpi/training/config.py` in the `_CONFIGS` list:

```python
# Import your model config at the top
import openpi.models.my_model_config as my_model_config

# Add to _CONFIGS list
_CONFIGS = [
    # ... existing configs
    
    # Your model config for ALOHA
    TrainConfig(
        name="my_model_aloha",
        model=my_model_config.MyModelConfig(
            action_dim=14,  # Adjust based on your robot
            action_horizon=50,
        ),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="your default prompt",  # Optional
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    
    # Your model config for other environments
    TrainConfig(
        name="my_model_droid",
        model=my_model_config.MyModelConfig(
            action_dim=8,
            action_horizon=10,
        ),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.MY_MODEL)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
]
```

### Step 6: Add Default Checkpoint (Optional)

If you want to provide a default checkpoint for your model, edit `scripts/serve_policy.py`:

```python
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="my_model_aloha",  # Use your config name
        dir="gs://your-bucket/checkpoints/my_model_aloha",
    ),
    # ... other environments
}
```

### Step 7: PyTorch Support (Optional)

If your model uses PyTorch instead of JAX:

1. Implement your model in `src/openpi/models_pytorch/my_model_pytorch.py`
2. Add a `load_pytorch` method to your config:

```python
@override
def load_pytorch(self, train_config, weight_path: str):
    """Load PyTorch model from checkpoint."""
    from openpi.models_pytorch import my_model_pytorch
    return my_model_pytorch.load_model(self, weight_path)
```

The serving system will automatically detect PyTorch models by checking for `model.safetensors` in the checkpoint directory.

## Usage

Once integrated, you can serve your model using:

```bash
# Using a trained checkpoint
python scripts/serve_policy.py \
    --policy.config my_model_aloha \
    --policy.dir /path/to/checkpoint \
    --port 8000

# Using default checkpoint (if configured)
python scripts/serve_policy.py \
    --env ALOHA \
    --port 8000

# With a custom default prompt
python scripts/serve_policy.py \
    --policy.config my_model_aloha \
    --policy.dir /path/to/checkpoint \
    --default-prompt "pick up the object" \
    --port 8000

# Enable recording for debugging
python scripts/serve_policy.py \
    --policy.config my_model_aloha \
    --policy.dir /path/to/checkpoint \
    --record
```

## Key Components Explained

### Policy Loading Flow

1. `serve_policy.py` calls `create_policy(args)`
2. This calls `policy_config.create_trained_policy()` with your train config
3. The train config's model config loads the model weights
4. Transforms are applied based on your config
5. A `Policy` object is created that wraps your model
6. The policy is served via WebSocket

### Data Transforms

Transforms convert raw data to model input format:
- **Repack Transforms**: Convert robot-specific formats
- **Data Transforms**: Robot-specific preprocessing
- **Normalize**: Apply normalization stats
- **Model Transforms**: Model-specific preprocessing (tokenization, etc.)

### Normalization Stats

Norm stats are computed during training and stored in the checkpoint:
- Located at: `checkpoint_dir/assets/{asset_id}/norm_stats.npz`
- Used to normalize states and actions during inference
- See [docs/norm_stats.md](norm_stats.md) for details

## Testing

Test your integration:

```python
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config

# Create policy
train_config = _config.get_config("my_model_aloha")
policy = _policy_config.create_trained_policy(
    train_config,
    "/path/to/checkpoint",
    default_prompt="test prompt"
)

# Test inference
import numpy as np
observation = {
    "image": {
        "base_0_rgb": np.random.rand(1, 224, 224, 3).astype(np.float32),
        "left_wrist_0_rgb": np.random.rand(1, 224, 224, 3).astype(np.float32),
        "right_wrist_0_rgb": np.random.rand(1, 224, 224, 3).astype(np.float32),
    },
    "image_mask": {
        "base_0_rgb": np.array([True]),
        "left_wrist_0_rgb": np.array([True]),
        "right_wrist_0_rgb": np.array([True]),
    },
    "state": np.zeros((1, 14), dtype=np.float32),
    "prompt": "test instruction",
}

actions = policy.sample_actions(observation)
print(f"Actions shape: {actions.shape}")
```

## Common Issues

1. **Model type not found**: Add your model type to `ModelType` enum
2. **Transform errors**: Check that your transforms match expected input/output format
3. **Checkpoint loading fails**: Ensure checkpoint directory has correct structure
4. **Normalization errors**: Verify norm stats exist in checkpoint assets
5. **Shape mismatches**: Check `inputs_spec` matches your model's expectations

## Examples

See existing model integrations for reference:
- [src/openpi/models/pi0.py](../src/openpi/models/pi0.py) - JAX implementation
- [src/openpi/models/pi0_fast.py](../src/openpi/models/pi0_fast.py) - FAST variant
- [src/openpi/models_pytorch/pi0_pytorch.py](../src/openpi/models_pytorch/pi0_pytorch.py) - PyTorch implementation

## Related Documentation

- [Remote Inference](remote_inference.md) - How to use the served policy
- [Normalization Stats](norm_stats.md) - Computing and using norm stats
- [Docker](docker.md) - Serving in Docker containers
