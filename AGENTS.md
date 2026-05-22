# OpenCode Agent Instructions

This file contains crucial context for agents working in this repository. 

## Project Layout & Architecture
- The package is structured under the `subspace_inference` namespace.
- **Top-level Import:** `import subspace_inference`
- **Sub-packages:**
  - `subspace_inference.curve_optimizer`: Contains training and optimization logic.
  - `subspace_inference.curve_optimizer.models`: Contains model implementations (e.g., Qwen JAX).
  - `subspace_inference.curve_optimizer.trainer`: Contains training entrypoints.
- **Main Entrypoint:** `subspace_inference/curve_optimizer/trainer/qwen_fineTuning_PtoC_fixCps.py`
- **Core Math:** `subspace_inference/curve_optimizer/subspace_curve.py`

## Transitional Codebase (Important!)
- This repository is currently being transformed into a pip-installable package.
- **Broken Imports:** Many files originally had `from src.X import Y` imports. These are being transitioned to absolute imports like `from subspace_inference.curve_optimizer.X import Y`. 
- **Proactive Fixes:** If you edit a file and see `from src...` or other broken relative imports, **proactively fix them** to use the `subspace_inference` absolute path.
- **Commented Dependencies:** `src.weight_matching` is currently missing from the repo. Related imports in `subspace_curve.py` are commented out and should remain so until the file is provided.

## Development Goals
- **Model Extensibility:** The `models/` directory is designed to be easily extended with new JAX implementations.
- **Dataset Injection:** The architecture aims to decouple dataset loading from training, allowing users to inject custom datasets easily.
- **Optimization Modes:** The framework supports Bézier curve optimization both with and without LoRA.

## Environment & Setup
- The project uses **Poetry** for dependency management. Requires Python `^3.10,<3.12`.
- **JAX/CUDA Quirk:** To properly install or upgrade JAX with CUDA 12 support, you must run:
  `poetry run pip install --upgrade "jax[cuda12_local]==0.5.3" "flax==0.10.5"`

## Verification & Testing
- **Smoke Test:** To verify changes, use the updated path:
  ```bash
  poetry run python subspace_inference/curve_optimizer/trainer/qwen_fineTuning_PtoC_fixCps.py --batch-size=4 --cp-fix 1 0 1 0 1 --smoke-test
  ```

## Adding New Models

The framework supports two types of model components:

### 1. Model Backbones (Task-Agnostic)

Model backbones are Flax `nn.Module` implementations that define the neural network architecture. They are registered in `subspace_inference/curve_optimizer/models/__init__.py`.

**Example: Adding a new backbone**

1. Create the model in `subspace_inference/curve_optimizer/models/my_model.py`:
```python
import jax.numpy as jnp
from flax import linen as nn


class MyModel(nn.Module):
    hidden_dim: int = 64
    out_dim: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x
```

2. Register it using the `@register_model` decorator in your model file:
```python
from subspace_inference.curve_optimizer.models import register_model

@register_model("my_model")
class MyModel(nn.Module):
    hidden_dim: int = 64
    out_dim: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x
```

The decorator automatically registers the model in `MODEL_REGISTRY`.

3. Use in config:
```python
config = {
    "net_kwargs": {
        "model_type": "my_model",
        "hidden_dim": 128,
        "out_dim": 1,
    },
    # ... other config ...
}
```

### 2. Subspace Models (Task-Specific)

Subspace models define task-specific loss functions and evaluation metrics. They are registered in `subspace_inference/curve_optimizer/subspace_curve.py` using the `@register_subspace_model` decorator.

**Example: Adding a new subspace model**

1. Create the subspace class in `subspace_curve.py`:
```python
@register_subspace_model("my_task")
class MyTaskSubspace(SubspaceBaseModel):
    def nll(self, params, state, t, x, y, train=True, key=None):
        # Define task-specific loss
        out, state = self(params, state, t, x, train=train, key=key)
        # ... compute loss ...
        return loss, state, out
    
    def evaluate(self, logits, y, key_prefix="", weights=None):
        # Define task-specific metrics
        # logits: (n_samples, n_data, output_dim)
        # y: (n_data, ...)
        # weights: (n_samples,) optional BMA weights
        # Returns dict with metrics
        
        # Always return the same keys for JAX lax.cond compatibility
        # Define empty_metric class attribute with all required keys
        ...
```

2. Define `empty_metric` class attribute with all metric keys (required for JAX consistency):
```python
@register_subspace_model("my_task")
class MyTaskSubspace(SubspaceBaseModel):
    empty_metric = {
        "ll": jnp.array(-jnp.inf),
        "acc": jnp.array(-jnp.inf),
        "mean_loss": jnp.array(jnp.inf),
        # ... all other metric keys ...
    }
    
    def evaluate(self, logits, y, key_prefix="", weights=None):
        # Compute and return metrics
        ...
```

2. Use in config:
```python
config = {
    "model_params": {
        "subspace_model": "my_task",
        # ... other subspace params ...
    },
    # ... other config ...
}
```

### Key Points

- **Backbones** (in `models/`) are task-agnostic neural network architectures
- **Subspace models** (in `subspace_curve.py`) define task-specific loss and evaluation
- Register both to make them available in configs
- The `evaluate()` method receives sampled logits with shape `(n_samples, n_data, output_dim)`
- For classification, logits are unnormalized (use softmax for probabilities)
- For regression, logits are direct predictions
- Models must accept `train: bool = True` parameter for API compatibility
- Subspace models must define `empty_metric` class attribute for JAX consistency
