# Subspace Inference

Bézier-curve parameterized neural networks with LoRA integration for fine-tuning models in a parameter subspace. This framework enables fine-tuning within a parameter subspace by learning weights as continuous Bézier curves rather than static points.

## Quick Start

### Installation

```bash
# Base installation (includes CUDA 12 support automatically)
uv sync

# With development tools
uv sync --group dev

# With visualization packages
uv sync --extra viz
```

### Basic Usage

The main training entry point is `subspace_inference/curve_optimizer/trainer/training_pipeline.py`. Here's a minimal example:

```python
from subspace_inference.curve_optimizer.trainer.training_pipeline import train, Config, DataSplits
import wandb

# 1. Load your dataset (model-specific)
data = DataSplits(train_x=..., train_y=..., val_x=..., val_y=..., test_x=..., test_y=...)

# 2. Configure the experiment
config_dict = {
    "rng_seed": 1,
    "model_params": {
        "num_curve_segment": 1,  # k=1 for single-segment curve
        "SegDeg": 1,             # degree per segment
        "Pretraining": False,
        "subspace_model": "lora_category",  # classification with LoRA
        "lora_params": {"r": 8, "lora_alpha": 16.0},
    },
    "train_hyper": {
        "batch_size": 4,
        "num_epochs": 10,
    },
    "optimizer_conf": {
        "name": "adamw",
        "kwargs": {"learning_rate": 1e-4},
    },
}
config = Config.from_dict(config_dict, data)

# 3. Train
logger = wandb.init()
env, params, config = train(logger, config, data)
```

### Single-Point Evaluation (k=0)

For standard fine-tuning without curves:

```python
config_dict = {
    "model_params": {
        "num_curve_segment": 0,
        "SegDeg": 1,
        "Pretraining": True,
    },
}
```

## Project Layout

```
.
├── subspace_inference/
│   ├── curve_optimizer/
│   │   ├── trainer/
│   │   │   ├── training_pipeline.py    # Main training entry point
│   │   │   └── qwen_fineTuning_PtoC_fixCps.py  # Qwen-specific example (deprecated)
│   │   ├── models/                     # DNN backbone architectures
│   │   │   ├── __init__.py            # Model registry
│   │   │   ├── MLP.py                 # MLP implementations
│   │   │   ├── ResNet.py              # ResNet implementations
│   │   │   └── qwen_jax.py            # Qwen JAX wrapper
│   │   └── subspace_curve.py          # Core subspace math & model registry
│   └── __init__.py
├── configs_newDS/                      # Wandb sweep configurations
└── ...
```

### Key Components

- **`training_pipeline.py`**: Generic training orchestrator for any model
- **`subspace_curve.py`**: Bézier math, SubspaceBaseModel, subspace model registry
- **`models/`**: Task-agnostic DNN backbones (MLP, ResNet, Qwen, etc.)
- **`models/__init__.py`**: Backbone model registry

## Training Modes

### 1. Pretraining Mode (k=0)

Train fixed control points independently before fitting the curve:

- Set `num_curve_segment=0`, `Pretraining=True`
- Each fixed CP (value=1 in `cp_fix`) is trained separately

### 2. Curve Fitting Mode (k>0)

Fit Bézier curve connecting the control points:

- `num_curve_segment=2`, `SegDeg=2` → k=5 control points
- `cp_fix=[1, 0, 1, 0, 1]` fixes endpoints, trains intermediate points

### 3. Evaluation

Automatically runs after training. Supports multiple evaluation modes via `curve_sampling_mode`:

- `combined_*`: Compute BMA weights on train/val, evaluate on test
- `per_leave`: Leave-one-out evaluation
- `noBMA`: Uniform averaging without Bayesian model averaging

## Extending the Framework

### Adding Custom DNN Backbones (using `@register_model`)

Create task-agnostic neural network architectures in `subspace_inference/curve_optimizer/models/` using the `@register_model` decorator:

```python
# subspace_inference/curve_optimizer/models/my_model.py
from flax import linen as nn
import jax.numpy as jnp
from subspace_inference.curve_optimizer.models import register_model

@register_model("my_mlp")
class MyMLP(nn.Module):
    """Custom MLP architecture."""
    hidden_dim: int = 128
    num_layers: int = 3
    output_dim: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(0.1)(x, deterministic=not train)
        x = nn.Dense(self.output_dim)(x)
        return x
```

**Usage in config:**

```python
config_dict = {
    "net_kwargs": {
        "model_type": "my_mlp",
        "hidden_dim": 256,
        "num_layers": 4,
        "output_dim": 10,
    },
}
```

**Key Points:**

- Must be a Flax `nn.Module`
- Must accept `train: bool = True` parameter
- Register with `@register_model("name")` decorator
- Import the module in `models/__init__.py` to auto-register

### Adding Custom Subspace Models

Define task-specific loss functions and evaluation metrics in `subspace_curve.py`:

```python
# subspace_inference/curve_optimizer/subspace_curve.py
import jax.numpy as jnp
from jax import random
import optax
from subspace_inference.curve_optimizer.subspace_curve import (
    register_subspace_model,
    SubspaceBaseModel,
)

@register_subspace_model("my_task")
class MyTaskSubspace(SubspaceBaseModel):
    """Custom subspace model for my task."""

    # Required: Define empty metric template
    empty_metric = {
        "ll": jnp.array(-jnp.inf, dtype=jnp.float32),
        "acc": jnp.array(-jnp.inf, dtype=jnp.float32),
        "mean_loss": jnp.array(jnp.inf, dtype=jnp.float32),
    }

    def nll(self, params, state, t, x, y, train: bool = True, key=None):
        """
        Compute negative log-likelihood loss.

        Args:
            params: Model parameters
            state: Flax state (BatchNorm stats, etc.)
            t: Curve parameter (scalar in [0, 1])
            x: Input data
            y: Target labels
            train: Training mode flag
            key: JAX random key

        Returns:
            (loss, state, logits) where logits shape is (n_samples, n_data, output_dim)
        """
        # Forward pass through subspace model (interpolates params based on t)
        out, state = self(params, state, t, x, train=train, key=key)

        # Compute loss (example: classification)
        loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits=out, labels=y)
        return jnp.mean(loss), state, out

    def evaluate(self, logits, y, key_prefix="", weights=None):
        """
        Compute evaluation metrics from sampled logits.

        Args:
            logits: (n_samples, n_data, n_classes) - unnormalized logits
            y: (n_data,) - integer labels
            weights: None/False for no averaging, True for uniform average,
                    or array of shape (n_samples,) for weighted BMA
            key_prefix: Prefix for metric keys (e.g., "test_", "val_")

        Returns:
            dict with metric keys like "ll", "acc", etc.
        """
        # Compute log_softmax
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        # Ensemble predictions based on weights
        if weights is None or weights is False:
            post_logits = log_probs.mean(axis=0)  # Uniform average
        elif weights is True:
            post_logits = jax.nn.logsumexp(log_probs, axis=0) - jnp.log(log_probs.shape[0])
        else:
            post_logits = jax.nn.logsumexp(log_probs, b=weights[:, None, None], axis=0)

        # Compute metrics
        acc = jnp.mean(jnp.argmax(post_logits, axis=-1) == y)
        ll = jnp.take_along_axis(post_logits, y[:, None], axis=-1).squeeze(-1).mean()

        return {
            f"{key_prefix}ll": ll,
            f"{key_prefix}acc": acc,
            f"{key_prefix}mean_loss": -ll,
        }
```

**Usage in config:**

```python
config_dict = {
    "model_params": {
        "subspace_model": "my_task",
    },
}
```

**Key Points:**

- Must inherit from `SubspaceBaseModel`
- Must define `empty_metric` class attribute (required for JAX consistency)
- `nll()` returns `(loss, state, logits)` where logits shape is `(n_samples, n_data, output_dim)`
- `evaluate()` receives sampled logits and returns metrics dict
- Use `@register_subspace_model("name")` decorator
- Mixins (LoRA, Repulsive, JSD, etc.) can be combined via multiple inheritance

## Contributing

### Development Setup

1.  Install the package with development dependencies:
    ```bash
    uv sync --group dev
    ```

2.  Install `pre-commit` hooks to ensure code quality (formatting with `ruff`):
    ```bash
    uv run pre-commit install
    ```

### Code Standards

We use `ruff` for linting and formatting. The pre-commit hooks will automatically check your changes. You can also run them manually:

```bash
uv run pre-commit run --all-files
```
