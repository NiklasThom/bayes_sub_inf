# OpenCode Agent Instructions

This file contains crucial context for agents working in this repository.

## Project Layout & Architecture
- The package is structured under the `subspace_inference` namespace.
- **Top-level Import:** `import subspace_inference`
- **Sub-packages:**
  - `subspace_inference.curve_optimizer`: Contains training and optimization logic.
  - `subspace_inference.curve_optimizer.models`: Contains model implementations (e.g., Qwen JAX).
  - `subspace_inference.curve_optimizer.trainer`: Contains training entrypoints.
- **Main Entrypoint:** `subspace_inference/curve_optimizer/trainer/training_pipeline.py` (generic training)
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
- **Generic Training Pipeline:** The `train()` function in `training_pipeline.py` handles all curve training logic (k=0 single-point, k>0 curve, fixed/free control points). Model-specific setup (loading, dataset preparation) should be done before calling `train()`.

## Environment & Setup
- The project uses **uv** for dependency management. Requires Python `^3.10,<3.12`.
- **CUDA 12 Support**: Automatically installed with JAX via `jax[cuda12]` extra. No manual setup needed!

## Command Reference

### Basic Commands
- `uv sync` - Install all dependencies (replaces `poetry install`)
- `uv run python script.py` - Run a script (replaces `poetry run python script.py`)
- `uv add package` - Add a dependency (replaces `poetry add package`)
- `uv remove package` - Remove a dependency (replaces `poetry remove package`)
- `uv lock` - Lock dependencies (replaces `poetry lock`)

### Optional Dependencies

**Development tools** (always available):
```bash
uv sync --group dev
```
Includes: `pytest`, `ipykernel` (for Jupyter notebooks)

**Visualization packages** (optional):
```bash
uv sync --extra viz
```
Includes: `ipywidgets`, `seaborn`, `matplotlib`, `pandas`, `arviz`, `hvplot`, `datashader`, etc.

**All extras**:
```bash
uv sync --all-extras --all-groups
```

### Python Version
uv will automatically use the first available Python 3.10 or 3.11. To specify a version:
```bash
uv sync --python 3.10
```

## Verification & Testing

**Smoke Test with Qwen example:**
```bash
uv run python subspace_inference/curve_optimizer/trainer/training_pipeline.py --smoke-test
```

**Note:** The `qwen_fineTuning_PtoC_fixCps.py` script is deprecated. Use `training_pipeline.py` as the main entry point.

## Single-Point (k=0) Evaluation

To run single-point evaluation (no curve, just a single model):

```python
config = {
    "model_params": {
        "num_curve_segment": 0,
        "SegDeg": 1,
        "Pretraining": True,
        # ... other params ...
    },
}
```

This configuration sets `k=0` (single control point) and trains/evaluates a standard model without Bézier curve optimization.

## Custom Model Development

### Custom DNN Backbones

To add a new neural network architecture:

1. Create model file in `subspace_inference/curve_optimizer/models/my_model.py`:

```python
from flax import linen as nn
import jax.numpy as jnp
from subspace_inference.curve_optimizer.models import register_model

@register_model("my_model")
class MyModel(nn.Module):
    hidden_dim: int = 128
    output_dim: int = 10
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x
```

2. Import in `models/__init__.py` to auto-register

3. Use in config:

```python
config_dict = {
    "net_kwargs": {
        "model_type": "my_model",
        "hidden_dim": 256,
        "output_dim": 10,
    },
}
```

### Custom Subspace Models

To define a custom task-specific loss function:

1. Add to `subspace_inference/curve_optimizer/subspace_curve.py`:

```python
import jax.numpy as jnp
from subspace_inference.curve_optimizer.subspace_curve import (
    register_subspace_model,
    SubspaceBaseModel,
)

@register_subspace_model("my_task")
class MyTaskSubspace(SubspaceBaseModel):
    # Required: Define empty metric template
    empty_metric = {
        "ll": jnp.array(-jnp.inf),
        "acc": jnp.array(-jnp.inf),
        "mean_loss": jnp.array(jnp.inf),
    }
    
    def nll(self, params, state, t, x, y, train=True, key=None):
        out, state = self(params, state, t, x, train=train, key=key)
        # Custom loss computation
        loss = ...  # Compute loss
        return loss, state, out
    
    def evaluate(self, logits, y, key_prefix="", weights=None):
        # Compute and return metrics
        # logits: (n_samples, n_data, output_dim)
        ...
```

2. Use in config:

```python
config_dict = {
    "model_params": {
        "subspace_model": "my_task",
    },
}
```

**Important:**
- Always define `empty_metric` class attribute with all metric keys
- `nll()` must return `(loss, state, logits)` with logits shape `(n_samples, n_data, output_dim)`
- `evaluate()` receives sampled logits and returns metrics dict
- For JAX `lax.cond` compatibility, always return the same metric keys
- Use `@register_subspace_model("name")` decorator
- Mixins (LoRA, Repulsive, JSD, etc.) can be combined via multiple inheritance
