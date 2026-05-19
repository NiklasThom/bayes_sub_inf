"""Model registry for generic model instantiation."""

from typing import Dict, Type
from flax import linen as nn

# Model registry for backbone models
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Class decorator that registers a Flax nn.Module model."""

    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Duplicate model name: '{name}'")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> Type[nn.Module]:
    """Get model class from registry by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


# Import models to register them
from subspace_inference.curve_optimizer.models.toy_mlp import MLP

# Register MLP
MODEL_REGISTRY["mlp"] = MLP
