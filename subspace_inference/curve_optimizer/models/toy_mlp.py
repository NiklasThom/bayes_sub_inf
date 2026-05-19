import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    """Simple MLP backbone for regression/classification tasks."""

    hidden_dim: int = 64
    out_dim: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x
