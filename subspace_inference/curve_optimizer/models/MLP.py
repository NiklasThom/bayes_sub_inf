from flax import linen as nn
import jax.numpy as jnp


class MLPFeatureModel(nn.Module):
    depth: int = 3
    width: int = 10
    activation: str = "relu"

    def setup(self) -> None:
        self.activation_fn = getattr(nn, self.activation)
        return super().setup()

    @nn.compact
    def __call__(self, x,):
        x = jnp.concat([x, x**2, x**3], axis=-1)
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x
    

class MLPModel(nn.Module):
    depth: int = 3
    width: int = 10
    activation: str = "relu"
    output_dim: int = 1

    def setup(self) -> None:
        self.activation_fn = getattr(nn, self.activation)
        return super().setup()

    @nn.compact
    def __call__(self, x,):
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = self.activation_fn(x)
        x = nn.Dense(self.output_dim)(x)
        return x
    