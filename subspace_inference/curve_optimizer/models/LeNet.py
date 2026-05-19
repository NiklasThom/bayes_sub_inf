from flax import linen as nn
import jax.numpy as jnp
from typing import Callable


class LeNetti(nn.Module):
    """
    A super simple LeNet version.
    """

    activation_fn: Callable = nn.sigmoid
    out_dim: int = 10
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        x = nn.Conv(
            features=1, kernel_size=(3, 3), strides=(1, 1), padding=2, name="conv1"
        )(x)
        x = self.activation_fn(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=8, use_bias=self.use_bias, name="fc1")(x)
        x = self.activation_fn(x)
        x = nn.Dense(features=8, use_bias=self.use_bias, name="fc2")(x)
        x = self.activation_fn(x)
        x = nn.Dense(features=8, use_bias=self.use_bias, name="fc3")(x)
        x = self.activation_fn(x)
        x = nn.Dense(features=self.out_dim, use_bias=self.use_bias, name="fc4")(x)
        return x


class LeNet(nn.Module):
    """
    Implementation of LeNet.
    """

    activation_fn: Callable = nn.sigmoid
    out_dim: int = 10
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        x = nn.Conv(
            features=6, kernel_size=(5, 5), strides=(1, 1), padding=2, name="conv1"
        )(x)
        x = self.activation_fn(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(
            features=16, kernel_size=(5, 5), strides=(1, 1), padding=0, name="conv2"
        )(x)
        x = self.activation_fn(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120, use_bias=self.use_bias, name="fc1")(x)
        x = self.activation_fn(x)
        x = nn.Dense(features=84, use_bias=self.use_bias, name="fc2")(x)
        x = self.activation_fn(x)
        x = nn.Dense(features=self.out_dim, use_bias=self.use_bias, name="fc3")(x)
        return x
