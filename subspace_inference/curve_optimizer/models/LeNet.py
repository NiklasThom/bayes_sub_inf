from flax import linen as nn
import jax.numpy as jnp
from subspace_inference.curve_optimizer.models import register_model
import jax


@register_model("lenetti")
class LeNetti(nn.Module):
    """
    A super simple LeNet version.
    """

    activation_fn: str = "sigmoid"
    out_dim: int = 10
    use_bias: bool = True

    def _activate(self, x: jnp.ndarray) -> jnp.ndarray:
        # Calling the function explicitly through a standard class method
        if self.activation_fn == "sigmoid":
            return jax.nn.sigmoid(x)
        elif self.activation_fn == "tanh":
            return jax.nn.tanh(x)
        elif self.activation_fn == "relu":
            return jax.nn.relu(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_fn}")

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True):
        """
        Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        x = nn.Conv(
            features=1, kernel_size=(3, 3), strides=(1, 1), padding=2, name="conv1"
        )(x)
        x = self._activate(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=8, use_bias=self.use_bias, name="fc1")(x)
        x = self._activate(x)
        x = nn.Dense(features=8, use_bias=self.use_bias, name="fc2")(x)
        x = self._activate(x)
        x = nn.Dense(features=8, use_bias=self.use_bias, name="fc3")(x)
        x = self._activate(x)
        x = nn.Dense(features=self.out_dim, use_bias=self.use_bias, name="fc4")(x)
        return x


@register_model("lenet")
class LeNet(nn.Module):
    """
    Implementation of LeNet.
    """

    activation_fn: str = "sigmoid"
    out_dim: int = 10
    use_bias: bool = True

    def _activate(self, x: jnp.ndarray) -> jnp.ndarray:
        # Calling the function explicitly through a standard class method
        if self.activation_fn == "sigmoid":
            return jax.nn.sigmoid(x)
        elif self.activation_fn == "tanh":
            return jax.nn.tanh(x)
        elif self.activation_fn == "relu":
            return jax.nn.relu(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_fn}")

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True):
        """
        Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        # Note: Flax Conv expects (batch, h, w, c) but we receive (batch, c, h, w)
        # So we need to transpose
        x = x.transpose(0, 2, 3, 1)  # (batch, h, w, c)

        x = nn.Conv(
            features=6, kernel_size=(5, 5), strides=(1, 1), padding=2, name="conv1"
        )(x)
        x = self._activate(x)
        x = jnp.mean(
            x.reshape(x.shape[0], x.shape[1] // 2, 2, x.shape[2] // 2, 2, x.shape[3]),
            axis=(2, 4),
        )
        x = nn.Conv(
            features=16, kernel_size=(5, 5), strides=(1, 1), padding=0, name="conv2"
        )(x)
        x = self._activate(x)
        x = jnp.mean(
            x.reshape(x.shape[0], x.shape[1] // 2, 2, x.shape[2] // 2, 2, x.shape[3]),
            axis=(2, 4),
        )
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120, use_bias=self.use_bias, name="fc1")(x)
        x = self._activate(x)
        x = nn.Dense(features=84, use_bias=self.use_bias, name="fc2")(x)
        x = self._activate(x)
        x = nn.Dense(features=self.out_dim, use_bias=self.use_bias, name="fc3")(x)
        return x
