from flax import linen as nn
import jax.numpy as jnp
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LeNetConfig(ModelConfig):
    """FCN Model Configuration."""

    model: str = 'LeNet'
    activation: Activation = field(
        default=Activation.SIGMOID,
        metadata={'description': 'Activation func. for hidden layers'},
    )
    out_dim: int = field(
        default=10,
        metadata={'description': 'Output dimension of the model'},
    )
    use_bias: bool = field(
        default=True,
        metadata={'description': 'Whether to include bias terms'},
    )


@dataclass(frozen=True)
class LeNettiConfig(ModelConfig):
    """FCN Model Configuration."""

    model: str = 'LeNetti'
    activation: Activation = field(
        default=Activation.SIGMOID,
        metadata={'description': 'Activation func. for hidden layers'},
    )
    out_dim: int = field(
        default=10,
        metadata={'description': 'Output dimension of the model'},
    )
    use_bias: bool = field(
        default=True,
        metadata={'description': 'Whether to include bias terms'},
    )


class LeNetti(nn.Module):
    """
    A super simple LeNet version.

    Args:
        config (LeNettiConfig): The configuration for the model.
    """

    config: LeNettiConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        activation = self.config.activation.flax_activation
        # x = x.transpose((0, 2, 3, 1))
        x = nn.Conv(
            features=1, kernel_size=(3, 3), strides=(1, 1), padding=2, name='conv1'
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=8, use_bias=self.config.use_bias, name='fc1')(x)
        x = activation(x)
        x = nn.Dense(features=8, use_bias=self.config.use_bias, name='fc2')(x)
        x = activation(x)
        x = nn.Dense(features=8, use_bias=self.config.use_bias, name='fc3')(x)
        x = activation(x)
        x = nn.Dense(
            features=self.config.out_dim, use_bias=self.config.use_bias, name='fc4'
        )(x)
        return x
    

class LeNet(nn.Module):
    """
    Implementation of LeNet.

    Args:
        config (LeNetConfig): The configuration for the model.
    """

    config: LeNetConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        activation = self.config.activation.flax_activation
        # x = x.transpose((0, 2, 3, 1))
        x = nn.Conv(
            features=6, kernel_size=(5, 5), strides=(1, 1), padding=2, name='conv1'
        )(x)
        x = activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2),
                        strides=(2, 2), padding='VALID')
        x = nn.Conv(
            features=16, kernel_size=(5, 5), strides=(1, 1), padding=0, name='conv2'
        )(x)
        x = activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2),
                        strides=(2, 2), padding='VALID')
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=120, use_bias=self.config.use_bias, name='fc1')(x)
        x = activation(x)
        x = nn.Dense(features=84, use_bias=self.config.use_bias, name='fc2')(x)
        x = activation(x)
        x = nn.Dense(
            features=self.config.out_dim, use_bias=self.config.use_bias, name='fc3'
        )(x)
        return x