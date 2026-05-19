"""Base Deep Learning Model Configuration Class."""

from dataclasses import dataclass, field
from enum import Enum
from flax import linen as nn


class Activation(Enum):
    """Activation Function."""

    SIGMOID = 'sigmoid'
    RELU = 'relu'
    GELU = 'gelu'
    TANH = 'tanh'
    SOFTMAX = 'softmax'
    LEAKY_RELU = 'leaky_relu'

    @property
    def flax_activation(self):
        """Get the Flax Activation Function."""
        return getattr(nn, self.value)


@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    """Base Model representing Deep Learning Model Configuration."""

    model: str = field(
        metadata={
            'description': (
                'The LiteralString must match actual class name'
                'implementing the model!'
            )
        },
    )

    @classmethod
    def get_name_mapping(cls):
        """Get the mapping of model names to the model classes."""
        return {c.model: c for c in cls.get_all_subclasses()}
    

