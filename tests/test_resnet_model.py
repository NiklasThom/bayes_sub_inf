"""Test ResNet model with subspace inference framework."""

import os
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import wandb
from subspace_inference.curve_optimizer.trainer.fineTuning_PtoC import (
    Config,
    DataSplits,
    _setup_training_env,
    run_training,
    run_evaluation,
)
from subspace_inference.curve_optimizer.models.ResNet import ResNetBlock


def create_toy_image_dataset(n_samples=1000):
    """Create a toy image classification dataset.

    Returns:
        x: (n_samples, 1, 28, 28) - grayscale images
        y: (n_samples,) - class labels (0-9)
    """
    key = random.PRNGKey(42)

    x = []
    y = []

    for class_id in range(10):
        n_per_class = n_samples // 10
        key, subkey = random.split(key)
        class_mean = random.normal(subkey, (1, 28, 28)) * 0.5 + class_id * 0.1
        key, subkey = random.split(key)
        images = class_mean + random.normal(subkey, (n_per_class, 1, 28, 28)) * 0.5
        x.append(images)
        y.extend([class_id] * n_per_class)

    x = jnp.concatenate(x, axis=0)
    y = jnp.array(y)

    key, subkey = random.split(key)
    perm = random.permutation(subkey, n_samples)
    x = x[perm]
    y = y[perm]

    return x, y


def test_resnet_model():
    """Test ResNet model with classification task."""
    wandb.init(project="resnet_test")

    x, y = create_toy_image_dataset(n_samples=1000)
    data = DataSplits(
        train_x=x[:500],
        train_y=y[:500],
        val_x=x[500:750],
        val_y=y[500:750],
        test_x=x[750:],
        test_y=y[750:],
    )

    config_dict = {
        "rng_seed": 42,
        "net_kwargs": {
            "model_type": "resnet",
            "num_classes": 10,
            "act_fn": lambda x: nn.relu(x),
            "block_class": ResNetBlock,
            "num_blocks": (2, 2, 2),
            "c_hidden": (16, 32, 64),
        },
        "train_hyper": {
            "batch_size": 10,
            "num_epochs": 2,
            "eval_every_n_batch": 5,
            "save_params": False,
            "smoke_test": False,
        },
        "model_params": {
            "num_curve_segment": 1,
            "SegDeg": 1,
            "Pretraining": False,
            "subspace_model": "category",
            "weight_decay": 1e-4,
            "curve_sampling_mode": "combined_noBMA",
            "filter_masks": [{"keys": ["Conv_0", "kernel"], "op": "all"}],
            "lora_params": None,
        },
        "optimizer_conf": {
            "name": "adam",
            "kwargs": {"learning_rate": 1e-3},
            "freeze_other_params": False,
        },
    }

    config = Config.from_dict(config_dict, data)

    rng_key = random.PRNGKey(0)
    env, params, rng_key = _setup_training_env(config, data, rng_key)

    print("Starting ResNet training...")
    rng_key, params, _ = run_training(
        rng_key, env, params, data, config, wandb, "resnet_"
    )

    print("Starting ResNet evaluation...")
    run_evaluation(
        rng_key=rng_key,
        env=env,
        params=params,
        data=data,
        config=config,
        logger=wandb,
    )

    print("ResNet model test successful!")
    wandb.finish()


if __name__ == "__main__":
    import flax.linen as nn

    test_resnet_model()
