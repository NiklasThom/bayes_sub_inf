"""Test MLP models with subspace inference framework."""

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
from subspace_inference.curve_optimizer.datasets.toy_regression import (
    load_toy_regression_dataset,
)


def test_mlp_model():
    """Test standard MLP model with regression task."""

    x, y, _ = load_toy_regression_dataset(n_samples=1000)
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
            "model_type": "mlp",
            "depth": 2,
            "width": 32,
            "output_dim": 1,
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
            "subspace_model": "regression",
            "weight_decay": 1e-4,
            "curve_sampling_mode": "combined_noBMA",
            "filter_masks": [{"keys": ["Dense_0", "kernel"], "op": "all"}],
            "lora_params": None,
        },
        "optimizer_conf": {
            "name": "adam",
            "kwargs": {"learning_rate": 1e-3},
            "freeze_other_params": False,
        },
    }

    wandb.init(project="mlp_test", config=config_dict)
    config = Config.from_dict(config_dict, data)

    rng_key = random.PRNGKey(0)
    env, params, rng_key = _setup_training_env(config, data, rng_key)

    print("Starting MLP training...")
    rng_key, params, _ = run_training(rng_key, env, params, data, config, wandb, "mlp_")

    print("Starting MLP evaluation...")
    run_evaluation(
        rng_key=rng_key,
        env=env,
        params=params,
        data=data,
        config=config,
        logger=wandb,
    )

    print("MLP model test successful!")
    wandb.finish()


def test_mlp_feature_model():
    """Test MLP feature model with polynomial features."""

    x, y, _ = load_toy_regression_dataset(n_samples=1000)
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
            "model_type": "mlp_feature",
            "depth": 2,
            "width": 32,
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
            "subspace_model": "regression",
            "weight_decay": 1e-4,
            "curve_sampling_mode": "combined_noBMA",
            "filter_masks": [{"keys": ["Dense_0", "kernel"], "op": "all"}],
            "lora_params": None,
        },
        "optimizer_conf": {
            "name": "adam",
            "kwargs": {"learning_rate": 1e-3},
            "freeze_other_params": False,
        },
    }

    wandb.init(project="mlp_feature_test", config=config_dict)
    config = Config.from_dict(config_dict, data)

    rng_key = random.PRNGKey(0)
    env, params, rng_key = _setup_training_env(config, data, rng_key)

    print("Starting MLP Feature training...")
    rng_key, params, _ = run_training(
        rng_key, env, params, data, config, wandb, "mlp_feature_"
    )

    print("Starting MLP Feature evaluation...")
    run_evaluation(
        rng_key=rng_key,
        env=env,
        params=params,
        data=data,
        config=config,
        logger=wandb,
    )

    # costum evaluation

    print("MLP Feature model test successful!")
    wandb.finish()


if __name__ == "__main__":
    # test_mlp_model()
    test_mlp_feature_model()
