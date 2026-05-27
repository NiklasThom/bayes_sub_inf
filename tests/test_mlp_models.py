"""Test MLP models with subspace inference framework."""

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import wandb
from subspace_inference.curve_optimizer.trainer.training_pipeline import (
    Config,
    DataSplits,
    train,
)
from subspace_inference.curve_optimizer.datasets.toy_regression import (
    load_toy_regression_dataset,
)


def test_mlp_model():
    """Test standard MLP model with regression task."""

    x, y, _ = load_toy_regression_dataset(n_samples=1000)
    x = (
        x - x.mean()
    ) / x.std()  # Center and scale the data for better polynomial fitting
    y = (y - y.mean()) / y.std()
    shuffle_idx = jax.random.permutation(random.PRNGKey(0), x.shape[0])
    x = x[shuffle_idx]
    y = y[shuffle_idx]
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

    logger = wandb.init(project="mlp_test", config=config_dict)

    print("Starting MLP training...")

    # Create config with data
    config = Config.from_dict(config_dict, data)

    # Call generic train function
    env, params, config = train(logger, config, data)

    print("Starting MLP evaluation and plotting...")
    s_model = env.s_model

    # Create prediction grid
    t_space = jnp.linspace(0, 1, 100)
    x_lin = jnp.linspace(-3, 3, 100)[:, None]

    # Get predictions at all t values
    # Note: s_model.__call__ expects the inner params dict (params["params"])
    def predict_at_t(t_single):
        out, _ = s_model(
            params["params"], {}, t_single, x_lin, train=False, key=random.PRNGKey(0)
        )
        return out.squeeze(axis=-1)

    out = jax.vmap(predict_at_t)(t_space)  # (100, 100)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(t_space)
    for o, c in zip(out, colors):
        ax.plot(x_lin, o, color=c, alpha=0.3)
    ax.plot(x_lin, out.mean(axis=0), label="mean", c="red", linewidth=2, alpha=0.8)
    ax.plot(data.test_x, data.test_y, "o", label="test", alpha=0.5, markersize=4)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MLP Regression Curve")

    # Log to wandb
    wandb.log({"Regression Plot": wandb.Image(fig)})
    plt.close(fig)

    print("MLP model test successful!")
    wandb.finish()


def test_mlp_feature_model():
    """Test MLP feature model with polynomial features."""

    x, y, _ = load_toy_regression_dataset(n_samples=1000)
    x = (
        x - x.mean()
    ) / x.std()  # Center and scale the data for better polynomial fitting
    y = (y - y.mean()) / y.std()
    shuffle_idx = jax.random.permutation(random.PRNGKey(0), x.shape[0])
    x = x[shuffle_idx]
    y = y[shuffle_idx]
    data = DataSplits(
        train_x=x[:500],
        train_y=y[:500],
        val_x=x[500:750],
        val_y=y[500:750],
        test_x=x[750:],
        test_y=y[750:],
    )

    print("Data shapes:", data.train_x.shape, data.train_y.shape)

    config_dict = {
        "rng_seed": 42,
        "net_kwargs": {
            "model_type": "mlp_feature",
            "depth": 4,
            "width": 32,
        },
        "train_hyper": {
            "batch_size": 10,
            "num_epochs": 10,
            "eval_every_n_batch": 2,
            "save_params": False,
            "smoke_test": False,
        },
        "model_params": {
            "num_curve_segment": 1,
            "SegDeg": 2,
            "Pretraining": True,
            "subspace_model": "regression",
            "weight_decay": 1e-5,
            "curve_sampling_mode": "combined_noBMA",
            "filter_masks": [{"keys": ["kernel"], "op": "all"}],
            "lora_params": None,
        },
        "optimizer_conf": {
            "name": "adam",
            "kwargs": {"learning_rate": 1e-1},
            "freeze_other_params": False,
        },
    }

    logger = wandb.init(project="mlp_test", config=config_dict)

    print("Starting MLP Feature training...")
    # Create config with data
    config = Config.from_dict(config_dict, data)

    # Call generic train function
    env, params, config = train(logger, config, data)

    print("Starting MLP Feature evaluation and plotting...")
    s_model = env.s_model

    # Create prediction grid
    t_space = jnp.linspace(0, 1, 100)
    x_lin = jnp.linspace(x.min() - 0.5, x.max() + 0.5, 100)[:, None]

    # Get predictions at all t values
    # Note: s_model.__call__ expects the inner params dict (params["params"])
    def predict_at_t(t_single):
        out, _ = s_model(
            params["params"], {}, t_single, x_lin, train=False, key=random.PRNGKey(0)
        )
        return out.squeeze(axis=-1)

    out = jax.vmap(predict_at_t)(t_space)  # (100, 100)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(t_space)
    for o, c in zip(out, colors):
        ax.plot(x_lin, o, color=c, alpha=0.3)
    ax.plot(x_lin, out.mean(axis=0), label="mean", c="red", linewidth=2, alpha=0.8)
    ax.plot(data.test_x, data.test_y, "o", label="test", alpha=0.5, markersize=4)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MLP Feature Regression Curve")

    # Log to wandb
    wandb.log({"Regression Plot": wandb.Image(fig)})
    plt.close(fig)

    print("MLP Feature model test successful!")
    wandb.finish()


def test_mlp_single_model():
    """Test MLP single model with polynomial features."""

    x, y, _ = load_toy_regression_dataset(n_samples=1000)
    x = (
        x - x.mean()
    ) / x.std()  # Center and scale the data for better polynomial fitting
    y = (y - y.mean()) / y.std()
    shuffle_idx = jax.random.permutation(random.PRNGKey(0), x.shape[0])
    x = x[shuffle_idx]
    y = y[shuffle_idx]
    data = DataSplits(
        train_x=x[:500],
        train_y=y[:500],
        val_x=x[500:750],
        val_y=y[500:750],
        test_x=x[750:],
        test_y=y[750:],
    )

    print("Data shapes:", data.train_x.shape, data.train_y.shape)

    config_dict = {
        "rng_seed": 42,
        "net_kwargs": {
            "model_type": "mlp_feature",
            "depth": 4,
            "width": 32,
        },
        "train_hyper": {
            "batch_size": 10,
            "num_epochs": 10,
            "eval_every_n_batch": 2,
            "save_params": False,
            "smoke_test": False,
        },
        "model_params": {
            "num_curve_segment": 0,  # single MLP no curve
            "SegDeg": 1,  # single MLP no curve
            "Pretraining": True,
            "subspace_model": "regression",
            "weight_decay": 1e-5,
            "curve_sampling_mode": "combined_noBMA",
            "filter_masks": [{"keys": ["kernel"], "op": "all"}],
            "lora_params": None,
        },
        "optimizer_conf": {
            "name": "adam",
            "kwargs": {"learning_rate": 1e-1},
            "freeze_other_params": False,
        },
    }

    logger = wandb.init(project="mlp_test", config=config_dict)

    print("Starting MLP Feature training...")
    # Create config with data
    config = Config.from_dict(config_dict, data)

    # Call generic train function
    env, params, config = train(logger, config, data)

    print("Starting MLP Feature evaluation and plotting...")
    s_model = env.s_model

    # Create prediction grid
    t_space = jnp.linspace(0, 1, 100)
    x_lin = jnp.linspace(x.min() - 0.5, x.max() + 0.5, 100)[:, None]

    # Get predictions at all t values
    # Note: s_model.__call__ expects the inner params dict (params["params"])
    def predict_at_t(t_single):
        out, _ = s_model(
            params["params"], {}, t_single, x_lin, train=False, key=random.PRNGKey(0)
        )
        return out.squeeze(axis=-1)

    out = jax.vmap(predict_at_t)(t_space)  # (100, 100)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(t_space)
    for o, c in zip(out, colors):
        ax.plot(x_lin, o, color=c, alpha=0.3)
    ax.plot(x_lin, out.mean(axis=0), label="mean", c="red", linewidth=2, alpha=0.8)
    ax.plot(data.test_x, data.test_y, "o", label="test", alpha=0.5, markersize=4)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MLP Feature Regression Curve")

    # Log to wandb
    wandb.log({"Regression Plot": wandb.Image(fig)})
    plt.close(fig)

    print("MLP Feature model test successful!")
    wandb.finish()


def test_mlp_DE_model():
    """Test MLP DE model with polynomial features."""

    x, y, _ = load_toy_regression_dataset(n_samples=1000)
    x = (
        x - x.mean()
    ) / x.std()  # Center and scale the data for better polynomial fitting
    y = (y - y.mean()) / y.std()
    shuffle_idx = jax.random.permutation(random.PRNGKey(0), x.shape[0])
    x = x[shuffle_idx]
    y = y[shuffle_idx]
    data = DataSplits(
        train_x=x[:500],
        train_y=y[:500],
        val_x=x[500:750],
        val_y=y[500:750],
        test_x=x[750:],
        test_y=y[750:],
    )

    print("Data shapes:", data.train_x.shape, data.train_y.shape)

    config_dict = {
        "rng_seed": 42,
        "net_kwargs": {
            "model_type": "mlp_feature",
            "depth": 4,
            "width": 32,
        },
        "train_hyper": {
            "batch_size": 10,
            "num_epochs": 10,
            "eval_every_n_batch": 2,
            "save_params": False,
            "smoke_test": False,
        },
        "model_params": {
            "num_curve_segment": 0,  # DE no curve
            "SegDeg": 5,
            "Pretraining": True,
            "subspace_model": "regression",
            "weight_decay": 1e-5,
            "curve_sampling_mode": "combined_noBMA",
            "filter_masks": [{"keys": ["kernel"], "op": "all"}],
            "lora_params": None,
        },
        "optimizer_conf": {
            "name": "adam",
            "kwargs": {"learning_rate": 1e-1},
            "freeze_other_params": False,
        },
    }

    logger = wandb.init(project="mlp_test", config=config_dict)

    print("Starting MLP Feature training...")
    # Create config with data
    config = Config.from_dict(config_dict, data)

    # Call generic train function
    env, params, config = train(logger, config, data)
    # params is params['params'] of list

    print("MLP Feature model test successful!")
    wandb.finish()


if __name__ == "__main__":
    # test_mlp_model()
    test_mlp_feature_model()
    # test_mlp_DE_model()
