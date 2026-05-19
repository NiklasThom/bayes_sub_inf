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


def test_toy_pipeline():
    # Initialize wandb in offline mode for testing
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project="toy_test")

    # 1. Load data
    x, y, _ = load_toy_regression_dataset(n_samples=1000)
    data = DataSplits(
        train_x=x[:500],
        train_y=y[:500],
        val_x=x[500:750],
        val_y=y[500:750],
        test_x=x[750:],
        test_y=y[750:],
    )

    # 2. Mock configuration
    config_dict = {
        "rng_seed": 42,
        "net_kwargs": {
            "model_type": "mlp",
            "hidden_dim": 32,
            "out_dim": 1,
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

    config = Config.from_dict(config_dict, data)

    # 3. Setup training env
    rng_key = random.PRNGKey(0)
    env, params, rng_key = _setup_training_env(config, data, rng_key)

    # 4. Run training
    print("Starting toy training...")
    rng_key, params, _ = run_training(rng_key, env, params, data, config, wandb, "toy_")

    # 5. Run evaluation
    print("Starting toy evaluation...")
    run_evaluation(
        rng_key=rng_key,
        env=env,
        params=params,
        data=data,
        config=config,
        logger=wandb,
    )

    print("Toy pipeline verification successful!")

    wandb.finish()


if __name__ == "__main__":
    test_toy_pipeline()
