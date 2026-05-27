import jax
from jax import random
import wandb
from subspace_inference.curve_optimizer.trainer.training_pipeline import (
    Config,
    DataSplits,
    train,
)
from subspace_inference.curve_optimizer.datasets.toy_regression import (
    load_toy_regression_dataset,
)


def test_toy_pipeline():
    x, y, _ = load_toy_regression_dataset(n_samples=1000)
    x = (x - x.mean()) / x.std()
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
            "depth": 3,
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

    logger = wandb.init(project="toy_test", config=config_dict)

    print("Starting toy training...")
    config = Config.from_dict(config_dict, data)
    env, params, config = train(logger, config, data)

    print("Toy pipeline verification successful!")

    wandb.log({"test_key": 1.0})
    wandb.finish()


if __name__ == "__main__":
    test_toy_pipeline()
