"""Test Qwen model with subspace inference framework."""

import os
import argparse
from jax import random
import wandb

from subspace_inference.curve_optimizer.trainer.training_pipeline import (
    Config,
    train,
)
from subspace_inference.curve_optimizer.datasets.text_classification import (
    prepare_text_classification_splits,
)

# Set environment variables
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"
# Central project prefix for artifacts/datasets
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "curve_optimizer")
WANDB_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else "curve_optimizer"


# 1. Add defaults to the function signature.
# Tip: Defaulting to smoke_test=True ensures pytest runs quickly!
def test_qwen_training(batch_size=4, smoke_test=True):
    """Main entry point for Qwen curve fine-tuning."""
    print("current path:", os.getcwd())

    config_dict = {
        # --- Run / experiment --------------------------------------------------
        "rng_seed": 2,
        "load_params_path": False,
        # --- Data --------------------------------------------------------------
        "data": {
            "dataset_path": "tests/qwen_data/helper_scripts/data_store/datasets/text_classification/ag_news",
            "val_percentage": 0.1,
        },
        # --- Base model --------------------------------------------------------------
        "net_kwargs": {
            "model_type": "qwen",
            "model_path": "tests/qwen_data/helper_scripts/data_store/models/qwen2.5_0.5B_float16",
        },
        # --- Training hyperparameters → TrainHyperparams ----------------------
        "train_hyper": {
            "batch_size": batch_size,  # Use the function argument
            "num_epochs": -1,
            "num_steps": 10 if smoke_test else args.num_steps,  # Use the function argument
            "eval_every_n_batch": 200,
            "temperature": [1.0],
            "dataset_sampling": {"minibatch": {}},
            "save_params": True,
            "smoke_test": smoke_test,  # Use the function argument
        },
        # --- Subspace model + LoRA architecture → ModelParams -----------------
        "model_params": {
            "n_samples_eval": 10,
            "num_curve_segment": 2,
            "SegDeg": 2,
            "Pretraining": False,
            "curve_sampling_mode": "combined_test",
            "subspace_model": "jsd_noise_sampling_dropout_category",
            "jitter_multiplier": 0.0,
            "natural_parameterization": False,
            "weight_decay": 0.0,
            "curve_parameterization": "bezier",
            "indepent_connected": False,
            "entropy_weight": 5.0,
            "target_jsd": 0.1,
            "noise_rate": 0.05,
            "filter_masks": [
                {"keys": ["self_attn", "q_proj", "kernel"], "op": "all"},
                {"keys": ["self_attn", "v_proj", "kernel"], "op": "all"},
                {"keys": ["lm_head", "kernel"], "op": "all"},
            ],
            "lora_params": {
                "use_lora": True,
                "r": 8,
                "lora_dtype": "float32",
                "lora_alpha": 16.0,
                "lora_mode": "A(t)eB(t)",
                "rho_scheduler_frequency": 50.0,
                "lora_rho": 0.25,
                "lora_rho_s": 0.0,
                "filter_masks": [
                    {
                        "keys": ["self_attn", "q_proj", "kernel"],
                        "op": "all",
                        "dims": [1, 0],
                    },
                    {
                        "keys": ["self_attn", "v_proj", "kernel"],
                        "op": "all",
                        "dims": [1, 0],
                    },
                    {"keys": ["lm_head", "kernel"], "op": "all", "dims": [1, 0]},
                ],
            },
        },
        # --- Optimizer ---------------------------------------------------------
        "optimizer_conf": {
            "name": "adamw",
            "kwargs": {
                "learning_rate": {
                    "name": "linear_onecycle_schedule",
                    "kwargs": {
                        "transition_steps": 10_000,
                        "peak_value": 1e-4,
                        "pct_start": 0.12,
                        "pct_final": 1.0,
                        "div_factor": 300.0,
                        "final_div_factor": 300.0,
                    },
                },
                "weight_decay": 0.001,
            },
            "freeze_other_params": True,
        },
    }

    logger = wandb.init(
        project=WANDB_PROJECT,
        name="agnews-qwen-lora",
        entity=WANDB_ENTITY,
        config=config_dict,
    )

    # Resolve model path from wandb if a local path is not given
    net_kwargs = dict(config_dict["net_kwargs"])
    if net_kwargs["model_path"].startswith("ddold/"):
        model_artifact = logger.use_artifact(net_kwargs["model_path"], type="model")
        net_kwargs["model_path"] = model_artifact.download()
    config_dict["net_kwargs"] = net_kwargs

    rng_key = random.PRNGKey(config_dict["rng_seed"])

    data, unique_target_ids, rng_key = prepare_text_classification_splits(
        rng_key,
        dataset_path=config_dict["data"]["dataset_path"],
        val_percentage=config_dict["data"].get("val_percentage", 0.0),
        batch_size=config_dict["train_hyper"]["batch_size"],
        smoke_test=config_dict["train_hyper"].get("smoke_test", False),
        logger=None,
    )

    config_dict["net_kwargs"]["target_token_ids"] = unique_target_ids
    config = Config.from_dict(config_dict, data)

    # Call generic train function
    env, params, config = train(logger, config, data)

    print("AG News Qwen-LoRA training successful!")
    wandb.finish()


if __name__ == "__main__":
    # Add requirement for wandb core
    wandb.require("core")
    os.makedirs("tmp_files", exist_ok=True)

    # 2. Move argparse down here so pytest ignores it
    parser = argparse.ArgumentParser(
        description="Train a single model or a curve model."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="batch size for training. Default is 4.",
    )
    parser.add_argument("--num-steps", type=int, default=500)

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test with reduced epochs (2) and batch iterations (3).",
    )

    args = parser.parse_args()

    # Pass the parsed arguments into the function
    test_qwen_training(batch_size=args.batch_size, smoke_test=args.smoke_test)
