
"""Train a single-point Qwen2.5-0.5B LoRA baseline on AG News."""

import os
import argparse

import wandb
from jax import random

from subspace_inference.curve_optimizer.trainer.training_pipeline import (
    Config,
    train,
)
from subspace_inference.curve_optimizer.datasets.text_classification import (
    prepare_text_classification_splits,
)

WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "curve_optimizer")


def train_qwen_lora_baseline(
    batch_size=4,
    num_steps=500,
    smoke_test=False,
    seed=2,
):

    print("current path:", os.getcwd())

    config_dict = {
        "rng_seed": seed,
        "load_params_path": False,

        "data": {
            "dataset_path": (
                "tests/qwen_data/helper_scripts/data_store/datasets/"
                "text_classification/ag_news"
            ),
            "val_percentage": 0.1,
        },

        "net_kwargs": {
            "model_type": "qwen",
            "model_path": (
                "tests/qwen_data/helper_scripts/data_store/models/"
                "qwen2.5_0.5B_float16"
            ),
        },

        "train_hyper": {
            "batch_size": batch_size,
            "num_epochs": -1,
            "num_steps": 10 if smoke_test else num_steps,
            "eval_every_n_batch": 200,
            "temperature": [1.0],
            "dataset_sampling": {"minibatch": {}},
            "save_params": True,
            "smoke_test": smoke_test,
        },

        "model_params": {
            # Single-point / standard fine-tuning
            "n_samples_eval": 1,
            "num_curve_segment": 0,
            "SegDeg": 1,
            "Pretraining": True,
            "curve_sampling_mode": "noBMA",

            # LoRA classification model
            "subspace_model": "lora_category",

            "natural_parameterization": False,
            "weight_decay": 0.0,

            "filter_masks": [
                {
                    "keys": ["self_attn", "q_proj", "kernel"],
                    "op": "all",
                },
                {
                    "keys": ["self_attn", "v_proj", "kernel"],
                    "op": "all",
                },
                {
                    "keys": ["lm_head", "kernel"],
                    "op": "all",
                },
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
                    {
                        "keys": ["lm_head", "kernel"],
                        "op": "all",
                        "dims": [1, 0],
                    },
                ],
            },
        },

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
        name=f"agnews-qwen-lora-baseline-seed-{seed}",
        entity=WANDB_ENTITY,
        config=config_dict,
    )

    rng_key = random.PRNGKey(seed)

    data, unique_target_ids, rng_key = prepare_text_classification_splits(
        rng_key,
        dataset_path=config_dict["data"]["dataset_path"],
        val_percentage=config_dict["data"]["val_percentage"],
        batch_size=config_dict["train_hyper"]["batch_size"],
        smoke_test=config_dict["train_hyper"]["smoke_test"],
        logger=None,
    )

    config_dict["net_kwargs"]["target_token_ids"] = unique_target_ids

    config = Config.from_dict(
        config_dict,
        data,
    )

    env, params, config = train(
        logger,
        config,
        data,
    )

    print("AG News Qwen-LoRA baseline training successful!")

    wandb.finish()

    return env, params, config


if __name__ == "__main__":

    wandb.require("core")

    os.makedirs(
        "tmp_files",
        exist_ok=True,
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    train_qwen_lora_baseline(
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
