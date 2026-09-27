import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from subspace_inference.curve_optimizer.trainer.training_pipeline import (
    Config,
    _setup_training_env,
    setup_metrics,
)

from subspace_inference.curve_optimizer.datasets.text_classification import (
    prepare_text_classification_splits,
)


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = "tmp_files/955nvvay_pretrained_params.npy"
SEED = 2
BATCH_SIZE = 4


config_dict = {
    "rng_seed": SEED,
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
        "batch_size": BATCH_SIZE,
        "num_epochs": -1,
        "num_steps": 500,
        "eval_every_n_batch": 200,
        "temperature": [1.0],
        "dataset_sampling": {"minibatch": {}},
        "save_params": False,
        "smoke_test": False,
    },

    "model_params": {
        "n_samples_eval": 1,
        "num_curve_segment": 0,
        "SegDeg": 1,
        "Pretraining": True,
        "curve_sampling_mode": "noBMA",
        "subspace_model": "lora_category",
        "natural_parameterization": False,
        "weight_decay": 0.0,

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


# ============================================================
# LOAD DATA
# ============================================================

rng_key = random.PRNGKey(SEED)

data, unique_target_ids, rng_key = prepare_text_classification_splits(
    rng_key,
    dataset_path=config_dict["data"]["dataset_path"],
    val_percentage=config_dict["data"]["val_percentage"],
    batch_size=BATCH_SIZE,
    smoke_test=False,
    logger=None,
)

config_dict["net_kwargs"]["target_token_ids"] = unique_target_ids

config = Config.from_dict(config_dict, data)


# ============================================================
# CREATE SAME SINGLE-POINT MODEL
# ============================================================

# Baseline was trained in pretraining / k=0 mode.
with config.as_pretrain() as pt_config:

    env, params, rng_key = _setup_training_env(
        config=pt_config,
        data=data,
        rng_key=rng_key,
    )

    # ========================================================
    # LOAD SAVED BEST TRAINABLE PARAMETERS
    # ========================================================

    saved = np.load(
        CHECKPOINT_PATH,
        allow_pickle=True,
    )

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Saved object shape:", saved.shape)

    # Only one fixed control point exists: cp_fix = [1]
    saved_trainable_params = saved[0]

    # Replace only the trainable LoRA/model parameters.
    params["params"] = jax.tree.map(
        lambda best, current, mask: best if mask else current,
        saved_trainable_params,
        params["params"],
        env.s_model.train_mask,
    )

    # Same evaluation setting as run_evaluation():
    # disable LoRA noise.
    params = env.s_model.set_lora_rho(
        rho_w=0.0,
        rho_s=0.0,
        params=params,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_x, val_y = data.get("val")

    val_metrics_fn = setup_metrics(
        env.s_model,
        max(pt_config.train_hyper.batch_size_eval, 1),
        val_x,
        val_y,
        n_samples=1,
        key_prefix="val_",
    )

    rng_key, val_key = random.split(rng_key)

    _, val_logits = val_metrics_fn(
        val_key,
        params,
    )

    val_metrics = env.s_model.evaluate(
        val_logits[0],
        val_y,
        key_prefix="val_",
        weights=False,
    )


    # ========================================================
    # TEST
    # ========================================================

    test_x, test_y = data.get("test")

    test_metrics_fn = setup_metrics(
        env.s_model,
        max(pt_config.train_hyper.batch_size_eval, 1),
        test_x,
        test_y,
        n_samples=1,
        key_prefix="test_",
    )

    rng_key, test_key = random.split(rng_key)

    _, test_logits = test_metrics_fn(
        test_key,
        params,
    )

    test_metrics = env.s_model.evaluate(
        test_logits[0],
        test_y,
        key_prefix="test_",
        weights=False,
    )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SAVED BEST CHECKPOINT - VALIDATION")
print("=" * 60)

for key, value in val_metrics.items():
    print(f"{key}: {float(value):.6f}")


print("\n" + "=" * 60)
print("SAVED BEST CHECKPOINT - TEST")
print("=" * 60)

for key, value in test_metrics.items():
    print(f"{key}: {float(value):.6f}")

