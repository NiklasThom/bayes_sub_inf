#!/usr/bin/env python3
"""
Save multiple classification datasets as numpy arrays for Time Transformer framework.
This script processes various datasets and saves them in compressed numpy format.

Datasets supported:
- boolq: Boolean Questions (2-class)
- winogrande_s: Winogrande Small (2-class)
- winogrande_m: Winogrande Medium (2-class)
- ARC-Easy: AI2 Reasoning Challenge Easy (5-class)
- ARC-Challenge: AI2 Reasoning Challenge Challenge (5-class)
- obqa: OpenBookQA (4-class)
- MMLU_chemistry: MMLU Chemistry (4-class)
- MMLU_physics: MMLU Physics (4-class)
"""

import importlib
import inspect
import os
import numpy as np
from argparse import Namespace
from accelerate import Accelerator
from tqdm import tqdm
import torch
import jax
import jax.numpy as jnp
from transformers import AutoModelForCausalLM
import sys

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from subspace_inference.curve_optimizer.models.qwen_jax import (
    QwenForCausalLM,
    create_qwen_config_from_pytorch,
    convert_pytorch_weights_to_jax,
    save_params_and_config,
)
import wandb


def get_all_dataset_names(script_dir):
    """Get all available dataset names from the dataset directory."""
    # Look for dataset directory in parent directory (tests/qwen_data/dataset/)
    dataset_parent_dir = os.path.dirname(script_dir)  # tests/qwen_data/
    dataset_dir = os.path.join(dataset_parent_dir, "dataset")
    if not os.path.exists(dataset_dir):
        print(f"Warning: Dataset directory not found at {dataset_dir}")
        return []
    return [
        dataset.split(".")[0]
        for dataset in os.listdir(dataset_dir)
        if not dataset.find("__") > -1 and "py" in dataset
    ]


def get_all_datasets(script_dir):
    """Load all dataset classes dynamically."""
    datasets = {}
    dataset_parent_dir = os.path.dirname(script_dir)  # tests/qwen_data/
    for dataset_name in get_all_dataset_names(script_dir):
        # Add parent directory so 'dataset' package can be imported
        sys.path.insert(0, dataset_parent_dir)
        try:
            mod = importlib.import_module(f"dataset.{dataset_name}")
            dataset_classes_name = [
                x
                for x in mod.__dir__()
                if "type" in str(type(getattr(mod, x)))
                and "DatasetBase" in str(inspect.getmro(getattr(mod, x))[1:])
            ]
            for d in dataset_classes_name:
                c = getattr(mod, d)
                datasets[c.NAME] = c
        finally:
            sys.path.remove(dataset_parent_dir)
    return datasets


def save_dataset_as_numpy(
    dataset_name,
    max_seq_len,
    model_name="Qwen/Qwen2.5-0.5B",
    batch_size=1,
    output_dir=None,
    save_wandb_artifact=False,
    wandb_project=None,
    script_dir=None,
):
    """
    Save a single dataset as numpy arrays.

    Args:
        dataset_name: Name of the dataset (e.g., 'boolq', 'winogrande_s')
        model_name: Hugging Face model name
        max_seq_len: Maximum sequence length
        batch_size: Batch size for processing
        output_dir: Base directory to save datasets (default: script_dir/data_store/datasets)
        save_wandb_artifact: Whether to save as wandb artifact
        wandb_project: wandb project name (required if save_wandb_artifact=True)
        script_dir: Directory containing the dataset folder (default: script directory)
    """
    if output_dir is None:
        if script_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "data_store", "datasets")

    print(f"\n{'=' * 60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")

    # Create output directory
    output_path = f"{output_dir}/text_classification/{dataset_name}/"
    os.makedirs(output_path, exist_ok=True)

    # Check if files already exist
    train_file_path = f"{output_path}train_data.npz"
    test_file_path = f"{output_path}test_data.npz"

    train_exists = os.path.exists(train_file_path)
    test_exists = os.path.exists(test_file_path)

    if (train_exists and test_exists) or (
        dataset_name.startswith("MMLU") and test_exists
    ):
        print(f"⏭️  Skipping {dataset_name} - files already exist:")
        print(f"   📄 {train_file_path}")
        print(f"   📄 {test_file_path}")

        # Still upload to wandb if requested
        if save_wandb_artifact and wandb_project:
            try:
                print("📤 Uploading existing files to wandb artifact...")
                run = wandb.init(
                    project=wandb_project,
                    name=f"dataset_{dataset_name}",
                    job_type="dataset",
                )
                artifact = wandb.Artifact(
                    name=f"{dataset_name}_dataset",
                    type="dataset",
                    description=f"Text classification dataset: {dataset_name}",
                )
                if not dataset_name.startswith("MMLU"):
                    artifact.add_file(train_file_path, name="train_data.npz")
                artifact.add_file(test_file_path, name="test_data.npz")
                run.log_artifact(artifact)
                run.finish()
                print(f"✅ Uploaded {dataset_name} to wandb project: {wandb_project}")
            except Exception as e:
                print(f"❌ Failed to upload to wandb: {e}")
        return True

    if not train_exists:
        print(f"📝 Train file missing: {train_file_path}")
    if not test_exists:
        print(f"📝 Test file missing: {test_file_path}")

    # Initialize accelerator
    accelerator = Accelerator(cpu=True)

    # Configure arguments
    args = Namespace()
    args.model = model_name
    args.dataset = dataset_name
    args.add_space = False
    args.max_seq_len = max_seq_len
    args.batch_size = batch_size
    args.is_s2s = False  # Qwen2 is decoder-only
    args.testing_set = "val"

    try:
        # Create dataset
        print("Creating dataset...")
        if script_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        all_datasets = get_all_datasets(script_dir)
        if "mcdataset" not in all_datasets:
            raise ValueError(
                f"mcdataset not found in available datasets: {list(all_datasets.keys())}"
            )
        dataset = all_datasets["mcdataset"](accelerator, args)
        print(f"✅ Successfully created dataset with {dataset.num_labels} labels")

        # Get dataloaders
        dataset.get_loaders()
        print(f"📊 Training samples: {dataset.num_samples}")

        # Process training data
        if not args.dataset.startswith("MMLU") and not train_exists:
            print("Processing training data...")
            train_tokens = {
                "input_ids": None,
                "attention_mask": None,
                "labels": None,
                "target_id": None,
            }
            first = True

            for input_batch, label_batch, target_id_batch in tqdm(
                dataset.train_dataloader, desc="Train batches"
            ):
                if first:
                    first = False
                    train_tokens["input_ids"] = (
                        input_batch["input_ids"].detach().cpu().numpy()
                    )
                    train_tokens["attention_mask"] = (
                        input_batch["attention_mask"].detach().cpu().numpy()
                    )
                    train_tokens["labels"] = label_batch.detach().cpu().numpy()
                    train_tokens["target_id"] = target_id_batch.detach().cpu().numpy()
                else:
                    train_tokens["input_ids"] = np.concatenate(
                        (
                            train_tokens["input_ids"],
                            input_batch["input_ids"].detach().cpu().numpy(),
                        ),
                        axis=0,
                    )
                    train_tokens["attention_mask"] = np.concatenate(
                        (
                            train_tokens["attention_mask"],
                            input_batch["attention_mask"].detach().cpu().numpy(),
                        ),
                        axis=0,
                    )
                    train_tokens["labels"] = np.concatenate(
                        (train_tokens["labels"], label_batch.detach().cpu().numpy()),
                        axis=0,
                    )
                    train_tokens["target_id"] = np.concatenate(
                        (
                            train_tokens["target_id"],
                            target_id_batch.detach().cpu().numpy(),
                        ),
                        axis=0,
                    )
            print(
                f"✅ Finished processing training data with shape: {train_tokens['input_ids'].shape}"
            )
            np.savez_compressed(
                train_file_path, target_ids=dataset.target_ids, **train_tokens
            )

        # Process test data
        if not test_exists:
            print("Processing test data...")
            test_tokens = {
                "input_ids": None,
                "attention_mask": None,
                "labels": None,
                "target_id": None,
            }
            first = True

            for input_batch, label_batch, target_id_batch in tqdm(
                dataset.test_dataloader, desc="Test batches"
            ):
                if first:
                    first = False
                    test_tokens["input_ids"] = (
                        input_batch["input_ids"].detach().cpu().numpy()
                    )
                    test_tokens["attention_mask"] = (
                        input_batch["attention_mask"].detach().cpu().numpy()
                    )
                    test_tokens["labels"] = label_batch.detach().cpu().numpy()
                    test_tokens["target_id"] = target_id_batch.detach().cpu().numpy()
                else:
                    test_tokens["input_ids"] = np.concatenate(
                        (
                            test_tokens["input_ids"],
                            input_batch["input_ids"].detach().cpu().numpy(),
                        ),
                        axis=0,
                    )
                    test_tokens["attention_mask"] = np.concatenate(
                        (
                            test_tokens["attention_mask"],
                            input_batch["attention_mask"].detach().cpu().numpy(),
                        ),
                        axis=0,
                    )
                    test_tokens["labels"] = np.concatenate(
                        (test_tokens["labels"], label_batch.detach().cpu().numpy()),
                        axis=0,
                    )
                    test_tokens["target_id"] = np.concatenate(
                        (
                            test_tokens["target_id"],
                            target_id_batch.detach().cpu().numpy(),
                        ),
                        axis=0,
                    )

            # Save files
            print(
                f"✅ Finished processing test data with shape: {test_tokens['input_ids'].shape}"
            )
            print(f"Saving to {output_path}...")
            np.savez_compressed(
                test_file_path, target_ids=dataset.target_ids, **test_tokens
            )

        # Upload to wandb if requested
        if save_wandb_artifact and wandb_project:
            try:
                print("📤 Uploading to wandb artifact...")
                run = wandb.init(
                    project=wandb_project,
                    name=f"dataset_{dataset_name}",
                    job_type="dataset",
                )
                artifact = wandb.Artifact(
                    name=f"{dataset_name}_dataset",
                    type="dataset",
                    description=f"Text classification dataset: {dataset_name}",
                    metadata={
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "max_seq_len": max_seq_len,
                        "num_labels": dataset.num_labels,
                        "target_ids": dataset.target_ids.squeeze().tolist(),
                    },
                )
                if not dataset_name.startswith("MMLU"):
                    artifact.add_file(train_file_path, name="train_data.npz")
                artifact.add_file(test_file_path, name="test_data.npz")
                run.log_artifact(artifact)
                run.finish()
                print(f"✅ Uploaded {dataset_name} to wandb project: {wandb_project}")
            except Exception as e:
                print(f"❌ Failed to upload to wandb: {e}")

        # Print summary
        print(f"✅ Successfully saved {dataset_name}")
        if not args.dataset.startswith("MMLU"):
            print(f"   Training data shape: {train_tokens['input_ids'].shape}")
        print(f"   Test data shape: {test_tokens['input_ids'].shape}")
        print(f"   Number of labels: {dataset.num_labels}")
        print(f"   Target tokens: {dataset.target_ids.squeeze().tolist()}")
        print(f"   Files saved to: {output_path}")

        return True

    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def save_qwen_jax_parameters(
    model_name="Qwen/Qwen2.5-0.5B",
    output_dir=None,
    dtype: str = "float16",
    save_wandb_artifact=False,
    wandb_project=None,
    script_dir=None,
):
    """
    Convert Qwen2.5 PyTorch model to JAX parameters and save them.

    Args:
        model_name: Hugging Face model name
            "Qwen/Qwen2.5-0.5B",   # 0.5 billion parameters
            "Qwen/Qwen2.5-1.5B",   # 1.5 billion parameters
            "Qwen/Qwen2.5-3B",     # 3 billion parameters
            "Qwen/Qwen2.5-7B",     # 7 billion parameters
            "Qwen/Qwen2.5-14B",    # 14 billion parameters
            "Qwen/Qwen2.5-32B",    # 32 billion parameters (large)
            "Qwen/Qwen2.5-72B",    # 72 billion parameters (very large)
        output_dir: Directory to save JAX parameters (default: script_dir/data_store/models)
        dtype: Data type for conversion (float16 or float32)
        save_wandb_artifact: Whether to save as wandb artifact
        wandb_project: wandb project name (required if save_wandb_artifact=True)
        script_dir: Script directory for default output path (default: script directory)

    Returns:
        bool: Success status
    """
    if output_dir is None:
        if script_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "data_store", "models")

    print(f"\n{'=' * 80}")
    print(f"Converting Qwen2.5 Model to JAX: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}")

    # Create output directory and build save path
    model_size = model_name.split("/")[-1].replace("Qwen2.5-", "")
    # include dtype in saved path to distinguish parameter precision
    save_path = os.path.join(output_dir, f"qwen2.5_{model_size}_{dtype}")
    os.makedirs(save_path, exist_ok=True)

    # Check if files already exist
    params_file = os.path.join(save_path, "jax_params.npy")
    config_file = os.path.join(save_path, "jax_config.json")

    if os.path.exists(params_file) and os.path.exists(config_file):
        print(f"⏭️  Skipping {model_name} - files already exist:")
        print(f"   📄 {params_file}")
        print(f"   📄 {config_file}")

        # Still upload to wandb if requested
        if save_wandb_artifact and wandb_project:
            try:
                print("📤 Uploading existing model to wandb artifact...")
                run = wandb.init(
                    project=wandb_project, name=f"qwen_{model_size}_{dtype}"
                )
                artifact = wandb.Artifact(
                    name=f"qwen2.5_{model_size}_{dtype}",
                    type="model",
                    description=f"Qwen2.5 {model_size} model converted to JAX ({dtype})",
                    metadata={
                        "model_name": model_name,
                        "model_size": model_size,
                        "dtype": dtype,
                    },
                )
                artifact.add_file(params_file, name="jax_params.npy")
                artifact.add_file(config_file, name="jax_config.json")
                run.log_artifact(artifact)
                run.finish()
                print(f"✅ Uploaded {model_name} to wandb project: {wandb_project}")
            except Exception as e:
                print(f"❌ Failed to upload to wandb: {e}")
        return True

    print(f"📝 Converting {model_name} (missing files in {save_path})")

    try:
        # Device configuration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Model loading configuration
        # Allow choosing conversion dtype (float16 or float32)
        model_config = {
            "device_map": "cpu",
            "torch_dtype": torch.float32,
            "trust_remote_code": False,
        }

        # Load PyTorch model
        print(f"Loading PyTorch model: {model_name}")
        pytorch_model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
        pytorch_model.eval()

        # Get parameter count
        param_count = sum(p.numel() for p in pytorch_model.parameters())
        print(f"Model parameters: {param_count:,}")

        # Create JAX model configuration from PyTorch config
        print("Creating JAX model configuration...")
        jax_config = create_qwen_config_from_pytorch(pytorch_model.config)
        if dtype == "float16":
            jax_dtype = jnp.float16
        elif dtype == "float32":
            jax_dtype = jnp.float32
        elif dtype == "bfloat16":
            jax_dtype = jnp.bfloat16
        else:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Choose from 'float16', 'float32', or 'bfloat16'."
            )

        print("JAX model configuration:")
        for key, value in jax_config.items():
            print(f"   {key}: {value}")

        # Initialize JAX model
        print("Initializing JAX model...")
        jax_model = QwenForCausalLM(**jax_config)

        # Initialize JAX parameters with dummy input
        rng = jax.random.PRNGKey(42)
        dummy_input_ids = jnp.ones((1, 10), dtype=jnp.int32)
        jax_params = jax_model.init(rng, dummy_input_ids)

        # Convert and transfer weights from PyTorch to JAX
        print("Converting PyTorch weights to JAX format...")
        jax_params = convert_pytorch_weights_to_jax(
            jax_params, pytorch_model, jax_config, jax_dtype
        )

        # Count JAX parameters
        jax_param_count = sum(p.size for p in jax.tree.leaves(jax_params))
        jax_memory_bytes = sum(p.nbytes for p in jax.tree.leaves(jax_params))
        print(f"JAX model parameters: {jax_param_count:,}")
        print(f"Parameter count match: {param_count == jax_param_count}")
        print(f"Parameter memory: {jax_memory_bytes / (1024**3):.2f} GB ({dtype})")

        # Save JAX parameters and config
        save_params_and_config(jax_params, jax_config, save_path)

        # Upload to wandb if requested
        if save_wandb_artifact and wandb_project:
            try:
                print("📤 Uploading model to wandb artifact...")
                run = wandb.init(
                    project=wandb_project, name=f"qwen_{model_size}_{dtype}"
                )
                artifact = wandb.Artifact(
                    name=f"qwen2.5_{model_size}_{dtype}",
                    type="model",
                    description=f"Qwen2.5 {model_size} model converted to JAX ({dtype})",
                    metadata={
                        "model_name": model_name,
                        "model_size": model_size,
                        "dtype": dtype,
                        "param_count": jax_param_count,
                        "jax_config": jax_config,
                    },
                )
                artifact.add_file(params_file, name="jax_params.npy")
                artifact.add_file(config_file, name="jax_config.json")
                run.log_artifact(artifact)
                run.finish()
                print(f"✅ Uploaded {model_name} to wandb project: {wandb_project}")
            except Exception as e:
                print(f"❌ Failed to upload to wandb: {e}")

        # # Verify with a simple forward pass (optional)
        # print("Verifying model with forward pass...")
        # test_input = jnp.array([[1, 2, 3, 4, 5]])
        # jax_output = jax_model.apply(jax_params, test_input, train=False)
        # print(f"Test output shape: {jax_output.shape}")
        # print(f"Expected shape: (1, 5, {jax_config['vocab_size']})")

        # Clean up PyTorch model to free memory
        del pytorch_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"✅ Successfully converted and saved {model_name} to {save_path}")
        return True

    except Exception as e:
        print(f"❌ Error converting {model_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def save_all_qwen_models(
    output_dir=None, save_wandb_artifact=False, wandb_project=None, script_dir=None
):
    """
    Save JAX parameters for multiple Qwen2.5 model sizes.

    Args:
        output_dir: Directory to save all model exports (default: script_dir/data_store/models)
        save_wandb_artifact: Whether to save as wandb artifact
        wandb_project: wandb project name for artifacts
        script_dir: Script directory for default output path (default: script directory)
    """
    if output_dir is None:
        if script_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "data_store", "models")
    # Available Qwen2.5 model sizes
    qwen_models = [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-72B",
    ]

    print("🚀 Starting Qwen2.5 Model Conversion to JAX")
    print(f"Models to convert: {len(qwen_models)}")
    print(f"Output directory: {output_dir}")

    success_count = 0
    for model_name in qwen_models:
        print(f"\n📦 Processing: {model_name}")

        # Skip very large models if not enough memory
        model_size = model_name.split("-")[-1]
        if model_size in ["32B", "72B"]:
            print(f"⚠️  Skipping {model_size} model (requires significant memory)")
            continue

        success = save_qwen_jax_parameters(
            model_name,
            output_dir,
            save_wandb_artifact=save_wandb_artifact,
            wandb_project=wandb_project,
        )
        if success:
            success_count += 1

        # Force garbage collection between models
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    print("🎉 QWEN2.5 CONVERSION COMPLETE")
    print(f"✅ Successfully converted: {success_count} models")
    print(f"📁 Models saved to: {output_dir}")
    print(f"{'=' * 80}")


def main():
    """Main function to process datasets and Qwen models."""
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Save datasets and Qwen2.5 models")
    parser.add_argument(
        "--mode",
        choices=["datasets", "qwen", "all"],
        default="all",
        help="What to save: datasets only, qwen models only, or all",
    )
    parser.add_argument(
        "--qwen-models",
        nargs="+",
        default=[
            "Qwen/Qwen2.5-0.5B",
        ],
        help="Specific Qwen models to convert",
    )
    parser.add_argument(
        "--dataset-output-dir",
        default=None,
        help="Output directory for datasets (default: script_dir/data_store/datasets)",
    )
    parser.add_argument(
        "--model-output-dir",
        default=None,
        help="Output directory for Qwen models (default: script_dir/data_store/models)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Data type for model conversion (float16 or float32)",
    )
    parser.add_argument(
        "--upload-to-wandb",
        action="store_true",
        help="Save datasets and models as wandb artifacts",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="qwen_PtoC",
        help="wandb project name for artifacts (required if --save-wandb-artifact is used)",
    )

    args = parser.parse_args()

    if args.mode in ["datasets", "all"]:
        # Dataset configurations
        datasets_config = {
            "ag_news": {
                "name": "ag_news",
                "description": "AG News (4-class text classification)",
                "max_seq_len": 128,
            },
            "boolq": {
                "name": "boolq",
                "description": "Boolean Questions (True/False)",
                "max_seq_len": 525,
            },
            "winogrande_s": {
                "name": "winogrande_s",
                "description": "Winogrande Small (pronoun resolution)",
                "max_seq_len": 64,
            },
            "winogrande_m": {
                "name": "winogrande_m",
                "description": "Winogrande Medium (pronoun resolution)",
                "max_seq_len": 69,
            },
            "ARC-Easy": {
                "name": "ARC-Easy",
                "description": "AI2 Reasoning Challenge Easy (science questions)",
                "max_seq_len": 194,
            },
            "ARC-Challenge": {
                "name": "ARC-Challenge",
                "description": "AI2 Reasoning Challenge Challenge (science questions)",
                "max_seq_len": 201,
            },
            "obqa": {
                "name": "obqa",
                "description": "OpenBookQA (elementary science)",
                "max_seq_len": 127,
            },
            "MMLU_chem": {
                "name": "MMLU_chem",
                "description": "MMLU Chemistry (multiple choice)",
                "max_seq_len": 239,
            },
            "MMLU_phy": {
                "name": "MMLU_phy",
                "description": "MMLU Physics (multiple choice)",
                "max_seq_len": 153,
            },
        }

        print("🚀 Starting dataset processing for Time Transformer framework")
        print(f"Total datasets to process: {len(datasets_config)}")

        # Force JAX to use CPU for dataset processing

        # Process each dataset
        success_count = 0
        for dataset_key, config in datasets_config.items():
            print(f"\n📝 {config['description']}")
            success = save_dataset_as_numpy(
                dataset_name=config["name"],
                model_name="Qwen/Qwen2.5-0.5B",
                max_seq_len=config["max_seq_len"],
                batch_size=1,
                output_dir=args.dataset_output_dir,
                save_wandb_artifact=args.upload_to_wandb,
                wandb_project=args.wandb_project,
                script_dir=script_dir,
            )
            if success:
                success_count += 1

        print(f"\n{'=' * 60}")
        print("🎉 DATASET PROCESSING COMPLETE")
        print(
            f"✅ Successfully processed: {success_count}/{len(datasets_config)} datasets"
        )
        output_dir = (
            args.dataset_output_dir
            if args.dataset_output_dir
            else os.path.join(script_dir, "data_store", "datasets")
        )
        print(f"📁 All datasets saved to: {output_dir}/text_classification/")
        print(f"{'=' * 60}")

        # List all created directories
        print("\n📂 Created dataset directories:")
        base_path = f"{output_dir}/text_classification/"
        if os.path.exists(base_path):
            for item in sorted(os.listdir(base_path)):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    train_file = os.path.join(item_path, "train_data.npz")
                    test_file = os.path.join(item_path, "test_data.npz")
                    if (
                        True if "MMLU" in item_path else os.path.exists(train_file)
                    ) and os.path.exists(test_file):
                        print(f"   ✅ {item}/")
                    else:
                        print(f"   ⚠️  {item}/ (incomplete)")

    if args.mode in ["qwen", "all"]:
        print("\n" + "=" * 80)
        print("🤖 STARTING QWEN2.5 MODEL CONVERSION")
        print("=" * 80)

        # Convert specified Qwen models
        qwen_success_count = 0
        for model_name in args.qwen_models:
            print(f"\n🔄 Converting: {model_name}")
            success = save_qwen_jax_parameters(
                model_name,
                output_dir=args.model_output_dir,
                dtype=args.dtype,
                save_wandb_artifact=args.upload_to_wandb,
                wandb_project=args.wandb_project,
                script_dir=script_dir,
            )
            if success:
                qwen_success_count += 1

        print(f"\n{'=' * 80}")
        print("🎉 QWEN2.5 CONVERSION COMPLETE")
        print(
            f"✅ Successfully converted: {qwen_success_count}/{len(args.qwen_models)} models"
        )
        output_dir = (
            args.model_output_dir
            if args.model_output_dir
            else os.path.join(script_dir, "data_store", "models")
        )
        print(f"📁 Models saved to: {output_dir}/")
        print(f"{'=' * 80}")

        # List converted models
        print("\n🤖 Converted Qwen models:")
        model_exports_path = f"{output_dir}/"
        if os.path.exists(model_exports_path):
            for item in sorted(os.listdir(model_exports_path)):
                item_path = os.path.join(model_exports_path, item)
                if os.path.isdir(item_path) and item.startswith("qwen"):
                    params_file = os.path.join(item_path, "jax_params.npy")
                    config_file = os.path.join(item_path, "jax_config.json")
                    if os.path.exists(params_file) and os.path.exists(config_file):
                        print(f"   ✅ {item}/")
                    else:
                        print(f"   ⚠️  {item}/ (incomplete)")


if __name__ == "__main__":
    main()
