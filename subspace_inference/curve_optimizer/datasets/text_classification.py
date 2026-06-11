import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Any


def load_text_classification_dataset(
    dataset_path: str, run=None, test=False
) -> Tuple[Any, Any, Any]:
    """
    Load text classification dataset from a directory containing .npz files.

    Args:
        dataset_path: Path to the dataset directory
        run: Optional wandb run object for downloading artifacts
        test: Whether to load the test set or train/val set

    Returns:
        Tuple of (x, y, target_ids) where x is a dict and y is an array
    """
    if run:
        artifact = run.use_artifact(dataset_path, type="dataset")
        dataset_path = artifact.download()

    file_name = "test_data.npz" if test else "train_data.npz"
    full_path = os.path.join(dataset_path, file_name)

    with np.load(full_path) as data:
        input_ids = jnp.array(data["input_ids"])
        attention_mask = jnp.array(data["attention_mask"])
        labels = jnp.array(data["labels"])
        # target_id and all_target_ids might be needed for specific models
        # but for the generic pipeline we group them into x and y
        target_ids = jnp.array(data["target_ids"])

    x = {"input_ids": input_ids, "attention_mask": attention_mask}
    y = labels

    return x, y, target_ids


def prepare_text_classification_splits(
    rng_key, dataset_path, val_percentage, batch_size, smoke_test, logger
) -> Tuple[Any, Any, Any]:
    """Load data and prepare train / val / test splits.

    Args:
        rng_key: JAX random key used for the train/val shuffle permutation.
        dataset_path: Path or wandb artifact reference for the dataset.
        val_percentage: Fraction of training data to reserve for validation.
        batch_size: Used to batch-align the train split so every epoch
            contains whole batches.  When *smoke_test* is True the dataset
            is also truncated to ``3 * batch_size`` training samples.
        smoke_test: Truncate dataset to a small size for fast iteration.
        logger: wandb run used when resolving artifact dataset paths.

    Returns:
        ``(data, unique_target_ids, rng_key)``
    """
    # Load train and test data
    x, y, unique_target_ids = load_text_classification_dataset(dataset_path, run=logger)
    n_samples = len(jax.tree.leaves(y)[0])
    print(f"Dataset loaded: {n_samples} samples, {len(unique_target_ids)} classes")

    test_x, test_y, _ = load_text_classification_dataset(
        dataset_path, run=logger, test=True
    )

    print(f"Target token IDs: {unique_target_ids}")

    # Shuffle and split into train / val
    rng_key, split_key = random.split(rng_key)
    perm_idx = random.permutation(split_key, n_samples)
    ordered_x = jax.tree.map(lambda leaf: leaf[perm_idx], x)
    ordered_y = jax.tree.map(lambda leaf: leaf[perm_idx], y)

    n_train = int(
        np.floor((1.0 - val_percentage) * n_samples / batch_size) * batch_size
    )
    train_x = jax.tree.map(lambda leaf: leaf[:n_train], ordered_x)
    train_y = jax.tree.map(lambda leaf: leaf[:n_train], ordered_y)

    has_val = val_percentage > 0.0 and (n_samples - n_train) > 0
    val_x = jax.tree.map(lambda leaf: leaf[n_train:], ordered_x) if has_val else None
    val_y = jax.tree.map(lambda leaf: leaf[n_train:], ordered_y) if has_val else None

    # Smoke-test truncation: keep only a few batches so the run finishes fast
    if smoke_test:
        n_smoke = 3 * batch_size
        train_x = jax.tree.map(lambda leaf: leaf[:n_smoke], train_x)
        train_y = jax.tree.map(lambda leaf: leaf[:n_smoke], train_y)
        if has_val:
            val_x = jax.tree.map(lambda leaf: leaf[:batch_size], val_x)
            val_y = jax.tree.map(lambda leaf: leaf[:batch_size], val_y)
        test_x = jax.tree.map(lambda leaf: leaf[:n_smoke], test_x)
        test_y = jax.tree.map(lambda leaf: leaf[:n_smoke], test_y)
        print(f"Smoke test: dataset truncated to {n_smoke} train samples")

    if has_val:
        print("Validation data are used")

    n_train_actual = len(jax.tree.leaves(train_y)[0])
    n_val_actual = len(jax.tree.leaves(val_y)[0]) if has_val else 0
    print(
        f"Training data: {n_train_actual} samples, Validation data: {n_val_actual} samples"
    )

    from subspace_inference.curve_optimizer.trainer.training_pipeline import DataSplits

    data = DataSplits(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )

    return data, unique_target_ids, rng_key
