import os
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Any


def load_text_classification_dataset(
    dataset_path: str, run=None, test=False
) -> Tuple[Any, Any]:
    """
    Load text classification dataset from a directory containing .npz files.

    Args:
        dataset_path: Path to the dataset directory
        run: Optional wandb run object for downloading artifacts
        test: Whether to load the test set or train/val set

    Returns:
        Tuple of (x, y) where x is a dict and y is an array
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
