import jax
import jax.numpy as jnp
import numpy as np


def load_toy_regression_dataset(n_samples=1000, seed=42):
    key = jax.random.PRNGKey(seed)
    x_key, y_key = jax.random.split(key)

    x = jax.random.uniform(x_key, (n_samples, 1), minval=-5.0, maxval=5.0)
    # y = sin(x) + noise
    noise = jax.random.normal(y_key, (n_samples, 1)) * 0.1
    y = (jnp.sin(x) + noise).reshape(-1)

    return x, y, None  # third arg is unique_target_ids which is None for regression
