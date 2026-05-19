import jax
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn
from typing import Any, Dict


class MLP(nn.Module):
    hidden_dim: int = 64
    out_dim: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class ToyMLPWrapper:
    def __init__(self, hidden_dim: int = 64, out_dim: int = 1, **kwargs):
        self.model = MLP(hidden_dim=hidden_dim, out_dim=out_dim)

    def init(self, key, x):
        if key is None:
            # For cases where we just want the structure, like in PtoC init
            key = jax.random.PRNGKey(0)
        return self.model.init(key, x)

    def apply(self, params, x, train=False, **kwargs):
        # PtoC calls this via s_model, which expects (params, x, train=False, ...)
        return self.model.apply(params, x)

    def evaluate(self, logits, target, key_prefix="", average=True, weights=None):
        """
        logits shape: (n_samples, n_data, out_dim)
        target shape: (n_data, out_dim)
        """
        # For regression, we compute mean over samples to get ensemble prediction
        if weights is not None:
            # weights shape (n_samples,)
            ensemble_pred = jnp.sum(logits * weights[:, None, None], axis=0)
        else:
            ensemble_pred = jnp.mean(logits, axis=0)

        mse = jnp.mean(jnp.square(ensemble_pred - target))
        mae = jnp.mean(jnp.abs(ensemble_pred - target))

        # PtoC also expects 'loss' (which it uses for BMA weights)
        # In PtoC, loss is typically NLL. For regression, we can use MSE as a proxy for -log_prob.
        # PtoC uses -mean_loss * n_data to get unnormalized log_like.

        metrics = {
            f"{key_prefix}mse": mse,
            f"{key_prefix}mae": mae,
            f"{key_prefix}loss": mse,  # Used as NLL proxy
        }

        if average:
            return metrics
        else:
            # PtoC setup_metrics calls evaluate with average=False
            # to get per-sample metrics if needed.
            # However, MSE is usually already averaged over batch.
            # If we want per-t-sample metrics:
            per_sample_mse = jnp.mean(
                jnp.square(logits - target[None, ...]), axis=(1, 2)
            )
            metrics[f"{key_prefix}mean_loss"] = per_sample_mse
            metrics[f"{key_prefix}mean_acc"] = -per_sample_mse  # Proxy for acc
            metrics[f"{key_prefix}mean_ece"] = jnp.zeros_like(per_sample_mse)
            return metrics
