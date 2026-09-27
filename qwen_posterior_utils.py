"""Shared, method-agnostic building blocks for Bayesian posterior
approximation experiments (MILE, MFVI, Laplace, ...) on the
Qwen2.5-0.5B LoRA / AG News setup in bayes_sub_inf.

Extracted from 02_methods.ipynb so MILE, MFVI, and Laplace notebooks
share one implementation of: the balanced posterior subset, the
memory-efficient per-example NLL, LoRA vectorization/rebuild, the
log-posterior function, and the SubspaceBaseModel._predict lax.cond
fix. Method-specific code (MCLMC sampling, ELBO optimization, Fisher/
Hessian estimation) stays in each method's own notebook.

Usage (after env/params/data are loaded, e.g. via
evaluate_saved_baseline.py):

    import sys
    sys.path.insert(0, str(QWEN_REPO / "experiments" / "ag_news_qwen_lora"))
    import qwen_posterior_utils as qpu

    qpu.patch_subspace_curve_predict()

    posterior = qpu.setup_qwen_posterior(
        env=env,
        params=params,
        data=data,
        n_per_class=32,
        seq_len=32,
        posterior_key_seed=2027,
        prior_std=1.0,
    )

    posterior.theta_map            # (540672,) flat MAP LoRA vector
    posterior.qwen_log_posterior   # theta -> scalar log posterior
    posterior.rebuild_full_params  # theta -> full Qwen param pytree

Also includes posterior-predictive evaluation, predictive/uncertainty
metrics, ECE, and a common save-run format, extracted from MILE's
POSTERIOR PREDICTIVE PROBABILITIES / METRICS / EXPECTED CALIBRATION
ERROR / LOG POSTERIOR VALUES FOR ALL SAMPLES / SAVE RUN cells. These
are method-agnostic: they take a (n_samples, n_params) array of flat
LoRA vectors, wherever they came from (MCMC samples, VI draws,
Laplace draws), plus x_eval/y_eval (not hardcoded to the posterior
subset, so the same functions work for held-out test-set evaluation
later).
"""

import copy
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.flatten_util import ravel_pytree


# ============================================================
# FIX: bypass lax.cond in SubspaceBaseModel._predict for
# concrete (non-traced) train flags.
#
# _predict always routes through jax.lax.cond(train, train_fn,
# eval_fn, inputs), even when `train` is a concrete Python bool.
# train_fn calls the real Qwen model with train=True (dropout path),
# which we never use/validate for posterior sampling. Under certain
# tracing contexts (e.g. inside jax.lax.scan) `train` loses its
# concrete-bool status, so JAX builds/differentiates the unused
# train_fn branch too, and an inf produced there can leak into the
# gradient of the actually-selected eval_fn branch via lax.cond's
# backward rule. This showed up as
# "FloatingPointError: invalid value (inf) encountered in cond"
# during MCLMC warmup.
#
# Fix: when `train` is a concrete Python bool, skip lax.cond
# entirely and call eval_fn/train_fn directly. No behavior change
# for genuinely traced `train` (real training).
# ============================================================

_PATCH_APPLIED = False


def patch_subspace_curve_predict():
    """Idempotent monkeypatch. Call once before any posterior
    log-density / gradient evaluation (before building
    qwen_log_posterior, and before loading the baseline via
    evaluate_saved_baseline.py to be safe)."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    from subspace_inference.curve_optimizer import subspace_curve as _sc

    def _patched_predict(self, params, state, inputs, train, key=None):
        kwargs = {"rngs": {"dropout": key}} if key is not None else {}

        if self.mutable_name:
            all_params = {"params": params, self.mutable_name: state}

            def train_fn(inputs_inner):
                if isinstance(inputs_inner, dict):
                    out, net_state = self.model.apply(
                        all_params, **inputs_inner, train=True,
                        mutable=self.mutable_name, **kwargs,
                    )
                elif isinstance(inputs_inner, tuple):
                    out, net_state = self.model.apply(
                        all_params, *inputs_inner, train=True,
                        mutable=self.mutable_name, **kwargs,
                    )
                else:
                    out, net_state = self.model.apply(
                        all_params, inputs_inner, train=True,
                        mutable=self.mutable_name, **kwargs,
                    )
                return out, net_state[self.mutable_name]
        else:
            all_params = {"params": params}

            def train_fn(inputs_inner):
                if isinstance(inputs_inner, dict):
                    out = self.model.apply(
                        all_params, **inputs_inner, train=True, mutable=False, **kwargs
                    )
                elif isinstance(inputs_inner, tuple):
                    out = self.model.apply(
                        all_params, *inputs_inner, train=True, mutable=False, **kwargs
                    )
                else:
                    out = self.model.apply(
                        all_params, inputs_inner, train=True, mutable=False, **kwargs
                    )
                return out, {}

        def eval_fn(inputs_inner):
            if isinstance(inputs_inner, dict):
                out = self.model.apply(
                    all_params, **inputs_inner, train=False, mutable=False, **kwargs
                )
            elif isinstance(inputs_inner, tuple):
                out = self.model.apply(
                    all_params, *inputs_inner, train=False, mutable=False, **kwargs
                )
            else:
                out = self.model.apply(
                    all_params, inputs_inner, train=False, mutable=False, **kwargs
                )
            return out, state

        if isinstance(train, bool):
            out, net_state = train_fn(inputs) if train else eval_fn(inputs)
        else:
            out, net_state = jax.lax.cond(train, train_fn, eval_fn, inputs)

        return out, net_state

    _sc.SubspaceBaseModel._predict = _patched_predict
    _PATCH_APPLIED = True
    print("Patched SubspaceBaseModel._predict: lax.cond bypass for concrete train flags active.")


# ============================================================
# BALANCED AG-NEWS POSTERIOR SUBSET
# ============================================================

def load_posterior_subset(data, n_per_class, seq_len, posterior_key_seed):
    """Balanced class subset of the training data used as the
    "posterior data" for MCMC / VI / Laplace (same subset should be
    reused across all methods for a fair comparison)."""

    train_inputs, train_labels = data.get("train")

    labels_host = np.asarray(jax.device_get(train_labels))

    class_ids = np.unique(labels_host)

    subset_indices_host = np.concatenate([
        np.where(labels_host == class_id)[0][:n_per_class]
        for class_id in class_ids
    ])

    subset_indices = jnp.asarray(subset_indices_host, dtype=jnp.int32)

    x_posterior = {
        key: value[subset_indices, -seq_len:]
        for key, value in train_inputs.items()
    }

    y_posterior = train_labels[subset_indices]

    n_posterior_examples = int(y_posterior.shape[0])

    posterior_example_keys = random.split(
        random.PRNGKey(posterior_key_seed),
        n_posterior_examples,
    )

    print("Subset indices:", subset_indices_host)
    print("Labels:", np.asarray(y_posterior))
    print("Number of examples:", n_posterior_examples)
    print("input_ids:", x_posterior["input_ids"].shape)
    print("attention_mask:", x_posterior["attention_mask"].shape)

    return (
        x_posterior,
        y_posterior,
        posterior_example_keys,
        n_posterior_examples,
        subset_indices_host,
    )


# ============================================================
# MEMORY-EFFICIENT QWEN NLL (lax.scan)
# ============================================================

def build_qwen_nll_fns(env):
    """Factory closing over `env` (the loaded model wrapper).
    Returns (qwen_logits, qwen_logits_remat, qwen_single_example_nll,
    qwen_subset_nll_sum)."""

    def qwen_logits(candidate_params, inputs, key):
        logits, _ = env.s_model(
            candidate_params,
            {},
            jnp.asarray(0.0, dtype=jnp.float32),
            inputs,
            train=False,
            key=key,
        )
        return logits

    qwen_logits_remat = jax.checkpoint(qwen_logits)

    def qwen_single_example_nll(candidate_params, inputs, labels, key):
        logits = qwen_logits_remat(candidate_params, inputs, key)

        log_probabilities = jax.nn.log_softmax(logits, axis=-1)

        true_log_probability = jnp.take_along_axis(
            log_probabilities, labels[:, None], axis=-1
        ).squeeze(axis=-1)

        return -true_log_probability.mean()

    def qwen_batch_nll(candidate_params, inputs_batch, labels_batch, key):
        """Per-example NLL for a genuine batch (leading dim > 1), one single
        model forward call instead of one call per example."""
        logits = qwen_logits_remat(candidate_params, inputs_batch, key)

        log_probabilities = jax.nn.log_softmax(logits, axis=-1)

        true_log_probability = jnp.take_along_axis(
            log_probabilities, labels_batch[:, None], axis=-1
        ).squeeze(axis=-1)

        return -true_log_probability

    def qwen_subset_nll_sum(candidate_params, inputs, labels, example_keys, batch_size=16):
        """Sum of per-example NLL over `inputs`/`labels`.

        Batches `batch_size` examples per model forward call (instead of one
        call per example) -- fewer, larger lax.scan steps means a shorter
        graph for JAX/XLA to trace and compile, and better GPU utilization
        per step. Falls back to per-example key handling is unnecessary here:
        with lora_rho_w=lora_rho_s=0 (noise disabled, see
        patch_subspace_curve_predict / evaluate_saved_baseline.py) the `key`
        argument does not affect the numerical result, so one key per batch
        is safe and behaviorally identical to one key per example.

        `batch_size` must evenly divide the number of examples -- fails
        loudly with an assertion otherwise rather than silently dropping or
        padding examples.
        """
        n_examples = int(labels.shape[0])

        assert n_examples % batch_size == 0, (
            f"qwen_subset_nll_sum: n_examples ({n_examples}) must be evenly "
            f"divisible by batch_size ({batch_size}). Pass a compatible "
            f"batch_size or adjust the posterior subset size."
        )

        n_batches = n_examples // batch_size

        def reshape_to_batches(array):
            return array.reshape((n_batches, batch_size) + array.shape[1:])

        inputs_batched = jax.tree.map(reshape_to_batches, inputs)
        labels_batched = reshape_to_batches(labels)
        # One key per batch is enough (see docstring) -- just take the first
        # key of each batch-sized chunk of the original per-example keys.
        keys_batched = example_keys[::batch_size][:n_batches]

        def one_batch_loss(cumulative_nll, batch_data):
            inputs_b, labels_b, key_b = batch_data

            example_nll = qwen_batch_nll(
                candidate_params, inputs_b, labels_b, key_b,
            )

            example_nll = example_nll.astype(jnp.float32)
            new_cumulative_nll = cumulative_nll + jnp.sum(example_nll)

            return new_cumulative_nll, example_nll

        initial_nll = jnp.asarray(0.0, dtype=jnp.float32)

        total_nll, individual_nll_batched = jax.lax.scan(
            one_batch_loss,
            initial_nll,
            (inputs_batched, labels_batched, keys_batched),
        )

        individual_nll = individual_nll_batched.reshape(n_examples)

        return total_nll, individual_nll

    return qwen_logits, qwen_logits_remat, qwen_single_example_nll, qwen_subset_nll_sum


# ============================================================
# VECTORIZE / REBUILD LORA PARAMETERS
# ============================================================

def vectorize_lora_params(full_params_template):
    """full_params_template = params["params"] of the loaded
    baseline checkpoint. Returns (theta_map, all_leaves, lora_indices,
    full_tree_definition, unravel_lora)."""

    from subspace_inference.curve_optimizer.subspace_curve import LoraAbstractParams

    all_leaves, full_tree_definition = jax.tree_util.tree_flatten(
        full_params_template,
        is_leaf=lambda value: isinstance(value, LoraAbstractParams),
    )

    lora_indices = [
        index
        for index, leaf in enumerate(all_leaves)
        if isinstance(leaf, LoraAbstractParams)
    ]

    lora_ab_tree = tuple(
        (all_leaves[index].A, all_leaves[index].B)
        for index in lora_indices
    )

    theta_map, unravel_lora = ravel_pytree(lora_ab_tree)

    print("LoRA objects:", len(lora_indices))
    print("Sampling vector shape:", theta_map.shape)
    print("Sampling parameters:", theta_map.size)
    print("All MAP parameters finite:", bool(jnp.all(jnp.isfinite(theta_map))))

    return theta_map, all_leaves, lora_indices, full_tree_definition, unravel_lora


def clone_lora_with_new_ab(old_lora, new_A, new_B):
    new_lora = copy.copy(old_lora)

    try:
        new_lora.A = new_A
        new_lora.B = new_B
    except (AttributeError, TypeError):
        object.__setattr__(new_lora, "A", new_A)
        object.__setattr__(new_lora, "B", new_B)

    return new_lora


def build_rebuild_fn(all_leaves, lora_indices, full_tree_definition, unravel_lora):
    """Returns rebuild_full_params(theta) -> full Qwen param pytree,
    with only the LoRA A/B leaves replaced by theta's content; every
    other field (incl. lora_rho_w/lora_rho_s) is preserved unchanged
    from the original (eval-mode, noise-disabled) checkpoint."""

    def rebuild_full_params(theta):
        candidate_ab_tree = unravel_lora(theta)

        candidate_leaves = list(all_leaves)

        for leaf_index, (new_A, new_B) in zip(lora_indices, candidate_ab_tree):
            candidate_leaves[leaf_index] = clone_lora_with_new_ab(
                all_leaves[leaf_index], new_A, new_B
            )

        return full_tree_definition.unflatten(candidate_leaves)

    return rebuild_full_params


# ============================================================
# LOG POSTERIOR OVER LORA VECTOR
# ============================================================

def build_log_posterior_fn(
    qwen_subset_nll_sum,
    rebuild_full_params,
    x_posterior,
    y_posterior,
    posterior_example_keys,
    prior_std,
):
    def qwen_log_posterior(theta):
        candidate_params = rebuild_full_params(theta)

        nll_sum, _ = qwen_subset_nll_sum(
            candidate_params,
            x_posterior,
            y_posterior,
            posterior_example_keys,
        )

        theta_float32 = theta.astype(jnp.float32)

        squared_norm = jnp.sum(jnp.square(theta_float32))

        log_likelihood = -nll_sum

        log_prior = -0.5 * squared_norm / jnp.asarray(prior_std**2, dtype=jnp.float32)

        return log_likelihood + log_prior

    return qwen_log_posterior


# ============================================================
# CONVENIENCE: bundle everything together
# ============================================================

@dataclass
class QwenPosterior:
    x_posterior: Any
    y_posterior: Any
    posterior_example_keys: Any
    n_posterior_examples: int
    subset_indices_host: Any
    theta_map: jnp.ndarray
    all_leaves: Any
    lora_indices: Any
    full_tree_definition: Any
    unravel_lora: Callable
    rebuild_full_params: Callable
    qwen_logits: Callable
    qwen_single_example_nll: Callable
    qwen_subset_nll_sum: Callable
    qwen_log_posterior: Callable


def setup_qwen_posterior(
    env,
    params,
    data,
    n_per_class,
    seq_len,
    posterior_key_seed,
    prior_std,
):
    """One-call setup used identically by every method notebook
    (MILE, MFVI, Laplace, ...). Call patch_subspace_curve_predict()
    before this."""

    (
        x_posterior,
        y_posterior,
        posterior_example_keys,
        n_posterior_examples,
        subset_indices_host,
    ) = load_posterior_subset(data, n_per_class, seq_len, posterior_key_seed)

    qwen_logits, qwen_logits_remat, qwen_single_example_nll, qwen_subset_nll_sum = (
        build_qwen_nll_fns(env)
    )

    theta_map, all_leaves, lora_indices, full_tree_definition, unravel_lora = (
        vectorize_lora_params(params["params"])
    )

    rebuild_full_params = build_rebuild_fn(
        all_leaves, lora_indices, full_tree_definition, unravel_lora
    )

    qwen_log_posterior = build_log_posterior_fn(
        qwen_subset_nll_sum,
        rebuild_full_params,
        x_posterior,
        y_posterior,
        posterior_example_keys,
        prior_std,
    )

    return QwenPosterior(
        x_posterior=x_posterior,
        y_posterior=y_posterior,
        posterior_example_keys=posterior_example_keys,
        n_posterior_examples=n_posterior_examples,
        subset_indices_host=subset_indices_host,
        theta_map=theta_map,
        all_leaves=all_leaves,
        lora_indices=lora_indices,
        full_tree_definition=full_tree_definition,
        unravel_lora=unravel_lora,
        rebuild_full_params=rebuild_full_params,
        qwen_logits=qwen_logits,
        qwen_single_example_nll=qwen_single_example_nll,
        qwen_subset_nll_sum=qwen_subset_nll_sum,
        qwen_log_posterior=qwen_log_posterior,
    )


# ============================================================
# POSTERIOR PREDICTIVE PROBABILITIES
# MEMORY-EFFICIENT BATCHED EVALUATION
# ============================================================

def compute_posterior_predictive_probabilities(
    env,
    rebuild_full_params,
    sample_thetas,
    x_eval,
    y_eval,
    rng_key,
    eval_batch_size=64,
):
    """sample_thetas: (n_samples, n_params) array of flat LoRA
    vectors -- MCMC samples for MILE, variational draws for MFVI,
    Gaussian draws for Laplace, whatever `rebuild_full_params` can
    consume. x_eval/y_eval: not hardcoded to the posterior subset,
    so pass the actual held-out test set here for the final thesis
    numbers, or the posterior subset for quick dev checks (as MILE's
    stability-test config currently does).

    Returns (sample_probabilities, mean_probabilities, rng_key).
    """
    gc.collect()
    jax.clear_caches()

    n_samples = int(sample_thetas.shape[0])
    n_eval_examples = int(y_eval.shape[0])

    print("Evaluation examples:", n_eval_examples)
    print("Evaluation batch size:", eval_batch_size)
    print("Posterior samples:", n_samples)

    sample_probabilities = []

    for sample_index in range(n_samples):
        print(f"Evaluating posterior sample {sample_index + 1}/{n_samples}")

        sample_theta = jnp.asarray(sample_thetas[sample_index])
        sample_params = rebuild_full_params(sample_theta)

        probability_batches = []

        for start in range(0, n_eval_examples, eval_batch_size):
            end = min(start + eval_batch_size, n_eval_examples)

            x_batch_eval = jax.tree.map(
                lambda array: array[start:end],
                x_eval,
            )

            rng_key, evaluation_key = random.split(rng_key)

            logits_batch, _ = env.s_model(
                sample_params,
                {},
                jnp.asarray(0.0, dtype=jnp.float32),
                x_batch_eval,
                train=False,
                key=evaluation_key,
            )

            probabilities_batch = jax.nn.softmax(logits_batch, axis=-1)

            # Direkt auf CPU holen, damit GPU-Speicher nicht anwaechst
            probabilities_batch_host = np.asarray(jax.device_get(probabilities_batch))
            probability_batches.append(probabilities_batch_host)

            del x_batch_eval, logits_batch, probabilities_batch

        probabilities_sample_host = np.concatenate(probability_batches, axis=0)
        sample_probabilities.append(probabilities_sample_host)

        del sample_theta, sample_params, probability_batches
        gc.collect()

    sample_probabilities = np.stack(sample_probabilities, axis=0)
    mean_probabilities = np.mean(sample_probabilities, axis=0)

    print("\nEvaluation completed")
    print("Probability tensor shape:", sample_probabilities.shape)
    print("Expected shape:", (n_samples, n_eval_examples, mean_probabilities.shape[-1]))
    print("All probabilities finite:", np.isfinite(sample_probabilities).all())

    return sample_probabilities, mean_probabilities, rng_key


# ============================================================
# METRICS
# ============================================================

def compute_predictive_metrics(sample_probabilities, mean_probabilities, y_true, epsilon=1e-8):
    """sample_probabilities: (n_samples, n_examples, n_classes).
    mean_probabilities: (n_examples, n_classes) -- Bayesian model
    average over samples. y_true: (n_examples,) integer labels.

    Returns a dict of scalar metrics (accuracy, lppd,
    posterior_predictive_nll, brier_score) and per-example arrays
    (predictive_entropy, expected_entropy, mutual_information).
    """
    mean_probabilities = jnp.asarray(mean_probabilities)
    sample_probabilities = jnp.asarray(sample_probabilities)
    y_true = jnp.asarray(y_true)

    n_classes = int(mean_probabilities.shape[-1])

    # Point predictions
    posterior_predictions = jnp.argmax(mean_probabilities, axis=-1)
    accuracy = jnp.mean(posterior_predictions == y_true)

    # LPPD, following the MILE paper (Appendix A.4, "Evaluation"), which in
    # turn cites Gelman et al. (2014): LPPD = (1/n_test) * sum_i log(mean
    # predictive probability_i) -- i.e. the MEAN over test points, not the
    # raw sum. (Earlier this was implemented as a pure sum per the thesis
    # draft's Ch. 4 formula; the MILE paper -- the actual methodological
    # reference -- normalizes by n_test, so this now matches the paper
    # instead. Update the thesis Ch. 4 formula to match.)
    #
    # NOTE: with this normalization, LPPD is algebraically identical to
    # -posterior_predictive_nll below (both are +/- the mean log true-class
    # probability). That is expected, not a bug -- it is a direct
    # consequence of the MILE paper's own definition, not a coincidence of
    # this implementation. Kept as two separate named metrics anyway, since
    # both are reported by convention (thesis Ch. 4 / MILE paper's Table 1).
    true_class_probabilities = jnp.take_along_axis(
        mean_probabilities, y_true[:, None], axis=-1
    ).squeeze(axis=-1)

    log_true_class_probabilities = jnp.log(true_class_probabilities + epsilon)

    lppd = jnp.mean(log_true_class_probabilities)

    posterior_predictive_nll = -jnp.mean(log_true_class_probabilities)

    # Brier Score
    one_hot_labels = jax.nn.one_hot(y_true, n_classes)
    brier_score = jnp.mean(
        jnp.sum(jnp.square(mean_probabilities - one_hot_labels), axis=-1)
    )

    # Predictive entropy
    predictive_entropy = -jnp.sum(
        mean_probabilities * jnp.log(mean_probabilities + epsilon), axis=-1
    )

    # Expected conditional entropy
    per_sample_entropy = -jnp.sum(
        sample_probabilities * jnp.log(sample_probabilities + epsilon), axis=-1
    )
    expected_entropy = per_sample_entropy.mean(axis=0)

    # Epistemic uncertainty
    mutual_information = predictive_entropy - expected_entropy

    metrics = {
        "accuracy": accuracy,
        "lppd": lppd,
        "posterior_predictive_nll": posterior_predictive_nll,
        "brier_score": brier_score,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
    }

    print("Accuracy:", float(accuracy))
    print("LPPD:", float(lppd))
    print("Posterior predictive NLL:", float(posterior_predictive_nll))
    print("Brier Score:", float(brier_score))
    print("Mean predictive entropy:", float(predictive_entropy.mean()))
    print("Mean expected entropy:", float(expected_entropy.mean()))
    print("Mean mutual information:", float(mutual_information.mean()))

    return metrics


# ============================================================
# EXPECTED CALIBRATION ERROR
# ============================================================

def multiclass_ece(probabilities, labels, n_bins=15):
    probabilities = jnp.asarray(probabilities)
    labels = jnp.asarray(labels)

    confidences = jnp.max(probabilities, axis=-1)
    predictions = jnp.argmax(probabilities, axis=-1)
    correct = (predictions == labels).astype(jnp.float32)

    bin_edges = jnp.linspace(0.0, 1.0, n_bins + 1)

    ece = jnp.asarray(0.0, dtype=jnp.float32)

    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        bin_count = jnp.sum(in_bin)
        safe_count = jnp.maximum(bin_count, 1)

        bin_accuracy = jnp.sum(correct * in_bin) / safe_count
        bin_confidence = jnp.sum(confidences * in_bin) / safe_count
        bin_weight = bin_count / labels.shape[0]

        ece = ece + bin_weight * jnp.abs(bin_accuracy - bin_confidence)

    return ece


# ============================================================
# LOG POSTERIOR VALUES / DISTANCES FROM MAP
# ============================================================

def compute_sample_log_posteriors(qwen_log_posterior, sample_positions):
    """sample_positions: (n_samples, n_params). Returns a
    (n_samples,) host numpy array."""
    sample_log_posteriors = []

    for sample_index in range(sample_positions.shape[0]):
        value = qwen_log_posterior(sample_positions[sample_index])
        sample_log_posteriors.append(float(value))

    print("Computed log posterior values:", len(sample_log_posteriors))
    print("Range:", min(sample_log_posteriors), "to", max(sample_log_posteriors))

    return np.asarray(sample_log_posteriors)


def compute_distances_from_map(theta_map, sample_positions):
    """Euclidean distance of every sample from the MAP point --
    useful sanity/exploration diagnostic for any method that produces
    posterior draws (MCMC samples, VI draws, Laplace draws)."""
    theta_map_host = np.asarray(jax.device_get(theta_map))
    samples_host = np.asarray(jax.device_get(sample_positions))

    distances = np.linalg.norm(samples_host - theta_map_host[None, :], axis=1)

    print("Distance summary:")
    print("Minimum:", distances.min())
    print("Mean:", distances.mean())
    print("Maximum:", distances.max())

    return distances


# ============================================================
# SAVE RUN (common format across methods)
# ============================================================

def summarize_metrics(metrics):
    """Split a metrics dict (mix of scalar and per-example array
    jax/np values) into (summary_scalars, metadata_arrays), matching
    MILE's original save format: scalar metrics go straight into the
    summary; array metrics contribute their per-example values to
    metadata_arrays and their mean (prefixed mean_) to the summary."""
    summary = {}
    metadata_arrays = {}

    for key, value in metrics.items():
        value_host = np.asarray(jax.device_get(value))
        if value_host.ndim == 0:
            summary[key] = float(value_host)
        else:
            metadata_arrays[key] = value_host
            summary[f"mean_{key}"] = float(value_host.mean())

    return summary, metadata_arrays


def save_method_run(
    result_dir,
    run_name,
    sample_positions,
    sample_probabilities,
    metadata_arrays,
    summary,
):
    """Common save format used by every method notebook (MILE, MFVI,
    Laplace, ...), so the final evaluation notebook can load them
    uniformly:

        {run_name}_samples.npy         -- (n_samples, n_params)
        {run_name}_probabilities.npy   -- (n_samples, n_examples, n_classes)
        {run_name}_metadata.npz        -- per-example arrays (entropy, MI, ...)
        {run_name}_summary.json        -- scalar metrics + config + method info

    `summary` and `metadata_arrays` are method-specific: build the
    common part with summarize_metrics(...) from
    compute_predictive_metrics(...), then add whatever
    method-specific fields you have (e.g. MILE's tuned step_size/L,
    MFVI's ELBO curve, Laplace's Hessian-approximation diagnostics)
    before calling this.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    samples_path = result_dir / f"{run_name}_samples.npy"
    probabilities_path = result_dir / f"{run_name}_probabilities.npy"
    metadata_path = result_dir / f"{run_name}_metadata.npz"
    summary_path = result_dir / f"{run_name}_summary.json"

    samples_host = np.asarray(jax.device_get(sample_positions))
    sample_probabilities_host = np.asarray(jax.device_get(sample_probabilities))

    np.save(samples_path, samples_host)
    np.save(probabilities_path, sample_probabilities_host)
    np.savez(metadata_path, **metadata_arrays)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n======================================")
    print("RUN SAVED")
    print("======================================")
    print("Run:", run_name)
    print("Samples:      ", samples_path)
    print("Probabilities:", probabilities_path)
    print("Metadata:     ", metadata_path)
    print("Summary:      ", summary_path)

    return {
        "samples_path": samples_path,
        "probabilities_path": probabilities_path,
        "metadata_path": metadata_path,
        "summary_path": summary_path,
    }
