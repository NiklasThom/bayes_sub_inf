# OpenCode Agent Instructions

This file contains crucial context for agents working in this repository. 

## Project Layout & Architecture (README is outdated)
- The codebase is located in the `curve_optimizer/` directory, **not** `src/` or the root as the `README.md` suggests.
- **Main Entrypoint:** `curve_optimizer/trainer/qwen_fineTuning_PtoC_fixCps.py` (Orchestrates training and evaluation).
- **Core Math & Models:** `curve_optimizer/subspace_curve.py` (Contains Bézier math, `SubspaceBaseModel`, and LoRA mixins).
- **Model Implementations:** `curve_optimizer/models/qwen_jax.py` (Qwen JAX model, PyTorch→JAX conversion).

## Environment & Setup
- The project uses **Poetry** for dependency management. Requires Python `^3.10,<3.12`.
- Run `poetry install` to set up the environment.
- **JAX/CUDA Quirk:** To properly install or upgrade JAX with CUDA 12 support, you must run the specific pip command noted in `pyproject.toml`:
  `poetry run pip install --upgrade "jax[cuda12_local]==0.5.3" "flax==0.10.5"`

## Verification & Testing
- There is currently no active `pytest` test suite, despite `pytest` being a dependency.
- **Smoke Test:** To verify changes to the training pipeline or model architecture, run the smoke test. Note the updated path compared to the README:
  ```bash
  poetry run python curve_optimizer/trainer/qwen_fineTuning_PtoC_fixCps.py --batch-size=4 --cp-fix 1 0 1 0 1 --smoke-test
  ```

## Development Conventions
- **Adding a New Subspace Model:** 
  1. Create a class inheriting from `SubspaceBaseModel` in `curve_optimizer/subspace_curve.py`.
  2. Decorate it with `@register_subspace_model("name")`.
  3. Implement the `nll(self, params, state, t, x, y, train, key)` method.
- **Adding a New Regularizer:** Create a mixin class inheriting from `SubspaceBaseModel`, override `compute_loss_t()`, and pass parameters via `ModelParams.get_model_kwargs()`.
