# OpenCode Agent Instructions

This file contains crucial context for agents working in this repository. 

## Project Layout & Architecture
- The package is structured under the `subspace_inference` namespace.
- **Top-level Import:** `import subspace_inference`
- **Sub-packages:**
  - `subspace_inference.curve_optimizer`: Contains training and optimization logic.
  - `subspace_inference.curve_optimizer.models`: Contains model implementations (e.g., Qwen JAX).
  - `subspace_inference.curve_optimizer.trainer`: Contains training entrypoints.
- **Main Entrypoint:** `subspace_inference/curve_optimizer/trainer/qwen_fineTuning_PtoC_fixCps.py`
- **Core Math:** `subspace_inference/curve_optimizer/subspace_curve.py`

## Transitional Codebase (Important!)
- This repository is currently being transformed into a pip-installable package.
- **Broken Imports:** Many files originally had `from src.X import Y` imports. These are being transitioned to absolute imports like `from subspace_inference.curve_optimizer.X import Y`. 
- **Proactive Fixes:** If you edit a file and see `from src...` or other broken relative imports, **proactively fix them** to use the `subspace_inference` absolute path.
- **Commented Dependencies:** `src.weight_matching` is currently missing from the repo. Related imports in `subspace_curve.py` are commented out and should remain so until the file is provided.

## Development Goals
- **Model Extensibility:** The `models/` directory is designed to be easily extended with new JAX implementations.
- **Dataset Injection:** The architecture aims to decouple dataset loading from training, allowing users to inject custom datasets easily.
- **Optimization Modes:** The framework supports Bézier curve optimization both with and without LoRA.

## Environment & Setup
- The project uses **Poetry** for dependency management. Requires Python `^3.10,<3.12`.
- **JAX/CUDA Quirk:** To properly install or upgrade JAX with CUDA 12 support, you must run:
  `poetry run pip install --upgrade "jax[cuda12_local]==0.5.3" "flax==0.10.5"`

## Verification & Testing
- **Smoke Test:** To verify changes, use the updated path:
  ```bash
  poetry run python subspace_inference/curve_optimizer/trainer/qwen_fineTuning_PtoC_fixCps.py --batch-size=4 --cp-fix 1 0 1 0 1 --smoke-test
  ```
