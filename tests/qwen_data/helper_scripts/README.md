# Helper Scripts for Qwen Data and Model Conversion

This directory contains scripts for converting PyTorch Qwen models to JAX and processing datasets for text classification tasks.

## Setup

### Install Dependencies

To use these scripts, you need to install the optional `torch` dependencies:

```bash
uv sync --extra torch
```

This will install:
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `accelerate>=0.20.0`

## Usage

### Converting Qwen Models to JAX

Convert Qwen2.5 models from PyTorch to JAX format:

```bash
# Convert a specific model (default: Qwen2.5-0.5B)
python save_datasets_and_params.py --mode qwen --qwen-models Qwen/Qwen2.5-0.5B

# Convert multiple models
python save_datasets_and_params.py --mode qwen --qwen-models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B

# Specify output directory
python save_datasets_and_params.py --mode qwen --model-output-dir /path/to/models

# Choose data type (float16, bfloat16, float32)
python save_datasets_and_params.py --mode qwen --dtype float32
```

### Processing Datasets

Process text classification datasets:

```bash
# Process all datasets
python save_datasets_and_params.py --mode datasets

# Specify output directory
python save_datasets_and_params.py --mode datasets --dataset-output-dir /path/to/datasets
```

### Combined Mode

Process both datasets and models:

```bash
python save_datasets_and_params.py --mode all
```

## Output Structure

By default, files are saved to:

- **Models**: `helper_scripts/data_store/models/`
  - `qwen2.5_{model_size}_{dtype}/`
    - `jax_params.npy`
    - `jax_config.json`

- **Datasets**: `helper_scripts/data_store/datasets/`
  - `text_classification/{dataset_name}/`
    - `train_data.npz`
    - `test_data.npz`

## Optional: Upload to WandB

To upload converted models and datasets as WandB artifacts:

```bash
python save_datasets_and_params.py --mode all --upload-to-wandb --wandb-project your_project_name
```

## Dataset Requirements

The script expects a `dataset/` directory containing:
- Python files with dataset classes that inherit from `DatasetBase`
- The `DatasetBase` class should be in `dataset/utils/datasetbase.py`

Currently supported datasets (configured in the script):
- boolq (Boolean Questions)
- winogrande_s, winogrande_m (Winogrande)
- ARC-Easy, ARC-Challenge (AI2 Reasoning Challenge)
- obqa (OpenBookQA)
- MMLU_chem, MMLU_phy (MMLU subsets)

## Notes

- The script automatically skips datasets/models that have already been converted
- Large models (32B, 72B) are skipped by default due to memory requirements
- The `tmp_files/` directory at the project root is used for intermediate artifacts during processing
