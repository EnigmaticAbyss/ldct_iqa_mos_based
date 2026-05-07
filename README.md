# LDCT IQA MOS-Based Model Training and Evaluation

This repository contains tools and scripts for low-dose CT (LDCT) image quality assessment (IQA) using visual language models (VLMs) and MOS regression. It includes data preprocessing, model training, evaluation, model comparison, and sweep automation for hyperparameter search.

## Project Overview

- `datasets/`: raw and processed LDCT image data, MOS annotations, and cache files.
- `src/`: core Python modules for data loading, formatting, training, regression, evaluation, and comparison.
- `scripts/`: command-line entrypoints for preprocessing, training, evaluation, model comparison, and dataset conversion.
- `config/`: reusable JSON configuration files for preprocessing, training, evaluation, and sweep experiments.
- `output/`: generated model checkpoints, evaluation results, and comparison outputs.
- `logs/`: training and preprocessing logs.

## Key Capabilities

- Convert TIFF CT scans to normalized PNG and JSONL training data.
- Train either:
  - MOS regression models (`sft_mode: regression`)
  - TRL fine-tuned (SFT) VLM models (`sft_mode: trl_sft`)
- Evaluate both regression and SFT models against test data.
- Compare multiple evaluated models using standard IQA metrics.
- Build TRL-compatible Arrow datasets from base JSONL sources.
- Run parameter sweep experiments with automatic run config generation.

## Recommended Environment

- Python 3.10
- GPU with CUDA support is recommended for training and inference.
- Core Python dependencies include:
  - `torch`
  - `transformers`
  - `trl`
  - `peft`
  - `datasets`
  - `Pillow`
  - `numpy`
  - `tqdm`

The repository includes `vlm-iqa-env.yml` for a Conda-style environment definition.

## Data Layout

- `datasets/ct_tif/`: original LDCT TIFF files for train/test.
- `datasets/ct_PNG/`: converted PNG images used for model input.
- `datasets/mos/`: JSON MOS annotations for train/test sets.
- `datasets/processed/`: processed dataset artifacts:
  - `base_train.jsonl`, `base_val.jsonl`, `base_test.jsonl`
  - `sft_*_dataset/`: Arrow datasets for TRL training.

## Preprocessing

Preprocessing converts TIFF images to normalized PNG files and writes JSONL dataset files.

Example:

```bash
python scripts/preprocess_ldct.py --config config/preprocess_config.json
```

This script reads `config/preprocess_config.json` and:

- converts TIFFs to PNGs using `minmax` or percentile normalization
- optionally resizes images
- writes `datasets/processed/base_train.jsonl`, `base_val.jsonl`, and `base_test.jsonl`
- creates a validation split from the training set when configured

## Building TRL Arrow Datasets

Build TRL-compatible Arrow datasets from a base JSONL dataset.

```bash
python scripts/build_sft_arrow_dataset.py --base_jsonl datasets/processed/base_train.jsonl --output_dir datasets/processed/sft_train_dataset
```

Repeat for validation and test JSONL files when needed.

## Training

Use `scripts/train.py` with a JSON config file. The config selects either regression or TRL SFT mode.

### Regression training

```bash
python scripts/train.py --config config/sft.json
```

Set `sft_mode` to `regression` in the config to train with `src.trainers.regression_trainer.LDCTRegressionTrainer`.

### TRL SFT training

```bash
python scripts/train.py --config config/sft.json
```

Set `sft_mode` to `trl_sft` in the config to train with `src.trainers.sft_trainer.LDCTSFTTrainer`.

### Important config fields

- `model_name`: base Vision-Language model checkpoint (for example `google/medgemma-1.5-4b-it`).
- `output_dir`: where to save checkpoints and adapter layers.
- `logging_dir`: TensorBoard/logging output location.
- `load_prebuilt_sft_dataset`: whether to load Arrow datasets instead of JSONL.
- `train_dataset_dir` / `val_dataset_dir` / `test_dataset_dir`: Arrow dataset paths.
- `use_4bit` / `use_8bit`: quantization settings.
- `lora_enabled`, `lora_scope`, `lora_coverage`: LoRA adapter configuration.

## Evaluation

Evaluate either a regression model or an SFT model.

```bash
python scripts/evaluate.py --config config/eval.json
```

Supported eval modes:

- `eval_mode: regression`
- `eval_mode: sft`

For SFT evaluation, set `is_peft_adapter` and `base_model_name` when loading a PEFT/LoRA adapter.

### Output

Evaluation writes a JSON file of metrics and predictions. Example default path:

- `output/eval/sft/sft_eval_results.json`

## Model Comparison

Compare multiple evaluation results with a single config.

```bash
python scripts/compare_models.py --config config/compare_models.json
```

This generates ranked comparison files in `output/eval/comparison/`, including:

- `model_comparison.json`
- `model_comparison.csv`

## Sweep Experiments

Run hyperparameter sweeps and generate per-run configs automatically.

```bash
python -m scripts.sweep_sft --config config/sft_sweep.json
```

Use `--dry-run` to preview commands and run configs without launching training.

## Outputs and Logs

- `output/model/`: trained model checkpoints, adapter configs, and training results.
- `output/eval/`: evaluation results and prediction CSVs.
- `logs/`: preprocessing and training logs.

## Notes

- The repository supports both base MOS regression and TRL-style supervised fine-tuning.
- SFT evaluation uses an image-chat prompt template and parses predicted MOS scores from generated text.
- The `src` package contains reusable components for data loading, collator construction, metric computation, and model comparison.

---

If you want, I can also add a condensed quick-start section or generate a sample `config/sft.json` for your preferred training setup.