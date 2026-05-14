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
- Run MOS-reward GRPO refinement from the SFT LoRA adapter (`train_mode: trl_grpo`).
- Evaluate regression, SFT, and GRPO models against test data.
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

Use `scripts/train.py` with a JSON config file. The config selects regression, TRL SFT, or TRL GRPO mode.

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

### TRL GRPO training

```bash
python scripts/train.py --config config/grpo.json
```

Set `train_mode` to `trl_grpo`. The default GRPO config starts from the existing SFT LoRA adapter at `output/model/medgemma15_iqa_sft_test`, uses the available SFT Arrow datasets, and rewards only MOS prediction closeness.

### Important config fields

- `model_name`: base Vision-Language model checkpoint (for example `google/medgemma-1.5-4b-it`).
- `output_dir`: where to save checkpoints and adapter layers.
- `logging_dir`: TensorBoard/logging output location.
- `load_prebuilt_sft_dataset`: whether to load Arrow datasets instead of JSONL.
- `dataset_format`: `arrow` for `save_to_disk` datasets or `json` / `jsonl` for base JSON files.
- `train_dataset_dir` / `val_dataset_dir`: training and validation Arrow dataset paths.
- `train_json_path` / `val_json_path`: training and validation JSON or JSONL dataset paths.
- `test_dataset_dir` / `test_json_path`: evaluation-only test dataset paths.
- `adapter_model_dir`: optional SFT PEFT adapter to continue from before GRPO.
- `use_4bit` / `use_8bit`: quantization settings.
- `lora_enabled`, `lora_scope`, `lora_coverage`: LoRA adapter configuration.

## Evaluation

Evaluate a regression, SFT, or GRPO model.

```bash
python scripts/evaluate.py --config config/eval.json
```

Supported eval modes:

- `eval_mode: regression`
- `eval_mode: sft`
- `eval_mode: grpo`

For SFT or GRPO evaluation, set `is_peft_adapter` and `base_model_name` when loading a PEFT/LoRA adapter.
SFT and GRPO have separate evaluator classes, both backed by the shared generative MOS evaluator.

### Output

Evaluation writes a JSON file of metrics and predictions. Example default path:

- `output/eval/sft/sft_eval_results.json`
- `output/eval/grpo/grpo_eval_results.json`

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

For GRPO parameter comparison:

```bash
python -m scripts.sweep_grpo --config config/grpo_sweep.json
```

Use `--dry-run` to preview commands and run configs without launching training.

## Outputs and Logs

- `output/model/`: trained model checkpoints, adapter configs, and training results.
- `output/eval/`: evaluation results and prediction CSVs.
- `logs/`: preprocessing and training logs.

## Notes

- The repository supports base MOS regression, TRL-style supervised fine-tuning, and MOS-only GRPO refinement.
- SFT and GRPO evaluation use an image-chat prompt template and parse predicted MOS scores from generated text.
- The `src` package contains reusable components for data loading, collator construction, metric computation, and model comparison.
