<p align="center">
  <img src="assets/medgemma_icon.svg" alt="MedGemma icon" width="360">
</p>

<h1 align="center">LDCT IQA MOS-Based Model Training and Evaluation</h1>

<p align="center">
  MedGemma-based low-dose CT image quality assessment with supervised fine-tuning,
  MOS-reward GRPO refinement, and reproducible evaluation sweeps.
</p>

This repository contains tools and scripts for low-dose CT (LDCT) image quality assessment (IQA) using visual language models (VLMs) and MOS regression. It includes data preprocessing, model training, evaluation, model comparison, and sweep automation for hyperparameter search.

## At a Glance

- **Task:** no-reference LDCT image quality assessment by predicting a radiologist mean opinion score (MOS) from a CT slice.
- **Base model:** `google/medgemma-1.5-4b-it`, adapted with LoRA.
- **Main comparison:** direct prompting, supervised fine-tuning (SFT), base-initialized GRPO, and SFT-initialized GRPO.
- **Dataset split:** 1,300 annotated LDCT images: 900 train, 100 validation, and 300 held-out test samples.
- **Main finding:** SFT is the strongest evaluated adaptation strategy; SFT-initialized GRPO improves over direct prompting but does not surpass SFT.

## Result Snapshot

The   table reports the following held-out test results:

| Model | MAE | RMSE | PLCC | SROCC | KROCC | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Direct base VLM | 4.453 | 5.395 | 0.014 | -0.020 | -0.016 | 100.0% |
| SFT initialization adapter | **0.347** | **0.451** | 0.916 | 0.917 | 0.787 | 100.0% |
| Best SFT | 0.393 | 0.482 | **0.930** | **0.932** | **0.808** | 100.0% |
| Best SFT+GRPO | 0.666 | 0.796 | 0.897 | 0.907 | 0.788 | 100.0% |

## Project Overview

- `datasets/`: raw and processed LDCT image data, MOS annotations, and cache files.
- `src/`: core Python modules for data loading, formatting, training, regression, evaluation, and comparison.
- `scripts/`: command-line entrypoints for preprocessing, training, evaluation, model comparison, and dataset conversion.
- `config/`: reusable JSON configuration files for preprocessing, training, evaluation, and sweep experiments.
- `output/`: generated model checkpoints, evaluation results, and comparison outputs.
- `logs/`: training and preprocessing logs.

## Key Capabilities

- Convert TIFF CT scans to normalized PNG and JSONL training data.
- Train MOS regression models (`train_mode` or `sft_mode`: `regression`).
- Train TRL fine-tuned (SFT) VLM models (`sft_mode: trl_sft`).
- Run MOS-reward GRPO refinement from an SFT LoRA adapter or directly from the base VLM (`train_mode: trl_grpo`).
- Evaluate regression, SFT, and GRPO models against test data.
- Compare multiple evaluated models using standard IQA metrics.
- Build TRL-compatible Arrow datasets from base JSONL sources.
- Run parameter sweep experiments with automatic run config generation.

## Recommended Environment

- Python 3.10
- GPU with CUDA support is recommended for training and inference.
- Access to the MedGemma checkpoint used in the configs is required for training and evaluation.
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

The LDCTIQAC-style data used by this project contain TIFF LDCT images and MOS annotations. The checked-in preprocessing configuration creates a 10% validation split from the 1,000-image development set, producing 900 train, 100 validation, and 300 test records.

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

The checked-in training configs are:

- `config/regression.json`: MOS regression training.
- `config/sft.json`: TRL SFT training.
- `config/grpo.json`: TRL GRPO training.

### Regression training

```bash
python scripts/train.py --config config/regression.json
```

Set `train_mode` or `sft_mode` to `regression`.

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
Set `adapter_model_dir` to `null` to run GRPO directly from the base model instead.

### Important config fields

- `model_name`: base Vision-Language model checkpoint (for example `google/medgemma-1.5-4b-it`).
- `output_dir`: where to save checkpoints and adapter layers.
- `logging_dir`: TensorBoard/logging output location.
- `data_dir`: fallback processed-data root used when explicit dataset paths are not set.
- `load_prebuilt_sft_dataset`: whether to load Arrow datasets instead of JSONL.
- `dataset_format`: `arrow` for `save_to_disk` datasets or `json` / `jsonl` for base JSON files.
- `train_dataset_dir` / `val_dataset_dir`: training and validation Arrow dataset paths.
- `train_json_path` / `val_json_path`: training and validation JSON or JSONL dataset paths.
- `adapter_model_dir`: optional SFT PEFT adapter to continue from before GRPO.
- `use_4bit` / `use_8bit`: quantization settings.
- `lora_enabled`, `lora_scope`, `lora_coverage`: LoRA adapter configuration.

## Evaluation

Evaluate a regression, SFT, or GRPO model.

```bash
python scripts/evaluate.py --config config/eval.json
```

Use `config/regression_eval.json` for regression, `config/eval.json` for the default SFT evaluator config, and `config/grpo_eval.json` for the GRPO evaluator config.

Supported eval modes:

- `eval_mode: regression`
- `eval_mode: sft`
- `eval_mode: grpo`

For regression, SFT, or GRPO evaluation, set `is_peft_adapter` and `base_model_name` when loading a PEFT/LoRA adapter.
SFT and GRPO have separate evaluator classes, both backed by the shared generative MOS evaluator.

Evaluation data fields:

- `data_dir`: fallback processed-data root. For JSON evaluation it defaults to `base_test.jsonl`; for Arrow evaluation it defaults to `test_dataset`.
- `dataset_format`: `arrow` for `save_to_disk` datasets or `json` / `jsonl` for base JSON files.
- `test_dataset_dir`: explicit Arrow test dataset path, such as `datasets/processed/sft_test_dataset`.
- `test_json_path`: explicit JSON/JSONL test dataset path, such as `datasets/processed/base_test.jsonl`.

### Output

Evaluation writes a JSON metrics file. The evaluators also write `predictions.csv`, `scatter.png`, and `error_hist.png` under the configured `output_dir`. Example default paths:

- `output/eval/sft/sft_eval_results.json`
- `output/eval/grpo/grpo_eval_results.json`
- `output/eval/regression/regression_eval_results.json`

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

Both sweep entrypoints use the shared implementation in `scripts/sweep_common.py`. Use `--dry-run` to preview commands and run configs without launching training. The GRPO sweep includes runs that start from the SFT adapter and runs that start directly from the base model.






## Outputs and Logs

- `output/model/`: trained model checkpoints, adapter configs, and training results.
- `output/eval/`: evaluation results and prediction CSVs.
- `logs/`: preprocessing and training logs.

## Notes


- The `src` package contains reusable components for data loading, collator construction, metric computation, and model comparison.
