# scripts/sweep_sft.py
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.evaluation.compare import compare_models


DEFAULT_METRICS = [
    "mae",
    "rmse",
    "plcc",
    "srocc",
    "krocc",
    "prediction_rate",
    "valid_predictions",
    "total_predictions",
]

METRIC_DIRECTIONS = {
    "mae": "min",
    "rmse": "min",
    "plcc": "max",
    "srocc": "max",
    "krocc": "max",
    "prediction_rate": "max",
    "valid_predictions": "max",
    "total_predictions": "max",
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_existing_path(value: str | Path, config_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return config_dir / path


def load_config_ref(value: str | Dict[str, Any], config_dir: Path, label: str) -> Dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, str):
        path = resolve_existing_path(value, config_dir)
        if not path.exists():
            raise FileNotFoundError(f"{label} config not found: {path}")
        return load_json(path)
    raise TypeError(f"{label} must be a JSON path or an inline object")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def expand_dotted_keys(values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not values:
        return {}

    out: Dict[str, Any] = {}
    for key, value in values.items():
        if "." not in key:
            out[key] = copy.deepcopy(value)
            continue

        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = copy.deepcopy(value)
    return out


def flatten_dict(values: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in values.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, flat_key))
        else:
            out[flat_key] = value
    return out


def short_param_name(key: str) -> str:
    aliases = {
        "learning_rate": "lr",
        "lora_r": "r",
        "lora_alpha": "alpha",
        "lora_dropout": "dropout",
        "gradient_accumulation_steps": "gas",
        "num_train_epochs": "epochs",
        "per_device_train_batch_size": "batch",
        "weight_decay": "wd",
        "warmup_ratio": "warmup",
    }
    return aliases.get(key.split(".")[-1], key.split(".")[-1])


def format_name_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    value = value.strip("-._")
    return value or "run"


def make_run_name(overrides: Dict[str, Any], index: int) -> str:
    flat = flatten_dict(overrides)
    if not flat:
        return f"run_{index:03d}"

    parts = [
        f"{short_param_name(key)}_{format_name_value(value)}"
        for key, value in flat.items()
    ]
    return slugify("-".join(parts))


def unique_name(name: str, seen: set[str]) -> str:
    candidate = slugify(name)
    if candidate not in seen:
        seen.add(candidate)
        return candidate

    index = 2
    while f"{candidate}_{index}" in seen:
        index += 1
    final = f"{candidate}_{index}"
    seen.add(final)
    return final


def grid_runs(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return []

    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    for key, choices in zip(keys, values):
        if not isinstance(choices, list) or len(choices) == 0:
            raise ValueError(f"grid.{key} must be a non-empty list")

    runs: List[Dict[str, Any]] = []
    for index, combo in enumerate(itertools.product(*values), start=1):
        raw_overrides = dict(zip(keys, combo))
        train_overrides = expand_dotted_keys(raw_overrides)
        runs.append(
            {
                "name": make_run_name(train_overrides, index),
                "train": train_overrides,
                "eval": {},
            }
        )
    return runs


def explicit_runs(raw_runs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_runs, start=1):
        if not isinstance(raw, dict):
            raise TypeError("Each item in runs must be an object")

        raw_train = raw.get("train", raw.get("overrides", raw.get("params", {})))
        raw_eval = raw.get("eval", {})
        train_overrides = expand_dotted_keys(raw_train)
        eval_overrides = expand_dotted_keys(raw_eval)
        name = raw.get("name") or make_run_name(train_overrides, index)

        runs.append(
            {
                "name": name,
                "train": train_overrides,
                "eval": eval_overrides,
            }
        )
    return runs


def build_run_specs(sweep_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = []
    runs.extend(grid_runs(sweep_cfg.get("grid", {})))
    runs.extend(explicit_runs(sweep_cfg.get("runs", [])))

    if not runs:
        raise ValueError("Sweep config must define either 'grid' or 'runs'")

    seen: set[str] = set()
    for run in runs:
        run["name"] = unique_name(str(run["name"]), seen)
    return runs


def is_peft_adapter_train(train_cfg: Dict[str, Any]) -> bool:
    return bool(train_cfg.get("lora_enabled", False)) and train_cfg.get("lora_coverage") != "full_finetune"


def build_train_eval_configs(
    *,
    base_train_cfg: Dict[str, Any],
    base_eval_cfg: Dict[str, Any],
    run: Dict[str, Any],
    run_dir: Path,
    force_separate_output_dirs: bool,
) -> tuple[Dict[str, Any], Dict[str, Any], Path, Path]:
    train_overrides = run.get("train", {})
    eval_overrides = run.get("eval", {})

    train_cfg = deep_merge(base_train_cfg, train_overrides)
    model_dir = run_dir / "model"
    logging_dir = run_dir / "logs"
    eval_dir = run_dir / "eval"
    eval_path = eval_dir / "eval_results.json"

    if force_separate_output_dirs or "output_dir" not in train_overrides:
        train_cfg["output_dir"] = str(model_dir)
    if force_separate_output_dirs or "logging_dir" not in train_overrides:
        train_cfg["logging_dir"] = str(logging_dir)

    eval_cfg = deep_merge(base_eval_cfg, eval_overrides)
    eval_cfg["model_dir"] = train_cfg["output_dir"]
    eval_cfg["output_dir"] = str(eval_dir)
    eval_cfg["output_path"] = str(eval_path)

    if "base_model_name" not in eval_overrides and train_cfg.get("model_name"):
        eval_cfg["base_model_name"] = train_cfg["model_name"]
    if "is_peft_adapter" not in eval_overrides:
        eval_cfg["is_peft_adapter"] = is_peft_adapter_train(train_cfg)

    return train_cfg, eval_cfg, model_dir, eval_path


def command_to_string(argv: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def run_command(argv: List[str], *, dry_run: bool) -> None:
    print(f"[sweep] {command_to_string(argv)}", flush=True)
    if dry_run:
        return
    subprocess.run(argv, check=True)


def load_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = load_json(path)
    metrics = data.get("metrics", data)
    return metrics if isinstance(metrics, dict) else {}


def is_valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def infer_metric_mode(metric: str, explicit_mode: Optional[str]) -> str:
    if explicit_mode:
        mode = explicit_mode.lower()
        if mode not in {"min", "max"}:
            raise ValueError("primary_mode must be 'min' or 'max'")
        return mode
    return METRIC_DIRECTIONS.get(metric, "max")


def sort_records(records: List[Dict[str, Any]], primary_metric: str, mode: str) -> List[Dict[str, Any]]:
    reverse = mode == "max"

    def key_fn(record: Dict[str, Any]):
        value = record.get("metrics", {}).get(primary_metric)
        if not is_valid_number(value):
            return (1, 0.0)
        numeric = float(value)
        return (0, -numeric if reverse else numeric)

    return sorted(records, key=key_fn)


def first_valid_best(records: List[Dict[str, Any]], primary_metric: str) -> Optional[Dict[str, Any]]:
    for record in records:
        if is_valid_number(record.get("metrics", {}).get(primary_metric)):
            return record
    return None


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def save_summary_csv(records: List[Dict[str, Any]], path: Path) -> None:
    param_keys = sorted({key for record in records for key in record.get("params", {})})
    metric_keys = sorted(
        set(DEFAULT_METRICS)
        | {key for record in records for key in record.get("metrics", {})}
    )

    fields = [
        "rank",
        "run_name",
        "status",
        "train_status",
        "eval_status",
        "model_dir",
        "eval_path",
        "train_config_path",
        "eval_config_path",
        "error",
    ] + param_keys + metric_keys

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, record in enumerate(records, start=1):
            row = {
                "rank": rank,
                "run_name": record["name"],
                "status": record.get("status", ""),
                "train_status": record.get("train_status", ""),
                "eval_status": record.get("eval_status", ""),
                "model_dir": record.get("model_dir", ""),
                "eval_path": record.get("eval_path", ""),
                "train_config_path": record.get("train_config_path", ""),
                "eval_config_path": record.get("eval_config_path", ""),
                "error": record.get("error", ""),
            }
            for key in param_keys:
                row[key] = csv_value(record.get("params", {}).get(key))
            for key in metric_keys:
                row[key] = csv_value(record.get("metrics", {}).get(key))
            writer.writerow(row)


def aggregate(values: List[float]) -> Dict[str, Optional[float] | int]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def save_parameter_effects(records: List[Dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    param_keys = sorted({key for record in records for key in record.get("params", {})})
    metric_keys = sorted(
        set(DEFAULT_METRICS)
        | {key for record in records for key in record.get("metrics", {})}
    )

    rows: List[Dict[str, Any]] = []
    for param in param_keys:
        values_seen = {
            json.dumps(record.get("params", {}).get(param), sort_keys=True)
            for record in records
            if param in record.get("params", {})
        }
        if len(values_seen) <= 1:
            continue

        for raw_value in sorted(values_seen):
            group = [
                record
                for record in records
                if json.dumps(record.get("params", {}).get(param), sort_keys=True) == raw_value
            ]
            for metric in metric_keys:
                numeric_values = [
                    float(record["metrics"][metric])
                    for record in group
                    if is_valid_number(record.get("metrics", {}).get(metric))
                ]
                stats = aggregate(numeric_values)
                rows.append(
                    {
                        "parameter": param,
                        "value": json.loads(raw_value),
                        "metric": metric,
                        **stats,
                    }
                )

    json_path = out_dir / "parameter_effects.json"
    csv_path = out_dir / "parameter_effects.csv"
    write_json(json_path, rows)

    fields = ["parameter", "value", "metric", "count", "mean", "std", "min", "max"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})

    return json_path, csv_path


def run_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    sweep_cfg_path = args.config
    sweep_cfg = load_json(sweep_cfg_path)
    config_dir = sweep_cfg_path.parent

    base_train_cfg = load_config_ref(
        sweep_cfg.get("base_train_config", sweep_cfg.get("train_config", "config/sft.json")),
        config_dir,
        "base_train_config",
    )
    base_eval_cfg = load_config_ref(
        sweep_cfg.get("base_eval_config", sweep_cfg.get("eval_config", "config/eval.json")),
        config_dir,
        "base_eval_config",
    )

    output_root = Path(sweep_cfg.get("output_root", "output/sweeps/sft"))
    output_root.mkdir(parents=True, exist_ok=True)

    skip_existing = bool(sweep_cfg.get("skip_existing", True)) and not args.force
    continue_on_error = bool(sweep_cfg.get("continue_on_error", False))
    dry_run = bool(args.dry_run)
    skip_train = bool(args.skip_train or sweep_cfg.get("skip_train", False) or args.only_compare)
    skip_eval = bool(args.skip_eval or sweep_cfg.get("skip_eval", False) or args.only_compare)
    force_separate_output_dirs = bool(sweep_cfg.get("force_separate_output_dirs", True))

    primary_metric = str(sweep_cfg.get("primary_metric", "srocc"))
    primary_mode = infer_metric_mode(primary_metric, sweep_cfg.get("primary_mode"))

    records: List[Dict[str, Any]] = []
    compare_entries: List[Dict[str, str]] = []

    for run in build_run_specs(sweep_cfg):
        run_name = run["name"]
        run_dir = output_root / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        train_cfg, eval_cfg, model_dir, eval_path = build_train_eval_configs(
            base_train_cfg=base_train_cfg,
            base_eval_cfg=base_eval_cfg,
            run=run,
            run_dir=run_dir,
            force_separate_output_dirs=force_separate_output_dirs,
        )

        train_config_path = run_dir / "train_config.json"
        eval_config_path = run_dir / "eval_config.json"
        write_json(train_config_path, train_cfg)
        write_json(eval_config_path, eval_cfg)

        record: Dict[str, Any] = {
            "name": run_name,
            "status": "pending",
            "train_status": "pending",
            "eval_status": "pending",
            "params": flatten_dict(run.get("train", {})),
            "model_dir": str(model_dir),
            "eval_path": str(eval_path),
            "train_config_path": str(train_config_path),
            "eval_config_path": str(eval_config_path),
            "metrics": {},
            "error": "",
        }

        try:
            train_done = Path(train_cfg["output_dir"]) / "training_results.json"
            if skip_train:
                record["train_status"] = "skipped"
            elif skip_existing and train_done.exists():
                record["train_status"] = "skipped_existing"
            else:
                run_command(
                    [sys.executable, "-m", "scripts.train", "--config", str(train_config_path)],
                    dry_run=dry_run,
                )
                record["train_status"] = "dry_run" if dry_run else "completed"

            if skip_eval:
                record["eval_status"] = "skipped"
            elif skip_existing and eval_path.exists():
                record["eval_status"] = "skipped_existing"
            else:
                run_command(
                    [sys.executable, "-m", "scripts.evaluate", "--config", str(eval_config_path)],
                    dry_run=dry_run,
                )
                record["eval_status"] = "dry_run" if dry_run else "completed"

            record["metrics"] = load_metrics(eval_path)
            record["status"] = "completed" if record["metrics"] else record["eval_status"]

        except subprocess.CalledProcessError as exc:
            record["status"] = "failed"
            record["error"] = f"Command failed with exit code {exc.returncode}: {command_to_string(exc.cmd)}"
            if not continue_on_error:
                records.append(record)
                raise
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            if not continue_on_error:
                records.append(record)
                raise

        records.append(record)
        compare_entries.append({"name": run_name, "path": str(eval_path)})

    ranked_records = sort_records(records, primary_metric, primary_mode)

    comparison_dir = output_root / "comparison"
    comparison_config_path = comparison_dir / "compare_models.json"
    write_json(
        comparison_config_path,
        {"output_dir": str(comparison_dir), "models": compare_entries},
    )
    comparison = compare_models(compare_entries, comparison_dir)

    summary_csv = output_root / "sweep_summary.csv"
    summary_json = output_root / "sweep_summary.json"
    effects_json, effects_csv = save_parameter_effects(ranked_records, output_root)

    best = first_valid_best(ranked_records, primary_metric)
    result = {
        "num_runs": len(ranked_records),
        "primary_metric": primary_metric,
        "primary_mode": primary_mode,
        "best_run": best["name"] if best else None,
        "best_metric_value": best.get("metrics", {}).get(primary_metric) if best else None,
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "parameter_effects_json": str(effects_json),
        "parameter_effects_csv": str(effects_csv),
        "comparison_config": str(comparison_config_path),
        "comparison": comparison,
        "runs": ranked_records,
    }

    write_json(summary_json, result)
    save_summary_csv(ranked_records, summary_csv)

    best_run_path = output_root / "best_run.json"
    if best:
        write_json(best_run_path, best)
    elif best_run_path.exists():
        best_run_path.unlink()

    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run multiple SFT trainings, evaluate each output, and compare metrics."
    )
    ap.add_argument("--config", type=Path, required=True, help="Path to SFT sweep config JSON")
    ap.add_argument("--dry-run", action="store_true", help="Write per-run configs and print commands only")
    ap.add_argument("--skip-train", action="store_true", help="Do not run training")
    ap.add_argument("--skip-eval", action="store_true", help="Do not run evaluation")
    ap.add_argument("--only-compare", action="store_true", help="Only rebuild comparison files from existing eval outputs")
    ap.add_argument("--force", action="store_true", help="Rerun even when existing train/eval outputs are present")
    args = ap.parse_args()

    result = run_sweep(args)
    print(json.dumps({key: result[key] for key in result if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
