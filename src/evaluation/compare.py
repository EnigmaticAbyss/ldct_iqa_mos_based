# src/evaluation/compare.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any


KEYS = [
    "mae",
    "rmse",
    "plcc",
    "srocc",
    "krocc",
    "prediction_rate",
    "valid_predictions",
    "total_predictions",
]


def load_eval_result(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def flatten_metrics(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts either:
      {"metrics": {...}, ...}
    or a flat metrics dict.
    """
    metrics = eval_result.get("metrics", eval_result)
    row = {}
    for k in KEYS:
        row[k] = metrics.get(k, None)
    return row


def build_comparison_rows(models):
    rows = []

    for m in models:
        path = Path(m["path"])
        if not path.exists():
            print(f"[WARNING] Missing eval file, skipping: {path}")
            continue

        result = load_eval_result(path)
        row = {"model_name": m["name"], "result_path": str(path)}
        row.update(flatten_metrics(result))
        rows.append(row)

    return rows


def save_comparison_json(rows: List[Dict[str, Any]], out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def save_comparison_csv(rows: List[Dict[str, Any]], out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model_name", "result_path"] + KEYS
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rank_models(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple sort:
    - lower is better: mae, rmse
    - higher is better: plcc, srocc, krocc, prediction_rate

    Default ranking priority:
      1) srocc desc
      2) plcc desc
      3) rmse asc
      4) mae asc
    """
    def key_fn(r: Dict[str, Any]):
        def safe(v, default):
            return default if v is None else v

        return (
            -safe(r.get("srocc"), -1e9),
            -safe(r.get("plcc"), -1e9),
            safe(r.get("rmse"), 1e9),
            safe(r.get("mae"), 1e9),
        )

    return sorted(rows, key=key_fn)


def compare_models(models: List[Dict[str, str]], out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_comparison_rows(models)
    ranked = rank_models(rows)

    json_path = out_dir / "model_comparison.json"
    csv_path = out_dir / "model_comparison.csv"

    save_comparison_json(ranked, json_path)
    save_comparison_csv(ranked, csv_path)

    return {
        "num_models": len(ranked),
        "best_model": ranked[0]["model_name"] if ranked else None,
        "rows": ranked,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }