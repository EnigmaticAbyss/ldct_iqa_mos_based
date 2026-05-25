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
    """
    Load a model evaluation result JSON file.

    Args:
        path: Path to an evaluation JSON file.

    Returns:
        Parsed evaluation payload.
    """
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
    """
    Build one flattened metric row for each configured model result.

    Args:
        models: Iterable of model entries with ``name`` and ``path`` keys.

    Returns:
        List of comparison rows containing model metadata and selected metrics.
    """
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
    """
    Write comparison rows to a JSON file.

    Args:
        rows: Ranked or unranked comparison rows.
        out_path: Destination JSON path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def save_comparison_csv(rows: List[Dict[str, Any]], out_path: str | Path):
    """
    Write comparison rows to a CSV file with stable metric columns.

    Args:
        rows: Ranked or unranked comparison rows.
        out_path: Destination CSV path.
    """
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
        """
        Build the ranking tuple for one model row.

        Args:
            r: Comparison row.

        Returns:
            Sort key prioritizing correlations before error metrics.
        """
        def safe(v, default):
            """
            Replace missing metric values with sortable defaults.

            Args:
                v: Metric value.
                default: Replacement used when ``v`` is ``None``.

            Returns:
                Original value or default fallback.
            """
            return default if v is None else v

        return (
            -safe(r.get("srocc"), -1e9),
            -safe(r.get("plcc"), -1e9),
            safe(r.get("rmse"), 1e9),
            safe(r.get("mae"), 1e9),
        )

    return sorted(rows, key=key_fn)


def compare_models(models: List[Dict[str, str]], out_dir: str | Path) -> Dict[str, Any]:
    """
    Rank model evaluation files and save JSON/CSV comparison artifacts.

    Args:
        models: Model entries with display names and evaluation-result paths.
        out_dir: Directory where comparison artifacts should be written.

    Returns:
        Summary containing the best model, ranked rows, and artifact paths.
    """
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
