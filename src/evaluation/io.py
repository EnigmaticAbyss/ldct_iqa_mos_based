# src/evaluation/io.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional, Sequence


def save_predictions_csv(
    image_paths: Sequence[str],
    y_true: Sequence[float],
    y_pred: Sequence[Optional[float]],
    out_path: str | Path,
    raw_outputs: Optional[Sequence[str]] = None,
):
    """
    Save prediction rows for later inspection.

    Columns:
      image_path, mos_true, mos_pred, abs_error, parsed_ok, raw_output
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_outputs is None:
        raw_outputs = [""] * len(image_paths)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "mos_true",
            "mos_pred",
            "abs_error",
            "parsed_ok",
            "raw_output",
        ])

        for img, yt, yp, raw in zip(image_paths, y_true, y_pred, raw_outputs):
            abs_error = "" if yp is None else abs(float(yp) - float(yt))
            parsed_ok = yp is not None
            writer.writerow([img, yt, yp, abs_error, parsed_ok, raw])


def save_results_json(results: dict, out_path: str | Path):
    """
    Save final evaluation summary JSON.

    Args:
        results: Evaluation summary payload to serialize.
        out_path: Destination JSON path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
