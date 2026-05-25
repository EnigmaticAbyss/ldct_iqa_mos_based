# src/evaluation/plotting.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _filter_valid_pairs(
    y_true: List[float],
    y_pred: List[Optional[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return numeric truth/prediction arrays for parsed predictions only."""
    pairs = [(float(t), float(p)) for t, p in zip(y_true, y_pred) if p is not None]
    if not pairs:
        return np.array([]), np.array([])
    yt, yp = zip(*pairs)
    return np.array(yt, dtype=np.float32), np.array(yp, dtype=np.float32)


def save_scatter_plot(
    y_true: List[float],
    y_pred: List[Optional[float]],
    out_path: str | Path,
    title: str = "Predicted vs Ground Truth MOS",
):
    """
    Save a predicted-versus-ground-truth MOS scatter plot.

    Args:
        y_true: Ground-truth MOS values.
        y_pred: Predicted MOS values, with ``None`` entries ignored.
        out_path: Destination image path.
        title: Plot title.
    """
    yt, yp = _filter_valid_pairs(y_true, y_pred)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    if len(yt) > 0:
        plt.scatter(yt, yp, alpha=0.7)
        lo = min(float(np.min(yt)), float(np.min(yp)))
        hi = max(float(np.max(yt)), float(np.max(yp)))
        plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Ground Truth MOS")
    plt.ylabel("Predicted MOS")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_error_histogram(
    y_true: List[float],
    y_pred: List[Optional[float]],
    out_path: str | Path,
    title: str = "Prediction Error Histogram",
):
    """
    Save a histogram of prediction errors for valid MOS predictions.

    Args:
        y_true: Ground-truth MOS values.
        y_pred: Predicted MOS values, with ``None`` entries ignored.
        out_path: Destination image path.
        title: Plot title.
    """
    yt, yp = _filter_valid_pairs(y_true, y_pred)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    if len(yt) > 0:
        err = yp - yt
        plt.hist(err, bins=20)
        plt.axvline(0.0, linestyle="--")
    plt.xlabel("Prediction Error (pred - true)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
