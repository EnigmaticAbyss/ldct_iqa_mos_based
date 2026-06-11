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
    title: str | None = None,
):
    """
    Save a predicted-versus-ground-truth MOS scatter plot.

    Args:
        y_true: Ground-truth MOS values.
        y_pred: Predicted MOS values, with ``None`` entries ignored.
        out_path: Destination image path.
        title: Optional plot title.
    """
    yt, yp = _filter_valid_pairs(y_true, y_pred)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))

    if len(yt) > 0:
        lo = min(0.0, float(np.min(yt)), float(np.min(yp)))
        hi = max(4.0, float(np.max(yt)), float(np.max(yp)))
        pad = 0.15 * max(hi - lo, 1.0)
        x_min = 0.0 if lo >= 0.0 else lo - pad
        x_max = hi + pad

        ax.scatter(
            yt,
            yp,
            s=24,
            color="#2f78b7",
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78,
            label="Predictions",
            zorder=3,
        )
        ax.plot(
            [x_min, x_max],
            [x_min, x_max],
            color="#666666",
            linewidth=1.1,
            linestyle="--",
            label="Ideal agreement",
            zorder=2,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)
        ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#dddddd",
            fontsize=7.5,
            borderpad=0.4,
            handlelength=1.6,
        )
    else:
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 4.0)

    ax.set_xlabel("Ground-truth MOS", fontsize=9)
    ax.set_ylabel("Predicted MOS", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, pad=6)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.75)
    ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
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
