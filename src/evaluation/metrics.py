# src/evaluation/metrics.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


def _filter_valid_pairs(
    y_true: List[float],
    y_pred: List[Optional[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only pairs where prediction is not None.
    """
    pairs = [(float(t), float(p)) for t, p in zip(y_true, y_pred) if p is not None]
    if not pairs:
        return np.array([]), np.array([])
    yt, yp = zip(*pairs)
    return np.array(yt, dtype=np.float32), np.array(yp, dtype=np.float32)


def compute_basic_errors(
    y_true: List[float],
    y_pred: List[Optional[float]],
) -> Dict[str, float]:
    """
    Compute MAE and RMSE on valid predictions only.
    """
    yt, yp = _filter_valid_pairs(y_true, y_pred)
    if len(yt) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "valid_predictions": 0,
            "total_predictions": len(y_true),
            "prediction_rate": 0.0,
        }

    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))

    return {
        "mae": mae,
        "rmse": rmse,
        "valid_predictions": int(len(yt)),
        "total_predictions": int(len(y_true)),
        "prediction_rate": float(len(yt) / len(y_true)) if len(y_true) > 0 else 0.0,
    }


def compute_rank_and_linear_metrics(
    y_true: List[float],
    y_pred: List[Optional[float]],
) -> Dict[str, float]:
    """
    Compute PLCC, SROCC, KROCC on valid predictions only.
    """
    yt, yp = _filter_valid_pairs(y_true, y_pred)

    if len(yt) < 2:
        return {
            "plcc": float("nan"),
            "plcc_p_value": float("nan"),
            "srocc": float("nan"),
            "srocc_p_value": float("nan"),
            "krocc": float("nan"),
            "krocc_p_value": float("nan"),
        }

    plcc, plcc_p = pearsonr(yt, yp)
    srocc, srocc_p = spearmanr(yt, yp)
    krocc, krocc_p = kendalltau(yt, yp)

    return {
        "plcc": float(plcc),
        "plcc_p_value": float(plcc_p),
        "srocc": float(srocc),
        "srocc_p_value": float(srocc_p),
        "krocc": float(krocc),
        "krocc_p_value": float(krocc_p),
    }


def compute_all_metrics(
    y_true: List[float],
    y_pred: List[Optional[float]],
) -> Dict[str, float]:
    """
    Full evaluation summary for MOS prediction.
    """
    out = {}
    out.update(compute_basic_errors(y_true, y_pred))
    out.update(compute_rank_and_linear_metrics(y_true, y_pred))
    return out