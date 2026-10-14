"""Forecasting metrics that respect a validity mask.

Metrics ignore positions where valid_mask is False, so suspected missing
readings (zeros in METR-LA) do not distort the numbers.
"""

from __future__ import annotations

import numpy as np


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray, valid: np.ndarray) -> float:
    """Mean absolute error over valid positions only."""
    if not (y_true.shape == y_pred.shape == valid.shape):
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}, valid={valid.shape}"
        )
    if not valid.any():
        return float("nan")
    return float(np.abs(y_true - y_pred)[valid].mean())


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, valid: np.ndarray) -> float:
    """Root mean squared error over valid positions only."""
    if not (y_true.shape == y_pred.shape == valid.shape):
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}, valid={valid.shape}"
        )
    if not valid.any():
        return float("nan")
    diff = (y_true - y_pred)[valid]
    return float(np.sqrt((diff**2).mean()))


def masked_mape(
    y_true: np.ndarray, y_pred: np.ndarray, valid: np.ndarray, eps: float = 1e-3
) -> float:
    """Mean absolute percentage error over valid positions only.

    Uses eps to avoid division by very small values.
    """
    if not (y_true.shape == y_pred.shape == valid.shape):
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}, valid={valid.shape}"
        )
    if not valid.any():
        return float("nan")
    denom = np.maximum(np.abs(y_true), eps)
    return float((np.abs(y_true - y_pred) / denom)[valid].mean() * 100)
