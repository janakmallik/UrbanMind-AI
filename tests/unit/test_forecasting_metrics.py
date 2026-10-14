"""Tests for masked forecasting metrics."""

from __future__ import annotations

import numpy as np
import pytest

from urbanintel.evaluation.forecasting import masked_mae, masked_mape, masked_rmse


def test_masked_mae_ignores_invalid() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 100.0, 100.0])  # last two are garbage
    valid = np.array([True, True, False, False])
    # Only first two count: |0.5|, |0.5| -> mean 0.5
    assert masked_mae(y_true, y_pred, valid) == pytest.approx(0.5)


def test_masked_rmse_ignores_invalid() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 4.0, 100.0])
    valid = np.array([True, True, False])
    # Errors: 1, 2 -> squared: 1, 4 -> mean 2.5 -> sqrt ~1.5811
    assert masked_rmse(y_true, y_pred, valid) == pytest.approx(np.sqrt(2.5))


def test_masked_metrics_return_nan_when_all_invalid() -> None:
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 2.5])
    valid = np.array([False, False])
    assert np.isnan(masked_mae(y_true, y_pred, valid))
    assert np.isnan(masked_rmse(y_true, y_pred, valid))
    assert np.isnan(masked_mape(y_true, y_pred, valid))


def test_masked_mape_matches_manual() -> None:
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([11.0, 18.0])
    valid = np.array([True, True])
    # Errors: 10% and 10% -> 10.0
    assert masked_mape(y_true, y_pred, valid) == pytest.approx(10.0)
