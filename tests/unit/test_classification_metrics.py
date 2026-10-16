"""Tests for classification metrics."""

import numpy as np
import pytest

from urbanintel.evaluation.classification import (
    masked_accuracy,
    masked_f1_macro,
    per_class_report,
)


def test_accuracy_perfect() -> None:
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 0])
    assert masked_accuracy(y_true, y_pred) == 1.0


def test_accuracy_zero() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([1, 2, 0])
    assert masked_accuracy(y_true, y_pred) == 0.0


def test_f1_macro_perfect() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    assert masked_f1_macro(y_true, y_pred, num_classes=3) == 1.0


def test_f1_macro_includes_unpredicted_classes() -> None:
    """Classes present in y_true but never predicted should still count.

    Hand computation:
      class 0: TP=2, FP=0, FN=0  -> P=1.0, R=1.0, F1=1.000
      class 1: TP=2, FP=2, FN=0  -> P=0.5, R=1.0, F1=0.667
      class 2: TP=0, FP=0, FN=2  -> P=0.0, R=0.0, F1=0.000
      macro-F1 = (1.000 + 0.667 + 0.000) / 3 = 0.556
    """
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 1])
    f1 = masked_f1_macro(y_true, y_pred, num_classes=3)
    assert f1 == pytest.approx(0.5556, abs=0.001)


def test_per_class_report_contains_labels() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    report = per_class_report(y_true, y_pred, ["a", "b", "c"])
    assert "a" in report
    assert "b" in report
    assert "c" in report
