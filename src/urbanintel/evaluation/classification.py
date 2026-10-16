"""Classification metrics used across the project."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, f1_score


def masked_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain accuracy (no masking needed for classification)."""
    return float((y_true == y_pred).mean())


def masked_f1_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Macro-F1 across all classes, including classes with no predictions."""
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )
    )


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> str:
    """Full classification report as a string."""
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    )
