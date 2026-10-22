"""Baselines for NYC 311 text classification.

Three baselines, each at a different level of sophistication:
  1. Majority class (predicts the most common label always)
  2. Descriptor lookup (maps each descriptor to its most common label)
  3. TF-IDF + Logistic Regression

The lookup baseline is the key one. If it achieves ~95% accuracy, the
task is trivially easy and DistilBERT cannot meaningfully beat it.

Logs all runs to MLflow under urbanintel-week4.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = (_REPO_ROOT / "mlflow.db").as_posix()
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_DB_PATH}"
sys.path.insert(0, str(_REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from urbanintel.config import dataset_path
from urbanintel.data.loaders import load_nyc311
from urbanintel.data.nyc311_text import prepare_text_data

EXPERIMENT_NAME = "urbanintel-week4"


def log_confusion(matrix, labels, filename):
    """Save a confusion matrix PNG and log it as an MLflow artifact."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(filename)
    os.remove(filename)


def majority_class_baseline(train_labels: list[int], test_labels: list[int]) -> list[int]:
    """Predict the most common training label for every test example."""
    most_common = Counter(train_labels).most_common(1)[0][0]
    return [most_common] * len(test_labels)


def descriptor_lookup_baseline(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    fallback: int,
) -> list[int]:
    """Map each descriptor to its most common training label.

    Unknown descriptors fall back to the majority class.
    """
    # Build lookup: descriptor -> most common label
    pairs: dict[str, Counter] = {}
    for text, label in zip(train_texts, train_labels):
        pairs.setdefault(text, Counter())[label] += 1

    lookup = {text: counter.most_common(1)[0][0] for text, counter in pairs.items()}
    return [lookup.get(text, fallback) for text in test_texts]


def evaluate_and_log(y_true: list[int], y_pred: list[int], labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))),
    }


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    csv_path = dataset_path("nyc311", "nyc311_sample.csv")
    df = load_nyc311(csv_path, validate=False)
    split = prepare_text_data(df)

    label_names = split.label_names
    print(f"Classes: {label_names}")
    print(
        f"Train: {len(split.train_texts):,}  Val: {len(split.val_texts):,}  Test: {len(split.test_texts):,}"
    )
    print(f"Train/test descriptor overlap: {split.overlap_report['test_row_overlap_fraction']:.4f}")
    print()

    # Log dataset identity once
    dataset_info = {
        "csv_path": str(csv_path),
        "train_rows": len(split.train_texts),
        "test_rows": len(split.test_texts),
        "label_names": label_names,
        "unique_train_descriptors": split.overlap_report["unique_train_descriptors"],
        "test_row_overlap_fraction": split.overlap_report["test_row_overlap_fraction"],
    }

    # --- Baseline 1: majority class ---
    print("=== Majority class ===")
    with mlflow.start_run(run_name="baseline-majority"):
        mlflow.log_params({"baseline": "majority_class"})
        mlflow.log_dict(dataset_info, "dataset.json")
        preds = majority_class_baseline(split.train_labels, split.test_labels)
        m = evaluate_and_log(split.test_labels, preds, label_names)
        mlflow.log_metrics({"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
        mlflow.log_text(m["report"], "classification_report.txt")
        log_confusion(m["confusion"], label_names, "confusion.png")
        print(f"Accuracy: {m['accuracy']:.4f}  F1 macro: {m['f1_macro']:.4f}")

    # --- Baseline 2: descriptor lookup ---
    print()
    print("=== Descriptor lookup ===")
    with mlflow.start_run(run_name="baseline-descriptor-lookup"):
        mlflow.log_params({"baseline": "descriptor_lookup"})
        mlflow.log_dict(dataset_info, "dataset.json")
        fallback = Counter(split.train_labels).most_common(1)[0][0]
        preds = descriptor_lookup_baseline(
            split.train_texts, split.train_labels, split.test_texts, fallback
        )
        m = evaluate_and_log(split.test_labels, preds, label_names)
        mlflow.log_metrics({"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
        mlflow.log_text(m["report"], "classification_report.txt")
        log_confusion(m["confusion"], label_names, "confusion.png")
        print(f"Accuracy: {m['accuracy']:.4f}  F1 macro: {m['f1_macro']:.4f}")

    # --- Baseline 3: TF-IDF + Logistic Regression ---
    print()
    print("=== TF-IDF + Logistic Regression ===")
    with mlflow.start_run(run_name="tfidf-logreg"):
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "vectorizer": "TfidfVectorizer",
                "ngram_range": "(1,2)",
                "max_features": 500,
            }
        )
        mlflow.log_dict(dataset_info, "dataset.json")
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        pipe.fit(split.train_texts, split.train_labels)
        preds = pipe.predict(split.test_texts)
        m = evaluate_and_log(split.test_labels, preds, label_names)
        mlflow.log_metrics({"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
        mlflow.log_text(m["report"], "classification_report.txt")
        log_confusion(m["confusion"], label_names, "confusion.png")
        print(f"Accuracy: {m['accuracy']:.4f}  F1 macro: {m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
