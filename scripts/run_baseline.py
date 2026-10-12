"""Week 1 baseline: classify NYC 311 complaint types with logistic regression.

Logs metrics and artifacts to MLflow.
"""

from __future__ import annotations

from pathlib import Path

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = (_REPO_ROOT / "mlflow.db").as_posix()  # forward slashes
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{_DB_PATH}")

import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from urbanintel.data.loaders import load_nyc311, summarize

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
DATA_PATH = Path("data/raw/nyc311/nyc311_sample.csv")
EXPERIMENT_NAME = "urbanintel-week1"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Use top-N complaint types to keep the task tractable
TOP_N_CLASSES = 5
MIN_SAMPLES_PER_CLASS = 50


def prepare_data():
    df = load_nyc311(DATA_PATH, validate=False)  # skip strict validation first time

    # Drop rows with missing labels
    df = df.dropna(subset=["complaint_type"]).copy()

    # Keep only the top-N classes
    counts = df["complaint_type"].value_counts()
    top = counts[counts >= MIN_SAMPLES_PER_CLASS].head(TOP_N_CLASSES).index.tolist()
    df = df[df["complaint_type"].isin(top)].copy()

    return df, top


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, classes = prepare_data()
    stats = summarize(df)

    print(f"Loaded {len(df)} rows, {len(classes)} classes")
    print(f"Classes: {classes}")

    X = df["complaint_type"].astype(str)  # placeholder text
    y = df["complaint_type"].astype(str)

    # NOTE: This is a first-pass baseline. We'll refine features next.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    with mlflow.start_run(run_name="tfidf-logreg-baseline"):
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "vectorizer": "TfidfVectorizer",
                "top_n_classes": TOP_N_CLASSES,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }
        )
        mlflow.log_metrics(stats)

        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=1000, n_jobs=-1)),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1_macro})

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")


if __name__ == "__main__":
    main()
