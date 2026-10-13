"""Leakage-free baseline: predict NYC 311 complaint type.

Features (all available at prediction time):
  - borough (categorical)
  - incident_zip (top-50 binned)
  - agency (categorical, if present)
  - location_type (categorical, if present)
  - created_date-derived calendar features (cyclical)
  - latitude / longitude

Target: complaint_type, filtered to top-5 classes.

Compares:
  1. Majority class
  2. Logistic Regression on full features
  3. LightGBM on full features

NOTE: 'descriptor' is NEVER used as a feature. It is a downstream
refinement of complaint_type and would cause leakage.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = (_REPO_ROOT / "mlflow.db").as_posix()
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_DB_PATH}"

sys.path.insert(0, str(_REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from urbanintel.config import dataset_path
from urbanintel.data.loaders import load_nyc311
from urbanintel.features.nyc311_features import (
    build_features,
    numeric_feature_columns,
)

EXPERIMENT_NAME = "urbanintel-week1"
TOP_N_CLASSES = 5
RANDOM_STATE = 42


def pick_top_classes(df, n: int):
    return df["complaint_type"].value_counts().head(n).index.tolist()


def split_data(df, top_classes):
    df = df[df["complaint_type"].isin(top_classes)].copy()
    train, temp = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["complaint_type"]
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp["complaint_type"],
    )
    return train, val, test


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()[:12]
    except Exception:
        return "unknown"


def evaluate(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


def log_confusion(matrix, labels, filename):
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


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    csv_path = dataset_path("nyc311", "nyc311_sample.csv")
    df = load_nyc311(csv_path, validate=False)
    df = df.dropna(subset=["complaint_type"]).copy()

    top_classes = pick_top_classes(df, TOP_N_CLASSES)
    train, val, test = split_data(df, top_classes)

    # Fit zip vocabulary on train only
    train = build_features(train)
    top_zips = train["incident_zip"].value_counts().head(50).index.tolist()
    val = build_features(val, top_zips=top_zips)
    test = build_features(test, top_zips=top_zips)

    num_cols = numeric_feature_columns()
    # Only features available BEFORE classification.
    # Excluded (leaky): descriptor, agency, location_type
    cat_cols = ["borough", "zip_binned"]

    print(f"Numeric features: {num_cols}")
    print(f"Categorical features: {cat_cols}")

    feature_cols = cat_cols + num_cols
    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_train = train["complaint_type"]
    y_test = test["complaint_type"]

    dataset_info = {
        "csv_path": str(csv_path),
        "csv_sha256_16": file_hash(csv_path),
        "total_rows_after_filter": len(df),
        "train_rows": len(train),
        "test_rows": len(test),
        "top_classes": top_classes,
        "categorical_features": cat_cols,
        "numeric_features": num_cols,
        "numeric_imputation": "median (fit on train only)",
        "excluded_features": [
            "descriptor",  # downstream label refinement
            "agency",  # chosen during classification
            "location_type",  # chosen during classification
            "additional_details",  # operator-written text
        ],
        "git_sha": git_sha(),
    }

    results = {}

    # Baseline 1: majority class
    with mlflow.start_run(run_name="baseline-majority"):
        mlflow.log_params({"model": "DummyClassifier", "strategy": "most_frequent"})
        mlflow.log_dict(dataset_info, "dataset.json")
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        preds = dummy.predict(X_test)
        m = evaluate(y_test, preds, top_classes)
        mlflow.log_metrics({"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
        mlflow.log_text(m["report"], "classification_report.txt")
        log_confusion(m["confusion"], top_classes, "confusion.png")
        results["majority"] = m

    # Preprocessor shared by both models
    # Median imputation for missing numerics (lat/lon have ~1.4% NaN)
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    # Baseline 2: Logistic Regression on full features
    with mlflow.start_run(run_name="logreg-location-time"):
        params = {
            "model": "LogisticRegression",
            "features": "borough + zip_binned + calendar + lat/lon",
            "excluded": "descriptor, agency, location_type",
            "max_iter": 1000,
        }
        mlflow.log_params(params)
        mlflow.log_dict(dataset_info, "dataset.json")
        clf = Pipeline(
            [
                ("pre", pre),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        m = evaluate(y_test, preds, top_classes)
        mlflow.log_metrics({"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
        mlflow.log_text(m["report"], "classification_report.txt")
        log_confusion(m["confusion"], top_classes, "confusion.png")
        results["logreg_full"] = m

    # Baseline 3: LightGBM on full features
    try:
        import lightgbm as lgb

        with mlflow.start_run(run_name="lightgbm-location-time"):
            params = {
                "model": "LightGBM",
                "features": "borough + zip_binned + calendar + lat/lon",
                "excluded": "descriptor, agency, location_type",
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 63,
            }
            mlflow.log_params(params)
            mlflow.log_dict(dataset_info, "dataset.json")
            clf = Pipeline(
                [
                    ("pre", pre),
                    (
                        "clf",
                        lgb.LGBMClassifier(
                            n_estimators=500,
                            learning_rate=0.05,
                            num_leaves=63,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            verbose=-1,
                        ),
                    ),
                ]
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            m = evaluate(y_test, preds, top_classes)
            mlflow.log_metrics({"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
            mlflow.log_text(m["report"], "classification_report.txt")
            log_confusion(m["confusion"], top_classes, "confusion.png")
            results["lightgbm_full"] = m
    except ImportError:
        print("LightGBM not installed — skipping")

    print("\n=== Summary (no descriptor — leakage-free) ===")
    for name, m in results.items():
        print(f"{name:20s}  acc={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
