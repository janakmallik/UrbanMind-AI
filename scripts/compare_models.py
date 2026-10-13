"""Compare 6 model families on the leakage-free NYC 311 task.

Uses the same train/test split and features as run_baseline_v2.py.
Logs all runs to the same MLflow experiment.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = (_REPO_ROOT / "mlflow.db").as_posix()
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_DB_PATH}"

sys.path.insert(0, str(_REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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


def build_preprocessor(cat_cols, num_cols, sparse_output: bool = True):
    """Build the shared preprocessing pipeline.

    Set sparse_output=False when the downstream estimator requires dense input
    (e.g., HistGradientBoostingClassifier, MLPClassifier).
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output),
                cat_cols,
            ),
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
        sparse_threshold=0.0 if not sparse_output else 0.3,
    )


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

    top_classes = df["complaint_type"].value_counts().head(TOP_N_CLASSES).index.tolist()
    df = df[df["complaint_type"].isin(top_classes)].copy()

    train, temp = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["complaint_type"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=RANDOM_STATE, stratify=temp["complaint_type"]
    )

    train = build_features(train)
    top_zips = train["incident_zip"].value_counts().head(50).index.tolist()
    test = build_features(test, top_zips=top_zips)

    num_cols = numeric_feature_columns()
    cat_cols = ["borough", "zip_binned"]
    feature_cols = cat_cols + num_cols

    X_train, X_test = train[feature_cols], test[feature_cols]

    # Encode string labels to integers (XGBoost requires this)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train["complaint_type"])
    y_test = label_encoder.transform(test["complaint_type"])
    class_names = list(label_encoder.classes_)

    dataset_info = {
        "csv_sha256_16": file_hash(csv_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "top_classes": top_classes,
        "label_encoding": "LabelEncoder (fit on train only)",
        "categorical_features": cat_cols,
        "numeric_features": num_cols,
        "excluded_features": ["descriptor", "agency", "location_type"],
        "git_sha": git_sha(),
    }

    # Each entry: (estimator, needs_dense)
    models = {
        "logreg": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            False,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
            ),
            False,
        ),
        "hist_gbdt": (
            HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.05, random_state=RANDOM_STATE
            ),
            True,
        ),
        "mlp": (
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=200,
                random_state=RANDOM_STATE,
                early_stopping=True,
            ),
            True,
        ),
    }

    try:
        import xgboost as xgb

        models["xgboost"] = (
            xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
                tree_method="hist",
            ),
            False,
        )
    except ImportError:
        pass

    try:
        import lightgbm as lgb

        models["lightgbm"] = (
            lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            False,
        )
    except ImportError:
        pass

    results = {}

    for name, (model, needs_dense) in models.items():
        with mlflow.start_run(run_name=f"cmp-{name}"):
            mlflow.log_params(
                {
                    "model": name,
                    "feature_set": "location+time",
                    "dense_input": str(needs_dense),
                }
            )
            mlflow.log_dict(dataset_info, "dataset.json")

            pre = build_preprocessor(cat_cols, num_cols, sparse_output=not needs_dense)
            pipe = Pipeline([("pre", pre), ("clf", model)])

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            # Convert encoded predictions back to class names for report/confusion
            preds_str = label_encoder.inverse_transform(preds.astype(int))
            y_test_str = label_encoder.inverse_transform(y_test.astype(int))
            m = evaluate(y_test_str, preds_str, class_names)

            mlflow.log_metrics(
                {
                    "accuracy": m["accuracy"],
                    "f1_macro": m["f1_macro"],
                }
            )
            mlflow.log_text(m["report"], "classification_report.txt")
            log_confusion(m["confusion"], top_classes, "confusion.png")

            results[name] = m
            print(f"{name:15s}  acc={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}")

    print("\n=== Model comparison ===")
    for name, m in sorted(results.items(), key=lambda x: -x[1]["f1_macro"]):
        print(f"{name:15s}  acc={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
