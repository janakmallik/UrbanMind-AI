"""Tune LightGBM with Optuna on the leakage-free NYC 311 task.

Search space:
  n_estimators, learning_rate, num_leaves, min_child_samples,
  subsample, colsample_bytree, reg_alpha, reg_lambda

Objective: maximize macro-F1 on the validation set.
Then retrain on train+val and report test metrics.

Logs the parent run + best params + best val metric + test metrics to MLflow.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = (_REPO_ROOT / "mlflow.db").as_posix()
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_DB_PATH}"

sys.path.insert(0, str(_REPO_ROOT / "src"))

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
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
N_TRIALS = 15


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()[:12]
    except Exception:
        return "unknown"


def build_preprocessor(cat_cols, num_cols):
    return ColumnTransformer(
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


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    csv_path = dataset_path("nyc311", "nyc311_sample.csv")
    df = load_nyc311(csv_path, validate=False).dropna(subset=["complaint_type"]).copy()

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
    val = build_features(val, top_zips=top_zips)
    test = build_features(test, top_zips=top_zips)

    num_cols = numeric_feature_columns()
    cat_cols = ["borough", "zip_binned"]
    feature_cols = cat_cols + num_cols

    X_train = train[feature_cols]
    X_val = val[feature_cols]
    X_test = test[feature_cols]
    y_train = train["complaint_type"].values
    y_val = val["complaint_type"].values
    y_test = test["complaint_type"].values

    pre = build_preprocessor(cat_cols, num_cols)
    X_train_p = pre.fit_transform(X_train)
    X_val_p = pre.transform(X_val)
    X_test_p = pre.transform(X_test)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, step=15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        clf = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        clf.fit(X_train_p, y_train)
        preds = clf.predict(X_val_p)
        return f1_score(y_val, preds, average="macro", zero_division=0)

    with mlflow.start_run(run_name="optuna-lightgbm") as parent:
        mlflow.log_params(
            {
                "n_trials": N_TRIALS,
                "objective": "f1_macro",
                "feature_set": "location+time",
                "git_sha": git_sha(),
            }
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

        best_params = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_f1_macro", study.best_value)

        # Retrain on train+val with best params, evaluate on test
        # Handle sparse vs dense correctly
        import scipy.sparse as sp

        if sp.issparse(X_train_p):
            X_trainval = sp.vstack([X_train_p, X_val_p]).tocsr()
        else:
            X_trainval = np.vstack([X_train_p, X_val_p])

        y_trainval = np.concatenate([y_train, y_val])

        best_clf = lgb.LGBMClassifier(
            **best_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        best_clf.fit(X_trainval, y_trainval)
        preds = best_clf.predict(X_test_p)

        test_acc = accuracy_score(y_test, preds)
        test_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_macro", test_f1)

        print()
        print(f"Best val f1_macro: {study.best_value:.4f}")
        print(f"Test accuracy:     {test_acc:.4f}")
        print(f"Test f1_macro:     {test_f1:.4f}")
        print(f"Best params: {best_params}")


if __name__ == "__main__":
    main()
