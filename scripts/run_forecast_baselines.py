"""Naive forecasting baselines for METR-LA.

Two baselines:
  1. Persistence: y_hat(t+h) = y(t)  [last input value]
  2. Sensor-hour historical mean: average speed at (sensor, hour) over train

Metrics are masked MAE / RMSE / MAPE per horizon. Results logged to MLflow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

# Zero-value policy (provisional):
#   - Targets equal to 0.0 are treated as SUSPECTED MISSING and excluded
#     from metrics via valid_mask.
#   - Inputs (the historical window) are used as-is, including zeros.
#   - The sensor-hour historical mean is computed over ALL training values,
#     including zeros.
#
# This policy is documented in docs/dataset_cards/metr_la.md, section 6.
# It will be revisited if the source dataset's zero convention is confirmed.

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = (_REPO_ROOT / "mlflow.db").as_posix()
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_DB_PATH}"
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlflow
import numpy as np

from urbanintel.data.metr_la import make_datasets
from urbanintel.evaluation.forecasting import masked_mae, masked_mape, masked_rmse

EXPERIMENT_NAME = "urbanintel-week2"
HORIZONS = (3, 6, 12)  # steps ahead


def _log_metrics(prefix: str, y_true, y_pred, valid) -> dict:
    """Compute and log masked MAE/RMSE/MAPE for each horizon."""
    metrics = {}
    for h_idx, h in enumerate(HORIZONS):
        mae = masked_mae(y_true[:, h_idx], y_pred[:, h_idx], valid[:, h_idx])
        rmse = masked_rmse(y_true[:, h_idx], y_pred[:, h_idx], valid[:, h_idx])
        mape = masked_mape(y_true[:, h_idx], y_pred[:, h_idx], valid[:, h_idx])
        mlflow.log_metric(f"{prefix}_mae_h{h}", mae)
        mlflow.log_metric(f"{prefix}_rmse_h{h}", rmse)
        mlflow.log_metric(f"{prefix}_mape_h{h}", mape)
        metrics[f"h{h}"] = {"mae": mae, "rmse": rmse, "mape": mape}
        print(f"  {prefix} h={h:2d}  MAE={mae:7.4f}  RMSE={rmse:7.4f}  MAPE={mape:6.2f}%")
    return metrics


def persistence_forecast(ds) -> np.ndarray:
    """Predict y(t+h) = last value of the input window, for all horizons."""
    last = ds.X[:, -1, :]  # (n_samples, n_sensors)
    n_samples, _, n_sensors = ds.y.shape
    # Broadcast to (n_samples, n_horizons, n_sensors)
    return np.broadcast_to(last[:, None, :], (n_samples, len(HORIZONS), n_sensors)).copy()


def sensor_hour_mean(train_df, test_timestamps) -> np.ndarray:
    """Predict each target using its sensor's training-only hour-of-day mean.

    test_timestamps contains the timestamp of the last input observation.

    For horizon h (steps ahead), the target timestamp is:
        t_last_input + 5 * h minutes

    Each horizon uses the hour-of-day of its own target timestamp.

    Returns:
        Predictions with shape (n_samples, n_horizons, n_sensors).
    """
    means = train_df.groupby(train_df.index.hour).mean()

    n_samples = len(test_timestamps)
    n_sensors = len(train_df.columns)

    preds = np.empty(
        (n_samples, len(HORIZONS), n_sensors),
        dtype=np.float32,
    )

    for i, ts in enumerate(test_timestamps):
        for h_idx, horizon in enumerate(HORIZONS):
            target_ts = pd.Timestamp(ts) + pd.Timedelta(minutes=5 * horizon)
            target_hour = target_ts.hour
            preds[i, h_idx] = means.loc[target_hour].to_numpy(dtype=np.float32)

    return preds


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading and windowing METR-LA ...")
    datasets = make_datasets()
    train_ds = datasets["train"]
    test_ds = datasets["test"]

    print(f"Train windows: {train_ds.X.shape}, Test windows: {test_ds.X.shape}")

    # --- Persistence ---
    print("\nBaseline: persistence")
    with mlflow.start_run(run_name="persistence"):
        mlflow.log_params(
            {
                "baseline": "persistence",
                "input_length": train_ds.X.shape[1],
                "horizons": str(HORIZONS),
                "test_samples": test_ds.X.shape[0],
            }
        )
        preds = persistence_forecast(test_ds)
        _log_metrics("persistence", test_ds.y, preds, test_ds.valid_mask)

    # --- Sensor-hour historical mean ---
    print("\nBaseline: sensor-hour historical mean")
    # Need raw DataFrames to compute hourly means
    from urbanintel.data.metr_la import chronological_split, load_metr_la

    df, _ = load_metr_la()
    train_df, _, _ = chronological_split(df)

    with mlflow.start_run(run_name="sensor-hour-mean"):
        mlflow.log_params(
            {
                "baseline": "sensor-hour historical mean",
                "input_length": train_ds.X.shape[1],
                "horizons": str(HORIZONS),
                "test_samples": test_ds.X.shape[0],
                "fit_source": "train only",
            }
        )
        preds = sensor_hour_mean(train_df, test_ds.timestamps)
        _log_metrics("sensor_hour", test_ds.y, preds, test_ds.valid_mask)

    print("\nDone.")


if __name__ == "__main__":
    main()
