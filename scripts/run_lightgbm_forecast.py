"""Train LightGBM per-horizon forecasters on METR-LA lag features.

Compares against persistence baseline. Logs to MLflow under urbanintel-week2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = (_REPO_ROOT / "mlflow.db").as_posix()
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_DB_PATH}"
sys.path.insert(0, str(_REPO_ROOT / "src"))

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from urbanintel.data.metr_la import (
    DEFAULT_HORIZONS,
    DEFAULT_INPUT_LENGTH,
    chronological_split,
    load_metr_la,
    make_datasets,
)
from urbanintel.evaluation.forecasting import masked_mae, masked_mape, masked_rmse
from urbanintel.features.metr_la_features import (
    build_training_table,
    compute_sensor_stats,
    feature_columns,
)

EXPERIMENT_NAME = "urbanintel-week2"
MAX_TRAIN_ROWS = 2_000_000  # subsample for tractability
SEED = 42


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading METR-LA and building windowed datasets ...")
    df, sensor_ids = load_metr_la()
    datasets = make_datasets(df)
    train_ds = datasets["train"]
    test_ds = datasets["test"]

    # Sensor stats from raw training data (zeros treated as missing)
    train_df, _, _ = chronological_split(df)
    stats = compute_sensor_stats(train_df)
    print(f"Sensor stats computed for {len(stats)} sensors")

    feat_cols = feature_columns(DEFAULT_INPUT_LENGTH)
    print(f"Feature columns ({len(feat_cols)}): {feat_cols}")

    # Build training and test tables per horizon
    print("Building training table (subsampled) ...")
    # Decide which training windows to use BEFORE building features.
    # This caps peak memory at roughly MAX_TRAIN_ROWS / (3 * 207) windows.
    rng = np.random.default_rng(SEED)
    n_train_samples = train_ds.X.shape[0]
    # Each window contributes 3 horizons × 207 sensors = 621 rows.
    rows_per_sample = len(DEFAULT_HORIZONS) * len(sensor_ids)
    max_samples = min(n_train_samples, MAX_TRAIN_ROWS // rows_per_sample)
    train_sample_idx = rng.choice(n_train_samples, size=max_samples, replace=False)
    train_sample_idx.sort()
    print(
        f"Selected {max_samples:,} / {n_train_samples:,} training windows "
        f"(≈ {max_samples * rows_per_sample:,} rows)"
    )

    train_table = build_training_table(
        train_ds,
        sensor_ids,
        DEFAULT_HORIZONS,
        stats,
        sample_idx=train_sample_idx,
    )
    print(f"Train rows: {len(train_table):,}")

    # NOTE: We do NOT build the full test table up front.
    # Per-horizon tables are built inside the loop below to keep memory bounded.

    # --- Train one LightGBM per horizon ---
    per_horizon_preds: dict[int, pd.DataFrame] = {}

    for h in DEFAULT_HORIZONS:
        with mlflow.start_run(run_name=f"lightgbm-h{h}"):
            mlflow.log_params(
                {
                    "model": "LightGBM",
                    "horizon_steps": h,
                    "horizon_minutes": 5 * h,
                    "input_length": DEFAULT_INPUT_LENGTH,
                    "max_train_rows": MAX_TRAIN_ROWS,
                    "n_features": len(feat_cols),
                    "seed": SEED,
                }
            )

            # Training rows for this horizon
            train_h = train_table[(train_table["horizon"] == h) & train_table["valid"]]
            X_train = train_h[feat_cols].to_numpy(dtype=np.float32)
            y_train = train_h["y"].to_numpy(dtype=np.float32)
            print(f"\n[horizon={h}] train rows={len(X_train):,}")

            model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=20,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_train, y_train)
            del X_train, y_train, train_h

            # Build test features for this horizon only
            test_h = build_training_table(
                test_ds,
                sensor_ids,
                (h,),
                stats,
                sample_idx=None,
            )
            X_test = test_h[feat_cols].to_numpy(dtype=np.float32)
            preds = model.predict(X_test).astype(np.float32)
            test_h["pred"] = preds
            del X_test

            # Reshape to (n_samples, 1, n_sensors)
            y_true = test_h.pivot(index="sample_idx", columns="sensor_idx", values="y").to_numpy()[
                :, None, :
            ]
            y_pred = test_h.pivot(
                index="sample_idx", columns="sensor_idx", values="pred"
            ).to_numpy()[:, None, :]
            valid = test_h.pivot(
                index="sample_idx", columns="sensor_idx", values="valid"
            ).to_numpy()[:, None, :]

            mae = masked_mae(y_true, y_pred, valid)
            rmse = masked_rmse(y_true, y_pred, valid)
            mape = masked_mape(y_true, y_pred, valid)

            mlflow.log_metrics(
                {
                    f"lightgbm_mae_h{h}": mae,
                    f"lightgbm_rmse_h{h}": rmse,
                    f"lightgbm_mape_h{h}": mape,
                }
            )
            print(f"[horizon={h}] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

            per_horizon_preds[h] = test_h[
                ["sample_idx", "sensor_idx", "y", "pred", "valid", "lag_h"]
            ].copy()
            del test_h

    # --- Summary: LightGBM vs persistence on the same test targets ---
    # Compute both predictions against the SAME (y_true, valid) arrays,
    # using the raw test_ds so shapes and masks align exactly.
    from urbanintel.data.metr_la import DEFAULT_HORIZONS as HORIZONS_REF

    print("\n=== Summary (LightGBM vs persistence, same test data) ===")
    print(f"{'horizon':>8}  {'lightgbm_mae':>13}  {'persistence_mae':>16}  {'improvement':>12}")

    for h_idx, h in enumerate(HORIZONS_REF):
        # Ground truth and validity for this horizon
        y_true = test_ds.y[:, h_idx, :][:, None, :]
        valid = test_ds.valid_mask[:, h_idx, :][:, None, :]

        # Persistence: last input step, broadcast
        y_per = test_ds.X[:, -1, :][:, None, :]

        # LightGBM: reconstruct from per_horizon_preds
        test_h = per_horizon_preds[h]
        y_lgb = test_h.pivot(index="sample_idx", columns="sensor_idx", values="pred").to_numpy()[
            :, None, :
        ]

        mae_lgb = masked_mae(y_true, y_lgb, valid)
        mae_per = masked_mae(y_true, y_per, valid)
        imp = 100 * (mae_per - mae_lgb) / mae_per if mae_per > 0 else float("nan")
        print(f"{h:>8}  {mae_lgb:>13.4f}  {mae_per:>16.4f}  {imp:>11.1f}%")


if __name__ == "__main__":
    main()
