"""Feature engineering for METR-LA forecasting.

Builds per-(sample, horizon, sensor) feature rows for tabular models.
All features use only information available at prediction time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from urbanintel.data.metr_la import METRLAWindowed


def build_training_table(
    ds: METRLAWindowed,
    sensor_ids: list[str],
    horizons: tuple[int, ...],
    sensor_stats: pd.DataFrame,
    sample_idx: np.ndarray | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a flat feature table for the given sample indices.

    If `sample_idx` is None, uses all samples. To subsample training data,
    pass a pre-selected array of indices — this avoids building the full
    table in memory before subsampling.
    """
    n_samples, input_len, n_sensors = ds.X.shape
    assert n_sensors == len(sensor_ids)

    if sample_idx is None:
        sample_idx = np.arange(n_samples, dtype=np.int32)
    n_use = len(sample_idx)

    sensor_mean = sensor_stats["mean"].reindex(sensor_ids).to_numpy(dtype=np.float32)
    sensor_std = sensor_stats["std"].reindex(sensor_ids).to_numpy(dtype=np.float32)

    timestamps = pd.to_datetime(ds.timestamps)

    rows = []
    for h_idx, h in enumerate(horizons):
        y_h = ds.y[sample_idx, h_idx, :]
        valid_h = ds.valid_mask[sample_idx, h_idx, :]

        target_ts = timestamps[sample_idx] + pd.Timedelta(minutes=5 * h)
        hour = target_ts.hour.to_numpy()
        dow = target_ts.dayofweek.to_numpy()
        is_weekend = (dow >= 5).astype(np.int8)

        lag_h = ds.X[sample_idx, -1, :]

        for s_idx, s_id in enumerate(sensor_ids):
            frame = pd.DataFrame(
                {
                    "sample_idx": np.arange(n_use, dtype=np.int32),
                    "sensor_idx": s_idx,
                    "horizon": h,
                    "lag_h": lag_h[:, s_idx],
                    "sensor_mean": sensor_mean[s_idx],
                    "sensor_std": sensor_std[s_idx],
                    "hour": hour,
                    "dow": dow,
                    "is_weekend": is_weekend,
                    "y": y_h[:, s_idx],
                    "valid": valid_h[:, s_idx],
                }
            )
            for lag in range(1, input_len + 1):
                frame[f"lag_{lag}"] = ds.X[sample_idx, -lag, s_idx]
            rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def compute_sensor_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor mean/std computed on training data only.

    Zeros are included here — this matches the raw data as recorded. The
    target-validity mask (used for evaluation) is a separate concern.
    """
    return pd.DataFrame(
        {
            "mean": train_df.mean(),
            "std": train_df.std(),
        }
    )


def feature_columns(input_len: int = 12) -> list[str]:
    """Return the list of numeric feature columns."""
    cols = [
        "lag_h",
        "sensor_mean",
        "sensor_std",
        "hour",
        "dow",
        "is_weekend",
        "sensor_idx",
        "horizon",
    ]
    cols += [f"lag_{i}" for i in range(1, input_len + 1)]
    return cols
