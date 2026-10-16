"""Tests for METR-LA tabular feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from urbanintel.data.metr_la import METRLAWindowed
from urbanintel.features.metr_la_features import (
    build_training_table,
    compute_sensor_stats,
    feature_columns,
)


def make_windowed() -> METRLAWindowed:
    """Create small deterministic windows for feature-alignment tests."""
    X = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    y = np.array(
        [
            [[101, 102], [111, 112]],
            [[201, 202], [211, 212]],
            [[301, 302], [311, 312]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.ones_like(y, dtype=bool)
    valid_mask[2, 1, 0] = False

    return METRLAWindowed(
        X=X,
        y=y,
        valid_mask=valid_mask,
        timestamps=pd.date_range("2024-01-05 23:40", periods=3, freq="5min"),
    )


def make_sensor_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {"mean": [10.0, 20.0], "std": [2.0, 4.0]},
        index=["s0", "s1"],
    )


def test_feature_columns_exclude_targets_and_masks() -> None:
    columns = feature_columns(input_len=4)

    assert "y" not in columns
    assert "valid" not in columns
    assert columns[:8] == [
        "lag_h",
        "sensor_mean",
        "sensor_std",
        "hour",
        "dow",
        "is_weekend",
        "sensor_idx",
        "horizon",
    ]
    assert columns[8:] == ["lag_1", "lag_2", "lag_3", "lag_4"]


def test_lags_and_latest_value_are_aligned() -> None:
    ds = make_windowed()
    table = build_training_table(ds, ["s0", "s1"], (1,), make_sensor_stats())

    # First sample, sensor s0: lag_1 is newest; lag_4 is oldest.
    row = table[(table["sample_idx"] == 0) & (table["sensor_idx"] == 0)].iloc[0]

    assert row["lag_h"] == ds.X[0, -1, 0]
    assert row["lag_1"] == ds.X[0, -1, 0]
    assert row["lag_2"] == ds.X[0, -2, 0]
    assert row["lag_3"] == ds.X[0, -3, 0]
    assert row["lag_4"] == ds.X[0, -4, 0]


def test_targets_and_validity_match_selected_windows() -> None:
    ds = make_windowed()
    table = build_training_table(ds, ["s0", "s1"], (1, 2), make_sensor_stats())

    row = table[
        (table["sample_idx"] == 2) & (table["sensor_idx"] == 0) & (table["horizon"] == 2)
    ].iloc[0]

    assert row["y"] == ds.y[2, 1, 0]
    assert bool(row["valid"]) == bool(ds.valid_mask[2, 1, 0])


def test_sampled_indices_are_remapped_to_local_row_ids() -> None:
    ds = make_windowed()
    table = build_training_table(
        ds,
        ["s0", "s1"],
        (1,),
        make_sensor_stats(),
        sample_idx=np.array([2, 0]),
    )

    # The first selected source window (2) is assigned local ID 0.
    row = table[(table["sample_idx"] == 0) & (table["sensor_idx"] == 0)].iloc[0]
    assert row["y"] == ds.y[2, 0, 0]

    # The second selected source window (0) is assigned local ID 1.
    row = table[(table["sample_idx"] == 1) & (table["sensor_idx"] == 0)].iloc[0]
    assert row["y"] == ds.y[0, 0, 0]


def test_target_calendar_features_use_target_time() -> None:
    ds = make_windowed()
    table = build_training_table(ds, ["s0", "s1"], (1,), make_sensor_stats())

    row = table[(table["sample_idx"] == 0) & (table["sensor_idx"] == 0)].iloc[0]
    target_time = ds.timestamps[0] + pd.Timedelta(minutes=5)

    assert row["hour"] == target_time.hour
    assert row["dow"] == target_time.dayofweek
    assert row["is_weekend"] == int(target_time.dayofweek >= 5)


def test_sensor_stats_include_zeros() -> None:
    train_df = pd.DataFrame({"s0": [0.0, 2.0], "s1": [4.0, 8.0]})

    stats = compute_sensor_stats(train_df)

    assert stats.loc["s0", "mean"] == 1.0
    assert stats.loc["s0", "std"] == np.std([0.0, 2.0], ddof=1)


def test_pivot_reconstruction_preserves_sample_and_sensor_order() -> None:
    ds = make_windowed()
    table = build_training_table(ds, ["s0", "s1"], (1,), make_sensor_stats())

    # Give each sample/sensor pair a distinct prediction.
    table["pred"] = table["y"] + 1000.0

    y_true = table.pivot(index="sample_idx", columns="sensor_idx", values="y").to_numpy()
    y_pred = table.pivot(index="sample_idx", columns="sensor_idx", values="pred").to_numpy()
    valid = table.pivot(index="sample_idx", columns="sensor_idx", values="valid").to_numpy()

    np.testing.assert_array_equal(y_true, ds.y[:, 0, :])
    np.testing.assert_array_equal(y_pred, ds.y[:, 0, :] + 1000.0)
    np.testing.assert_array_equal(valid, ds.valid_mask[:, 0, :])
