"""Tests for METR-LA loader, split, and windowing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from urbanintel.data.metr_la import (
    build_windows,
    chronological_split,
    load_metr_la,
    make_datasets,
)


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Synthetic 500-timestep, 5-sensor DataFrame.

    500 timesteps is enough for a 70/10/20 split to give every partition
    more than input_length + max(horizons) = 24 timesteps.
    """
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=500, freq="5min")
    data = rng.uniform(10, 70, size=(500, 5)).astype(np.float32)
    cols = [f"s{i}" for i in range(5)]
    return pd.DataFrame(data, index=idx, columns=cols)


def test_chronological_split_boundaries(tiny_df: pd.DataFrame) -> None:
    train, val, test = chronological_split(tiny_df, train_frac=0.7, val_frac=0.1)
    assert len(train) == 350
    assert len(val) == 50
    assert len(test) == 100
    # Order preserved, no overlap
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_chronological_split_rejects_bad_fractions(tiny_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        chronological_split(tiny_df, train_frac=1.0, val_frac=0.1)
    with pytest.raises(ValueError):
        chronological_split(tiny_df, train_frac=0.6, val_frac=0.5)


def test_build_windows_shapes(tiny_df: pd.DataFrame) -> None:
    w = build_windows(tiny_df, input_length=12, horizons=(3, 6, 12))
    expected_n = 500 - 12 - 12 + 1
    assert w.X.shape == (expected_n, 12, 5)
    assert w.y.shape == (expected_n, 3, 5)
    assert w.valid_mask.shape == (expected_n, 3, 5)
    assert len(w.timestamps) == expected_n


def test_build_windows_target_alignment(tiny_df: pd.DataFrame) -> None:
    """Verify target at horizon h equals value at t + input_length + h - 1."""
    input_length = 12
    horizons = (3, 6, 12)
    w = build_windows(tiny_df, input_length=input_length, horizons=horizons)
    values = tiny_df.to_numpy(dtype=np.float32)

    # Sample 0: input starts at index 0
    for h_idx, h in enumerate(horizons):
        expected = values[input_length + h - 1]
        np.testing.assert_allclose(w.y[0, h_idx], expected, rtol=1e-6)


def test_valid_mask_flags_zeros(tiny_df: pd.DataFrame) -> None:
    """If any target is zero, valid_mask should be False at that position."""
    df = tiny_df.copy()
    df.iloc[20, 0] = 0.0  # zero out a specific cell
    w = build_windows(df, input_length=12, horizons=(3, 6, 12))

    # Find windows whose target includes row 20 at any horizon
    for i in range(len(w.X)):
        for h_idx, h in enumerate((3, 6, 12)):
            target_row = i + 12 + h - 1
            if target_row == 20:
                assert w.valid_mask[i, h_idx, 0] == False  # noqa: E712


def test_make_datasets_no_cross_partition_leakage(tiny_df: pd.DataFrame) -> None:
    """Verify no window spans across partition boundaries."""
    datasets = make_datasets(tiny_df, input_length=12, horizons=(3, 6, 12))

    train_ts_max = datasets["train"].timestamps.max()
    val_ts_min = datasets["val"].timestamps.min()
    val_ts_max = datasets["val"].timestamps.max()
    test_ts_min = datasets["test"].timestamps.min()

    # Each partition's windows start after the previous partition's windows end
    assert train_ts_max < val_ts_min
    assert val_ts_max < test_ts_min


def test_load_metr_la_raises_when_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_metr_la(tmp_path / "nonexistent.h5")
