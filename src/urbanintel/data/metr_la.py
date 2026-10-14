"""METR-LA traffic dataset loader and preprocessing.

Design principles:
  - Keep raw values unchanged (including zeros).
  - Expose an explicit validity mask so downstream code can decide how to
    treat suspected missing readings.
  - Split chronologically, never randomly.
  - Build sliding windows within each partition to prevent cross-boundary leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Standard METR-LA benchmark parameters (DCRNN, Graph WaveNet)
DEFAULT_INPUT_LENGTH = 12  # 12 steps × 5 min = 60 min input window
DEFAULT_HORIZONS = (3, 6, 12)  # 15, 30, 60 minutes ahead

# Chronological split fractions
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.10
TEST_FRACTION = 0.20


@dataclass(frozen=True)
class METRLAWindowed:
    """Sliding-window dataset for METR-LA forecasting.

    Attributes
    ----------
    X : np.ndarray, shape (n_samples, input_length, n_sensors)
        Historical speed windows.
    y : np.ndarray, shape (n_samples, n_horizons, n_sensors)
        Future speed targets at each horizon.
    valid_mask : np.ndarray, shape (n_samples, n_horizons, n_sensors)
        True where the target is considered valid (currently: nonzero).
    timestamps : np.ndarray, shape (n_samples,)
        Timestamp of the **last input step** for each sample.
        For horizon h (steps ahead), the target timestamp is
        timestamps[i] + 5 * h minutes.
    """

    X: np.ndarray
    y: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray


def load_metr_la(
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load METR-LA HDF5 file as a (timesteps × sensors) DataFrame.

    Parameters
    ----------
    path : str or Path, optional
        Path to `metr-la.h5`. If None, resolves via URBANINTEL_DATA_ROOT.

    Returns
    -------
    df : pd.DataFrame
        Index = timestamps, columns = sensor IDs (strings). Values in mph.
    sensor_ids : list[str]
        Sensor IDs in column order (matches adjacency and locations files).
    """
    import os

    if path is None:
        data_root = os.environ.get("URBANINTEL_DATA_ROOT")
        if data_root is None:
            data_root = Path("C:/Users/Janak Mallik/ml-data")
        path = Path(data_root) / "metr-la" / "metr-la.h5"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"METR-LA HDF5 file not found at {path}. Run scripts/download_metr_la.ps1 first."
        )

    # NOTE: pd.read_hdf() is incompatible with this file when using
    # pandas 3.x + PyTables 3.11 (TypeError in pytables.py). Read with
    # h5py directly — the file structure is:
    #   df/axis0        -> sensor IDs (bytes)
    #   df/axis1        -> timestamps in nanoseconds (int64)
    #   df/block0_values-> speed matrix (float64, shape [T, S])
    import h5py

    with h5py.File(path, "r") as f:
        if "df" not in f:
            raise ValueError(f"Expected HDF5 group 'df' not found in {path}")
        group = f["df"]
        required = {"axis0", "axis1", "block0_values"}
        missing = required - set(group.keys())
        if missing:
            raise ValueError(f"HDF5 file is missing required keys: {missing}")

        raw_ids = group["axis0"][:]
        raw_ts = group["axis1"][:]
        values = group["block0_values"][:]

    # Decode sensor IDs (bytes -> str)
    sensor_ids = [v.decode().strip() if isinstance(v, bytes) else str(v).strip() for v in raw_ids]

    # Build timestamps. METR-LA stores nanoseconds since epoch.
    timestamps = pd.to_datetime(raw_ts, unit="ns")

    df = pd.DataFrame(values, index=timestamps, columns=sensor_ids)
    return df, sensor_ids


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRACTION,
    val_frac: float = VAL_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame chronologically.

    No shuffling. No overlap. The order of rows is preserved.
    """
    if not (0 < train_frac < 1) or not (0 < val_frac < 1):
        raise ValueError("split fractions must be in (0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def build_windows(
    df: pd.DataFrame,
    input_length: int = DEFAULT_INPUT_LENGTH,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> METRLAWindowed:
    """Construct sliding windows for supervised forecasting.

    For each time index `t` such that `t + input_length + max(horizons) <= len(df)`:
      X[i] = df[t : t+input_length]
      y[i, h_idx] = df[t + input_length + (horizons[h_idx] - 1)]

    The last valid start index ensures the longest horizon is reachable.
    """
    if len(horizons) == 0:
        raise ValueError("horizons must be non-empty")
    if any(h <= 0 for h in horizons):
        raise ValueError("all horizons must be positive")
    if input_length <= 0:
        raise ValueError("input_length must be positive")

    values = df.to_numpy(dtype=np.float32)
    n_timesteps, n_sensors = values.shape

    max_horizon = max(horizons)
    n_samples = n_timesteps - input_length - max_horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough timesteps ({n_timesteps}) for "
            f"input_length={input_length} and horizons={horizons}"
        )

    X = np.empty((n_samples, input_length, n_sensors), dtype=np.float32)
    y = np.empty((n_samples, len(horizons), n_sensors), dtype=np.float32)

    for i in range(n_samples):
        X[i] = values[i : i + input_length]
        for h_idx, h in enumerate(horizons):
            y[i, h_idx] = values[i + input_length + (h - 1)]

    # Validity mask: target is valid if all values at that step are nonzero.
    # We treat 0 as suspected missing (documented as a modeling choice).
    valid_mask = y != 0.0

    timestamps = df.index[input_length - 1 : input_length - 1 + n_samples].to_numpy()

    return METRLAWindowed(X=X, y=y, valid_mask=valid_mask, timestamps=timestamps)


def make_datasets(
    df: pd.DataFrame | None = None,
    input_length: int = DEFAULT_INPUT_LENGTH,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, METRLAWindowed]:
    """Split chronologically, then build sliding windows within each partition.

    If `df` is None, loads METR-LA from disk via `load_metr_la()`.
    """
    if df is None:
        df, _ = load_metr_la()
    train_df, val_df, test_df = chronological_split(df)
    return {
        "train": build_windows(train_df, input_length, horizons),
        "val": build_windows(val_df, input_length, horizons),
        "test": build_windows(test_df, input_length, horizons),
    }
