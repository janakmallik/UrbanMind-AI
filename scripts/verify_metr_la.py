"""Verify METR-LA files: shapes, timestamps, sensor IDs, and consistency.

Exit code 0 if all checks pass, 1 otherwise.
Run after scripts/download_metr_la.ps1.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DATA_ROOT = Path("C:/Users/Janak Mallik/ml-data/metr-la")
H5_PATH = DATA_ROOT / "metr-la.h5"
ADJ_PATH = DATA_ROOT / "adj_mx.pkl"
LOC_PATH = DATA_ROOT / "graph_sensor_locations.csv"

EXPECTED_TIMESTEPS = 34272
EXPECTED_SENSORS = 207
EXPECTED_INTERVAL_NS = 300 * 1_000_000_000


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def info(msg: str) -> None:
    print(f"INFO: {msg}")


def main() -> None:
    # --- File existence ---
    for path in (H5_PATH, ADJ_PATH, LOC_PATH):
        if not path.is_file():
            fail(f"missing file: {path}")
        ok(f"exists: {path.name} ({path.stat().st_size / 1e6:.2f} MB)")

    # --- HDF5 data ---
    try:
        with h5py.File(H5_PATH, "r") as file:
            if "df" not in file:
                fail("HDF5 file does not contain the expected 'df' group")

            group = file["df"]
            required_keys = {"axis0", "axis1", "block0_values"}

            if not required_keys.issubset(group.keys()):
                fail(f"HDF5 group is missing keys: {required_keys - set(group.keys())}")

            x = group["block0_values"][:]
            raw_timestamps = group["axis1"][:]
            ids_h5 = [
                value.decode().strip() if isinstance(value, bytes) else str(value).strip()
                for value in group["axis0"][:]
            ]
    except (OSError, KeyError, ValueError) as exc:
        fail(f"cannot read HDF5 data: {exc}")

    if x.shape != (EXPECTED_TIMESTEPS, EXPECTED_SENSORS):
        fail(f"unexpected HDF5 shape: {x.shape}")
    ok(f"HDF5 shape: {x.shape}")

    if len(ids_h5) != EXPECTED_SENSORS:
        fail(f"unexpected number of sensor IDs: {len(ids_h5)}")

    if len(set(ids_h5)) != EXPECTED_SENSORS:
        fail("HDF5 contains duplicate sensor IDs")
    ok(f"sensor IDs: {len(ids_h5)} unique")

    if not np.isfinite(x).all():
        fail("HDF5 contains NaN or infinite values")
    ok("all traffic values are finite")

    # --- Timestamps ---
    if len(raw_timestamps) != EXPECTED_TIMESTEPS:
        fail(f"unexpected timestamp count: {len(raw_timestamps)}")

    try:
        timestamps = pd.to_datetime(raw_timestamps)
    except (ValueError, TypeError, OverflowError) as exc:
        fail(f"cannot parse timestamps: {exc}")

    intervals = np.diff(timestamps.asi8)

    if not np.all(intervals == EXPECTED_INTERVAL_NS):
        fail("timestamps contain gaps, duplicates, or irregular intervals")
    ok(f"regular 5-minute sampling: {len(timestamps)} timestamps")

    info(f"time range: {timestamps.min()} -> {timestamps.max()}")

    # --- Zero-value diagnostics ---
    zero_fraction = float((x == 0).mean())
    all_zero_rows = int(np.all(x == 0, axis=1).sum())

    info(f"zero-valued readings: {zero_fraction:.2%}")
    info(f"timestamps where all sensors are zero: {all_zero_rows}")
    info("Zero values are reported for investigation, not automatically treated as missing.")

    # --- Adjacency matrix ---
    try:
        with open(ADJ_PATH, "rb") as file:
            obj = pickle.load(file, encoding="latin1")
    except (OSError, pickle.UnpicklingError, EOFError, ValueError) as exc:
        fail(f"cannot read adjacency file: {exc}")

    if not isinstance(obj, (list, tuple)) or len(obj) != 3:
        fail("unexpected adjacency pickle structure; expected three elements")

    adj_ids, mapping, adj = obj
    adj = np.asarray(adj, dtype=float)

    if adj.shape != (EXPECTED_SENSORS, EXPECTED_SENSORS):
        fail(f"unexpected adjacency shape: {adj.shape}")
    ok(f"adjacency shape: {adj.shape}")

    if not np.isfinite(adj).all():
        fail("adjacency matrix contains NaN or infinite values")
    ok(f"adjacency values are finite; nonzero entries: {np.count_nonzero(adj)}")

    adj_ids = [
        value.decode().strip() if isinstance(value, bytes) else str(value).strip()
        for value in adj_ids
    ]

    if len(set(adj_ids)) != EXPECTED_SENSORS:
        fail("adjacency data contains duplicate sensor IDs")

    if adj_ids != ids_h5:
        fail("adjacency ID order does not match HDF5 column order")
    ok("adjacency sensor IDs match HDF5 order")

    # --- Sensor locations ---
    try:
        locs = pd.read_csv(LOC_PATH)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        fail(f"cannot read locations CSV: {exc}")

    required_columns = {"sensor_id", "latitude", "longitude"}
    if not required_columns.issubset(locs.columns):
        fail(f"locations CSV is missing columns: {required_columns - set(locs.columns)}")

    location_ids = locs["sensor_id"].astype(str).str.strip().tolist()

    if len(locs) != EXPECTED_SENSORS:
        fail(f"unexpected locations row count: {len(locs)}")

    if len(set(location_ids)) != EXPECTED_SENSORS:
        fail("locations CSV contains duplicate sensor IDs")

    if location_ids != ids_h5:
        fail("locations CSV sensor order does not match HDF5 column order")
    ok("locations CSV sensor IDs match HDF5 order")

    coordinates = locs[["latitude", "longitude"]].to_numpy(dtype=float)

    if not np.isfinite(coordinates).all():
        fail("locations CSV contains non-finite coordinates")

    latitudes = coordinates[:, 0]
    longitudes = coordinates[:, 1]

    if not np.all((-90 <= latitudes) & (latitudes <= 90)):
        fail("locations CSV contains invalid latitudes")

    if not np.all((-180 <= longitudes) & (longitudes <= 180)):
        fail("locations CSV contains invalid longitudes")
    ok("location coordinates are finite and within valid ranges")

    # --- Sensor-to-index mapping ---
    if not isinstance(mapping, dict):
        fail("adjacency sensor mapping is not a dictionary")

    mapping_ids = {
        key.decode().strip() if isinstance(key, bytes) else str(key).strip(): value
        for key, value in mapping.items()
    }

    if set(mapping_ids) != set(ids_h5):
        fail("mapping sensor IDs do not match HDF5 sensor IDs")

    if any(not isinstance(index, (int, np.integer)) for index in mapping_ids.values()):
        fail("mapping contains non-integer indices")

    if sorted(mapping_ids.values()) != list(range(EXPECTED_SENSORS)):
        fail("mapping indices do not cover 0 through 206")

    for index, sensor_id in enumerate(adj_ids):
        if mapping_ids[sensor_id] != index:
            fail(f"mapping index mismatch for sensor {sensor_id}")
    ok("sensor-to-index mapping is consistent with adjacency order")

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
