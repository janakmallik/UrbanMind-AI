"""Graph utilities for METR-LA."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def load_adjacency(
    path: str | Path | None = None,
    add_self_loops: bool = False,
) -> np.ndarray:
    """Load METR-LA sensor adjacency matrix.

    Parameters
    ----------
    path : str or Path, optional
        Path to adj_mx.pkl. If None, resolves via URBANINTEL_DATA_ROOT.
    add_self_loops : bool, default False
        If True, adds identity before normalization so each node also
        attends to itself. Explicit choice — not applied silently.

    Returns
    -------
    adj : np.ndarray, shape (207, 207)
        Row-normalized adjacency (each row sums to 1 if nonzero).
    """
    import os

    if path is None:
        data_root = os.environ.get("URBANINTEL_DATA_ROOT")
        if data_root is None:
            data_root = Path("C:/Users/Janak Mallik/ml-data")
        path = Path(data_root) / "metr-la" / "adj_mx.pkl"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Adjacency matrix not found at {path}. Run scripts/download_metr_la.ps1 first."
        )

    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    if not isinstance(obj, (list, tuple)) or len(obj) != 3:
        raise ValueError("Unexpected adjacency pickle structure")

    sensor_ids, id_to_index, adj = obj
    adj = np.asarray(adj, dtype=np.float32)

    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got {adj.shape}")

    if add_self_loops:
        adj = adj + np.eye(adj.shape[0], dtype=adj.dtype)

    # Row-normalize
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    adj_norm = adj / row_sum

    return adj_norm
