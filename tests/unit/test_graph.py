"""Tests for graph utilities (self-contained, no local dataset needed)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from urbanintel.data.graph import load_adjacency


@pytest.fixture
def synthetic_adj_pkl(tmp_path: Path) -> Path:
    """Create a tiny synthetic adjacency file for testing."""
    sensor_ids = ["s0", "s1", "s2"]
    id_map = {"s0": 0, "s1": 1, "s2": 2}
    adj = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "adj.pkl"
    with open(path, "wb") as f:
        pickle.dump([sensor_ids, id_map, adj], f)
    return path


def test_load_adjacency_shape_and_normalization(synthetic_adj_pkl: Path) -> None:
    adj = load_adjacency(synthetic_adj_pkl, add_self_loops=False)
    assert adj.shape == (3, 3)
    row_sums = adj.sum(axis=1)
    assert np.allclose(row_sums[row_sums > 0], 1.0, atol=1e-5)
    assert np.isfinite(adj).all()


def test_add_self_loops_changes_result(synthetic_adj_pkl: Path) -> None:
    without = load_adjacency(synthetic_adj_pkl, add_self_loops=False)
    with_loops = load_adjacency(synthetic_adj_pkl, add_self_loops=True)
    # Diagonal entries should be higher with self-loops
    assert with_loops[0, 0] > without[0, 0]
    # Rows still sum to 1
    assert np.allclose(with_loops.sum(axis=1), 1.0, atol=1e-5)


def test_isolated_node_handled(tmp_path: Path) -> None:
    """A node with no edges should stay all-zero after normalization."""
    sensor_ids = ["s0", "s1"]
    adj = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    path = tmp_path / "isolated.pkl"
    with open(path, "wb") as f:
        pickle.dump([sensor_ids, {"s0": 0, "s1": 1}, adj], f)
    result = load_adjacency(path)
    assert np.allclose(result, 0.0)
