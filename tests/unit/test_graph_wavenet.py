"""Tests for Graph WaveNet: shapes, gradients, and message passing."""

import torch

from urbanintel.models.graph_wavenet import GraphConv, GraphWaveNet


def test_graph_conv_shapes() -> None:
    gc = GraphConv(channels=8, n_nodes=5)
    x = torch.randn(2, 5, 8)
    adj = torch.eye(5)
    y = gc(x, adj)
    assert y.shape == (2, 5, 8)
    assert torch.isfinite(y).all()


# verifies actual message-passing behavior, not just shapes
def test_graph_conv_with_identity_adj_matches_fixed_path() -> None:
    """With identity adjacency and alpha large positive, output = fixed path.

    We force sigmoid(alpha) -> 1 so the adaptive path contributes ~0,
    then verify output equals linear(x) applied through identity adjacency
    (which is just linear(x)).
    """
    torch.manual_seed(0)
    gc = GraphConv(channels=4, n_nodes=3)
    # sigmoid(+20) ~= 1, so fixed path dominates
    with torch.no_grad():
        gc.alpha.fill_(20.0)
    x = torch.randn(1, 3, 4)
    eye = torch.eye(3)
    y = gc(x, eye)
    expected = gc.linear(x)
    assert torch.allclose(y, expected, atol=1e-4)


# confirms learnable parameters train
def test_adaptive_adjacency_receives_gradient() -> None:
    """Verify nodevec parameters get gradients during a forward+backward."""
    gc = GraphConv(channels=4, n_nodes=3)
    x = torch.randn(1, 3, 4, requires_grad=True)
    adj = torch.eye(3)
    y = gc(x, adj)
    y.sum().backward()
    assert gc.nodevec1.grad is not None
    assert gc.nodevec1.grad.abs().sum() > 0


def test_graph_wavenet_output_shape() -> None:
    model = GraphWaveNet(n_sensors=10, n_horizons=3, input_length=12, hidden=16, n_blocks=2)
    x = torch.randn(4, 12, 10)
    adj = torch.eye(10)
    y = model(x, adj)
    assert y.shape == (4, 3, 10)
    assert torch.isfinite(y).all()


# proves the graph actually influences predictions
def test_graph_wavenet_changing_adj_changes_output() -> None:
    """A different adjacency must change the output."""
    torch.manual_seed(0)
    model = GraphWaveNet(n_sensors=5, n_horizons=3, input_length=12, hidden=8, n_blocks=2)
    model.eval()
    x = torch.randn(1, 12, 5)
    adj_a = torch.eye(5)
    adj_b = torch.ones(5, 5) / 5.0
    with torch.no_grad():
        y_a = model(x, adj_a)
        y_b = model(x, adj_b)
    assert not torch.allclose(y_a, y_b), "Graph structure had no effect on output"


def test_graph_wavenet_runs_with_sparse_adj() -> None:
    model = GraphWaveNet(n_sensors=5, n_horizons=3, input_length=12, hidden=8, n_blocks=2)
    x = torch.randn(2, 12, 5)
    adj = torch.tensor(
        [
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.5, 0.5],
        ]
    )
    y = model(x, adj)
    assert y.shape == (2, 3, 5)
