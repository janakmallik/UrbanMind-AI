"""Graph WaveNet-lite: spatial graph convolution + temporal TCN.

Correct flow per block:
    Input:  (B, N, C, T)
    Step 1: Graph conv at EVERY timestep -> (B, N, C, T)
    Step 2: Temporal causal conv per node   -> (B, N, C, T)
    Step 3: Residual + activation + dropout

This design lets each sensor exchange information with its neighbors at
every historical moment, then temporal convolution learns how the
spatially-enriched signal evolves.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from urbanintel.models.tcn import CausalConv1d


class GraphConv(nn.Module):
    """Graph convolution: H = A_fixed @ X + A_adaptive @ X, then Linear.

    Both adjacencies are applied to the SAME linear projection so their
    scales are comparable. Adaptive adjacency is softmax-normalized so it
    stays bounded in [0, 1].
    """

    def __init__(self, channels: int, n_nodes: int, embed_dim: int = 10):
        super().__init__()
        self.linear = nn.Linear(channels, channels)
        # Adaptive adjacency via learnable node embeddings
        self.nodevec1 = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.1)
        self.nodevec2 = nn.Parameter(torch.randn(embed_dim, n_nodes) * 0.1)
        # Blend weight between fixed and adaptive adjacency.
        # The forward pass applies sigmoid(alpha), so an initial value of 0.5
        # gives a starting blend of ~62% fixed / 38% adaptive. The parameter
        # is learnable, so the initial value is not critical.
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C) -> (B, N, C)"""
        h = self.linear(x)

        # Fixed adjacency message
        h_fixed = torch.einsum("nm,bmc->bnc", adj, h)

        # Adaptive adjacency message (bounded by softmax)
        adapt = torch.softmax(torch.relu(self.nodevec1 @ self.nodevec2), dim=-1)
        h_adapt = torch.einsum("nm,bmc->bnc", adapt, h)

        # Blend with learnable weight
        alpha = torch.sigmoid(self.alpha)
        return alpha * h_fixed + (1 - alpha) * h_adapt


class GraphWaveNetBlock(nn.Module):
    """Spatial (graph) at each timestep, then temporal (causal conv)."""

    def __init__(self, channels: int, n_nodes: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.graph = GraphConv(channels, n_nodes)
        self.temporal = CausalConv1d(channels, channels, kernel=3, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C, T) -> (B, N, C, T)"""
        residual = x
        B, N, C, T = x.shape

        # --- Spatial: graph conv at every timestep ---
        # Reshape to put time in batch: (B*T, N, C)
        x_perm = x.permute(0, 3, 1, 2).reshape(B * T, N, C)
        g = self.graph(x_perm, adj)  # (B*T, N, C)
        g = g.reshape(B, T, N, C).permute(0, 2, 3, 1)  # back to (B, N, C, T)

        # --- Temporal: per-node causal conv over time ---
        # (B, N, C, T) -> (B*N, C, T) -> conv -> back
        g_flat = g.reshape(B * N, C, T)
        h = self.temporal(g_flat).reshape(B, N, C, T)

        out = self.act(h)
        out = self.drop(out)
        return residual + out


class GraphWaveNet(nn.Module):
    """Input projection + stacked spatiotemporal blocks + output head."""

    def __init__(
        self,
        n_sensors: int,
        n_horizons: int,
        input_length: int = 12,
        hidden: int = 64,
        n_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.n_horizons = n_horizons

        # Project each node's 1D time series to hidden channels
        self.input_proj = nn.Conv1d(1, hidden, kernel_size=1)

        dilations = [2**i for i in range(n_blocks)]
        self.blocks = nn.ModuleList(
            [GraphWaveNetBlock(hidden, n_sensors, d, dropout) for d in dilations]
        )

        # Head: hidden -> n_horizons, applied to the last timestep
        self.head = nn.Linear(hidden, n_horizons)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N) -> (B, n_horizons, N)"""
        B, T, N = x.shape
        # (B, T, N) -> (B, N, T) -> (B*N, 1, T)
        x = x.transpose(1, 2).reshape(B * N, 1, T)

        # Per-node projection along time
        h = self.input_proj(x)  # (B*N, C, T)
        h = h.reshape(B, N, -1, T)  # (B, N, C, T)

        for block in self.blocks:
            h = block(h, adj)

        # Last timestep -> (B, N, C)
        h_last = h[:, :, :, -1]
        out = self.head(h_last)  # (B, N, H)
        return out.transpose(1, 2)  # (B, H, N)
