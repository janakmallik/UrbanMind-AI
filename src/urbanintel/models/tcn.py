"""Temporal Convolutional Network for multi-horizon forecasting.

Architecture (per sensor, shared across all sensors):
    Input:  (batch, input_length, n_sensors)
    Output: (batch, n_horizons, n_sensors)

Design:
    - Dilated causal convolutions preserve temporal order.
    - Residual connections for stable training.
    - Weights shared across sensors (so few parameters, generalizes well).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1D convolution that never looks ahead in time."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_ch, time)
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual block: two causal convs + GELU + dropout + skip connection.

    No normalization — GroupNorm/BatchNorm would leak future information
    across the time dimension in a causal architecture.
    """

    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel, dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel, dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.act(h)
        h = self.drop(h)

        h = self.conv2(h)
        h = self.act(h)
        h = self.drop(h)

        return x + h


class TCNForecaster(nn.Module):
    """Multivariate temporal CNN for multi-horizon speed forecasting.

    All 207 sensors are treated as input channels; the network learns
    cross-sensor relationships through shared convolutional filters.
    Weights are NOT shared independently across sensors — the model is
    multivariate, not a per-sensor independent forecaster.

    Architecture:
        Input:  (batch, input_length, n_sensors)
        Output: (batch, n_horizons, n_sensors)
    """

    def __init__(
        self,
        n_sensors: int,
        n_horizons: int,
        input_length: int = 12,
        hidden: int = 64,
        kernel: int = 3,
        n_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.n_horizons = n_horizons

        self.input_proj = nn.Conv1d(n_sensors, hidden, kernel_size=1)

        # Dilated causal blocks: 1, 2 (2 blocks by default).
        # expand to 4, 8 if validation supports deeper model.
        dilations = [2**i for i in range(n_blocks)]
        self.blocks = nn.ModuleList([TCNBlock(hidden, kernel, d, dropout) for d in dilations])

        self.head = nn.Conv1d(hidden, n_sensors * n_horizons, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_length, n_sensors) -> (batch, n_horizons, n_sensors)"""
        x = x.transpose(1, 2)  # (B, S, T)
        h = self.input_proj(x)  # (B, H, T)
        for block in self.blocks:
            h = block(h)
        h_last = h[:, :, -1]  # (B, H)
        out = self.head(h_last.unsqueeze(-1)).squeeze(-1)  # (B, S*H_out)
        out = out.view(-1, self.n_sensors, self.n_horizons)
        return out.transpose(1, 2)  # (B, H_out, S)
