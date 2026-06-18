"""TCN: Temporal Convolutional Network with dilated causal convolutions (Bai et al. 2018)."""

import torch
import torch.nn as nn
from typing import List


class CausalConv1d(nn.Module):
    """Causal convolution with padding to maintain sequence length."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Trim right padding to enforce causality
        return out[:, :, :-self.padding] if self.padding else out


class TCNBlock(nn.Module):
    """Single TCN residual block: 2× (CausalConv1d → BN → ReLU → Dropout) + residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.residual(x))


class TCN(nn.Module):
    """TCN for LOB forecasting.

    Dilations [1, 2, 4, 8, 16, 32] → receptive field = 2×(3-1)×(1+2+4+8+16+32) = 252 > 100.
    Dropout enabled for MC Dropout inference.
    """

    def __init__(
        self,
        input_channels: int = 40,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        n_classes: int = 3,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64] * 6
        dilations = [2 ** i for i in range(len(num_channels))]

        layers = []
        in_ch = input_channels
        for out_ch, d in zip(num_channels, dilations):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, d, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 40) → (B, 40, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.network(x)          # (B, 64, T)
        x = x[:, :, -1]             # last time step: (B, 64)
        return self.fc(x)            # (B, 3)

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
