"""DeepLOB: Deep CNN + LSTM for LOB forecasting (Zhang et al. 2019)."""

import torch
import torch.nn as nn
from typing import Tuple


class _InceptionBlock(nn.Module):
    """Inception-style block: 3 parallel convolutions concatenated."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # 3 parallel branches; each outputs out_ch // 3 channels
        branch_ch = out_ch // 3
        self.b1 = nn.Sequential(nn.Conv1d(in_ch, branch_ch, kernel_size=1, padding=0), nn.LeakyReLU())
        self.b2 = nn.Sequential(nn.Conv1d(in_ch, branch_ch, kernel_size=3, padding=1), nn.LeakyReLU())
        self.b3 = nn.Sequential(nn.Conv1d(in_ch, branch_ch, kernel_size=5, padding=2), nn.LeakyReLU())
        self.out_ch = branch_ch * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)


class DeepLOB(nn.Module):
    """Inception-style CNN backbone + LSTM + FC for 3-class LOB prediction.

    Architecture:
        Input: (B, T, 40) where T=100 (sequence), 40=LOB features
        CNN: 3 inception blocks → MaxPool → (B, 96, T')
        LSTM: 1 layer, hidden 64
        Dropout + FC: Linear(64, 3)

    Dropout is applied after LSTM for MC Dropout inference.
    """

    def __init__(
        self,
        input_channels: int = 40,
        cnn_out_channels: int = 96,
        lstm_hidden: int = 64,
        n_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        # CNN: 3 inception blocks (input→96→96→96) with MaxPool between blocks
        self.cnn = nn.Sequential(
            _InceptionBlock(input_channels, cnn_out_channels),
            nn.MaxPool1d(kernel_size=2, stride=2),
            _InceptionBlock(cnn_out_channels, cnn_out_channels),
            nn.MaxPool1d(kernel_size=2, stride=2),
            _InceptionBlock(cnn_out_channels, cnn_out_channels),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 40) → (B, 40, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.cnn(x)                       # (B, 96, T')
        x = x.permute(0, 2, 1)               # (B, T', 96) for LSTM
        _, (h_n, _) = self.lstm(x)           # h_n: (1, B, hidden)
        out = self.dropout(h_n.squeeze(0))   # (B, hidden)
        return self.fc(out)                  # (B, 3)

    def enable_dropout(self):
        """Enable dropout layers for MC Dropout inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
