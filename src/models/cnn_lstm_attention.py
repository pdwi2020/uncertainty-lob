"""CNN-LSTM with Bahdanau attention for LOB forecasting (ablation model)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        # lstm_outputs: (B, T, hidden)
        scores = self.v(torch.tanh(self.W(lstm_outputs)))   # (B, T, 1)
        attn = F.softmax(scores, dim=1)                      # (B, T, 1)
        context = (attn * lstm_outputs).sum(dim=1)           # (B, hidden)
        return context


class CNNLSTMAttention(nn.Module):
    """CNN feature extractor + LSTM + Bahdanau attention for LOB forecasting."""

    def __init__(
        self,
        input_channels: int = 40,
        cnn_filters: Optional[List[int]] = None,
        cnn_kernel: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 3,
        use_attention: bool = True,
    ):
        super().__init__()
        if cnn_filters is None:
            cnn_filters = [32, 64, 64]
        self.use_attention = use_attention

        # CNN stack: 3 × (Conv1d → BN → ReLU)
        cnn_layers = []
        in_ch = input_channels
        for out_ch in cnn_filters:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1], hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = BahdanauAttention(lstm_hidden) if use_attention else None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 40) → (B, 40, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.cnn(x)                    # (B, 64, T)
        x = x.permute(0, 2, 1)            # (B, T, 64)
        lstm_out, (h_n, _) = self.lstm(x) # lstm_out: (B, T, hidden)
        if self.use_attention and self.attention is not None:
            context = self.attention(lstm_out)   # (B, hidden)
        else:
            context = h_n[-1]                    # last layer hidden: (B, hidden)
        return self.fc(self.dropout(context))    # (B, 3)

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
