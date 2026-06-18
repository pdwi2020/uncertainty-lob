"""LOB-Transformer: Transformer encoder with [CLS] token for LOB forecasting."""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class LOBTransformer(nn.Module):
    """2-layer Transformer encoder for LOB sequence classification.

    Uses [CLS] token prepended to sequence; classification from CLS output.
    Dropout enabled for MC Dropout inference.
    """

    def __init__(
        self,
        input_dim: int = 40,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        n_classes: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.classifier = nn.Linear(d_model, n_classes)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        proj = self.input_proj(x)                              # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, d_model)
        seq = torch.cat([cls, proj], dim=1)                   # (B, T+1, d_model)
        seq = self.pos_enc(seq)
        out = self.encoder(seq)                               # (B, T+1, d_model)
        return self.classifier(out[:, 0])                     # CLS → (B, 3)

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
