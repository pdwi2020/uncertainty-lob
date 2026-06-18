"""RAC-MoE: Regime-Adaptive Calibration Mixture of Experts (ablation comparator).

Architecture:
  2 experts (low-vol / high-vol regime) + softmax gate from HMM regime features.
  Each expert is a CNN-LSTM-Attention head; gate is a 2-layer MLP over the
  same microstructure feature vector m(x) used by EvLOB.
  Per-expert temperature T_r (learned scalar, initialized to 1.0).

Purpose:
  Ablation comparator for EvLOB. Shows that naive regime-gated MoE is NOT
  equivalent to microstructure-gated evidential learning:
    - MoE soft mixture vs. evidential modulation of evidence
    - Fixed expert heads vs. Dirichlet uncertainty
    - Regime gate vs. OFI gate

This is *not* a novel contribution — it serves as a hard baseline.
Three seeds, identical train loop to EvLOB.

Forward:
  logits = Σ_r gate_r(m) * T_r^{-1} * expert_r(x)   (soft mixture of scaled logits)
  (gate from sigmoid if 2 experts, normalized to sum to 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .cnn_lstm_attention import CNNLSTMAttention


class ExpertHead(nn.Module):
    """Single classification expert sharing a backbone."""

    def __init__(self, feat_dim: int, n_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)   # (B, n_classes), raw logits


class RegimeGate(nn.Module):
    """MLP gate: m(x) → softmax mixing weights over n_experts."""

    def __init__(self, n_micro: int, n_experts: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_micro, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(m), dim=-1)   # (B, n_experts)


class RAC_MoE(nn.Module):
    """Regime-Adaptive Calibration Mixture of Experts.

    Args:
        n_micro:       dimensionality of microstructure feature m(x).
        n_experts:     number of expert heads (default 2 for low/high-vol).
        freeze_backbone: if True, freeze the shared CNN-LSTM backbone weights.
    """

    def __init__(
        self,
        input_channels: int = 40,
        lstm_hidden: int = 64,
        n_classes: int = 3,
        n_micro: int = 7,
        n_experts: int = 2,
        gate_hidden: int = 32,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_experts = n_experts

        # Shared backbone (feature extractor)
        self.backbone = CNNLSTMAttention(
            input_channels=input_channels,
            cnn_filters=[32, 64, 64],
            lstm_hidden=lstm_hidden,
            lstm_layers=2,
            dropout=dropout,
            n_classes=n_classes,
            use_attention=True,
        )
        feat_dim = lstm_hidden
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Per-expert classification heads
        self.experts = nn.ModuleList([
            ExpertHead(feat_dim, n_classes, dropout)
            for _ in range(n_experts)
        ])

        # Per-expert learnable temperature (log-parameterized for positivity)
        self.log_temps = nn.Parameter(torch.zeros(n_experts))   # T = exp(log_T) > 0, init=1

        # Regime gate
        self.gate = RegimeGate(n_micro, n_experts, gate_hidden)

    def forward(
        self,
        x: torch.Tensor,
        m: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, gate_weights).

        logits: (B, n_classes) — soft mixture of temperature-scaled expert logits
        gate_weights: (B, n_experts) — for analysis / logging
        """
        feat = self.backbone(x)   # (B, lstm_hidden)

        # Expert logits scaled by per-expert temperature
        expert_logits = []
        temps = torch.exp(self.log_temps)   # (n_experts,), all > 0
        for i, expert in enumerate(self.experts):
            raw = expert(feat)              # (B, n_classes)
            scaled = raw / temps[i]
            expert_logits.append(scaled)
        expert_logits = torch.stack(expert_logits, dim=1)   # (B, n_experts, n_classes)

        # Gate weights
        if m is not None:
            gate_w = self.gate(m)           # (B, n_experts)
        else:
            # Uniform mixture if no microstructure features
            gate_w = torch.ones(feat.size(0), self.n_experts, device=feat.device) / self.n_experts

        # Weighted sum: (B, n_experts, 1) * (B, n_experts, n_classes) → (B, n_classes)
        logits = (gate_w.unsqueeze(-1) * expert_logits).sum(dim=1)
        return logits, gate_w

    def predict(
        self,
        x: torch.Tensor,
        m: Optional[torch.Tensor] = None,
    ) -> dict:
        logits, gate_w = self.forward(x, m)
        probs = F.softmax(logits, dim=-1)
        temps = torch.exp(self.log_temps)
        return {
            "logits":      logits,
            "p_hat":       probs,
            "pred":        probs.argmax(dim=1),
            "gate_weights": gate_w,
            "temperatures": temps.detach(),
        }
