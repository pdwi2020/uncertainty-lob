"""EvLOB: Microstructure-gated Evidential LOB model (HEADLINE Track-A architecture).

Loss variants (select via --loss-fn):
  'bayes_risk'      (DEFAULT/WINNER) Sensoy 2018 MSE-based + annealed KL.
                    Best convergence in ablation (69.3% at 20 epochs vs UCE 56.8%).
  'ce_warmup_bayes' TWO-PHASE: pure CE for warm_epochs, then bayes_risk.
                    Closes gap to baseline by letting backbone converge with CE first.
  'uce'             UCE: NLL on Dirichlet mean + annealed KL (Prior et al. 2022).
                    Underperforms bayes_risk in ablation; kept for comparison.
  'ce_warmup'       Two-phase: pure CE for warm_epochs, then UCE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .cnn_lstm_attention import CNNLSTMAttention


class MicrostructureGate(nn.Module):
    def __init__(self, n_micro: int, n_classes: int = 3, hidden: int = 32, init_bias: float = 3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_micro, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes), nn.Sigmoid(),
        )
        # Init gate near 1.0 (sigmoid(3.0)≈0.95) so backbone trains freely early on
        with torch.no_grad():
            self.net[-2].bias.fill_(init_bias)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.net(m)


class EvidentialHead(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int = 3):
        super().__init__()
        self.W = nn.Linear(feat_dim, n_classes, bias=True)

    def forward(self, feat: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        e = F.softplus(self.W(feat))
        if gate is not None:
            e = e * gate
        return e


class EvLOB(nn.Module):
    """Microstructure-gated Evidential LOB model."""

    def __init__(
        self,
        input_channels: int = 40,
        lstm_hidden: int = 64,
        n_classes: int = 3,
        n_micro: int = 7,
        gate_hidden: int = 32,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.use_gate  = n_micro > 0

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

        self.evid_head = EvidentialHead(feat_dim, n_classes)
        self.gate = MicrostructureGate(n_micro, n_classes, gate_hidden) if self.use_gate else None

    def forward(
        self,
        x: torch.Tensor,
        m: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat  = self.backbone(x)
        gate  = self.gate(m) if (self.use_gate and m is not None) else None
        e     = self.evid_head(feat, gate)
        alpha = e + 1.0
        S     = alpha.sum(dim=1, keepdim=True)
        p_hat = alpha / S
        logits = torch.log(p_hat + 1e-8)
        return alpha, S, p_hat, logits

    def predict(self, x: torch.Tensor, m: Optional[torch.Tensor] = None) -> dict:
        alpha, S, p_hat, _ = self.forward(x, m)
        K = self.n_classes
        vacuity = K / S.squeeze(1)
        b = (alpha - 1.0) / S
        bal = torch.zeros(b.size(0), device=b.device)
        for i in range(K):
            for j in range(K):
                if i != j:
                    bal += b[:, i] * b[:, j] / (b[:, i] + b[:, j]).clamp(min=1e-8)
        dissonance = bal / (K * (K - 1) / 2)
        return {
            "alpha": alpha, "S": S.squeeze(1), "p_hat": p_hat,
            "vacuity": vacuity, "dissonance": dissonance,
            "pred": p_hat.argmax(dim=1),
        }


# ── Loss functions ─────────────────────────────────────────────────────────────

def edl_bayes_risk_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
    n_epochs: int,
    lambda_kl: float = 1.0,
) -> torch.Tensor:
    """Sensoy 2018 Bayes-risk loss.

    Uses n_epochs//4 ramp (vs //2) so KL builds more slowly, preserving accuracy.
    """
    B, K  = alpha.shape
    S     = alpha.sum(dim=1, keepdim=True)
    p_hat = alpha / S
    y_oh  = F.one_hot(labels, K).float()
    risk  = ((y_oh - p_hat) ** 2 + p_hat * (1 - p_hat) / (S + 1)).sum(dim=1)
    alpha_hat = y_oh + (1.0 - y_oh) * alpha
    kl    = _kl_dirichlet_uniform(alpha_hat)
    anneal = min(1.0, epoch / max(1, n_epochs // 4))   # slower KL ramp → better accuracy
    return (risk + anneal * lambda_kl * kl).mean()


def edl_ce_warmup_bayes_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
    n_epochs: int,
    lambda_kl: float = 1.0,
    warm_epochs: int = 20,
) -> torch.Tensor:
    """Two-phase loss: evidence-based CE for warm_epochs, then Bayes-risk EDL.

    Phase 1 (epoch < warm_epochs): F.cross_entropy(evidence, labels) where evidence
    = alpha - 1 = softplus(W*feat)*gate. This gives IDENTICAL gradient magnitude to
    standard softmax-CE — eliminates the 1/alpha attenuation of Dirichlet-CE.
    Phase 2 (epoch >= warm_epochs): switch to Bayes-risk EDL for evidential UQ.
    """
    if epoch < warm_epochs:
        # Evidence as CE logits: full gradient signal, no 1/alpha attenuation
        e = (alpha - 1.0).clamp(min=0.0)
        return F.cross_entropy(e, labels)
    return edl_bayes_risk_loss(alpha, labels, epoch - warm_epochs,
                               n_epochs - warm_epochs, lambda_kl)


def edl_uce_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
    n_epochs: int,
    lambda_kl: float = 1.0,
) -> torch.Tensor:
    """UCE loss: NLL on Dirichlet mean + annealed KL (Prior et al. NeurIPS 2022)."""
    B, K  = alpha.shape
    S     = alpha.sum(dim=1, keepdim=True)
    p_hat = alpha / S
    nll   = -torch.log(p_hat[torch.arange(B), labels] + 1e-8)
    y_oh      = F.one_hot(labels, K).float()
    alpha_hat = y_oh + (1.0 - y_oh) * alpha
    kl     = _kl_dirichlet_uniform(alpha_hat)
    anneal = min(1.0, epoch / max(1, n_epochs // 2))
    return (nll + anneal * lambda_kl * kl).mean()


def edl_ce_warmup_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
    n_epochs: int,
    lambda_kl: float = 1.0,
    warm_epochs: int = 20,
) -> torch.Tensor:
    """Two-phase: pure CE for warm_epochs, then UCE (legacy; use ce_warmup_bayes instead)."""
    B, K  = alpha.shape
    S     = alpha.sum(dim=1, keepdim=True)
    p_hat = alpha / S
    if epoch < warm_epochs:
        return F.nll_loss(torch.log(p_hat + 1e-8), labels)
    return edl_uce_loss(alpha, labels, epoch - warm_epochs,
                        n_epochs - warm_epochs, lambda_kl)


def _kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL[Dir(alpha) || Dir(1)] per sample. Shape: (B,)"""
    K           = alpha.shape[1]
    S           = alpha.sum(dim=1)
    log_B_alpha = torch.lgamma(alpha).sum(1) - torch.lgamma(S)
    log_B_1_val = -math.lgamma(K)
    psi_alpha   = torch.digamma(alpha) - torch.digamma(S.unsqueeze(1))
    kl = log_B_1_val - log_B_alpha + (psi_alpha * (alpha - 1.0)).sum(1)
    return kl.clamp(min=0.0)
