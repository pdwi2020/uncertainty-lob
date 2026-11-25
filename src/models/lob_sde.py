"""LOB-SDE: Neural SDE with microstructure-driven diffusion for LOB forecasting.

Architecture:
  Latent Neural SDE  dz = f_θ(z,t)dt + σ_φ(z,t,m(x))dW
  where:
    f_θ  = drift MLP (2-layer, hidden=64)
    σ_φ  = diffusion MLP conditioned on microstructure m(x): aleatoric UQ source
    z(0) = h_0 from CNN-LSTM encoder (same backbone as EvLOB)
    z(T) → classification head

UQ quantities (N rollouts, see predict()):
  Predictive distribution: mean over {softmax(clf(z_i(T)))}
  Aleatoric (per-sample):  sum_k E_W[p̂_k (1 - p̂_k)] over the rollouts
  Epistemic (per-sample):  sum_k Var_W[p̂_k] across the rollouts
  Trajectory signal:       H¹ path energy ∫||dz/dt||²dt, reported separately

Training:
  Fixed-step Euler-Maruyama (dt=0.05, T=1.0 → 20 steps) for speed + stability
  Gradient clipping at 1.0; LR ≤ 1e-4
  Loss = CE(mean logits, y) + β·path_energy_reg (β=1e-4)

3-epoch verify gate:
  Call LOBSDETrainer.verify(loader) → {"stable": bool, "reason": str}
  If not stable, abort training and report honest negative.

Reference: Li et al., AISTATS 2020 (arXiv:2001.01338); Kidger et al., NeurIPS 2020
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .cnn_lstm_attention import CNNLSTMAttention


class DriftNet(nn.Module):
    """f_θ(z, t): latent drift, time-conditioned."""

    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, t: float) -> torch.Tensor:
        t_vec = torch.full((z.size(0), 1), t, device=z.device, dtype=z.dtype)
        return self.net(torch.cat([z, t_vec], dim=1))


class DiffusionNet(nn.Module):
    """σ_φ(z, t, m(x)): microstructure-driven diagonal diffusion.

    Output: positive diagonal of diffusion matrix ∈ R^{latent_dim}.
    Conditioning on m(x) gives per-sample aleatoric modulation.
    """

    def __init__(self, latent_dim: int, n_micro: int, hidden: int = 64):
        super().__init__()
        self.use_micro = n_micro > 0
        in_dim = latent_dim + 1 + (n_micro if self.use_micro else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
            nn.Softplus(),   # ensure σ > 0
        )

    def forward(
        self,
        z: torch.Tensor,
        t: float,
        m: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_vec = torch.full((z.size(0), 1), t, device=z.device, dtype=z.dtype)
        parts = [z, t_vec]
        if self.use_micro and m is not None:
            parts.append(m)
        return self.net(torch.cat(parts, dim=1))   # (B, latent_dim), σ > 0


class LOB_SDE(nn.Module):
    """Neural SDE with microstructure-driven diffusion for LOB classification.

    Args:
        n_micro:  dimensionality of m(x). 0 = pure drift SDE (ablation).
        n_rollouts: number of SDE rollouts for MC UQ at inference.
        dt:       Euler-Maruyama step size (T=1.0, steps=1/dt).
    """

    def __init__(
        self,
        input_channels: int = 40,
        lstm_hidden: int = 64,
        n_classes: int = 3,
        n_micro: int = 7,
        latent_dim: int = 64,
        sde_hidden: int = 64,
        dropout: float = 0.2,
        n_rollouts: int = 20,
        dt: float = 0.05,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.n_classes  = n_classes
        self.latent_dim = latent_dim
        self.n_rollouts = n_rollouts
        self.dt         = dt
        self.n_steps    = max(1, round(1.0 / dt))

        # CNN-LSTM encoder → initial latent state z(0)
        self.backbone = CNNLSTMAttention(
            input_channels=input_channels,
            cnn_filters=[32, 64, 64],
            lstm_hidden=lstm_hidden,
            lstm_layers=2,
            dropout=dropout,
            n_classes=n_classes,
            use_attention=True,
        )
        self.backbone.fc = nn.Identity()   # remove classification head
        # project lstm_hidden → latent_dim
        self.z0_proj = nn.Linear(lstm_hidden, latent_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # SDE components
        self.drift     = DriftNet(latent_dim, sde_hidden)
        self.diffusion = DiffusionNet(latent_dim, n_micro, sde_hidden)

        # Classification head on z(T)
        self.clf = nn.Sequential(
            nn.Linear(latent_dim, sde_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sde_hidden, n_classes),
        )

    def _euler_maruyama(
        self,
        z0: torch.Tensor,
        m: Optional[torch.Tensor],
        return_path_energy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single EM trajectory. Returns (z_T, path_energy)."""
        z = z0
        sqrt_dt = math.sqrt(self.dt)
        path_sq = torch.zeros(z.size(0), device=z.device)

        for step in range(self.n_steps):
            t = step * self.dt
            f = self.drift(z, t)
            sigma = self.diffusion(z, t, m)   # (B, latent_dim)
            dW = torch.randn_like(z) * sqrt_dt
            dz = f * self.dt + sigma * dW
            if return_path_energy:
                path_sq = path_sq + (dz ** 2).sum(dim=1)
            z = z + dz

        path_energy = path_sq / (self.n_steps * self.dt)   # time-normalized H¹
        return z, path_energy

    def forward(
        self,
        x: torch.Tensor,
        m: Optional[torch.Tensor] = None,
        n_rollouts: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward: single rollout, returns (logits, path_energy).

        For training: use single rollout to save memory.
        For inference: use predict() which runs n_rollouts.
        """
        feat = self.backbone(x)          # (B, lstm_hidden)
        z0   = self.z0_proj(feat)        # (B, latent_dim)
        z_T, path_energy = self._euler_maruyama(z0, m, return_path_energy=True)
        logits = self.clf(z_T)           # (B, n_classes)
        return logits, path_energy

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        m: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """MC UQ inference: runs n_rollouts, returns UQ dictionary."""
        feat = self.backbone(x)
        z0   = self.z0_proj(feat)

        all_probs   = []
        all_h1      = []

        for _ in range(self.n_rollouts):
            z_T, pe = self._euler_maruyama(z0, m, return_path_energy=True)
            probs = F.softmax(self.clf(z_T), dim=-1)
            all_probs.append(probs)
            all_h1.append(pe)

        probs_stack = torch.stack(all_probs, dim=0)    # (R, B, K)
        h1_stack    = torch.stack(all_h1,    dim=0)    # (R, B)

        mean_probs  = probs_stack.mean(0)              # (B, K)
        # Aleatoric: mean of per-rollout variance = E[p(1-p)]
        aleatoric   = (probs_stack * (1 - probs_stack)).mean(0).sum(-1)   # (B,)
        # Epistemic: variance of mean
        epistemic   = probs_stack.var(0).sum(-1)       # (B,)
        # H¹ path energy (mean over rollouts)
        h1_energy   = h1_stack.mean(0)                 # (B,)

        return {
            "p_hat":      mean_probs,
            "pred":       mean_probs.argmax(dim=1),
            "aleatoric":  aleatoric,
            "epistemic":  epistemic,
            "h1_energy":  h1_energy,
            "logits":     torch.log(mean_probs + 1e-8),
        }


# ── LOB-SDE Loss ──────────────────────────────────────────────────────────────

def lob_sde_loss(
    logits: torch.Tensor,
    path_energy: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 1e-4,
) -> torch.Tensor:
    """CE loss + path-energy regularization (prevents SDE from collapsing to ODE)."""
    ce   = F.cross_entropy(logits, labels)
    reg  = path_energy.mean()
    return ce + beta * reg


# ── Verify gate ───────────────────────────────────────────────────────────────

class LOBSDEVerifyGate:
    """3-epoch stability check before committing to full training.

    Passes if:
      - Loss decreases over 3 epochs (not diverging)
      - No NaN/Inf in gradients
      - Training accuracy > 40% (better than random on 3-class)
    """

    @staticmethod
    def run(
        model: LOB_SDE,
        loader,
        device: torch.device,
        n_epochs: int = 3,
        lr: float = 1e-4,
    ) -> Dict:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        grad_ok = True

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_correct  = 0
            n_total    = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits, pe = model(x)
                loss = lob_sde_loss(logits, pe, y)
                loss.backward()

                # Check gradient health
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            grad_ok = False
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                if total_norm > 1e6:
                    grad_ok = False

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item() * x.size(0)
                n_correct  += (logits.argmax(1) == y).sum().item()
                n_total    += x.size(0)

            avg_loss = epoch_loss / max(n_total, 1)
            losses.append(avg_loss)

        loss_decreasing = len(losses) < 2 or losses[-1] < losses[0]
        acc_ok = (n_correct / max(n_total, 1)) > 0.40

        stable = grad_ok and loss_decreasing and acc_ok
        reason_parts = []
        if not grad_ok:         reason_parts.append("NaN/Inf gradients or norm > 1e6")
        if not loss_decreasing: reason_parts.append(f"loss not decreasing ({losses[0]:.4f} → {losses[-1]:.4f})")
        if not acc_ok:          reason_parts.append(f"accuracy below 40% threshold")

        return {
            "stable":  stable,
            "reason":  "; ".join(reason_parts) if reason_parts else "OK",
            "losses":  losses,
            "n_epochs": n_epochs,
        }
