"""Neural CDE for LOB forecasting — stabilised version.

    dZ(t) = f_θ(Z(t)) dX(t),   X(t) = cubic spline through [time | LOB features].

Stability fixes vs the original (adjoint=True, unbounded field) that exploded to loss≈5e5
and, once over-damped with clamp(±3)/3 → [-1,1], plateaued at ~54% accuracy:
  1. Hermite backward-difference spline coefficients — local, non-ringing derivatives that
     prevent the ±2.5M coefficient blow-up of natural cubic splines (falls back to natural
     cubic if the installed torchcde lacks the Hermite helper).
  2. Magnitude-preserving clamp(±7) instead of clamp(±3)/3 — keeps the LOB volume-spike
     information the baselines exploit (the [-1,1] squashing was the accuracy ceiling).
  3. tanh-bounded vector field, tanh initial state, LayerNorm before readout, grad-clip (runner).
  4. adjoint=False by default (adjoint backprop was a second instability source).

This MUST stay in sync with ``compute/runpod_ncde.py`` (the self-contained pod trainer);
the checkpoint produced there loads into this class — identical layer names and shapes.

Uncertainty proxy: H¹ trajectory norm ∫‖dZ/dt‖² dt — a high-norm trajectory means the model
is traversing an unusual region of hidden-state space → defer / abstain.
"""

import torch
import torch.nn as nn
import torchcde

CLAMP = 7.0  # keep in sync with compute/runpod_ncde.py
N_STEPS = 20  # CDE integration steps (fix: previously a single coarse RK4 step via t=X.interval)


def _build_spline(x: torch.Tensor, T: int, device):
    """Magnitude-preserving clamp + prepended normalised time + cubic spline.

    Shared by ``NeuralCDE.forward`` AND ``compute_h1_norms`` so the H¹ trajectory norm is
    computed on exactly the path the model was trained on (these previously diverged).
    """
    x = torch.clamp(x, -CLAMP, CLAMP)
    t = torch.arange(T, dtype=torch.float32, device=device)  # dt=1 → dX/dt sane
    t_ch = t[None, :, None].expand(x.shape[0], T, 1)
    x_aug = torch.cat([t_ch, x], dim=-1)  # (B, T, 41)
    if hasattr(torchcde, "hermite_cubic_coefficients_with_backward_differences"):
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_aug, t)
    else:
        coeffs = torchcde.natural_cubic_spline_coeffs(x_aug, t)
    return torchcde.CubicSpline(coeffs), x


class CDEFunc(nn.Module):
    """Vector field f_θ: R^hidden → R^(hidden × input_channels), tanh-bounded."""

    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.Tanh(),
            nn.Linear(hidden_channels * 2, hidden_channels * input_channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z)).view(
            z.shape[0], self.hidden_channels, self.input_channels
        )


class NeuralCDE(nn.Module):
    """Neural CDE for 3-class LOB mid-price prediction."""

    def __init__(
        self,
        input_channels: int = 40,
        hidden_channels: int = 128,
        output_channels: int = 3,
        adjoint: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cde_input_channels = input_channels + 1  # +1 time channel
        self.hidden_channels = hidden_channels
        self.adjoint = adjoint

        self.cde_func = CDEFunc(self.cde_input_channels, hidden_channels)
        self.initial = nn.Linear(input_channels, hidden_channels)
        self.readout = nn.Linear(hidden_channels, output_channels)
        self.drop = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 40) → logits (B, 3). Returns plain logits (matches the other models).
        B, T, _ = x.shape
        X, xc = _build_spline(x, T, x.device)
        z0 = torch.tanh(self.initial(xc[:, 0, :]))
        t_grid = torch.linspace(0, T - 1, N_STEPS + 1, device=x.device)  # N_STEPS RK4 steps, not 1
        z_path = torchcde.cdeint(
            X=X, func=self.cde_func, z0=z0, t=t_grid,
            adjoint=self.adjoint, method="rk4", rtol=1e-3, atol=1e-4,
        )
        return self.readout(self.drop(self.final_norm(z_path[:, -1])))


def compute_h1_norms(model: "NeuralCDE", loader, device):
    """H¹ trajectory norm as an uncertainty proxy. Higher = more uncertain.

    Uses the SAME ``_build_spline`` preprocessing and tanh initial state as ``forward``.
    """
    import numpy as np

    model.eval()
    norms = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            B, T, _ = x.shape
            X, xc = _build_spline(x, T, device)
            z0 = torch.tanh(model.initial(xc[:, 0, :]))
            t_eval = torch.linspace(0, float(T - 1), N_STEPS + 1, device=device)  # match forward integration density
            z_path = torchcde.cdeint(
                X=X, func=model.cde_func, z0=z0, t=t_eval,
                adjoint=False, method="rk4", rtol=1e-3, atol=1e-4,
            )  # (B, 10, hidden)
            dz = z_path[:, 1:] - z_path[:, :-1]
            norms.append((dz ** 2).sum(dim=-1).sum(dim=-1).cpu().numpy())
    return np.concatenate(norms)
