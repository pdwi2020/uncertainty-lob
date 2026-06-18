"""E2b — Learned microstructure-gated conformal head (HEADLINE / TNNLS component).

Trains ψ=MLP_θ(m(x)) with ConfTS-style set-efficiency loss on saved val logits + features.
No base-model retraining — purely post-hoc.

Nonconformity score: s(x,y) = 1 − p̂_y + ψ(x)
ψ(x) = σ(MLP(m(x))),  MLP: R^7 → R^1  (sigmoid output, ψ ∈ (0,1))

Training objective (on cal_val, with q̂ fixed from cal_fit):
  L = C · max(0, (1−α) − cov_soft)²  +  β · E_x[Σ_k σ((q̂ − s(x,k))/τ)]
Gradient flows through ψ(x) on cal_val; q̂ is a detached float from cal_fit.

Split: same as E2a — calib = first N//2 val samples, eval = second N//2.
Within calib: 80% cal_fit / 20% cal_val.
Features standardized on calib (no eval leakage).

Comparison ladder (α=0.10, 90% target coverage):
  1. LAC           — s = 1−p̂_y, λ=0
  2. Toxicity-closed— s = 1−p̂_y + λ*·ψ_cf(x)  (λ* loaded from E2a JSON)
  3. Learned gate  — s = 1−p̂_y + ψ_MLP(x)     (this script, HEADLINE)

Metrics: marginal_coverage, mean_set_size, singleton_rate, coverage_gap (Mondrian diagnostic).
Also reports per-regime ψ distribution to validate microstructure grounding.

Saves:
  results/learned_gate_results.json
  results/learned_gate_psi_weights.pt   (state_dict + scaler per model)

Usage:
  python experiments/learned_gate.py
  python experiments/learned_gate.py --alpha 0.10 --n-epochs 300 --beta 0.10
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"


# ─── Conformal utilities ─────────────────────────────────────────────────────

def get_q_hat(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(scores, min(level, 1.0), method="higher"))


def make_sets(probs: np.ndarray, psi: np.ndarray, q_hat: float) -> np.ndarray:
    """Hard prediction sets: include class k if 1 − p̂_k + ψ(x) ≤ q̂."""
    return (1.0 - probs + psi[:, None]) <= q_hat


def coverage_stats(
    sets: np.ndarray,
    labels: np.ndarray,
    regimes: np.ndarray,
    psi_eval: np.ndarray = None,
) -> dict:
    N = len(labels)
    marginal = float(sets[np.arange(N), labels].mean())
    sizes = sets.sum(axis=1)
    per = {}
    psi_by_regime = {}
    if regimes is not None:
        for r in sorted(np.unique(regimes)):
            mask = regimes == r
            name = "high_vol" if r == 1 else "low_vol"
            per[name] = {
                "coverage":      round(float(sets[np.arange(N), labels][mask].mean()), 4),
                "n":             int(mask.sum()),
                "mean_set_size": round(float(sizes[mask].mean()), 4),
                "singleton":     round(float((sizes[mask] == 1).mean()), 4),
            }
            if psi_eval is not None:
                psi_by_regime[name] = {
                    "psi_mean": round(float(psi_eval[mask].mean()), 4),
                    "psi_std":  round(float(psi_eval[mask].std()),  4),
                }
    gap = 0.0
    if "low_vol" in per and "high_vol" in per:
        gap = abs(per["low_vol"]["coverage"] - per["high_vol"]["coverage"])
    result = {
        "marginal_coverage": round(marginal, 4),
        "mean_set_size":     round(float(sizes.mean()), 4),
        "singleton_rate":    round(float((sizes == 1).mean()), 4),
        "coverage_gap":      round(gap, 4),
        "per_regime":        per,
    }
    if psi_by_regime:
        result["psi_by_regime"] = psi_by_regime
    return result


# ─── MLP ψ-gate ──────────────────────────────────────────────────────────────

class PsiGate(nn.Module):
    """Learned microstructure toxicity gate: m(x) → ψ ∈ (0,1)."""

    def __init__(self, in_dim: int = 7, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(m)).squeeze(-1)


# ─── Differentiable training loss ────────────────────────────────────────────

def soft_coverage_and_size(
    probs_t: torch.Tensor,
    labels_t: torch.Tensor,
    psi_t: torch.Tensor,
    q_hat: float,
    tau: float,
) -> tuple:
    """Differentiable coverage + set size via sigmoid approximation."""
    p_true = probs_t[torch.arange(len(labels_t)), labels_t]
    s_true = 1.0 - p_true + psi_t
    soft_cov = torch.sigmoid((q_hat - s_true) / tau).mean()

    s_all = 1.0 - probs_t + psi_t.unsqueeze(1)   # (N, K)
    soft_size = torch.sigmoid((q_hat - s_all) / tau).sum(dim=1).mean()

    return soft_cov, soft_size


def train_psi_gate(
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    calib_m: np.ndarray,
    alpha: float = 0.10,
    n_epochs: int = 200,
    beta: float = 0.10,
    tau: float = 0.05,
    cov_penalty: float = 100.0,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple:
    """Train MLP ψ-gate. Returns (gate, m_mean, m_std, q_hat_final)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    m_mean = calib_m.mean(axis=0)
    m_std  = calib_m.std(axis=0) + 1e-8
    m_norm = (calib_m - m_mean) / m_std

    n_fit  = int(len(calib_labels) * 0.8)
    fit_probs  = torch.from_numpy(calib_probs[:n_fit]).float()
    fit_labels = torch.from_numpy(calib_labels[:n_fit]).long()
    fit_m      = torch.from_numpy(m_norm[:n_fit]).float()
    val_probs  = torch.from_numpy(calib_probs[n_fit:]).float()
    val_labels = torch.from_numpy(calib_labels[n_fit:]).long()
    val_m      = torch.from_numpy(m_norm[n_fit:]).float()

    gate = PsiGate(in_dim=calib_m.shape[1])
    opt  = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=1e-5)

    best_loss  = float("inf")
    best_state = {k: v.clone() for k, v in gate.state_dict().items()}
    no_improve = 0

    for epoch in range(n_epochs):
        # q̂ fixed from cal_fit (no gradient)
        gate.eval()
        with torch.no_grad():
            psi_fit_np = gate(fit_m).numpy()
        lac_fit = 1.0 - fit_probs[torch.arange(len(fit_labels)), fit_labels].numpy()
        scores_fit = lac_fit + psi_fit_np
        q_hat = get_q_hat(scores_fit, alpha)

        # Gradient signal on cal_val
        gate.train()
        opt.zero_grad()
        psi_val = gate(val_m)
        soft_cov, soft_size = soft_coverage_and_size(
            val_probs, val_labels, psi_val, q_hat, tau
        )
        target_cov = 1.0 - alpha
        cov_viol = F.relu(torch.tensor(target_cov, dtype=torch.float32) - soft_cov)
        loss = cov_penalty * cov_viol ** 2 + beta * soft_size
        loss.backward()
        nn.utils.clip_grad_norm_(gate.parameters(), max_norm=1.0)
        opt.step()

        l_val = loss.item()
        if l_val < best_loss - 1e-6:
            best_loss  = l_val
            best_state = {k: v.clone() for k, v in gate.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 50 == 0:
            print(f"    ep {epoch+1:3d}  q̂={q_hat:.4f}  cov≈{soft_cov.item():.4f}  "
                  f"|C|≈{soft_size.item():.4f}  loss={l_val:.4f}")

        if no_improve >= 40:
            print(f"    Early stop ep {epoch+1}")
            break

    gate.load_state_dict(best_state)
    gate.eval()

    # Final q̂ on full calib with learned ψ
    m_norm_t = torch.from_numpy(m_norm).float()
    with torch.no_grad():
        psi_calib = gate(m_norm_t).numpy()
    lac_c = 1.0 - calib_probs[np.arange(len(calib_labels)), calib_labels]
    q_hat_final = get_q_hat(lac_c + psi_calib, alpha)

    return gate, m_mean, m_std, q_hat_final


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--alpha",    type=float, default=0.10)
    parser.add_argument("--n-epochs", type=int,   default=200)
    parser.add_argument("--beta",     type=float, default=0.10,
                        help="Efficiency penalty weight")
    parser.add_argument("--tau",      type=float, default=0.05,
                        help="Sigmoid temperature for differentiable set size")
    parser.add_argument("--cov-penalty", type=float, default=100.0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    alpha = args.alpha
    print(f"α={alpha}  epochs={args.n_epochs}  β={args.beta}  τ={args.tau}")

    # Load MACC features (val set)
    macc   = np.load(results_dir / "macc_features.npz")
    r_val  = macc["r_val"].astype(np.int64)
    m_val  = macc["m_val"].astype(np.float32)
    psi_cf = macc["psi_val"].astype(np.float32)   # closed-form ψ ∈ (0,1)

    # Load E2a lambda_star values
    e2a_lambda = {}
    e2a_path = results_dir / "micro_conformal_results.json"
    if e2a_path.exists():
        with open(e2a_path) as f:
            e2a = json.load(f)
        for mn, mres in e2a.items():
            e2a_lambda[mn] = mres.get("lambda_star", 0.30)
    print(f"E2a λ*: {e2a_lambda}")

    model_names = ["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention"]
    all_results = {}
    all_weights = {}

    for model_name in model_names:
        ckpt = results_dir / f"{model_name}_k10_logits.pt"
        if not ckpt.exists():
            print(f"[SKIP] {model_name}: {ckpt.name} missing")
            continue

        d      = torch.load(ckpt, map_location="cpu", weights_only=False)
        logits = d["logits"].float()
        labels = d["labels"].long().numpy()
        probs  = F.softmax(logits, dim=-1).numpy()

        N   = len(labels)
        n_c = N // 2
        calib_probs,  eval_probs   = probs[:n_c],  probs[n_c:]
        calib_labels, eval_labels  = labels[:n_c], labels[n_c:]
        calib_m,      eval_m       = m_val[:n_c],  m_val[n_c:]
        calib_psi_cf, eval_psi_cf  = psi_cf[:n_c], psi_cf[n_c:]
        calib_reg,    eval_reg     = r_val[:n_c],   r_val[n_c:]

        lam_cf = e2a_lambda.get(model_name, 0.30)
        print(f"\n══ {model_name}  N={N:,}  calib={n_c:,}  eval={N-n_c:,} ══")

        # ── 1. LAC ──────────────────────────────────────────────────────────
        lac_c    = 1.0 - calib_probs[np.arange(n_c), calib_labels]
        q_lac    = get_q_hat(lac_c, alpha)
        sets_lac = make_sets(eval_probs, np.zeros(len(eval_labels)), q_lac)
        stats_lac = coverage_stats(sets_lac, eval_labels, eval_reg)
        print(f"  LAC       q̂={q_lac:.4f}: cov={stats_lac['marginal_coverage']:.4f}  "
              f"size={stats_lac['mean_set_size']:.4f}  "
              f"sing={stats_lac['singleton_rate']:.4f}  "
              f"gap={stats_lac['coverage_gap']:.4f}")

        # ── 2. Closed-form toxicity (E2a λ*) ────────────────────────────────
        cf_scores = lac_c + lam_cf * calib_psi_cf
        q_cf      = get_q_hat(cf_scores, alpha)
        sets_cf   = make_sets(eval_probs, lam_cf * eval_psi_cf, q_cf)
        stats_cf  = coverage_stats(sets_cf, eval_labels, eval_reg)
        print(f"  Tox-CF    q̂={q_cf:.4f}: cov={stats_cf['marginal_coverage']:.4f}  "
              f"size={stats_cf['mean_set_size']:.4f}  "
              f"sing={stats_cf['singleton_rate']:.4f}  "
              f"gap={stats_cf['coverage_gap']:.4f}  λ*={lam_cf}")

        # ── 3. Learned gate ──────────────────────────────────────────────────
        print(f"  Training MLP ψ-gate (epochs={args.n_epochs}, β={args.beta})...")
        gate, m_mean, m_std, q_lg = train_psi_gate(
            calib_probs, calib_labels, calib_m,
            alpha=alpha,
            n_epochs=args.n_epochs,
            beta=args.beta,
            tau=args.tau,
            cov_penalty=args.cov_penalty,
        )

        # Evaluate on eval set
        m_eval_norm = ((eval_m - m_mean) / (m_std + 1e-8)).astype(np.float32)
        with torch.no_grad():
            psi_eval_lg = gate(torch.from_numpy(m_eval_norm)).numpy()
        sets_lg   = make_sets(eval_probs, psi_eval_lg, q_lg)
        stats_lg  = coverage_stats(sets_lg, eval_labels, eval_reg, psi_eval=psi_eval_lg)
        psi_mean  = float(psi_eval_lg.mean())
        psi_std   = float(psi_eval_lg.std())

        print(f"  Learned   q̂={q_lg:.4f}: cov={stats_lg['marginal_coverage']:.4f}  "
              f"size={stats_lg['mean_set_size']:.4f}  "
              f"sing={stats_lg['singleton_rate']:.4f}  "
              f"gap={stats_lg['coverage_gap']:.4f}  "
              f"ψ̄={psi_mean:.4f}±{psi_std:.4f}")

        # Per-regime ψ distribution (validates microstructure grounding)
        if "psi_by_regime" in stats_lg:
            for rname, rstats in stats_lg["psi_by_regime"].items():
                print(f"    ψ_{rname}: mean={rstats['psi_mean']:.4f}  std={rstats['psi_std']:.4f}")

        # Size / gap reduction vs LAC
        lac_size = stats_lac["mean_set_size"]
        lac_gap  = stats_lac["coverage_gap"]
        lg_size  = stats_lg["mean_set_size"]
        lg_gap   = stats_lg["coverage_gap"]
        size_red = (lac_size - lg_size) / max(lac_size, 1e-6)
        gap_red  = (lac_gap  - lg_gap)  / max(lac_gap,  1e-6)
        print(f"  Δsize vs LAC: {size_red*100:+.1f}%  Δgap vs LAC: {gap_red*100:+.1f}%")

        # Size reduction vs closed-form
        cf_size  = stats_cf["mean_set_size"]
        size_red_cf = (cf_size - lg_size) / max(cf_size, 1e-6)
        print(f"  Δsize vs Tox-CF: {size_red_cf*100:+.1f}%")

        all_results[model_name] = {
            "alpha":   alpha,
            "n_calib": n_c,
            "n_eval":  N - n_c,
            "lambda_cf": lam_cf,
            "lac": {
                **stats_lac,
                "q_hat": round(q_lac, 4),
            },
            "toxicity_closed": {
                **stats_cf,
                "q_hat":  round(q_cf, 4),
                "lambda": lam_cf,
            },
            "learned_gate": {
                **stats_lg,
                "q_hat":                     round(q_lg, 4),
                "psi_mean":                  round(psi_mean, 4),
                "psi_std":                   round(psi_std, 4),
                "size_reduction_vs_lac":     round(float(size_red), 4),
                "gap_reduction_vs_lac":      round(float(gap_red),  4),
                "size_reduction_vs_tox_cf":  round(float(size_red_cf), 4),
            },
        }

        all_weights[model_name] = {
            "state_dict": {k: v.cpu() for k, v in gate.state_dict().items()},
            "m_mean":     torch.from_numpy(m_mean),
            "m_std":      torch.from_numpy(m_std),
            "q_hat":      q_lg,
            "alpha":      alpha,
        }

    # ── Save ─────────────────────────────────────────────────────────────────
    out_json = results_dir / "learned_gate_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_json.name}")

    out_pt = results_dir / "learned_gate_psi_weights.pt"
    torch.save(all_weights, out_pt)
    print(f"Saved → {out_pt.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Summary: mean set size at α={alpha} ──")
    print(f"{'Model':<26} {'LAC':>8} {'Tox-CF':>8} {'Learned':>8} {'Δvs LAC':>8} {'Δvs CF':>8}")
    for mn, res in all_results.items():
        lac_sz  = res["lac"]["mean_set_size"]
        cf_sz   = res["toxicity_closed"]["mean_set_size"]
        lg_sz   = res["learned_gate"]["mean_set_size"]
        d_lac   = res["learned_gate"]["size_reduction_vs_lac"]
        d_cf    = res["learned_gate"]["size_reduction_vs_tox_cf"]
        print(f"  {mn:<24} {lac_sz:>8.4f} {cf_sz:>8.4f} {lg_sz:>8.4f} "
              f"{d_lac*100:>+7.1f}% {d_cf*100:>+7.1f}%")

    print(f"\n── Summary: coverage gap (Mondrian diagnostic, ↓ better) ──")
    print(f"{'Model':<26} {'LAC':>8} {'Tox-CF':>8} {'Learned':>8} {'Δvs LAC':>8}")
    for mn, res in all_results.items():
        lac_g = res["lac"]["coverage_gap"]
        cf_g  = res["toxicity_closed"]["coverage_gap"]
        lg_g  = res["learned_gate"]["coverage_gap"]
        d_gap = res["learned_gate"]["gap_reduction_vs_lac"]
        print(f"  {mn:<24} {lac_g:>8.4f} {cf_g:>8.4f} {lg_g:>8.4f} "
              f"{d_gap*100:>+7.1f}%")


if __name__ == "__main__":
    main()
