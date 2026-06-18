"""
EXP-G1: ACI/AgACI coverage-restore experiment.

Demonstrates that standard LAC undercovers under temporal LOB drift
and that adaptive conformal inference restores marginal coverage.

Uses: LOBTransformerV2 val/test logits (time-ordered, shuffle=False).
Outputs: results/aci_recalibration_results.json + aci_recalibration_table.csv
"""

import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
LOGIT_DIR = Path("/tmp/lobtransformerv2_train_results/targeted")
OUT_DIR   = Path("/tmp/dl_paper_experiments")
OUT_DIR.mkdir(exist_ok=True)

# ── Load logits ───────────────────────────────────────────────────────────────
val_ck  = torch.load(LOGIT_DIR / "lobt_v2_k10_val_logits.pt",  map_location="cpu", weights_only=False)
test_ck = torch.load(LOGIT_DIR / "lobt_v2_k10_test_logits.pt", map_location="cpu", weights_only=False)

val_logits  = val_ck["logits"].float()   # (M, 3)
val_labels  = val_ck["labels"].long()    # (M,)
test_logits = test_ck["logits"].float()  # (N, 3)
test_labels = test_ck["labels"].long()   # (N,)

val_probs  = F.softmax(val_logits,  dim=-1).numpy()
test_probs = F.softmax(test_logits, dim=-1).numpy()
val_y  = val_labels.numpy()
test_y = test_labels.numpy()

N_val  = len(val_y)
N_test = len(test_y)
print(f"Val: {N_val:,}   Test: {N_test:,}")

ALPHA = 0.10   # target miscoverage

# ── Helper: LAC nonconformity score ──────────────────────────────────────────
def nc_scores(probs, labels):
    """1 - p̂_{y_true}  (LAC)"""
    return 1.0 - probs[np.arange(len(labels)), labels]

def lac_quantile(cal_scores, alpha, n):
    """Finite-sample-corrected quantile: ceil((n+1)(1-alpha))/n."""
    lv = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(cal_scores, min(lv, 1.0)))

def coverage_and_size(probs, labels, q_hat):
    """Coverage and mean prediction-set size for a fixed q̂."""
    sets   = (1.0 - probs) <= q_hat          # shape (N, 3)
    cov    = float(sets[np.arange(len(labels)), labels].mean())
    mss    = float(sets.sum(axis=1).mean())
    sing   = float((sets.sum(axis=1) == 1).mean())
    return cov, mss, sing

# ── 1. Standard LAC (one-shot val calibration) ───────────────────────────────
val_scores  = nc_scores(val_probs, val_y)
q_hat_fixed = lac_quantile(val_scores, ALPHA, N_val)
lac_cov, lac_mss, lac_sing = coverage_and_size(test_probs, test_y, q_hat_fixed)
print(f"Standard LAC   : q̂={q_hat_fixed:.4f}  cov={lac_cov:.4f}  mss={lac_mss:.4f}  sing={lac_sing:.4f}")

# ── 2. ACI — Adaptive Conformal Inference (Gibbs & Candès, 2021) ─────────────
# Online update: q_{t+1} = q_t + η * (α - err_t)
# err_t = 1{y_t ∉ C_t(x_t)}
def run_aci(test_probs, test_labels, q_init, alpha, eta):
    n = len(test_labels)
    qt = q_init
    covs, qts, errs = [], [], []
    for t in range(n):
        p = test_probs[t]
        y = test_labels[t]
        # prediction set at current qt
        in_set = bool((1.0 - p[y]) <= qt)
        err_t  = 0.0 if in_set else 1.0
        errs.append(err_t)
        covs.append(in_set)
        qts.append(qt)
        # update: q_{t+1} = q_t + η*(err_t − α)  [Gibbs & Candès 2021, Eq. 3]
        # miss (err_t=1) → increase q̂; hit (err_t=0) → decrease q̂ slightly
        qt = qt + eta * (err_t - alpha)
        qt = float(np.clip(qt, 0.0, 1.0))
    cov   = float(np.mean(covs))
    mss   = float(np.mean([(1.0 - test_probs[t] <= qts[t]).sum() for t in range(n)]))
    sing  = float(np.mean([(1.0 - test_probs[t] <= qts[t]).sum() == 1 for t in range(n)]))
    return cov, mss, sing, np.array(qts)

# η sweep
aci_results = {}
for eta in [0.005, 0.01, 0.05, 0.10]:
    cov, mss, sing, qts = run_aci(test_probs, test_y, q_hat_fixed, ALPHA, eta)
    aci_results[f"ACI_eta{eta}"] = {"coverage": cov, "mean_set_size": mss, "singleton_rate": sing,
                                     "q_init": q_hat_fixed, "eta": eta}
    print(f"ACI (η={eta:.3f}) : cov={cov:.4f}  mss={mss:.4f}  sing={sing:.4f}")

# Best ACI (η=0.05, canonical)
aci_cov, aci_mss, aci_sing, aci_qts = run_aci(test_probs, test_y, q_hat_fixed, ALPHA, 0.05)

# ── 3. AgACI — Aggregated ACI (Zaffran et al., 2022) ─────────────────────────
# Run K experts at different η; average their prediction-set indicators
def run_agaci(test_probs, test_labels, q_init, alpha, etas):
    n   = len(test_labels)
    K   = len(etas)
    qts = [q_init] * K
    covs, sets_sum = [], []
    for t in range(n):
        p  = test_probs[t]
        y  = test_labels[t]
        # each expert k makes its own prediction set; aggregate by majority vote
        in_sets = np.zeros(3)
        for k, eta in enumerate(etas):
            set_k = (1.0 - p) <= qts[k]
            in_sets += set_k.astype(float)
        # majority vote (> K/2 experts include class c)
        agg_set = in_sets >= (K / 2.0)
        in_set  = bool(agg_set[y])
        covs.append(in_set)
        sets_sum.append(int(agg_set.sum()))
        # each expert updates based on its own coverage indicator
        for k, eta in enumerate(etas):
            err_k = 0.0 if bool((1.0 - p[y]) <= qts[k]) else 1.0
            qts[k] = float(np.clip(qts[k] + eta * (err_k - alpha), 0.0, 1.0))
    cov  = float(np.mean(covs))
    mss  = float(np.mean(sets_sum))
    sing = float(np.mean([s == 1 for s in sets_sum]))
    return cov, mss, sing

etas_agaci = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
ag_cov, ag_mss, ag_sing = run_agaci(test_probs, test_y, q_hat_fixed, ALPHA, etas_agaci)
print(f"AgACI (K={len(etas_agaci)}  ) : cov={ag_cov:.4f}  mss={ag_mss:.4f}  sing={ag_sing:.4f}")

# ── 4. Coverage over time (window analysis) ──────────────────────────────────
WIN = 10000
n_windows = N_test // WIN
lac_window_covs = []
aci_window_covs = []
# Rebuild ACI coverage sequence
_, _, _, aci_qts_seq = run_aci(test_probs, test_y, q_hat_fixed, ALPHA, 0.05)
for w in range(n_windows):
    sl = slice(w * WIN, (w + 1) * WIN)
    tp = test_probs[sl]; ty = test_y[sl]; qts_w = aci_qts_seq[sl]
    # LAC window coverage
    lac_sets = (1.0 - tp) <= q_hat_fixed
    lac_window_covs.append(float(lac_sets[np.arange(WIN), ty].mean()))
    # ACI window coverage — check if true label is in the adaptive set
    aci_in_set = np.array([(1.0 - tp[t] <= qts_w[t])[ty[t]] for t in range(WIN)])
    aci_window_covs.append(float(aci_in_set.mean()))

print(f"\nWindow analysis ({WIN:,} samples/window, {n_windows} windows):")
print(f"  LAC  min/mean/max: {min(lac_window_covs):.3f}/{np.mean(lac_window_covs):.3f}/{max(lac_window_covs):.3f}")
print(f"  ACI  min/mean/max: {min(aci_window_covs):.3f}/{np.mean(aci_window_covs):.3f}/{max(aci_window_covs):.3f}")

# ── 5. Results ────────────────────────────────────────────────────────────────
results = {
    "model": "LOBTransformerV2",
    "dataset": "FI-2010 NoAuction Zscore k=10",
    "alpha": ALPHA,
    "n_val": int(N_val),
    "n_test": int(N_test),
    "standard_lac": {
        "q_hat": float(q_hat_fixed),
        "coverage": lac_cov,
        "mean_set_size": lac_mss,
        "singleton_rate": lac_sing,
    },
    "aci_eta005": {
        "eta": 0.05,
        "coverage": float(aci_cov),
        "mean_set_size": float(aci_mss),
        "singleton_rate": float(aci_sing),
        "q_init": float(q_hat_fixed),
    },
    "agaci_K6": {
        "etas": etas_agaci,
        "coverage": float(ag_cov),
        "mean_set_size": float(ag_mss),
        "singleton_rate": float(ag_sing),
    },
    "aci_eta_sweep": aci_results,
    "window_analysis": {
        "window_size": WIN,
        "n_windows": n_windows,
        "lac_window_coverages": lac_window_covs,
        "aci_window_coverages": aci_window_covs,
        "lac_min_window_cov": float(min(lac_window_covs)),
        "lac_mean_window_cov": float(np.mean(lac_window_covs)),
        "aci_min_window_cov":  float(min(aci_window_covs)),
        "aci_mean_window_cov": float(np.mean(aci_window_covs)),
    },
}

out_json = OUT_DIR / "aci_recalibration_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out_json}")

# ── CSV table (for paper) ─────────────────────────────────────────────────────
import csv
table_rows = [
    ["Method",         "Coverage", "Avg Set Size", "Singleton %"],
    ["LAC (standard)", f"{lac_cov:.3f}", f"{lac_mss:.3f}", f"{lac_sing:.3f}"],
    ["ACI (η=0.05)",   f"{aci_cov:.3f}", f"{aci_mss:.3f}", f"{aci_sing:.3f}"],
    ["AgACI (K=6)",    f"{ag_cov:.3f}",  f"{ag_mss:.3f}",  f"{ag_sing:.3f}"],
]
out_csv = OUT_DIR / "aci_recalibration_table.csv"
with open(out_csv, "w", newline="") as f:
    csv.writer(f).writerows(table_rows)
print(f"Saved → {out_csv}")
print("\nACI_RECALIBRATION_DONE")
