"""EXP1 — Second-market (crypto) MACC-LOB conformal benchmark.

Tests whether the FI-2010 methodology — LAC -> closed-form toxicity gate ->
learned MLP gate, plus ACI/AgACI coverage restoration — generalizes to a SECOND
market (crypto BTC/ETH/SOL), upgrading the cross-market story from
temperature-only recalibration to the full learned-conformal contribution.

Pure post-hoc: uses saved per-model crypto logits + raw crypto LOB windows
(crypto_<SYM>_k10_features.npz). The 7-dim microstructure vector m(x) is derived
from the window-end LOB state with the SAME definitions used on FI-2010
(experiments/build_macc_features.py), computed on the Z-scored values for
cross-market consistency. No GPU, no base-model retraining.

Ladder (alpha=0.10, 90% target), calib/eval = temporal 50/50 split per symbol:
  1. LAC                  s = 1 - p_y
  2. Toxicity-closed      s = 1 - p_y + lambda* psi_cf(x)   (lambda* tuned on calib)
  3. Learned gate         s = 1 - p_y + psi_MLP(x)          (HEADLINE; trained on calib)
  4. ACI  (eta=0.05)      online quantile update on eval stream
  5. AgACI (K=6 experts)  aggregated ACI

Metrics: marginal coverage, mean set size, singleton rate, Mondrian regime gap.

Output:
  results/crossmarket_conformal_results.json
  results/crossmarket_conformal_table.csv
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/Volumes/Crucial X9/Research Projects/DL_Paper")
EXPDIR = ROOT / "experiments"
RESULTS = ROOT / "results"          # read inputs (npz + logits) from X9
OUT = Path("/tmp/dl_paper_experiments")  # write outputs to /tmp (X9 new-file writes blocked under fskit mount)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(EXPDIR))
sys.path.insert(0, str(ROOT))

# Reuse the exact FI-2010 microstructure + gate implementations
from build_macc_features import (  # noqa: E402
    extract_microstructure,
    compute_toxicity,
    build_hmm_features,
)
from learned_gate import (  # noqa: E402
    get_q_hat,
    make_sets,
    coverage_stats,
    train_psi_gate,
)

ALPHA = 0.10
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MODELS = ["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention"]


# ── Regime labels (per symbol) ────────────────────────────────────────────────
def assign_regimes(cur_states: np.ndarray):
    """2-state low/high-vol regime per window-end state.

    Mirrors run_regime.py (Gaussian HMM on [vol_ratio, spread, ofi_mean]) when
    hmmlearn is available; otherwise falls back to a reproducible median split on
    the volatility-ratio signal. Returns (int8 {0=low,1=high}, method).
    """
    hmm_f = build_hmm_features(cur_states)
    try:
        from hmmlearn import hmm as hmmlearn_hmm
        model = hmmlearn_hmm.GaussianHMM(n_components=2, covariance_type="full",
                                         n_iter=100, random_state=42)
        model.fit(hmm_f)
        raw = model.predict(hmm_f)
        high = int(np.argmax(model.means_[:, 0]))   # higher vol_ratio mean
        return np.where(raw == high, 1, 0).astype(np.int8), "hmm"
    except Exception:
        vr = hmm_f[:, 0]
        return (vr > np.median(vr)).astype(np.int8), "median_split"


# ── Closed-form toxicity lambda* (tuned on calib) ─────────────────────────────
def tune_lambda(calib_probs, calib_labels, calib_psi, alpha, grid=None):
    """Pick lambda minimizing calib mean set size s.t. calib coverage >= 1-alpha."""
    if grid is None:
        grid = np.round(np.arange(0.0, 0.61, 0.04), 2)
    n = len(calib_labels)
    lac = 1.0 - calib_probs[np.arange(n), calib_labels]
    best_lam, best_size = 0.0, np.inf
    for lam in grid:
        scores = lac + lam * calib_psi
        q = get_q_hat(scores, alpha)
        sets = make_sets(calib_probs, lam * calib_psi, q)
        cov = float(sets[np.arange(n), calib_labels].mean())
        size = float(sets.sum(1).mean())
        if cov >= (1 - alpha) - 1e-9 and size < best_size:
            best_size, best_lam = size, float(lam)
    return best_lam


# ── ACI / AgACI on the eval stream ────────────────────────────────────────────
def run_aci(probs, labels, q_init, alpha, eta):
    n = len(labels)
    qt = q_init
    in_set, sizes = np.zeros(n, bool), np.zeros(n)
    for t in range(n):
        p, y = probs[t], labels[t]
        s_set = (1.0 - p) <= qt
        in_set[t] = bool(s_set[y]); sizes[t] = int(s_set.sum())
        err = 0.0 if in_set[t] else 1.0
        qt = float(np.clip(qt + eta * (err - alpha), 0.0, 1.0))
    return float(in_set.mean()), float(sizes.mean()), float((sizes == 1).mean())


def run_agaci(probs, labels, q_init, alpha, etas):
    n, K = len(labels), len(etas)
    qts = [q_init] * K
    cov, ssz = np.zeros(n, bool), np.zeros(n)
    for t in range(n):
        p, y = probs[t], labels[t]
        inc = np.zeros(3)
        for k in range(K):
            inc += ((1.0 - p) <= qts[k]).astype(float)
        agg = inc >= (K / 2.0)
        cov[t] = bool(agg[y]); ssz[t] = int(agg.sum())
        for k in range(K):
            err = 0.0 if bool((1.0 - p[y]) <= qts[k]) else 1.0
            qts[k] = float(np.clip(qts[k] + etas[k] * (err - alpha), 0.0, 1.0))
    return float(cov.mean()), float(ssz.mean()), float((ssz == 1).mean())


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    results = {}
    for sym in SYMBOLS:
        npz = RESULTS / f"crypto_{sym}_k10_features.npz"
        if not npz.exists():
            print(f"[SKIP] {sym}: {npz.name} missing"); continue
        d = np.load(npz)
        X, Y = d["X"], d["Y"].astype(np.int64)
        cur = X[:, -1, :].astype(np.float32)          # window-end state (N,40)
        m = extract_microstructure(cur)               # (N,7)
        psi_cf = compute_toxicity(m)                  # (N,) closed-form toxicity
        regimes, reg_method = assign_regimes(cur)
        print(f"\n##### {sym}  N={len(Y):,}  regime={reg_method} "
              f"(low={int((regimes==0).sum()):,} high={int((regimes==1).sum()):,}) #####")

        results[sym] = {"regime_method": reg_method, "n": int(len(Y)), "models": {}}

        for model in MODELS:
            lgp = RESULTS / f"crypto_{sym}_{model}_k10_logits.pt"
            if not lgp.exists():
                print(f"  [SKIP] {model}: logits missing"); continue
            ck = torch.load(lgp, map_location="cpu", weights_only=False)
            logits = ck["logits"].float()
            labels = ck["labels"].long().numpy() if "labels" in ck else Y
            probs = F.softmax(logits, dim=-1).numpy()
            n = len(labels)
            if n != len(Y):
                k = min(n, len(Y))
                probs, labels = probs[:k], labels[:k]
                m_s, psi_s, reg_s = m[:k], psi_cf[:k], regimes[:k]
            else:
                m_s, psi_s, reg_s = m, psi_cf, regimes

            # temporal 50/50 calib/eval split
            nc = len(labels) // 2
            cp, ep = probs[:nc], probs[nc:]
            cl, el = labels[:nc], labels[nc:]
            cm, em = m_s[:nc], m_s[nc:]
            cpsi, epsi = psi_s[:nc], psi_s[nc:]
            creg, ereg = reg_s[:nc], reg_s[nc:]

            # 1. LAC
            q_lac = get_q_hat(1.0 - cp[np.arange(nc), cl], ALPHA)
            st_lac = coverage_stats(make_sets(ep, np.zeros(len(el)), q_lac), el, ereg)

            # 2. closed-form toxicity gate
            lam = tune_lambda(cp, cl, cpsi, ALPHA)
            q_cf = get_q_hat(1.0 - cp[np.arange(nc), cl] + lam * cpsi, ALPHA)
            st_cf = coverage_stats(make_sets(ep, lam * epsi, q_cf), el, ereg)

            # 3. learned gate (trained on calib, eval held out)
            gate, mu, sd, q_lg = train_psi_gate(cp, cl, cm, alpha=ALPHA,
                                                n_epochs=200, beta=0.10)
            with torch.no_grad():
                psi_lg = gate(torch.from_numpy(((em - mu) / (sd + 1e-8)).astype(np.float32))).numpy()
            st_lg = coverage_stats(make_sets(ep, psi_lg, q_lg), el, ereg, psi_eval=psi_lg)

            # 4-5. ACI / AgACI for coverage restoration on the eval stream
            aci_cov, aci_sz, aci_sing = run_aci(ep, el, q_lac, ALPHA, 0.05)
            ag_cov, ag_sz, ag_sing = run_agaci(ep, el, q_lac, ALPHA,
                                               [0.005, 0.01, 0.02, 0.05, 0.10, 0.20])

            print(f"  {model:20s} LAC cov={st_lac['marginal_coverage']:.3f} gap={st_lac['coverage_gap']:.3f} | "
                  f"CF(l={lam:.2f}) gap={st_cf['coverage_gap']:.3f} | "
                  f"Learned cov={st_lg['marginal_coverage']:.3f} gap={st_lg['coverage_gap']:.3f} | "
                  f"ACI cov={aci_cov:.3f} AgACI cov={ag_cov:.3f}")

            results[sym]["models"][model] = {
                "lac": st_lac,
                "toxicity_closed": {**st_cf, "lambda": lam},
                "learned_gate": st_lg,
                "aci_eta005": {"coverage": round(aci_cov, 4), "mean_set_size": round(aci_sz, 4),
                               "singleton_rate": round(aci_sing, 4)},
                "agaci_K6": {"coverage": round(ag_cov, 4), "mean_set_size": round(ag_sz, 4),
                             "singleton_rate": round(ag_sing, 4)},
                "gap_reduction_learned_vs_lac": round(
                    (st_lac["coverage_gap"] - st_lg["coverage_gap"]) / max(st_lac["coverage_gap"], 1e-6), 4),
            }

    out_json = OUT / "crossmarket_conformal_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_json}")

    rows = [["Symbol", "Model", "LAC_cov", "LAC_gap", "Learned_cov", "Learned_gap",
             "gap_reduction_%", "ACI_cov", "AgACI_cov"]]
    for sym, sres in results.items():
        for model, r in sres["models"].items():
            rows.append([
                sym, model,
                f"{r['lac']['marginal_coverage']:.3f}", f"{r['lac']['coverage_gap']:.3f}",
                f"{r['learned_gate']['marginal_coverage']:.3f}", f"{r['learned_gate']['coverage_gap']:.3f}",
                f"{r['gap_reduction_learned_vs_lac']*100:+.1f}",
                f"{r['aci_eta005']['coverage']:.3f}", f"{r['agaci_K6']['coverage']:.3f}",
            ])
    out_csv = OUT / "crossmarket_conformal_table.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved -> {out_csv}")
    print("\nCROSSMARKET_CONFORMAL_DONE")


if __name__ == "__main__":
    main()
