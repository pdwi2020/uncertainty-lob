"""Conformal-score robustness: does the microstructure gate's Mondrian
coverage-gap reduction hold beyond LAC?  (Reviewer pre-empt: "why only LAC?")

We compare three base nonconformity scores on the SAME val 50/50 split and the
SAME microstructure signal used by the headline learned-gate experiment:

  LAC  : s(x,y) = 1 - p_y                         (Least Ambiguous Classifier)
  APS  : s(x,y) = cumulative prob mass up to y    (Romano et al., NeurIPS 2020)
  RAPS : APS + lambda_raps * max(0, rank(y)-k_reg) (Angelopoulos et al., ICLR 2021)

For each base score we report the ungated conformal set and the
microstructure-GATED version s(x,y) + lambda* * psi_cf(x), where psi_cf is the
closed-form order-flow toxicity proxy and lambda* is the per-model weight already
tuned for LAC (results/micro_conformal_results.json). Using the SAME gate for all
three base scores is the point: if it reduces the Mondrian coverage gap under
APS and RAPS too, the benefit is a property of microstructure conditioning, not
of LAC.

Reuses learned_gate.{get_q_hat, make_sets, coverage_stats} for LAC (so the LAC
row matches Table~\\ref{tab:learned_gate} exactly) and implements APS/RAPS set
construction here.

Outputs:
  results/aps_raps_robustness.json
  results/aps_raps_robustness.csv
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
RESULTS = ROOT / "results"

from learned_gate import get_q_hat, make_sets, coverage_stats  # noqa: E402

ALPHA = 0.10
MODELS = ["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention"]
RAPS_LAMBDA = 0.10      # RAPS regularization weight
RAPS_KREG = 1           # penalize ranks beyond the top-1


# ─── APS / RAPS scores (regularized cumulative mass) ─────────────────────────

def _reg_term(ranks_1indexed: np.ndarray, reg_lambda: float, k_reg: int) -> np.ndarray:
    return reg_lambda * np.maximum(0, ranks_1indexed - k_reg)


def aps_cal_scores(probs, labels, reg_lambda, k_reg, psi=None, lam_gate=0.0):
    """Regularized cumulative conformity score for the TRUE label (calibration)."""
    N, K = probs.shape
    order = np.argsort(-probs, axis=1)                       # desc prob -> class idx
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    rank = np.argsort(order, axis=1)                         # 0-indexed rank per class
    o_y = rank[np.arange(N), labels]                         # rank of true label
    s = cumsum[np.arange(N), o_y] + _reg_term(o_y + 1, reg_lambda, k_reg)
    if psi is not None and lam_gate > 0:
        s = s + lam_gate * psi
    return s


def aps_build_sets(probs, q_hat, reg_lambda, k_reg, psi=None, lam_gate=0.0):
    """Non-randomized (R)APS sets: smallest top-set whose reg. cumulative score
    reaches q_hat (the crossing class is included to guarantee coverage)."""
    N, K = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    ranks1 = np.arange(1, K + 1)[None, :]
    score_sorted = cumsum + _reg_term(ranks1, reg_lambda, k_reg)
    if psi is not None and lam_gate > 0:
        score_sorted = score_sorted + lam_gate * psi[:, None]
    leq = score_sorted <= q_hat                              # positions below threshold
    L = np.clip(leq.sum(axis=1) + 1, 1, K)                   # +1 = include crossing class
    col = np.arange(K)[None, :]
    mask_sorted = col < L[:, None]
    sets = np.zeros((N, K), dtype=bool)
    np.put_along_axis(sets, order, mask_sorted, axis=1)
    return sets


def run_base(method, calib_probs, calib_labels, eval_probs, eval_labels, eval_reg,
             alpha, psi_calib=None, psi_eval=None, lam_gate=0.0):
    """Return coverage_stats dict for one base score, optionally gated."""
    if method == "LAC":
        lac_c = 1.0 - calib_probs[np.arange(len(calib_labels)), calib_labels]
        if psi_calib is not None and lam_gate > 0:
            lac_c = lac_c + lam_gate * psi_calib
        q = get_q_hat(lac_c, alpha)
        psi_term = (lam_gate * psi_eval) if (psi_eval is not None and lam_gate > 0) \
            else np.zeros(len(eval_labels))
        sets = make_sets(eval_probs, psi_term, q)
    else:
        reg_lambda = RAPS_LAMBDA if method == "RAPS" else 0.0
        cal = aps_cal_scores(calib_probs, calib_labels, reg_lambda, RAPS_KREG,
                             psi_calib, lam_gate)
        q = get_q_hat(cal, alpha)
        sets = aps_build_sets(eval_probs, q, reg_lambda, RAPS_KREG, psi_eval, lam_gate)
    stats = coverage_stats(sets, eval_labels, eval_reg)
    stats["q_hat"] = round(float(q), 4)
    return stats


def main():
    macc = np.load(RESULTS / "macc_features.npz")
    r_val = macc["r_val"].astype(np.int64)
    psi_cf = macc["psi_val"].astype(np.float32)

    lam_star = {}
    e2a = RESULTS / "micro_conformal_results.json"
    if e2a.exists():
        with open(e2a) as f:
            for mn, mres in json.load(f).items():
                lam_star[mn] = mres.get("lambda_star", 0.30)
    print(f"lambda* (closed-form gate, per model): {lam_star}", flush=True)

    out = {}
    for mn in MODELS:
        ckpt = RESULTS / f"{mn}_k10_logits.pt"
        if not ckpt.exists():
            print(f"[SKIP] {mn}: {ckpt.name} missing", flush=True)
            continue
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        labels = d["labels"].long().numpy()
        probs = F.softmax(d["logits"].float(), dim=-1).numpy()
        N = len(labels)
        n_c = N // 2
        cp, ep = probs[:n_c], probs[n_c:]
        cl, el = labels[:n_c], labels[n_c:]
        cpsi, epsi = psi_cf[:n_c], psi_cf[n_c:]
        creg, ereg = r_val[:n_c], r_val[n_c:]   # noqa: F841 (creg unused; reg from eval)
        lam = lam_star.get(mn, 0.30)

        print(f"\n══ {mn}  N={N:,}  lambda*={lam} ══", flush=True)
        out[mn] = {"lambda_star": lam, "n_calib": n_c, "n_eval": N - n_c, "methods": {}}
        for method in ["LAC", "APS", "RAPS"]:
            base = run_base(method, cp, cl, ep, el, ereg, ALPHA)
            gated = run_base(method, cp, cl, ep, el, ereg, ALPHA,
                             psi_calib=cpsi, psi_eval=epsi, lam_gate=lam)
            gap_red = (base["coverage_gap"] - gated["coverage_gap"]) / max(base["coverage_gap"], 1e-6)
            out[mn]["methods"][method] = {
                "base": base, "gated": gated,
                "gap_reduction": round(float(gap_red), 4),
            }
            print(f"  {method:<4}  base: cov={base['marginal_coverage']:.3f} "
                  f"sz={base['mean_set_size']:.3f} sing={base['singleton_rate']:.3f} "
                  f"gap={base['coverage_gap']:.4f}   gated: cov={gated['marginal_coverage']:.3f} "
                  f"sz={gated['mean_set_size']:.3f} gap={gated['coverage_gap']:.4f} "
                  f"(Δgap {gap_red*100:+.0f}%)", flush=True)

    with open(RESULTS / "aps_raps_robustness.json", "w") as f:
        json.dump(out, f, indent=2)

    with open(RESULTS / "aps_raps_robustness.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "variant", "coverage", "set_size",
                    "singleton", "gap", "q_hat"])
        for mn, mres in out.items():
            for method, mm in mres["methods"].items():
                for variant in ("base", "gated"):
                    s = mm[variant]
                    w.writerow([mn, method, variant, s["marginal_coverage"],
                                s["mean_set_size"], s["singleton_rate"],
                                s["coverage_gap"], s["q_hat"]])

    print("\n── gap (Mondrian) summary: base → gated ──", flush=True)
    print(f"{'model':<20}{'LAC':>22}{'APS':>22}{'RAPS':>22}", flush=True)
    for mn, mres in out.items():
        cells = []
        for method in ["LAC", "APS", "RAPS"]:
            mm = mres["methods"][method]
            cells.append(f"{mm['base']['coverage_gap']:.3f}->{mm['gated']['coverage_gap']:.3f}")
        print(f"{mn:<20}" + "".join(f"{c:>22}" for c in cells), flush=True)
    print("\nAPS_RAPS_ROBUSTNESS_DONE", flush=True)


if __name__ == "__main__":
    main()
