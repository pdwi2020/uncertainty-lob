"""Phase 2a -- Per-stock cross-instrument generalization of the learned conformal gate.

Rebuts the single-pooled-dataset criticism by showing the learned microstructure
conformal gate, *calibrated on a held-out set of FI-2010 instruments*, transfers to a
different (held-out) instrument -- i.e. it generalises ACROSS the five equities, not
just on the pooled split. Pure post-hoc: no base-model retraining.

Two analyses (alpha=0.10, 90% target coverage):
  A. Leave-one-stock-out (LOSO) calibration transfer        [HEADLINE]
       calib = pooled OTHER 4 stocks ; eval = held-out stock
  B. Per-instrument self-calibration breakdown              [SUPPORTING]
       within each stock, temporal 50/50 calib/eval

Per (model, stock) we report, for LAC vs the learned gate: marginal coverage, the
Mondrian per-regime coverage-gap (|cov_low - cov_high|, low/high-vol by a median
split on the short/long volatility ratio), and mean prediction-set size.

Inputs (already on disk; no GPU):
  results/<model>_k10_test_logits.pt          saved TEST logits (354,725 x 3) + labels
  results/data_cache/Test_Dst_NoAuction_ZScore_CF_{1..9}.npy  raw FI-2010 test windows

Per-stock boundaries are the mathematically-verified partition of the 354,725-sample
pooled test set (zero gaps / zero overlaps). Microstructure m(x) is computed on the
concatenated test series exactly as in build_macc_features. We use BOUNDARY-PURE
windows -- dropping the first (seq_len-1) logits of every contiguous stock segment, so
no 100-step model input spans a stock boundary.

Outputs:
  results/per_stock_generalization.json
  results/per_stock_generalization_table.csv
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
RESULTS = ROOT / "results"
CACHE = RESULTS / "data_cache"
sys.path.insert(0, str(EXPDIR))
sys.path.insert(0, str(ROOT))

from build_macc_features import extract_microstructure, compute_toxicity  # noqa: E402
from learned_gate import (  # noqa: E402
    get_q_hat,
    make_sets,
    coverage_stats,
    train_psi_gate,
)

ALPHA = 0.10
SEQ_LEN = 100                 # FI-2010 window length
N_TEST = 354_725              # pooled test logits

# model name -> saved test-logit filename
MODELS = {
    "cnn_lstm_attention": "cnn_lstm_attn_k10_test_logits.pt",
    "tcn": "tcn_k10_test_logits.pt",
    "lob_transformer": "lobtransformer_k10_test_logits.pt",
    "neural_cde": "neural_cde_k10_test_logits.pt",
}

# Verified per-stock LOGIT-index partition of the 354,725 test samples
# (logit j belongs to the stock of raw label-row j+99; segments are contiguous
#  within each of the 9 CF files). Totals: 26191/61617/57976/75886/133055.
STOCK_LOGIT_RANGES = {
    1: [(0, 4980), (38298, 42201), (66833, 69639), (103856, 106886),
        (138641, 140904), (177793, 180594), (215139, 217786),
        (270617, 272490), (322789, 324677)],
    2: [(4980, 11102), (42201, 47612), (69639, 76631), (106886, 113614),
        (140904, 147147), (180594, 187654), (217786, 226448),
        (272490, 281761), (324677, 329805)],
    3: [(11102, 17067), (47612, 51639), (76631, 81973), (113614, 119560),
        (147147, 152754), (187654, 194394), (226448, 235039),
        (281761, 291797), (329805, 335527)],
    4: [(17067, 23779), (51639, 56825), (81973, 92792), (119560, 125938),
        (152754, 159761), (194394, 202248), (235039, 248268),
        (291797, 304677), (335527, 341348)],
    5: [(23779, 38298), (56825, 66833), (92792, 103856), (125938, 138641),
        (159761, 177793), (202248, 215139), (248268, 270617),
        (304677, 322789), (341348, 354725)],
}


def build_pure_masks(n=N_TEST, drop=SEQ_LEN - 1):
    """Boolean mask per stock; drops the first `drop` logits of every contiguous
    segment so that no model input window [j, j+99] spans a stock boundary."""
    masks = {}
    for s, ranges in STOCK_LOGIT_RANGES.items():
        m = np.zeros(n, dtype=bool)
        for (a, b) in ranges:
            lo = min(a + drop, b)
            m[lo:b] = True
        masks[s] = m
    return masks


def median_regimes(vol_ratio: np.ndarray) -> np.ndarray:
    """Low/high-vol regime by a median split on the short/long vol ratio."""
    med = np.median(vol_ratio)
    return (vol_ratio > med).astype(np.int8)


def eval_ladder(calib_probs, calib_labels, calib_m,
                eval_probs, eval_labels, eval_m, eval_vol):
    """LAC vs learned gate on one (calib, eval) split. Returns a dict."""
    eval_reg = median_regimes(eval_vol)
    n_c = len(calib_labels)

    # 1. LAC (split conformal, no microstructure)
    lac_c = 1.0 - calib_probs[np.arange(n_c), calib_labels]
    q_lac = get_q_hat(lac_c, ALPHA)
    sets_lac = make_sets(eval_probs, np.zeros(len(eval_labels)), q_lac)
    st_lac = coverage_stats(sets_lac, eval_labels, eval_reg)

    # 2. Learned microstructure gate (trained on calib only)
    gate, mu, sd, q_lg = train_psi_gate(
        calib_probs, calib_labels, calib_m, alpha=ALPHA, n_epochs=200, beta=0.10,
    )
    m_eval_norm = ((eval_m - mu) / (sd + 1e-8)).astype(np.float32)
    with torch.no_grad():
        psi_eval = gate(torch.from_numpy(m_eval_norm)).numpy()
    sets_lg = make_sets(eval_probs, psi_eval, q_lg)
    st_lg = coverage_stats(sets_lg, eval_labels, eval_reg, psi_eval=psi_eval)

    gap_lac = st_lac["coverage_gap"]
    gap_lg = st_lg["coverage_gap"]
    gap_red = (gap_lac - gap_lg) / max(gap_lac, 1e-6)
    return {
        "n_calib": int(n_c),
        "n_eval": int(len(eval_labels)),
        "lac": {**st_lac, "q_hat": round(q_lac, 4)},
        "learned_gate": {**st_lg, "q_hat": round(q_lg, 4),
                         "psi_mean": round(float(psi_eval.mean()), 4)},
        "gap_reduction_vs_lac": round(float(gap_red), 4),
    }


def main():
    print("Loading raw FI-2010 test windows + microstructure ...")
    npys = sorted(
        f for f in CACHE.iterdir()
        if f.name.startswith("Test_Dst_NoAuction_ZScore_CF_")
        and f.suffix == ".npy" and not f.name.startswith("._")
    )
    feat = np.concatenate([np.load(str(f))[:40].T for f in npys], axis=0)  # (354825,40)
    m_all = extract_microstructure(feat)                                   # (354825,7)
    m_logit = m_all[SEQ_LEN - 1: SEQ_LEN - 1 + N_TEST]                     # align j<->j+99
    assert len(m_logit) == N_TEST, f"m_logit={len(m_logit)} != {N_TEST}"
    vol_logit = m_logit[:, 2]

    pure = build_pure_masks()
    total_pure = int(sum(m.sum() for m in pure.values()))
    print(f"  pure samples retained: {total_pure:,} / {N_TEST:,} "
          f"({100*total_pure/N_TEST:.1f}%)")

    results = {"alpha": ALPHA, "seq_len": SEQ_LEN, "models": {}}
    for model, fname in MODELS.items():
        fp = RESULTS / fname
        if not fp.exists():
            print(f"[SKIP] {model}: {fname} missing")
            continue
        d = torch.load(fp, map_location="cpu", weights_only=False)
        logits = d["logits"].float()
        labels = d["labels"].long().numpy()
        probs = F.softmax(logits, dim=-1).numpy()
        if len(labels) != N_TEST:
            print(f"[SKIP] {model}: {len(labels)} != {N_TEST}")
            continue
        print(f"\n===== {model}  (acc={np.mean(probs.argmax(1)==labels):.4f}) =====")

        loso, selfcal = {}, {}
        for s in range(1, 6):
            em = pure[s]                          # eval = held-out stock s (pure)
            cm = np.zeros(N_TEST, bool)           # calib = pooled other 4 stocks (pure)
            for s2 in range(1, 6):
                if s2 != s:
                    cm |= pure[s2]

            # --- A. Leave-one-stock-out transfer ---
            a = eval_ladder(
                probs[cm], labels[cm], m_logit[cm],
                probs[em], labels[em], m_logit[em], vol_logit[em],
            )
            loso[f"stock_{s}"] = a

            # --- B. Per-instrument self-calibration (temporal 50/50 within stock) ---
            idx = np.where(em)[0]
            half = len(idx) // 2
            ci, ei = idx[:half], idx[half:]
            b = eval_ladder(
                probs[ci], labels[ci], m_logit[ci],
                probs[ei], labels[ei], m_logit[ei], vol_logit[ei],
            )
            selfcal[f"stock_{s}"] = b

            print(f"  stock {s}: LOSO  LAC cov={a['lac']['marginal_coverage']:.3f} "
                  f"gap={a['lac']['coverage_gap']:.3f} -> gate cov="
                  f"{a['learned_gate']['marginal_coverage']:.3f} "
                  f"gap={a['learned_gate']['coverage_gap']:.3f} "
                  f"(gap_red {a['gap_reduction_vs_lac']*100:+.0f}%)")

        # aggregate: mean coverage-gap across stocks (LOSO)
        lac_gaps = [loso[f"stock_{s}"]["lac"]["coverage_gap"] for s in range(1, 6)]
        lg_gaps = [loso[f"stock_{s}"]["learned_gate"]["coverage_gap"] for s in range(1, 6)]
        lac_cov = [loso[f"stock_{s}"]["lac"]["marginal_coverage"] for s in range(1, 6)]
        lg_cov = [loso[f"stock_{s}"]["learned_gate"]["marginal_coverage"] for s in range(1, 6)]
        n_improved = int(sum(lg < lac for lg, lac in zip(lg_gaps, lac_gaps)))
        results["models"][model] = {
            "loso": loso,
            "self_cal": selfcal,
            "loso_summary": {
                "mean_gap_lac": round(float(np.mean(lac_gaps)), 4),
                "mean_gap_gate": round(float(np.mean(lg_gaps)), 4),
                "mean_cov_lac": round(float(np.mean(lac_cov)), 4),
                "mean_cov_gate": round(float(np.mean(lg_cov)), 4),
                "stocks_gap_improved": n_improved,
                "mean_gap_reduction": round(
                    float((np.mean(lac_gaps) - np.mean(lg_gaps)) / max(np.mean(lac_gaps), 1e-6)), 4),
            },
        }
        sm = results["models"][model]["loso_summary"]
        print(f"  LOSO summary: mean gap {sm['mean_gap_lac']:.3f} -> "
              f"{sm['mean_gap_gate']:.3f} ({sm['mean_gap_reduction']*100:+.0f}%); "
              f"gate cov {sm['mean_cov_gate']:.3f}; improved {sm['stocks_gap_improved']}/5")

    # ---- save ----
    out_json = RESULTS / "per_stock_generalization.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_json}")

    rows = [["model", "stock", "analysis", "LAC_cov", "LAC_gap",
             "gate_cov", "gate_gap", "gap_reduction_%"]]
    for model, mres in results["models"].items():
        for tag, block in (("LOSO", mres["loso"]), ("self_cal", mres["self_cal"])):
            for s in range(1, 6):
                r = block[f"stock_{s}"]
                rows.append([
                    model, s, tag,
                    f"{r['lac']['marginal_coverage']:.3f}",
                    f"{r['lac']['coverage_gap']:.3f}",
                    f"{r['learned_gate']['marginal_coverage']:.3f}",
                    f"{r['learned_gate']['coverage_gap']:.3f}",
                    f"{r['gap_reduction_vs_lac']*100:+.1f}",
                ])
    out_csv = RESULTS / "per_stock_generalization_table.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved -> {out_csv}")
    print("\nPER_STOCK_GENERALIZATION_DONE")


if __name__ == "__main__":
    main()
