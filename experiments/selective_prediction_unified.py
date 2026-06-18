"""E8 — Unified selective prediction analysis across all models.

For each model, computes accuracy vs coverage curves using three
abstention signals:
  1. Max softmax confidence (1 - max_p)
  2. Softmax entropy H(p)
  3. LAC nonconformity score (1 - p_y_hat, argmax class)

Also plots the accuracy-coverage tradeoff and reports key breakpoints.

Models: TCN, LOBTransformer, CNN-LSTM-Attn, NeuralCDE (354,725 test)
        (DeepLOB val logits also included at 337,974)

Output: results/selective_pred_unified_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

MODEL_FILES = {
    "DeepLOB":        "deeplob_k10_logits.pt",
    "TCN":            "tcn_k10_test_logits.pt",
    "LOBTransformer": "lobtransformer_k10_test_logits.pt",
    "CNN-LSTM-Attn":  "cnn_lstm_attn_k10_test_logits.pt",
    "NeuralCDE":      "neural_cde_k10_test_logits.pt",
}


def load_logits(fname):
    path = RESULTS / fname
    if not path.exists():
        return None, None
    d = torch.load(path, map_location="cpu", weights_only=False)
    logits = d["logits"].float() if isinstance(d, dict) else d.float()
    labels = d["labels"].long().numpy() if isinstance(d, dict) else None
    return logits, labels


def selective_curve(scores_ascending, correct, coverage_points):
    """
    scores_ascending: higher = more confident (or lower nonconformity score)
    Returns accuracy at each coverage point.
    """
    n = len(correct)
    sorted_idx = np.argsort(-scores_ascending)  # descending confidence
    accs = []
    for cov in coverage_points:
        k = max(1, int(np.ceil(cov * n)))
        retained = sorted_idx[:k]
        accs.append(float(correct[retained].mean()))
    return accs


def breakpoints(scores_asc, correct, fracs=(0.10, 0.25, 0.50, 0.75)):
    """Accuracy at fixed retained fractions."""
    n = len(correct)
    sorted_idx = np.argsort(-scores_asc)
    base_acc = correct.mean()
    out = {}
    for frac in fracs:
        k = max(1, int(frac * n))
        acc = correct[sorted_idx[:k]].mean()
        out[f"top_{int(frac*100)}pct"] = {
            "accuracy": float(acc),
            "delta_pp": float((acc - base_acc) * 100),
        }
    return out


def area_under_selective_curve(accs, coverages):
    """AUC of accuracy-coverage curve (normalised by coverage range)."""
    return float(np.trapz(accs, coverages) / (coverages[-1] - coverages[0]))


def main():
    print("=" * 60)
    print("E8 — Unified selective prediction analysis")
    print("=" * 60)

    coverage_grid = np.linspace(0.10, 1.0, 91)  # 10% to 100% in 1% steps
    results = {}

    for name, fname in MODEL_FILES.items():
        logits, labels = load_logits(fname)
        if logits is None:
            print(f"  [SKIP] {name}: {fname} missing")
            continue

        probs  = F.softmax(logits, dim=-1).numpy()
        preds  = probs.argmax(axis=1)
        correct = (preds == labels).astype(float)
        base_acc = correct.mean()

        # Abstention signals (higher = more confident)
        max_conf  = probs.max(axis=1)
        neg_entropy = (probs * np.log(probs + 1e-12)).sum(axis=1)  # negative entropy
        # LAC score: 1 - p_argmax → lower = more confident → negate
        lac_score = -(1.0 - max_conf)   # negate so higher = more confident

        signals = {
            "max_conf":    max_conf,
            "neg_entropy": neg_entropy,
            "neg_lac":     lac_score,
        }

        model_results = {
            "base_acc":  float(base_acc),
            "n":         int(len(labels)),
            "signals":   {},
            "coverage_grid": coverage_grid.tolist(),
        }

        print(f"\n  {name} (base acc={base_acc:.4f})")
        for sig_name, scores in signals.items():
            curve = selective_curve(scores, correct, coverage_grid)
            bp    = breakpoints(scores, correct)
            ausc  = area_under_selective_curve(curve, coverage_grid)

            model_results["signals"][sig_name] = {
                "accuracy_curve": curve,
                "breakpoints":    bp,
                "ausc":           ausc,
            }

            delta25 = bp["top_25pct"]["delta_pp"]
            delta10 = bp["top_10pct"]["delta_pp"]
            print(f"    {sig_name:15s}  AUSC={ausc:.4f}  "
                  f"top10%Δ={delta10:+.2f}pp  top25%Δ={delta25:+.2f}pp")

        results[name] = model_results

    # Cross-model summary: best signal per model at 25% retention
    print("\n── Best signal per model at 25% retention ──")
    for name, r in results.items():
        best_sig, best_delta = None, -999
        for sig, sr in r["signals"].items():
            delta = sr["breakpoints"]["top_25pct"]["delta_pp"]
            if delta > best_delta:
                best_delta = delta
                best_sig = sig
        print(f"  {name:20s}  best={best_sig}  Δ={best_delta:+.2f}pp")

    # Rank models by AUSC of best signal
    print("\n── Models ranked by AUSC (best signal) ──")
    ranking = []
    for name, r in results.items():
        best_ausc = max(sr["ausc"] for sr in r["signals"].values())
        ranking.append((name, best_ausc))
    for name, ausc in sorted(ranking, key=lambda x: -x[1]):
        print(f"  {name:20s}  best AUSC={ausc:.4f}")

    out = RESULTS / "selective_pred_unified_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
