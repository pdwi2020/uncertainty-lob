"""E7 — Per-horizon calibration analysis.

Uses DeepLOB logits at k=1,2,3,5,10 to study how calibration quality
(ECE, NLL, Brier) and accuracy degrade as the prediction horizon lengthens.
Validates the claim that longer-horizon forecasting is harder and less calibrated.

Output: results/per_horizon_ece_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, cohen_kappa_score

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

HORIZONS = [1, 2, 3, 5, 10]


def load_logits(fname):
    path = RESULTS / fname
    if not path.exists():
        return None, None
    d = torch.load(path, map_location="cpu", weights_only=False)
    logits = d["logits"].float() if isinstance(d, dict) else d.float()
    labels = (d["labels"].long().numpy() if isinstance(d, dict) else None)
    return logits, labels


def ece(probs, labels, n_bins=15):
    conf  = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    corr  = (preds == labels).astype(float)
    bins  = np.linspace(0, 1, n_bins + 1)
    val   = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        val += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(val)


def entropy_mean(probs):
    return float(-(probs * np.log(probs + 1e-12)).sum(axis=1).mean())


def nll(probs, labels):
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())


def brier(probs, labels):
    oh = np.eye(3)[labels]
    return float(((probs - oh) ** 2).sum(axis=1).mean())


def class_imbalance(labels):
    counts = np.bincount(labels, minlength=3)
    return {
        "down": int(counts[0]), "stationary": int(counts[1]), "up": int(counts[2]),
        "pct_stationary": float(counts[1] / len(labels) * 100),
    }


def main():
    print("=" * 60)
    print("E7 — Per-horizon ECE / calibration analysis (DeepLOB)")
    print("=" * 60)

    results = {}

    for k in HORIZONS:
        fname = f"deeplob_k{k}_logits.pt"
        logits, labels = load_logits(fname)
        if logits is None:
            print(f"  k={k:2d}: MISSING ({fname})")
            continue

        probs = F.softmax(logits, dim=-1).numpy()
        preds = probs.argmax(axis=1)
        acc   = (preds == labels).mean()
        f1    = f1_score(labels, preds, average="macro")
        kap   = cohen_kappa_score(labels, preds)
        e     = ece(probs, labels)
        n     = nll(probs, labels)
        b     = brier(probs, labels)
        h     = entropy_mean(probs)
        ci    = class_imbalance(labels)

        results[f"k{k}"] = {
            "k": k,
            "n": int(len(labels)),
            "acc": float(acc),
            "f1_macro": float(f1),
            "kappa": float(kap),
            "ece": e,
            "nll": n,
            "brier": b,
            "mean_entropy": h,
            "class_counts": ci,
        }

        print(f"  k={k:2d}  n={len(labels):7d}  acc={acc:.4f}  F1={f1:.4f}  κ={kap:.4f}  "
              f"ECE={e:.4f}  NLL={n:.4f}  Brier={b:.4f}  H={h:.4f}  "
              f"stat%={ci['pct_stationary']:.1f}%")

    if len(results) >= 2:
        # Deltas from k=1 to k=10
        print("\n── Δ from k=1 to k=10 ──")
        r1  = results.get("k1",  {})
        r10 = results.get("k10", {})
        for metric in ["acc", "ece", "nll", "brier", "mean_entropy"]:
            if metric in r1 and metric in r10:
                delta = r10[metric] - r1[metric]
                print(f"  {metric:15s}: {r1[metric]:.4f} → {r10[metric]:.4f}  (Δ={delta:+.4f})")

    out = RESULTS / "per_horizon_ece_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
