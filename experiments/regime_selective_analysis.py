"""E10 — Regime-conditional selective prediction + class-conditional calibration.

Uses val logits (337,974 samples) aligned with macc_features.npz regime labels.
Answers:
  (a) Is selective prediction especially valuable in high-vol regimes?
  (b) Which class (down/stat/up) is most miscalibrated per model?
  (c) Does high-vol regime predict harder samples (lower confidence)?

Output: results/regime_selective_results.json
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

# Val-set logit files (337,974 samples — aligned with macc_features)
VAL_MODEL_FILES = {
    "DeepLOB":       "deeplob_k10_logits.pt",
}
# These are test-set logits; they don't align with macc_features (which is val)
# So we only use DeepLOB val logits for regime analysis (337,974 samples match macc)
# For other models, we do class-conditional ECE on their test logits separately.
TEST_MODEL_FILES = {
    "TCN":            "tcn_k10_test_logits.pt",
    "LOBTransformer": "lobtransformer_k10_test_logits.pt",
    "CNN-LSTM-Attn":  "cnn_lstm_attn_k10_test_logits.pt",
    "NeuralCDE":      "neural_cde_k10_test_logits.pt",
}
CLASS_NAMES = ["Down", "Stationary", "Up"]


def load_logits(fname):
    path = RESULTS / fname
    if not path.exists():
        return None, None
    d = torch.load(path, map_location="cpu", weights_only=False)
    logits = d["logits"].float() if isinstance(d, dict) else d.float()
    labels = d["labels"].long().numpy() if isinstance(d, dict) else None
    return logits, labels


def ece_per_class(probs, labels, n_bins=15):
    """Per-class ECE: confidence of predicting class c vs actual rate."""
    n_classes = probs.shape[1]
    result = {}
    for c in range(n_classes):
        conf_c = probs[:, c]
        is_c   = (labels == c).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece_c = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf_c >= lo) & (conf_c < hi)
            if mask.sum() == 0:
                continue
            ece_c += mask.mean() * abs(is_c[mask].mean() - conf_c[mask].mean())
        result[CLASS_NAMES[c]] = float(ece_c)
    return result


def regime_analysis(probs, labels, regimes):
    """Selective accuracy and calibration split by regime."""
    preds   = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    conf    = probs.max(axis=1)

    out = {}
    regime_names = {0: "low_vol", 1: "high_vol"}
    for r, rname in regime_names.items():
        mask = regimes == r
        if mask.sum() == 0:
            continue
        r_probs   = probs[mask]
        r_labels  = labels[mask]
        r_correct = correct[mask]
        r_conf    = conf[mask]
        r_preds   = preds[mask]

        acc  = r_correct.mean()
        f1   = f1_score(r_labels, r_preds, average="macro", zero_division=0)
        # ECE
        bins = np.linspace(0, 1, 16)
        ece_r = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (r_conf >= lo) & (r_conf < hi)
            if m.sum() == 0: continue
            ece_r += m.mean() * abs(r_correct[m].mean() - r_conf[m].mean())

        # Selective at 25% abstain
        k25 = max(1, int(0.75 * mask.sum()))
        top_idx = np.argsort(-r_conf)[:k25]
        sel_acc = r_correct[top_idx].mean()

        out[rname] = {
            "n":         int(mask.sum()),
            "frac":      float(mask.mean()),
            "acc":       float(acc),
            "f1_macro":  float(f1),
            "ece":       float(ece_r),
            "mean_conf": float(r_conf.mean()),
            "sel_acc_25pct": float(sel_acc),
            "sel_delta_pp":  float((sel_acc - acc) * 100),
        }
    return out


def main():
    print("=" * 60)
    print("E10 — Regime-conditional selective + per-class calibration")
    print("=" * 60)

    # Load macc features (337,974 samples)
    macc = np.load(RESULTS / "macc_features.npz")
    regimes = macc["r_val"].astype(int)
    print(f"Regime distribution: low={( regimes==0).sum()}  high={(regimes==1).sum()}")

    results = {}

    # ── Part A: regime analysis on DeepLOB val logits ─────────────────────────
    print("\n── Part A: Regime-conditional analysis (DeepLOB val) ──")
    logits, labels = load_logits("deeplob_k10_logits.pt")
    if logits is not None and len(labels) == len(regimes):
        probs = F.softmax(logits, dim=-1).numpy()
        r_results = regime_analysis(probs, labels, regimes)
        results["DeepLOB_regime"] = r_results
        for rname, r in r_results.items():
            print(f"  {rname:10s}: n={r['n']:6d}  acc={r['acc']:.4f}  "
                  f"ECE={r['ece']:.4f}  conf={r['mean_conf']:.4f}  "
                  f"sel@75%={r['sel_acc_25pct']:.4f}  Δ={r['sel_delta_pp']:+.2f}pp")
        if "low_vol" in r_results and "high_vol" in r_results:
            low, high = r_results["low_vol"], r_results["high_vol"]
            print(f"\n  Regime gap  acc_low={low['acc']:.4f}  acc_high={high['acc']:.4f}  "
                  f"Δ={abs(low['acc']-high['acc'])*100:.2f}pp")
            print(f"  ECE gap     ece_low={low['ece']:.4f}  ece_high={high['ece']:.4f}  "
                  f"Δ={abs(low['ece']-high['ece'])*100:.4f}")
            print(f"  Conf gap    conf_low={low['mean_conf']:.4f}  conf_high={high['mean_conf']:.4f}")
            print(f"  Selective gain in high-vol: {high['sel_delta_pp']:+.2f}pp  "
                  f"vs low-vol: {low['sel_delta_pp']:+.2f}pp")
    else:
        print("  Skipping regime analysis: size mismatch")

    # ── Part B: Per-class ECE for all models (test set) ───────────────────────
    print("\n── Part B: Per-class ECE (test logits) ──")
    all_files = {**{"DeepLOB_val": "deeplob_k10_logits.pt"}, **TEST_MODEL_FILES}
    for name, fname in all_files.items():
        logits, labels = load_logits(fname)
        if logits is None:
            continue
        probs = F.softmax(logits, dim=-1).numpy()
        pc_ece = ece_per_class(probs, labels)
        worst = max(pc_ece, key=pc_ece.get)
        print(f"  {name:20s}: Down={pc_ece['Down']:.4f}  "
              f"Stat={pc_ece['Stationary']:.4f}  Up={pc_ece['Up']:.4f}  "
              f"[worst={worst}]")
        results[f"{name}_per_class_ece"] = pc_ece

        # Class imbalance
        counts = np.bincount(labels, minlength=3)
        preds  = probs.argmax(axis=1)
        pred_counts = np.bincount(preds, minlength=3)

    # ── Part C: Confidence distribution by regime for DeepLOB ─────────────────
    print("\n── Part C: Confidence histogram by regime (DeepLOB val) ──")
    logits, labels = load_logits("deeplob_k10_logits.pt")
    if logits is not None and len(labels) == len(regimes):
        probs = F.softmax(logits, dim=-1).numpy()
        conf  = probs.max(axis=1)
        for r, rname in [(0, "low_vol"), (1, "high_vol")]:
            mask = regimes == r
            c = conf[mask]
            print(f"  {rname}: mean={c.mean():.4f}  median={np.median(c):.4f}  "
                  f"<0.5={( c<0.5).mean()*100:.1f}%  "
                  f">0.9={(c>0.9).mean()*100:.1f}%")
        results["DeepLOB_conf_by_regime"] = {
            "low_vol":  {"mean": float(conf[regimes==0].mean()),
                         "median": float(np.median(conf[regimes==0])),
                         "frac_low_conf": float((conf[regimes==0]<0.5).mean()),
                         "frac_high_conf": float((conf[regimes==0]>0.9).mean())},
            "high_vol": {"mean": float(conf[regimes==1].mean()),
                         "median": float(np.median(conf[regimes==1])),
                         "frac_low_conf": float((conf[regimes==1]<0.5).mean()),
                         "frac_high_conf": float((conf[regimes==1]>0.9).mean())},
        }

    out = RESULTS / "regime_selective_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
