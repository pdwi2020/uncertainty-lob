"""E6 — Deep Ensemble UQ baseline.

Combines TCN + LOBTransformer + CNN-LSTM-Attn (all share 354,725 test samples)
into a simple average-logit ensemble, then computes UQ metrics and compares
to individual models and the EvLOB aggregate (from track_a_full_analysis.json).

Metrics:
  Accuracy, F1-macro, Cohen's kappa
  ECE (15 bins), NLL, Brier score
  Error-detection AUROC (entropy score, 1-max_softmax)
  Selective prediction: accuracy at 25% / 50% abstention

Output: results/deep_ensemble_uq_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

# ── Logit files ────────────────────────────────────────────────────────────────
MODEL_FILES = {
    "TCN":            "tcn_k10_test_logits.pt",
    "LOBTransformer": "lobtransformer_k10_test_logits.pt",
    "CNN-LSTM-Attn":  "cnn_lstm_attn_k10_test_logits.pt",
    "NeuralCDE":      "neural_cde_k10_test_logits.pt",
}


def load_logits(fname):
    d = torch.load(RESULTS / fname, map_location="cpu", weights_only=False)
    logits = d["logits"].float() if isinstance(d, dict) else d.float()
    labels = d["labels"].long().numpy() if isinstance(d, dict) else None
    return logits, labels


def ece(probs, labels, n_bins=15):
    """Expected Calibration Error (equal-width bins)."""
    confidence = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        acc_b = correct[mask].mean()
        conf_b = confidence[mask].mean()
        ece_val += mask.mean() * abs(acc_b - conf_b)
    return float(ece_val)


def nll(probs, labels):
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())


def brier(probs, labels):
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def auroc_error_detection(probs, labels):
    """AUROC for detecting wrong predictions (error=1) using entropy."""
    preds = probs.argmax(axis=1)
    errors = (preds != labels).astype(int)
    if errors.sum() == 0 or errors.sum() == len(errors):
        return float("nan")
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    conf    = probs.max(axis=1)
    auroc_ent  = roc_auc_score(errors, entropy)
    auroc_conf = roc_auc_score(errors, -conf)
    return {"entropy": float(auroc_ent), "neg_max_conf": float(auroc_conf)}


def selective_accuracy(probs, labels, abstain_fracs=(0.10, 0.25, 0.50)):
    """Accuracy after abstaining on the least-confident abstain_frac fraction."""
    confidence = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels)
    results = {}
    for frac in abstain_fracs:
        thresh = np.quantile(confidence, frac)
        retained = confidence >= thresh
        if retained.sum() == 0:
            results[f"abstain_{int(frac*100)}pct"] = None
            continue
        acc = correct[retained].mean()
        results[f"abstain_{int(frac*100)}pct"] = {
            "accuracy": float(acc),
            "retained": float(retained.mean()),
            "delta_pp": float((acc - correct.mean()) * 100),
        }
    return results


def evaluate(probs, labels, name):
    preds = probs.argmax(axis=1)
    acc   = (preds == labels).mean()
    f1    = f1_score(labels, preds, average="macro")
    kap   = cohen_kappa_score(labels, preds)
    return {
        "name":      name,
        "n":         int(len(labels)),
        "acc":       float(acc),
        "f1_macro":  float(f1),
        "kappa":     float(kap),
        "ece":       ece(probs, labels),
        "nll":       nll(probs, labels),
        "brier":     brier(probs, labels),
        "auroc":     auroc_error_detection(probs, labels),
        "selective": selective_accuracy(probs, labels),
    }


def main():
    print("=" * 60)
    print("E6 — Deep Ensemble UQ baseline")
    print("=" * 60)

    # Load all models
    data = {}
    for name, fname in MODEL_FILES.items():
        path = RESULTS / fname
        if not path.exists():
            print(f"  [SKIP] {name}: {fname} missing")
            continue
        logits, labels = load_logits(fname)
        data[name] = (logits, labels)
        print(f"  Loaded {name}: {logits.shape}")

    if not data:
        print("No logit files found.")
        sys.exit(1)

    # Verify labels match across models (use TCN labels as reference)
    ref_labels = None
    for name, (logits, labels) in data.items():
        if ref_labels is None:
            ref_labels = labels
        elif not np.array_equal(labels, ref_labels):
            print(f"  WARNING: {name} labels differ from reference! Sizes: {len(labels)} vs {len(ref_labels)}")

    results = {}

    # Evaluate each model individually
    print("\n── Individual models ──")
    for name, (logits, labels) in data.items():
        probs = F.softmax(logits, dim=-1).numpy()
        r = evaluate(probs, labels, name)
        results[name] = r
        print(f"  {name:20s}  acc={r['acc']:.4f}  ECE={r['ece']:.4f}  NLL={r['nll']:.4f}  kappa={r['kappa']:.4f}")

    # 3-model ensemble (TCN + LOBTransformer + CNN-LSTM-Attn)
    ensemble_names = ["TCN", "LOBTransformer", "CNN-LSTM-Attn"]
    available_ens = [n for n in ensemble_names if n in data]
    if len(available_ens) >= 2:
        print(f"\n── Ensemble: {' + '.join(available_ens)} ──")
        ens_logits = torch.stack([data[n][0] for n in available_ens]).mean(0)
        ens_probs  = F.softmax(ens_logits, dim=-1).numpy()
        ens_labels = data[available_ens[0]][1]
        r = evaluate(ens_probs, ens_labels, "Ensemble-3")
        results["Ensemble_3model"] = r
        print(f"  {'Ensemble-3':20s}  acc={r['acc']:.4f}  ECE={r['ece']:.4f}  NLL={r['nll']:.4f}  kappa={r['kappa']:.4f}")

        # 4-model ensemble (add NeuralCDE if available)
        if "NeuralCDE" in data:
            all4 = available_ens + ["NeuralCDE"]
            print(f"\n── Ensemble: {' + '.join(all4)} ──")
            ens4_logits = torch.stack([data[n][0] for n in all4]).mean(0)
            ens4_probs  = F.softmax(ens4_logits, dim=-1).numpy()
            r4 = evaluate(ens4_probs, ens_labels, "Ensemble-4")
            results["Ensemble_4model"] = r4
            print(f"  {'Ensemble-4':20s}  acc={r4['acc']:.4f}  ECE={r4['ece']:.4f}  NLL={r4['nll']:.4f}  kappa={r4['kappa']:.4f}")

    # ECE improvement summary
    print("\n── ECE vs individual models ──")
    if "Ensemble_3model" in results:
        ens_ece = results["Ensemble_3model"]["ece"]
        for name in available_ens:
            indiv_ece = results[name]["ece"]
            delta = (indiv_ece - ens_ece) / indiv_ece * 100
            print(f"  {name}: ECE {indiv_ece:.4f} → {ens_ece:.4f} ({delta:+.1f}%)")

    # Selective prediction summary
    print("\n── Selective prediction (abstain 25%) ──")
    for name, r in results.items():
        sel = r.get("selective", {}).get("abstain_25pct")
        if sel:
            print(f"  {name:25s}  acc@75%={sel['accuracy']:.4f}  Δ={sel['delta_pp']:+.2f}pp")

    # Save
    out = RESULTS / "deep_ensemble_uq_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
