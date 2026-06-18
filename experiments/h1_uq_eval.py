"""E3 — H¹ continuous-time UQ evaluation for NeuralCDE.

Uses existing test logits and h1_norms.npy (zero new compute).
Evaluates H¹ path energy as an error-detection / selective-prediction
signal vs softmax confidence and MC-Dropout entropy.

Saves: results/h1_uq_results.json
"""

import sys, json, math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))


def compute_ece(probs_np, labels_np, n_bins=15):
    confs = probs_np.max(axis=1)
    preds = probs_np.argmax(axis=1)
    correct = (preds == labels_np).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() > 0:
            ece += mask.sum() / len(labels_np) * abs(correct[mask].mean() - confs[mask].mean())
    return float(ece)


def bootstrap_auroc(y_true, scores, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def conformal_selective_curves(scores, labels_np, probs_np):
    """Compute selective-prediction curves using different uncertainty signals."""
    results = {}
    thresholds = np.linspace(0.3, 0.99, 50)
    base_acc = (probs_np.argmax(axis=1) == labels_np).mean()

    for signal_name, signal in [
        ("softmax_conf", probs_np.max(axis=1)),
        ("h1_norm_inv",  -scores),   # low H¹ = abstain (keep high-H¹ samples)
    ]:
        curve = []
        for t in thresholds:
            mask = signal >= t
            if mask.sum() < 100:
                break
            acc = (probs_np.argmax(axis=1)[mask] == labels_np[mask]).mean()
            curve.append({
                "threshold": round(float(t), 4),
                "coverage":  round(float(mask.mean()), 4),
                "accuracy":  round(float(acc), 4),
                "delta_acc": round(float(acc - base_acc), 4),
            })
        results[signal_name] = curve
    return results


def main():
    print("E3 — H¹ UQ Evaluation")

    # Load test logits
    saved = torch.load(RESULTS / "neural_cde_k10_test_logits.pt", map_location="cpu", weights_only=False)
    logits = saved["logits"].float()
    labels = saved["labels"].long().numpy()
    probs  = F.softmax(logits, dim=-1).numpy()
    preds  = probs.argmax(axis=1)
    is_wrong = (preds != labels).astype(int)
    n_test = len(labels)
    base_acc = float((preds == labels).mean())
    print(f"  Test set: n={n_test}, base_acc={base_acc:.4f}")

    # Load H¹ norms
    h1 = np.load(RESULTS / "neural_cde_k10_h1_norms.npy")
    print(f"  H¹ norms: n={len(h1)}, mean={h1.mean():.1f}, median={np.median(h1):.1f}")
    assert len(h1) == n_test, f"H¹ size mismatch: {len(h1)} vs {n_test}"

    # ── Signals ──────────────────────────────────────────────────────────────
    softmax_conf    = probs.max(axis=1)
    softmax_entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    h1_norm         = h1

    # AUROC for misclassification detection (higher score → more likely wrong)
    print("\n  AUROC (misclassification detection):")
    signals = {
        "1 - softmax_conf":    (1.0 - softmax_conf,      "high 1-conf → wrong"),
        "softmax_entropy":     (softmax_entropy,          "high entropy → wrong"),
        "h1_norm":             (h1_norm,                  "high H¹ → wrong?"),
        "1 - h1_norm_normed":  (1.0 / (h1_norm + 1.0),   "low H¹ → wrong"),
    }

    auroc_results = {}
    for name, (scores, desc) in signals.items():
        try:
            auc = roc_auc_score(is_wrong, scores)
            ci_lo, ci_hi = bootstrap_auroc(is_wrong, scores, n_boot=500)
            ap = average_precision_score(is_wrong, scores)
            auroc_results[name] = {
                "auroc":   round(float(auc), 4),
                "ci_95":   [round(ci_lo, 4), round(ci_hi, 4)],
                "avg_prec": round(float(ap), 4),
                "desc":    desc,
            }
            print(f"    {name:30s}: AUROC={auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  AP={ap:.4f}")
        except Exception as e:
            print(f"    {name}: ERROR {e}")

    # ── H¹ distribution statistics ────────────────────────────────────────────
    h1_stats = {
        "n":      int(n_test),
        "mean":   round(float(h1.mean()), 2),
        "median": round(float(np.median(h1)), 2),
        "std":    round(float(h1.std()), 2),
        "p25":    round(float(np.percentile(h1, 25)), 2),
        "p75":    round(float(np.percentile(h1, 75)), 2),
        "p95":    round(float(np.percentile(h1, 95)), 2),
        "min":    round(float(h1.min()), 2),
        "max":    round(float(h1.max()), 2),
    }

    # ── Conformalize H¹ as alternative nonconformity score ───────────────────
    # Load val logits for calibration
    saved_val = torch.load(RESULTS / "neural_cde_k10_logits.pt", map_location="cpu", weights_only=False)
    val_logits = saved_val["logits"].float()
    val_labels = saved_val["labels"].long().numpy()
    val_probs  = F.softmax(val_logits, dim=-1).numpy()

    # Standard LAC conformal on test (from val q̂)
    alpha = 0.10
    val_scores = 1.0 - val_probs[np.arange(len(val_labels)), val_labels]
    n_val = len(val_labels)
    q_hat_lac = float(np.quantile(val_scores, min(math.ceil((n_val+1)*(1-alpha))/n_val, 1.0),
                                   method="higher"))
    test_scores_lac = 1.0 - probs[np.arange(n_test), labels]
    cov_lac  = float(((1.0 - probs) <= q_hat_lac)[np.arange(n_test), labels].mean())

    # H¹-based conformalized score: s(x,y) = 1 - p̂_y + λ * (h1 / median_h1)
    lam = 0.1
    h1_normed = h1 / float(np.median(h1) + 1e-8)
    test_h1_scores = test_scores_lac + lam * h1_normed
    val_h1_scores  = val_scores  # val H¹ not available — use same LAC val q̂ as approx

    conformal_results = {
        "lac_standard": {
            "q_hat":      round(q_hat_lac, 4),
            "coverage":   round(cov_lac, 4),
            "set_size":   round(float(((1.0 - probs) <= q_hat_lac).sum(axis=1).mean()), 4),
            "singleton":  round(float((((1.0 - probs) <= q_hat_lac).sum(axis=1) == 1).mean()), 4),
        }
    }

    # ── Selective curves ──────────────────────────────────────────────────────
    selective_curves = conformal_selective_curves(h1_norm, labels, probs)

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "experiment": "E3_h1_uq_eval",
        "model":      "neural_cde",
        "horizon":    10,
        "n_test":     int(n_test),
        "base_acc":   round(base_acc, 4),
        "h1_stats":   h1_stats,
        "auroc_results": auroc_results,
        "conformal_results": conformal_results,
        "selective_curves":  selective_curves,
    }

    out_path = RESULTS / "h1_uq_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  H¹ AUROC (misclassification):  {auroc_results.get('h1_norm', {}).get('auroc', 'N/A')}")
    print(f"  1-conf AUROC:                  {auroc_results.get('1 - softmax_conf', {}).get('auroc', 'N/A')}")
    print(f"  Softmax entropy AUROC:         {auroc_results.get('softmax_entropy', {}).get('auroc', 'N/A')}")
    return out


if __name__ == "__main__":
    main()
