"""
EXP-G3: Selective-prediction leakage fix.

Original bug: threshold θ was fit on the test set (in-sample → inflated accuracy).
Fix: fit θ on val logits only, evaluate on held-out test set.

Selective prediction: abstain if max_softmax_prob < θ.
  - Coverage (C): fraction of test samples predicted on = 1 - abstention_rate
  - Selective accuracy (Acc_sel): accuracy on predicted-on samples
  - Unconditional accuracy (Acc_all): accuracy on all test samples (abstained = wrong)

Outputs:
  results/selective_fix_results.json   — full sweep + chosen θ
  results/selective_fix_table.csv      — Table 8 replacement rows
"""

import json, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
LOGIT_DIR = Path("/tmp/lobtransformerv2_train_results/targeted")
OUT_DIR   = Path("/tmp/dl_paper_experiments")
OUT_DIR.mkdir(exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
val_ck  = torch.load(LOGIT_DIR / "lobt_v2_k10_val_logits.pt",  map_location="cpu", weights_only=False)
test_ck = torch.load(LOGIT_DIR / "lobt_v2_k10_test_logits.pt", map_location="cpu", weights_only=False)

val_probs  = F.softmax(val_ck["logits"].float(),  dim=-1).numpy()  # (M, 3)
val_labels = val_ck["labels"].long().numpy()
test_probs  = F.softmax(test_ck["logits"].float(), dim=-1).numpy()  # (N, 3)
test_labels = test_ck["labels"].long().numpy()

val_conf  = val_probs.max(axis=1)    # max-softmax confidence
test_conf = test_probs.max(axis=1)
val_preds  = val_probs.argmax(axis=1)
test_preds = test_probs.argmax(axis=1)

N_val  = len(val_labels)
N_test = len(test_labels)
print(f"Val: {N_val:,}  Test: {N_test:,}")

# Baseline (no abstention)
base_acc = float((test_preds == test_labels).mean())
print(f"Baseline test accuracy (θ=0): {base_acc:.4f}")

# ── Fit θ on val sweep ────────────────────────────────────────────────────────
thresholds = np.arange(0.30, 0.991, 0.01)

def sel_metrics(probs, labels, confs, theta):
    mask = confs >= theta
    n_sel = mask.sum()
    if n_sel == 0:
        return {"coverage": 0.0, "selective_acc": float("nan"), "n_selected": 0}
    sel_acc = float((probs[mask].argmax(axis=1) == labels[mask]).mean())
    cov     = float(n_sel / len(labels))
    return {"coverage": cov, "selective_acc": sel_acc, "n_selected": int(n_sel)}

val_sweep = {}
for theta in thresholds:
    m = sel_metrics(val_probs, val_labels, val_conf, theta)
    val_sweep[float(f"{theta:.2f}")] = m

# ── Choose θ* on val ──────────────────────────────────────────────────────────
# Strategy: highest val selective_acc subject to coverage ≥ 50%
#           Also report the 80%, 90%, 95% coverage operating points
def find_theta_for_coverage(sweep, target_cov, thresholds):
    """Largest θ such that coverage ≥ target_cov (val set)."""
    best_theta = None
    for theta in sorted(thresholds, reverse=True):
        m = sweep.get(float(f"{theta:.2f}"), {})
        if m.get("coverage", 0) >= target_cov:
            best_theta = theta
            break
    return best_theta

# Coverage operating points
coverage_points = {}
for target in [0.50, 0.60, 0.70, 0.80, 0.90]:
    theta = find_theta_for_coverage(val_sweep, target, thresholds)
    if theta is not None:
        val_m  = val_sweep[float(f"{theta:.2f}")]
        test_m = sel_metrics(test_probs, test_labels, test_conf, theta)
        coverage_points[f"cov{int(target*100)}"] = {
            "theta": float(theta),
            "val_coverage": val_m["coverage"],
            "val_selective_acc": val_m["selective_acc"],
            "test_coverage": test_m["coverage"],
            "test_selective_acc": test_m["selective_acc"],
            "test_n_selected": test_m["n_selected"],
        }
        print(f"  cov≥{int(target*100)}%: θ={theta:.2f}  val_acc={val_m['selective_acc']:.4f} "
              f"test_cov={test_m['coverage']:.4f}  test_acc={test_m['selective_acc']:.4f}")

# Max-accuracy operating point (val, coverage≥50%)
best_theta_acc = None
best_val_acc   = -1.0
for theta in thresholds:
    m = val_sweep.get(float(f"{theta:.2f}"), {})
    if m.get("coverage", 0) >= 0.50 and m.get("selective_acc", 0) > best_val_acc:
        best_val_acc   = m["selective_acc"]
        best_theta_acc = theta

test_opt = sel_metrics(test_probs, test_labels, test_conf, best_theta_acc)
print(f"\nMax-acc θ*={best_theta_acc:.2f}:")
print(f"  Val  cov={val_sweep[float(f'{best_theta_acc:.2f}')]['coverage']:.4f}  "
      f"acc={val_sweep[float(f'{best_theta_acc:.2f}')]['selective_acc']:.4f}")
print(f"  Test cov={test_opt['coverage']:.4f}  acc={test_opt['selective_acc']:.4f}")

# ── Full sweep on test (using val-selected θ at each coverage point) ──────────
# Also show what the leaky (in-sample test) curve would look like for comparison
test_sweep_corrected = {}
test_sweep_leaky     = {}
for theta in thresholds:
    t = float(f"{theta:.2f}")
    test_sweep_corrected[t] = sel_metrics(test_probs, test_labels, test_conf, theta)
    # "leaky" = what you'd report if θ was fit on test (same data used for eval)
    test_sweep_leaky[t]     = sel_metrics(test_probs, test_labels, test_conf, theta)
    # Note: leaky and corrected are identical here because we compute coverage/acc on same test set.
    # The leakage difference is in which θ you choose: a leaky approach picks θ to maximize
    # test accuracy (test-oracle θ) rather than val-derived θ.

# Leaky baseline: oracle θ from test set (what the bug produced)
best_theta_leaky = None
best_test_acc_leaky = -1.0
for theta in thresholds:
    m = test_sweep_leaky.get(float(f"{theta:.2f}"), {})
    if m.get("coverage", 0) >= 0.50 and m.get("selective_acc", 0) > best_test_acc_leaky:
        best_test_acc_leaky  = m["selective_acc"]
        best_theta_leaky = theta

print(f"\nLeaky (oracle θ on test): θ*={best_theta_leaky:.2f}  test_acc={best_test_acc_leaky:.4f}")
print(f"Corrected (val-fit θ):    θ*={best_theta_acc:.2f}  test_acc={test_opt['selective_acc']:.4f}")
print(f"Overfit gap: {best_test_acc_leaky - test_opt['selective_acc']:.4f}")

# ── Results ───────────────────────────────────────────────────────────────────
results = {
    "model":  "LOBTransformerV2",
    "horizon": 10,
    "n_val":  int(N_val),
    "n_test": int(N_test),
    "baseline_test_acc_no_abstention": base_acc,
    "leaky_oracle": {
        "theta": float(best_theta_leaky),
        "test_coverage": test_sweep_leaky[float(f"{best_theta_leaky:.2f}")]["coverage"],
        "test_selective_acc": best_test_acc_leaky,
        "note": "θ chosen by maximizing test accuracy — invalid for publication",
    },
    "corrected_max_acc": {
        "theta": float(best_theta_acc),
        "val_coverage": float(val_sweep[float(f"{best_theta_acc:.2f}")]["coverage"]),
        "val_selective_acc": float(val_sweep[float(f"{best_theta_acc:.2f}")]["selective_acc"]),
        "test_coverage": float(test_opt["coverage"]),
        "test_selective_acc": float(test_opt["selective_acc"]),
        "test_n_selected": test_opt["n_selected"],
        "note": "θ chosen by maximizing val accuracy (coverage≥50%) — valid",
    },
    "coverage_operating_points": coverage_points,
    "overfit_gap": float(best_test_acc_leaky - test_opt["selective_acc"]),
}

out_json = Path(OUT_DIR) / "selective_fix_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out_json}")

# ── Table 8 replacement CSV ──────────────────────────────────────────────────
rows = [
    ["θ source", "θ", "Coverage (C)", "Selective Acc.", "Baseline Acc.", "Note"],
    ["Test (leaked — invalid)",
     f"{best_theta_leaky:.2f}",
     f"{test_sweep_leaky[float(f'{best_theta_leaky:.2f}')]['coverage']:.3f}",
     f"{best_test_acc_leaky:.4f}",
     f"{base_acc:.4f}",
     "θ fit on test; overestimates"],
    ["Val (corrected)",
     f"{best_theta_acc:.2f}",
     f"{test_opt['coverage']:.3f}",
     f"{test_opt['selective_acc']:.4f}",
     f"{base_acc:.4f}",
     "θ fit on val; valid estimate"],
]
# Coverage operating points
for label, cp in coverage_points.items():
    rows.append([
        f"Val-fit ({label})",
        f"{cp['theta']:.2f}",
        f"{cp['test_coverage']:.3f}",
        f"{cp['test_selective_acc']:.4f}",
        f"{base_acc:.4f}",
        "",
    ])

out_csv = Path(OUT_DIR) / "selective_fix_table.csv"
with open(out_csv, "w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"Saved → {out_csv}")
print("\nSELECTIVE_FIX_DONE")
