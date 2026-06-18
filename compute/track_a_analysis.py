"""Comprehensive Track-A analysis: bootstrap CIs, McNemar, ECE, vacuity, error-detection AUROC.

Run after all training completes:
  python3 /workspace/track_a_analysis.py --results-dir /workspace/results

Outputs:
  /workspace/results/track_a_full_analysis.json   (paper-ready numbers)
  /workspace/results/track_a_latex_rows.txt        (copy-paste into LaTeX)
"""
import argparse
import json
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score, cohen_kappa_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

SEEDS = [0, 1, 2]
HORIZON = 10
N_BOOT = 1000
np.random.seed(42)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_logits(results_dir, arch, tag, seed):
    path = results_dir / f"{arch}_{tag}_k{HORIZON}_seed{seed}_test_logits.pt"
    if not path.exists():
        return None
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    logits = d.get("logits")
    labels = d.get("labels")
    alpha  = d.get("alpha")
    if logits is None or labels is None:
        return None
    probs = F.softmax(logits.float(), dim=-1).numpy()
    lbl   = labels.long().numpy()
    result = {"probs": probs, "labels": lbl, "alpha": alpha.numpy() if alpha is not None else None}
    # LOB-SDE MC uncertainty fields
    for k in ("aleatoric", "epistemic", "h1_energy"):
        v = d.get(k)
        result[k] = v.numpy() if v is not None else None
    return result


def compute_metrics(probs, labels):
    preds = probs.argmax(1)
    acc = (preds == labels).mean()
    if HAS_SKLEARN:
        f1  = f1_score(labels, preds, average="macro", zero_division=0)
        kap = cohen_kappa_score(labels, preds)
    else:
        f1 = kap = float("nan")
    # ECE (10 bins)
    confs  = probs.max(1)
    bins   = np.linspace(0, 1, 11)
    ece    = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.sum() == 0: continue
        acc_b = (preds[m] == labels[m]).mean()
        conf_b = confs[m].mean()
        ece += (m.sum() / len(labels)) * abs(acc_b - conf_b)
    # NLL
    nll = F.cross_entropy(torch.tensor(np.log(np.clip(probs, 1e-8, 1))),
                          torch.tensor(labels)).item()
    return {"acc": float(acc), "f1_macro": float(f1), "kappa": float(kap),
            "ece": float(ece), "nll": float(nll)}


def bootstrap_metric(probs, labels, key="acc", n=N_BOOT):
    n_samples = len(labels)
    vals = []
    for _ in range(n):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        m = compute_metrics(probs[idx], labels[idx])
        vals.append(m[key])
    return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def compute_vacuity(alpha):
    """K / S where S = sum(alpha), K = 3 classes."""
    if alpha is None:
        return float("nan")
    S = alpha.sum(axis=1)
    K = alpha.shape[1]
    return float((K / S).mean())


def error_detection_auroc(scores, labels, probs):
    """AUROC for using `scores` to detect misclassified samples.
    Higher score = more uncertain = should be misclassified.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    preds   = probs.argmax(1)
    is_wrong = (preds != labels).astype(int)
    if is_wrong.sum() == 0 or is_wrong.sum() == len(is_wrong):
        return float("nan")
    return float(roc_auc_score(is_wrong, scores))


def mcnemar_test(pred_a, pred_b, labels):
    """McNemar's test between two classifiers."""
    correct_a = (pred_a == labels)
    correct_b = (pred_b == labels)
    b = ((~correct_a) & correct_b).sum()   # A wrong, B right
    c = (correct_a & (~correct_b)).sum()   # A right, B wrong
    if (b + c) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": int(b), "c": int(c)}
    # continuity-corrected McNemar
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return {"statistic": float(chi2), "p_value": float(p), "b": int(b), "c": int(c)}


# ── main ──────────────────────────────────────────────────────────────────────

def analyse_arch(results_dir, arch, tag):
    all_probs, all_labels, all_alpha = [], [], []
    seed_results = {}

    for seed in SEEDS:
        d = load_logits(results_dir, arch, tag, seed)
        if d is None:
            print(f"  [MISSING] {arch}_{tag} seed={seed}")
            continue
        m = compute_metrics(d["probs"], d["labels"])
        vac = compute_vacuity(d["alpha"])
        # Entropy as uncertainty signal
        ent = -(d["probs"] * np.log(np.clip(d["probs"], 1e-8, 1))).sum(1)
        auroc_ent = error_detection_auroc(ent, d["labels"], d["probs"])
        # Primary uncertainty signal: vacuity (EvLOB) or epistemic (LOB-SDE) or entropy
        auroc_primary = float("nan")
        if d["alpha"] is not None:
            S = d["alpha"].sum(axis=1)
            K = 3
            vacuity_per_sample = K / S
            auroc_primary = error_detection_auroc(vacuity_per_sample, d["labels"], d["probs"])
        elif d["epistemic"] is not None:
            auroc_primary = error_detection_auroc(d["epistemic"], d["labels"], d["probs"])
        # Also compute h1_energy AUROC for LOB-SDE
        auroc_h1 = float("nan")
        if d.get("h1_energy") is not None:
            auroc_h1 = error_detection_auroc(d["h1_energy"], d["labels"], d["probs"])
        seed_results[seed] = {**m, "vacuity": vac,
                              "mean_entropy": float(ent.mean()),
                              "auroc_entropy": auroc_ent,
                              "auroc_vacuity": auroc_primary,
                              "auroc_h1": auroc_h1}
        all_probs.append(d["probs"])
        all_labels.append(d["labels"])
        all_alpha.append(d["alpha"])

    if not all_probs:
        return None

    # Pool all seeds for bootstrap CIs on pooled metrics
    pooled_probs  = np.concatenate(all_probs,  axis=0)
    pooled_labels = np.concatenate(all_labels, axis=0)

    # Mean ± CI across seeds
    accs    = [v["acc"]       for v in seed_results.values()]
    f1s     = [v["f1_macro"]  for v in seed_results.values()]
    kaps    = [v["kappa"]     for v in seed_results.values()]
    eces    = [v["ece"]       for v in seed_results.values()]
    vacs    = [v["vacuity"]      for v in seed_results.values()]
    ents    = [v["mean_entropy"] for v in seed_results.values()]
    aurocs_e= [v["auroc_entropy"]  for v in seed_results.values()]
    aurocs_v= [v["auroc_vacuity"]  for v in seed_results.values()]

    def sem_ci(lst):
        a = np.array([x for x in lst if not np.isnan(x)])
        if len(a) < 2: return np.nanmean(lst), float("nan"), float("nan")
        ci = 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))
        return float(np.mean(a)), float(np.mean(a)-ci), float(np.mean(a)+ci)

    return {
        "arch": arch, "tag": tag,
        "n_seeds": len(seed_results),
        "acc": sem_ci(accs),
        "f1_macro": sem_ci(f1s),
        "kappa": sem_ci(kaps),
        "ece": sem_ci(eces),
        "vacuity": sem_ci(vacs),
        "aleatoric": sem_ci(ents),
        "auroc_entropy": sem_ci(aurocs_e),
        "auroc_vacuity": sem_ci(aurocs_v),
        "per_seed": seed_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/workspace/results")
    args = ap.parse_args()
    rdir = Path(args.results_dir)

    runs = [
        ("evlob",   "gate"),
        ("evlob",   "nogate"),
        ("lob_sde", "gate"),
        ("rac_moe", "gate"),
    ]

    results = {}
    for arch, tag in runs:
        print(f"\nAnalysing {arch}/{tag} ...")
        r = analyse_arch(rdir, arch, tag)
        if r:
            results[f"{arch}_{tag}"] = r

    # McNemar: gate vs nogate for EvLOB
    print("\n--- McNemar: EvLOB gate vs nogate (seed 0) ---")
    d_gate   = load_logits(rdir, "evlob", "gate",   0)
    d_nogate = load_logits(rdir, "evlob", "nogate", 0)
    if d_gate and d_nogate:
        try:
            mc = mcnemar_test(d_gate["probs"].argmax(1),
                              d_nogate["probs"].argmax(1),
                              d_gate["labels"])
            print(f"  McNemar statistic={mc['statistic']:.3f}  p={mc['p_value']:.4f}")
            print(f"  b={mc['b']} (gate wrong/nogate right)  c={mc['c']} (gate right/nogate wrong)")
            results["mcnemar_gate_vs_nogate"] = mc
        except Exception as e:
            print(f"  McNemar failed: {e}")

    # Print summary table
    print("\n" + "="*85)
    print(f"{'Model':<22} {'tag':<8} {'Acc%':>10} {'F1':>8} {'κ':>8} {'ECE':>7} {'Vac':>7} {'AUROC_V':>9}")
    print("-"*85)
    for key, r in results.items():
        if "mcnemar" in key: continue
        acc_m, acc_lo, acc_hi = r["acc"]
        f1_m  = r["f1_macro"][0]
        kap_m = r["kappa"][0]
        ece_m = r["ece"][0]
        vac_m = r["vacuity"][0]
        av_m  = r["auroc_vacuity"][0]
        ci_w  = (acc_hi - acc_lo) / 2 if not np.isnan(acc_hi) else float("nan")
        print(f"{r['arch']:<22} {r['tag']:<8} {acc_m*100:6.2f}±{ci_w*100:.2f}  "
              f"{f1_m:.4f}  {kap_m:.4f}  {ece_m:.4f}  {vac_m:.4f}  {av_m:.4f}")

    # LaTeX rows
    latex_lines = []
    arch_display = {
        "evlob_gate":   r"EvLOB (gate)",
        "evlob_nogate": r"EvLOB (no gate)",
        "lob_sde_gate": r"LOB-SDE",
        "rac_moe_gate": r"RAC-MoE",
    }
    for key, disp in arch_display.items():
        r = results.get(key)
        if r is None:
            latex_lines.append(f"% {key}: MISSING")
            continue
        acc_m, acc_lo, acc_hi = r["acc"]
        f1_m  = r["f1_macro"][0]
        kap_m = r["kappa"][0]
        ece_m = r["ece"][0]
        vac_m = r["vacuity"][0]
        ci_w  = (acc_hi - acc_lo) / 2 if not np.isnan(acc_hi) else 0
        latex_lines.append(
            f"{disp} & {acc_m*100:.2f}$\\pm${ci_w*100:.2f} & "
            f"{f1_m:.4f} & {kap_m:.4f} & {ece_m:.4f} & {vac_m:.4f} \\\\"
        )

    latex_str = "\n".join(latex_lines)
    print("\n--- LaTeX rows (copy-paste) ---")
    print(latex_str)

    # Save JSON
    def safe_serial(obj):
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, (np.float32, np.float64)):
            return round(float(obj), 6)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    out = rdir / "track_a_full_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=safe_serial)
    print(f"\nSaved full analysis → {out}")

    latex_out = rdir / "track_a_latex_rows.txt"
    with open(latex_out, "w") as f:
        f.write(latex_str + "\n")
    print(f"Saved LaTeX rows → {latex_out}")


if __name__ == "__main__":
    main()
