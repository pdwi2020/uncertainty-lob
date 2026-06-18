"""Bootstrap 95% confidence intervals + McNemar pairwise significance tests.

Addresses the top-tier-reviewer gap: all headline metrics were single point estimates.
This computes distribution-free CIs by resampling the held-out logits, and pairwise
McNemar tests for whether accuracy differences between models are significant.

NO retraining — operates on the saved per-model logits already in results/.

Usage:
    python experiments/run_bootstrap_ci.py                 # val split (saved *_k10_logits.pt)
    python experiments/run_bootstrap_ci.py --split test    # test split (*_k10_test_logits.pt)
    python experiments/run_bootstrap_ci.py --n-boot 2000 --seed 0

Outputs:
    results/bootstrap_ci_results.json   — per-model {acc,f1_macro,kappa,ece_pre,ece_post} mean+95% CI
    results/significance_tests.json     — pairwise McNemar chi2 + p-values (accuracy)

Caveat (documented in the paper): bootstrap over a fixed evaluation set captures
test-sampling variance, NOT training-initialisation variance. Multi-seed retraining
would capture the latter; deferred unless a reviewer requires it.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, cohen_kappa_score

MODELS = ["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention", "neural_cde"]


def softmax_np(logits, T=1.0):
    z = logits / float(T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error, 15 equal-width confidence bins (matches the paper)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += abs(correct[m].mean() - conf[m].mean()) * (m.sum() / n)
    return float(ece)


def metrics_from_idx(logits, labels, idx, T):
    """All scalar metrics on a (bootstrap) index subset."""
    lg, lb = logits[idx], labels[idx]
    pred = lg.argmax(axis=1)
    acc = float((pred == lb).mean())
    f1m = float(f1_score(lb, pred, average="macro", zero_division=0))
    kap = float(cohen_kappa_score(lb, pred))
    ece_pre = compute_ece(softmax_np(lg, 1.0), lb)
    ece_post = compute_ece(softmax_np(lg, T), lb) if T and T != 1.0 else ece_pre
    return acc, f1m, kap, ece_pre, ece_post


def ci(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()),
            "ci_low": float(np.percentile(a, 2.5)),
            "ci_high": float(np.percentile(a, 97.5))}


def mcnemar(correct_a, correct_b):
    """McNemar test (with continuity correction). chi2 df=1 → p = erfc(sqrt(chi2/2))."""
    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2.0))      # survival fn of chi2_1
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": float(p)}


def load_logits(results_dir, model, split, k):
    suffix = f"_k{k}_logits.pt" if split == "val" else f"_k{k}_test_logits.pt"
    path = results_dir / f"{model}{suffix}"
    if not path.exists():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return saved["logits"].float().numpy(), saved["labels"].long().numpy()


def load_temperatures(results_dir):
    """Per-model optimal temperature from calibration_results.json, if available."""
    p = results_dir / "calibration_results.json"
    if not p.exists():
        return {}
    data = json.load(open(p))
    temps = {}
    for key, v in (data.items() if isinstance(data, dict) else []):
        if isinstance(v, dict):
            name = v.get("model", key).replace(f"_k10", "").replace("_k1", "")
            T = v.get("temperature") or v.get("T_opt") or v.get("T")
            if T:
                temps[v.get("model", name)] = float(T)
    return temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else \
        Path(__file__).resolve().parent.parent / "results"
    rng = np.random.default_rng(args.seed)
    temps = load_temperatures(results_dir)

    loaded, ci_out = {}, {}
    for m in args.models:
        d = load_logits(results_dir, m, args.split, args.horizon)
        if d is None:
            print(f"  SKIP {m}: no {args.split} logits found")
            continue
        logits, labels = d
        loaded[m] = (logits, labels)
        T = temps.get(m, 1.0)
        n = len(labels)
        accs, f1s, kaps, epre, epost = [], [], [], [], []
        # point estimate (full set)
        pa, pf, pk, pep, peo = metrics_from_idx(logits, labels, np.arange(n), T)
        for _ in range(args.n_boot):
            idx = rng.integers(0, n, n)
            a, f, k, ep, eo = metrics_from_idx(logits, labels, idx, T)
            accs.append(a); f1s.append(f); kaps.append(k); epre.append(ep); epost.append(eo)
        ci_out[m] = {
            "n": n, "temperature": T,
            "accuracy": {"point": pa, **ci(accs)},
            "f1_macro": {"point": pf, **ci(f1s)},
            "kappa": {"point": pk, **ci(kaps)},
            "ece_pre": {"point": pep, **ci(epre)},
            "ece_post": {"point": peo, **ci(epost)},
        }
        print(f"  {m:20s} acc={pa:.4f} [{ci_out[m]['accuracy']['ci_low']:.4f},"
              f"{ci_out[m]['accuracy']['ci_high']:.4f}]  kappa={pk:.4f}  ece_post={peo:.4f}")

    # McNemar pairwise (accuracy) — requires identical sample ordering across models
    sig = {}
    names = list(loaded.keys())
    ref_labels = loaded[names[0]][1] if names else None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            la, lb = loaded[a][1], loaded[b][1]
            if len(la) != len(lb) or not np.array_equal(la, lb):
                sig[f"{a}__vs__{b}"] = {"error": "label/length mismatch — different split or ordering"}
                continue
            ca = (loaded[a][0].argmax(1) == la)
            cb = (loaded[b][0].argmax(1) == lb)
            sig[f"{a}__vs__{b}"] = mcnemar(ca, cb)

    (results_dir / "bootstrap_ci_results.json").write_text(
        json.dumps({"split": args.split, "n_boot": args.n_boot, "seed": args.seed,
                    "models": ci_out}, indent=2))
    (results_dir / "significance_tests.json").write_text(
        json.dumps({"split": args.split, "test": "mcnemar_accuracy", "pairs": sig}, indent=2))
    print(f"\nWrote bootstrap_ci_results.json + significance_tests.json ({args.split} split)")


if __name__ == "__main__":
    main()
