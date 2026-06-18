#!/usr/bin/env python3
"""Unified LOB-SDE no-gate metrics across seeds 0/1/2 from test logits.
Validates the metric definitions by reproducing the recorded seed0/seed1 values,
then reports the clean 3-seed mean +/- std for tab:lobsde.
Run from a LOCAL dir (no X9 cwd) to avoid the torch getcwd EPERM.
"""
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score

SEEDS = {
    0: "results/lob_sde_nogate_k10_seed0_test_logits.pt",
    1: "results/lob_sde_nogate_k10_seed1_test_logits.pt",
    2: "results/lob_sde_nogate_k10_seed2_test_logits.pt",
}
# recorded values (from prior session inflight notes) for validation
RECORDED = {
    0: dict(acc=0.7388, ece=0.097, err=0.747, h1=0.364),
    1: dict(acc=0.7012, ece=0.053, err=0.734, h1=0.350),
}


def ece_score(probs, labels, n_bins=15):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1); ece = 0.0; N = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return ece


def metrics_for(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    logits = d["logits"].numpy().astype(np.float64)
    labels = d["labels"].numpy().astype(np.int64)
    h1 = d["h1_energy"].numpy().astype(np.float64)
    ale = d["aleatoric"].numpy().astype(np.float64)
    epi = d["epistemic"].numpy().astype(np.float64)
    probs = np.exp(logits); probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    err = (pred != labels).astype(int)
    return dict(
        acc=float((pred == labels).mean()),
        f1=float(f1_score(labels, pred, average="macro")),
        kappa=float(cohen_kappa_score(labels, pred)),
        ece=float(ece_score(probs, labels)),
        err_auroc=float(roc_auc_score(err, ale + epi)),
        h1_auroc=float(roc_auc_score(err, h1)),
    )


def main():
    res = {s: metrics_for(p) for s, p in SEEDS.items()}
    print(f"{'seed':>4} {'acc':>8} {'f1':>7} {'kappa':>7} {'ece':>7} {'errAUC':>7} {'h1AUC':>7}")
    for s, m in res.items():
        print(f"{s:>4} {m['acc']:8.4f} {m['f1']:7.4f} {m['kappa']:7.4f} "
              f"{m['ece']:7.4f} {m['err_auroc']:7.4f} {m['h1_auroc']:7.4f}")
    print("\n=== validation vs recorded ===")
    ok = True
    for s, rec in RECORDED.items():
        m = res[s]
        for key, rk in [("acc", "acc"), ("ece", "ece"), ("err_auroc", "err"), ("h1_auroc", "h1")]:
            diff = abs(m[key] - rec[rk])
            tol = 0.002 if key != "ece" else 0.003
            flag = "OK" if diff <= tol else "MISMATCH"
            if diff > tol:
                ok = False
            print(f"seed{s} {key:10s} computed={m[key]:.4f} recorded={rec[rk]:.4f} d={diff:.4f} {flag}")
    print(f"\nVALIDATION {'PASSED' if ok else 'FAILED'}")

    print("\n=== clean 3-seed (mean +/- sample std) ===")
    keys = ["acc", "f1", "kappa", "ece", "err_auroc", "h1_auroc"]
    summ = {}
    for k in keys:
        a = np.array([res[s][k] for s in (0, 1, 2)])
        summ[k] = dict(mean=float(a.mean()), std=float(a.std(ddof=1)),
                       seeds=[round(float(x), 4) for x in a])
        print(f"{k:10s} {a.mean():.4f} +/- {a.std(ddof=1):.4f}   {summ[k]['seeds']}")
    with open("lobsde_3seed_final.json", "w") as f:
        json.dump({"per_seed": res, "summary": summ}, f, indent=2)
    print("\nwrote lobsde_3seed_final.json")


if __name__ == "__main__":
    main()
