#!/usr/bin/env python3
"""LAC split conformal prediction on saved FI-2010 LOB logits.

Nonconformity score: s(x,y) = 1 - softmax(logits)[true_class]
Threshold: q_hat = quantile((1-alpha)(1 + 1/n), calibration scores)
Prediction set: C(x) = {y : 1 - softmax(logits)[y] <= q_hat}

Coverage guarantee: P(Y in C(X)) >= 1 - alpha  (marginal, finite-sample).
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RESULTS_DIR = Path(__file__).parent / "results"
ALPHAS = [0.05, 0.10, 0.15, 0.20]
HORIZONS = [1, 2, 3, 5, 10]
MODELS_K10 = ["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention"]
CLASS_NAMES = ["down", "stationary", "up"]


def load_logits(path: Path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    logits = saved["logits"].float()   # (N, C)
    labels = saved["labels"].long()    # (N,)
    return logits, labels


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """Compute the (1-alpha)(1+1/n) quantile of calibration nonconformity scores."""
    n = len(cal_scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(cal_scores, level, method="higher"))


def evaluate_conformal(logits: torch.Tensor, labels: torch.Tensor, alpha: float) -> dict:
    """Fit and evaluate LAC conformal predictor on the same set (split-conformal approximation)."""
    n = len(labels)
    probs = F.softmax(logits, dim=-1).numpy()      # (N, C)
    true_probs = probs[np.arange(n), labels.numpy()]
    cal_scores = 1.0 - true_probs                  # nonconformity scores

    q_hat = conformal_threshold(cal_scores, alpha)

    # Prediction sets: include class y if 1 - p(y) <= q_hat
    in_set = (1.0 - probs) <= q_hat               # (N, C) bool
    set_sizes = in_set.sum(axis=1)                 # (N,)

    # Coverage: true label is in prediction set
    covered = in_set[np.arange(n), labels.numpy()]

    return {
        "alpha": alpha,
        "q_hat": round(float(q_hat), 6),
        "coverage": round(float(covered.mean()), 4),
        "avg_set_size": round(float(set_sizes.mean()), 4),
        "singleton_rate": round(float((set_sizes == 1).mean()), 4),
        "empty_set_rate": round(float((set_sizes == 0).mean()), 4),
        "n_cal": n,
    }


def run_model(name: str, logits: torch.Tensor, labels: torch.Tensor) -> list:
    results = []
    for alpha in ALPHAS:
        r = evaluate_conformal(logits, labels, alpha)
        r["model"] = name
        results.append(r)
    return results


def print_table(all_results: dict):
    header = f"{'Model':<26} {'α':>5} {'q̂':>7} {'Cov':>6} {'SetSz':>6} {'Single':>7} {'Empty':>6}"
    print("\n" + "=" * len(header))
    print("LAC Conformal Prediction — FI-2010 LOB (k=10 unless noted)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, rows in all_results.items():
        for r in rows:
            print(f"{name:<26} {r['alpha']:>5.2f} {r['q_hat']:>7.4f} "
                  f"{r['coverage']:>6.4f} {r['avg_set_size']:>6.4f} "
                  f"{r['singleton_rate']:>7.4f} {r['empty_set_rate']:>6.4f}")
        print("-" * len(header))


def main():
    all_results = {}

    # --- DeepLOB across all horizons ---
    for k in HORIZONS:
        path = RESULTS_DIR / f"deeplob_k{k}_logits.pt"
        if not path.exists():
            print(f"[SKIP] {path.name} not found")
            continue
        logits, labels = load_logits(path)
        name = f"deeplob_k{k}"
        all_results[name] = run_model(name, logits, labels)
        print(f"[OK] {name}: {len(labels)} samples")

    # --- All models at k=10 ---
    for model in MODELS_K10:
        if model == "deeplob":
            continue  # already done above
        path = RESULTS_DIR / f"{model}_k10_logits.pt"
        if not path.exists():
            print(f"[SKIP] {path.name} not found")
            continue
        logits, labels = load_logits(path)
        name = f"{model}_k10"
        all_results[name] = run_model(name, logits, labels)
        print(f"[OK] {name}: {len(labels)} samples")

    if not all_results:
        print("No logit files found. Check RESULTS_DIR path.")
        return

    print_table(all_results)

    out_path = RESULTS_DIR / "conformal_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
