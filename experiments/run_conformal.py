"""Exp 10 — LAC split conformal prediction on saved val logits.

Requires: run_calibration.py (or run_deeplob.py + run_baselines.py) to have run first.

Usage:
    python experiments/run_conformal.py
    python experiments/run_conformal.py --output-dir /kaggle/working --alphas 0.05 0.10 0.20

Saves:
    results/conformal_results.json  — coverage / set-size per model × alpha
"""

import sys
import math
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.utils import resolve_paths, save_results, set_seed


MODELS_K10 = ["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention", "neural_cde"]
HORIZONS   = [1, 2, 3, 5, 10]


def load_logits(path: Path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return saved["logits"].float(), saved["labels"].long()


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    n = len(cal_scores)
    level = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(cal_scores, level, method="higher"))


def evaluate(logits: torch.Tensor, labels: torch.Tensor, alpha: float) -> dict:
    n = len(labels)
    probs      = F.softmax(logits, dim=-1).numpy()
    true_probs = probs[np.arange(n), labels.numpy()]
    cal_scores = 1.0 - true_probs

    q_hat    = conformal_threshold(cal_scores, alpha)
    in_set   = (1.0 - probs) <= q_hat
    set_sizes = in_set.sum(axis=1)
    covered  = in_set[np.arange(n), labels.numpy()]

    return {
        "alpha":          alpha,
        "q_hat":          round(float(q_hat), 6),
        "coverage":       round(float(covered.mean()), 4),
        "avg_set_size":   round(float(set_sizes.mean()), 4),
        "singleton_rate": round(float((set_sizes == 1).mean()), 4),
        "empty_set_rate": round(float((set_sizes == 0).mean()), 4),
        "n_cal":          n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",  default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    _, output_dir = resolve_paths(args.data_root, args.output_dir)
    results_dir = output_dir / "results"

    all_results = {}

    # DeepLOB across all horizons
    for k in HORIZONS:
        path = results_dir / f"deeplob_k{k}_logits.pt"
        if not path.exists():
            print(f"[SKIP] {path.name} — run run_deeplob.py first")
            continue
        logits, labels = load_logits(path)
        key = f"deeplob_k{k}"
        all_results[key] = [evaluate(logits, labels, a) for a in args.alphas]
        print(f"[OK] {key}: n={len(labels)}")

    # All models at k=10
    for model in MODELS_K10:
        if model == "deeplob":
            continue
        path = results_dir / f"{model}_k10_logits.pt"
        if not path.exists():
            print(f"[SKIP] {path.name}")
            continue
        logits, labels = load_logits(path)
        key = f"{model}_k10"
        all_results[key] = [evaluate(logits, labels, a) for a in args.alphas]
        print(f"[OK] {key}: n={len(labels)}")

    if not all_results:
        print("No logit files found — run training experiments first.")
        return

    save_results(all_results, results_dir / "conformal_results.json")

    # Summary
    print(f"\n{'Model':<26} {'α':>5} {'Cov':>6} {'SetSz':>6} {'Single':>7}")
    print("-" * 55)
    for key, rows in all_results.items():
        for r in rows:
            print(f"{key:<26} {r['alpha']:>5.2f} {r['coverage']:>6.4f} "
                  f"{r['avg_set_size']:>6.4f} {r['singleton_rate']:>7.4f}")
        print()


if __name__ == "__main__":
    main()
