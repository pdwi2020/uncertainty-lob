"""Shared loaders and metrics for recomputing the leakage-free fold-9 analyses.

`use_retrained=True` swaps the four fold-9 seed-0 smoke-test runs (3 epochs) for their
80-epoch retrains; everything else reads the original campaign runs on X9.
"""
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax

_X9_RUNS = "/Volumes/Crucial X9/Research Projects/DL_Paper/experiments/new/DL_Paper_v2_complete/runs/"
RUNS = Path(os.environ.get("LF_RUNS", _X9_RUNS + "protocol_v2_long"))
RETRAINED = Path(os.environ.get("LF_RETRAINED_RUNS", _X9_RUNS + "protocol_v2_long_smokefix_20261111"))
SMOKE = {("tcn", 9, 0), ("lob_transformer", 9, 0), ("lob_transformer_v2", 9, 0), ("rac_moe", 9, 0)}
BASE = ["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention"]
ALL6 = ["deeplob", "tcn", "lob_transformer", "lob_transformer_v2", "cnn_lstm_attention", "rac_moe"]


def run_dir(model, fold, seed, use_retrained):
    name = f"anchored_references__{model}__f{fold}__k10__s{seed}__none"
    if use_retrained and (model, fold, seed) in SMOKE:
        return RETRAINED / name
    return RUNS / name


def load_logits(model, seed, split, use_retrained, fold=9):
    d = torch.load(run_dir(model, fold, seed, use_retrained) / f"{split}_logits.pt",
                   map_location="cpu", weights_only=False)
    return d["logits"].double().numpy(), d["labels"].long().numpy()


def ece(probs, labels, n_bins=15):
    """Same binning as anchored_runner.compute_ece (last bin closed)."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & ((conf <= hi) if i == n_bins - 1 else (conf < hi))
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def nll(probs, labels):
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())


def fit_temperature(logits, labels):
    """NLL-optimal scalar temperature, bounded search on log T."""
    def f(log_t):
        return nll(softmax(logits / math.exp(log_t), axis=1), labels)
    r = minimize_scalar(f, bounds=(math.log(0.05), math.log(20.0)), method="bounded",
                        options={"xatol": 1e-6})
    return math.exp(r.x)


def lac_qhat(cal_probs, cal_labels, alpha=0.10):
    scores = 1.0 - cal_probs[np.arange(len(cal_labels)), cal_labels]
    n = len(scores)
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def lac_eval(probs, labels, qhat):
    sets = (1.0 - probs) <= qhat
    size = sets.sum(1)
    return {"coverage": float(sets[np.arange(len(labels)), labels].mean()),
            "mean_size": float(size.mean()), "singleton": float((size == 1).mean())}


def ci95(values):
    v = np.asarray(values, dtype=float)
    return 1.96 * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
