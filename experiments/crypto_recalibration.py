"""E5 — Cross-market vol-normalized recalibration (MACC-LOB Component 6).

Evaluates FI-2010-trained baselines on three crypto symbols (BTCUSDT/ETHUSDT/SOLUSDT)
and one equity (LOBSTER AAPL) under vol-normalized labels, then measures how quickly
tiny-budget temperature scaling and conformal calibration restores calibration
on each target market.

Protocol:
  For each target dataset (crypto + LOBSTER) and each model:
    1. Zero-shot (no recalibration): ECE/NLL/Brier/acc on all N samples.
    2. Tiny-budget global-T: fit T on first K samples → evaluate on remainder.
       K ∈ {10, 50, 100, 200, 500, 1_000, 2_000, 5_000}
    3. Tiny-budget conformal: fit q̂ on first K samples (LAC) → report coverage/set-size
       on remainder at α=0.10.

Key finding: characterize the minimum K needed for ECE ≤ 0.05 (adequate calibration).

FI-2010 source model context:
  Pre-fitted T values (from E1 regime_temp_scaling_results.json) are loaded for
  context, then re-fitted from scratch on target data at each K.

Saves:
  results/crypto_recalibration_results.json   — full metrics
  results/recalibration_curves.npy            — (n_K, n_models, n_datasets, 3) for plotting
                                               3 = [ECE_T, ECE_zeroshot, conformal_coverage]

Usage:
  python experiments/crypto_recalibration.py
  python experiments/crypto_recalibration.py --alpha 0.10 --max-k 5000
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

K_GRID = [10, 50, 100, 200, 500, 1_000, 2_000, 5_000]


# ── Calibration metrics ────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(labels) * abs(correct[mask].mean() - confs[mask].mean())
    return float(ece)


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    K = probs.shape[1]
    y_oh = np.eye(K)[labels]
    return float(((probs - y_oh) ** 2).sum(axis=1).mean())


def metrics_dict(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    return {
        "acc":   round(float((preds == labels).mean()), 4),
        "ece":   round(compute_ece(probs, labels), 4),
        "nll":   round(compute_nll(probs, labels), 4),
        "brier": round(compute_brier(probs, labels), 4),
    }


# ── Temperature fitting ────────────────────────────────────────────────────────

def fit_temperature(logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Scalar temperature T by NLL minimization (LBFGS)."""
    logits_t = torch.from_numpy(logits_np).float()
    labels_t = torch.from_numpy(labels_np).long()
    T = nn.Parameter(torch.tensor([1.5]))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)

    def step():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        return loss

    opt.step(step)
    return max(0.01, float(T.item()))


def apply_temperature(logits_np: np.ndarray, T: float) -> np.ndarray:
    return F.softmax(torch.from_numpy(logits_np).float() / T, dim=-1).numpy()


# ── Conformal utilities ────────────────────────────────────────────────────────

def get_q_hat(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(scores, min(level, 1.0), method="higher"))


def conformal_stats(
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    eval_probs: np.ndarray,
    eval_labels: np.ndarray,
    alpha: float,
) -> dict:
    """Fit LAC conformal on calib, report coverage + set size on eval."""
    scores = 1.0 - calib_probs[np.arange(len(calib_labels)), calib_labels]
    q_hat  = get_q_hat(scores, alpha)
    sets   = (1.0 - eval_probs) <= q_hat
    cov    = float(sets[np.arange(len(eval_labels)), eval_labels].mean())
    sizes  = sets.sum(axis=1)
    return {
        "q_hat":         round(q_hat, 4),
        "coverage":      round(cov, 4),
        "mean_set_size": round(float(sizes.mean()), 4),
        "singleton_rate": round(float((sizes == 1).mean()), 4),
        "n_calib":       int(len(calib_labels)),
    }


# ── Dataset registry ──────────────────────────────────────────────────────────

def get_target_datasets(results_dir: Path) -> list:
    """Return list of (name, logit_pattern) for all available target datasets."""
    targets = []
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        key = f"crypto_{sym}"
        # Check first model to see if logits exist
        probe = results_dir / f"crypto_{sym}_deeplob_k10_logits.pt"
        if probe.exists():
            targets.append(("crypto", sym, key))
    lobster_probe = results_dir / "lobster_AAPL_deeplob_k10_test_logits.pt"
    if lobster_probe.exists():
        targets.append(("lobster", "AAPL", "lobster_AAPL"))
    return targets


def load_target_logits(results_dir: Path, dset_type: str, sym: str, model_name: str):
    """Load logits + labels for a target dataset and model."""
    if dset_type == "crypto":
        path = results_dir / f"crypto_{sym}_{model_name}_k10_logits.pt"
    else:
        path = results_dir / f"lobster_{sym}_{model_name}_k10_test_logits.pt"
    if not path.exists():
        return None, None
    d      = torch.load(path, map_location="cpu", weights_only=False)
    logits = d["logits"].float().numpy()
    labels = d["labels"].long().numpy()
    return logits, labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--alpha",       type=float, default=0.10)
    parser.add_argument("--max-k",       type=int,   default=5000,
                        help="Maximum K for recalibration (must be << dataset size)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    alpha = args.alpha
    k_grid = [k for k in K_GRID if k <= args.max_k]

    print(f"α={alpha}  K-grid={k_grid}")

    model_names = ["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention"]
    targets     = get_target_datasets(results_dir)
    if not targets:
        print("No target datasets found — run build_crypto_dataset.py / build_lobster_dataset.py first")
        sys.exit(1)

    all_results = {}

    for dset_type, sym, dset_key in targets:
        print(f"\n══ {dset_key} ══")
        dset_res = {}

        for model_name in model_names:
            logits, labels = load_target_logits(results_dir, dset_type, sym, model_name)
            if logits is None:
                print(f"  [SKIP] {model_name}: logits missing")
                continue

            N = len(labels)
            probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

            # Zero-shot (no recalibration)
            zs = metrics_dict(probs, labels)
            print(f"  {model_name} zero-shot: acc={zs['acc']:.4f} ECE={zs['ece']:.4f} NLL={zs['nll']:.4f}")

            model_res = {
                "n_total":   N,
                "zero_shot": zs,
                "by_k":      {},
            }

            for K in k_grid:
                if K >= N - 100:
                    continue

                calib_logits = logits[:K]
                calib_labels = labels[:K]
                calib_probs  = probs[:K]
                eval_logits  = logits[K:]
                eval_labels  = labels[K:]
                eval_probs   = probs[K:]

                # Tiny-budget temperature scaling
                T = fit_temperature(calib_logits, calib_labels)
                scaled_probs = apply_temperature(eval_logits, T)
                ts_metrics = metrics_dict(scaled_probs, eval_labels)

                # Conformal on eval
                conf_stats = conformal_stats(
                    calib_probs, calib_labels,
                    eval_probs,  eval_labels,
                    alpha,
                )

                model_res["by_k"][str(K)] = {
                    "T":          round(T, 4),
                    "temp_scale": ts_metrics,
                    "conformal":  conf_stats,
                    "n_eval":     int(len(eval_labels)),
                }

            # Print ladder for key K values
            print(f"  {'K':>6} {'T':>7} {'ECE_zs':>8} {'ECE_T':>8} {'Δ_ECE':>8} {'cov':>8} {'sz':>7}")
            for K_str, entry in model_res["by_k"].items():
                K_int = int(K_str)
                if K_int in {50, 200, 1000, 5000}:
                    ece_t  = entry["temp_scale"]["ece"]
                    ece_zs = zs["ece"]
                    cov    = entry["conformal"]["coverage"]
                    sz     = entry["conformal"]["mean_set_size"]
                    T_val  = entry["T"]
                    print(f"  {K_int:>6} {T_val:>7.3f} {ece_zs:>8.4f} {ece_t:>8.4f} {ece_t - ece_zs:>+8.4f} {cov:>8.4f} {sz:>7.3f}")

            dset_res[model_name] = model_res

        all_results[dset_key] = {"alpha": alpha, "k_grid": k_grid, "models": dset_res}

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = results_dir / "crypto_recalibration_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path.name}")

    # ── Build plotting array ──────────────────────────────────────────────────
    # Shape: (n_K, n_models, n_datasets) for ECE after T-scaling
    n_K   = len(k_grid)
    n_mod = len(model_names)
    n_dset = len(targets)
    ece_curves = np.full((n_K, n_mod, n_dset), np.nan)
    cov_curves = np.full((n_K, n_mod, n_dset), np.nan)

    for di, (_, _, dset_key) in enumerate(targets):
        if dset_key not in all_results:
            continue
        for mi, model_name in enumerate(model_names):
            if model_name not in all_results[dset_key]["models"]:
                continue
            mres = all_results[dset_key]["models"][model_name]
            for ki, K in enumerate(k_grid):
                if str(K) in mres["by_k"]:
                    ece_curves[ki, mi, di] = mres["by_k"][str(K)]["temp_scale"]["ece"]
                    cov_curves[ki, mi, di] = mres["by_k"][str(K)]["conformal"]["coverage"]

    np.save(str(results_dir / "recalibration_ece_curves.npy"),  ece_curves)
    np.save(str(results_dir / "recalibration_cov_curves.npy"),  cov_curves)
    np.save(str(results_dir / "recalibration_k_grid.npy"),      np.array(k_grid))
    print(f"Saved → recalibration_{{ece,cov}}_curves.npy  (shape={ece_curves.shape})")

    # ── Summary: minimum K for ECE ≤ 0.05 ────────────────────────────────────
    print("\n── Minimum K for ECE ≤ 0.05 after T-scaling ──")
    for di, (_, _, dset_key) in enumerate(targets):
        if dset_key not in all_results:
            continue
        for mi, model_name in enumerate(model_names):
            if model_name not in all_results[dset_key]["models"]:
                continue
            mres = all_results[dset_key]["models"][model_name]
            zs_ece = mres["zero_shot"]["ece"]
            min_K  = "never"
            for K in k_grid:
                if str(K) in mres["by_k"]:
                    if mres["by_k"][str(K)]["temp_scale"]["ece"] <= 0.05:
                        min_K = K
                        break
            print(f"  {dset_key:20s} {model_name:26s}: zs_ECE={zs_ece:.4f}  minK={min_K}")


if __name__ == "__main__":
    main()
