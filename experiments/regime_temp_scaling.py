"""E1 — Regime-adaptive temperature scaling (MACC-LOB Component 1).

Ladder: raw → global-T → regime-T → neural-T (optional).
Reports ECE, NLL, Brier on calibration split AND per-regime.

Data flow:
  - Val partition: split 50/50 → calib (fit T) + eval (report metrics)
  - When test logits present (post C-stat): val = calib, test = eval
  - Regime labels from macc_features.npz (r_val, r_test)

Saves: results/regime_temp_scaling_results.json

Usage:
  python experiments/regime_temp_scaling.py
  python experiments/regime_temp_scaling.py --use-test  # requires test logits
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"


# ── Calibration metrics ────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
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
    acc = float((preds == labels).mean())
    return {
        "acc":   round(acc, 4),
        "ece":   round(compute_ece(probs, labels), 4),
        "nll":   round(compute_nll(probs, labels), 4),
        "brier": round(compute_brier(probs, labels), 4),
    }


# ── Temperature fitting ────────────────────────────────────────────────────────

def fit_temperature(logits_t: torch.Tensor, labels_t: torch.Tensor) -> float:
    """Fit a single scalar temperature T by minimizing NLL (LBFGS)."""
    T = nn.Parameter(torch.tensor([1.5]))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)
    def step():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        return loss
    opt.step(step)
    return max(0.01, float(T.item()))


def scale_logits(logits: np.ndarray, T: float) -> np.ndarray:
    probs = torch.softmax(torch.from_numpy(logits) / T, dim=-1).numpy()
    return probs


# ── Neural temperature (tiny MLP on regime features) ──────────────────────────

class NeuralTemperature(nn.Module):
    """T(x) = softplus(MLP(m(x))) > 0."""
    def __init__(self, n_micro: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_micro, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Softplus(),
        )

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.net(m).squeeze(1) + 0.01   # T > 0.01

    def fit(self, logits_t, labels_t, m_t, epochs=50, lr=1e-3):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            T  = self.forward(m_t)
            scaled = logits_t / T.unsqueeze(1)
            loss = F.cross_entropy(scaled, labels_t)
            loss.backward()
            opt.step()

    def predict_probs(self, logits: np.ndarray, m: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits_t = torch.from_numpy(logits)
            m_t      = torch.from_numpy(m)
            T        = self.forward(m_t).unsqueeze(1)
            return torch.softmax(logits_t / T, dim=-1).numpy()


# ── Per-regime metrics ────────────────────────────────────────────────────────

def regime_metrics(probs, labels, regimes):
    out = {}
    for r in sorted(np.unique(regimes)):
        mask = regimes == r
        name = "high_vol" if r == 1 else "low_vol"
        if mask.sum() < 50:
            continue
        out[name] = {**metrics_dict(probs[mask], labels[mask]), "n": int(mask.sum())}
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",  default=str(RESULTS))
    parser.add_argument("--use-test",     action="store_true",
                        help="Use test logits as eval (requires C-stat #36 completion)")
    parser.add_argument("--neural-t",     action="store_true",
                        help="Also run neural-T variant (slow on CPU)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # ── Load features ─────────────────────────────────────────────────────────
    macc = np.load(results_dir / "macc_features.npz")
    r_val    = macc["r_val"].astype(np.int64)
    m_val    = macc["m_val"].astype(np.float32)
    psi_val  = macc["psi_val"].astype(np.float32)

    model_names = ["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention"]
    all_results = {}

    for model_name in model_names:
        # Load logits
        if args.use_test:
            ckpt_val  = results_dir / f"{model_name}_k10_logits.pt"
            ckpt_test = results_dir / f"{model_name}_k10_test_logits.pt"
            if not ckpt_test.exists():
                print(f"[SKIP] {model_name}: test logits not found (C-stat pending)")
                continue
            val_data   = torch.load(ckpt_val,  map_location="cpu", weights_only=False)
            test_data  = torch.load(ckpt_test, map_location="cpu", weights_only=False)
            calib_logits = val_data["logits"].float().numpy()
            calib_labels = val_data["labels"].long().numpy()
            eval_logits  = test_data["logits"].float().numpy()
            eval_labels  = test_data["labels"].long().numpy()
            calib_regimes = r_val
            eval_regimes  = macc["r_test"].astype(np.int64) if "r_test" in macc else None
            calib_m = m_val
            eval_m  = macc["m_test"].astype(np.float32) if "m_test" in macc else None
            split_label = "val→test"
        else:
            ckpt = results_dir / f"{model_name}_k10_logits.pt"
            if not ckpt.exists():
                print(f"[SKIP] {model_name}: {ckpt.name} not found")
                continue
            data   = torch.load(ckpt, map_location="cpu", weights_only=False)
            logits = data["logits"].float().numpy()
            labels = data["labels"].long().numpy()
            # 50/50 val split for within-val evaluation
            n_calib = len(logits) // 2
            calib_logits, eval_logits = logits[:n_calib], logits[n_calib:]
            calib_labels, eval_labels = labels[:n_calib], labels[n_calib:]
            calib_regimes = r_val[:n_calib]
            eval_regimes  = r_val[n_calib:]
            calib_m  = m_val[:n_calib]
            eval_m   = m_val[n_calib:]
            split_label = "val_calib→val_eval"

        print(f"\n── {model_name} ({split_label}) ──")
        print(f"  Calib: {len(calib_labels):,}  Eval: {len(eval_labels):,}")
        if eval_regimes is not None:
            dist = np.bincount(eval_regimes, minlength=2)
            print(f"  Eval regime dist: low={dist[0]:,} high={dist[1]:,}")

        # ── Ladder ────────────────────────────────────────────────────────────
        ladder = {}

        # 0. Raw
        raw_probs = torch.softmax(torch.from_numpy(eval_logits), dim=-1).numpy()
        ladder["raw"] = {
            "T": None, "metrics": metrics_dict(raw_probs, eval_labels),
            "regime_metrics": regime_metrics(raw_probs, eval_labels, eval_regimes) if eval_regimes is not None else {},
        }

        # 1. Global-T
        T_global = fit_temperature(
            torch.from_numpy(calib_logits), torch.from_numpy(calib_labels)
        )
        scaled_probs = scale_logits(eval_logits, T_global)
        ladder["global_T"] = {
            "T": round(T_global, 4),
            "metrics": metrics_dict(scaled_probs, eval_labels),
            "regime_metrics": regime_metrics(scaled_probs, eval_labels, eval_regimes) if eval_regimes is not None else {},
        }

        # 2. Regime-T: fit separate T for each regime on calib, apply on eval
        T_per_regime = {}
        for r in [0, 1]:
            mask = calib_regimes == r
            if mask.sum() < 100:
                T_per_regime[r] = T_global  # fallback to global
                continue
            T_r = fit_temperature(
                torch.from_numpy(calib_logits[mask]),
                torch.from_numpy(calib_labels[mask]),
            )
            T_per_regime[r] = T_r
        print(f"  T_low={T_per_regime[0]:.4f}  T_high={T_per_regime[1]:.4f}")

        regime_T_probs = np.zeros_like(raw_probs)
        if eval_regimes is not None:
            for r in [0, 1]:
                mask = eval_regimes == r
                if mask.sum() > 0:
                    regime_T_probs[mask] = scale_logits(eval_logits[mask], T_per_regime[r])
        else:
            regime_T_probs = scale_logits(eval_logits, T_global)  # fallback

        ladder["regime_T"] = {
            "T_low":  round(T_per_regime[0], 4),
            "T_high": round(T_per_regime[1], 4),
            "metrics": metrics_dict(regime_T_probs, eval_labels),
            "regime_metrics": regime_metrics(regime_T_probs, eval_labels, eval_regimes) if eval_regimes is not None else {},
        }

        # 3. Neural-T (optional, slow on CPU)
        if args.neural_t:
            neural_t = NeuralTemperature(n_micro=7)
            neural_t.fit(
                torch.from_numpy(calib_logits), torch.from_numpy(calib_labels),
                torch.from_numpy(calib_m), epochs=100,
            )
            neural_probs = neural_t.predict_probs(eval_logits, eval_m)
            ladder["neural_T"] = {
                "metrics": metrics_dict(neural_probs, eval_labels),
                "regime_metrics": regime_metrics(neural_probs, eval_labels, eval_regimes) if eval_regimes is not None else {},
            }

        all_results[model_name] = {"split": split_label, "ladder": ladder}

        # Print ladder summary
        print(f"  {'Method':<12} {'ECE':>8} {'NLL':>8} {'Brier':>8} {'Acc':>8}")
        print(f"  {'-'*48}")
        for method, entry in ladder.items():
            m = entry["metrics"]
            T_str = f" T={entry['T']:.3f}" if entry.get("T") else ""
            print(f"  {method:<12} {m['ece']:>8.4f} {m['nll']:>8.4f} {m['brier']:>8.4f} {m['acc']:>8.4f}{T_str}")

        # Per-regime ECE
        if ladder["regime_T"]["regime_metrics"]:
            print("  Per-regime (regime_T):")
            for rname, rm in ladder["regime_T"]["regime_metrics"].items():
                print(f"    {rname:10s}: ECE={rm['ece']:.4f} NLL={rm['nll']:.4f} n={rm['n']:,}")

    # Save
    out_path = results_dir / "regime_temp_scaling_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path.name}")


if __name__ == "__main__":
    main()
