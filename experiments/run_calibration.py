"""Exp 4 — Calibration benchmark: ECE + temperature scaling + MC Dropout for all models.

Requires: run_deeplob.py, run_baselines.py (and optionally run_neural_cde.py) to have run first.

Usage:
    python experiments/run_calibration.py
    python experiments/run_calibration.py --output-dir /kaggle/working

Saves:
    results/calibration_results.json  — ECE before/after temp scaling per model
    results/mc_dropout_results.json   — predictive entropy distributions per model
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.deeplob import DeepLOB
from src.models.tcn import TCN
from src.models.lob_transformer import LOBTransformer
from src.models.cnn_lstm_attention import CNNLSTMAttention
from src.data.fi2010_loader import FI2010DataModule
from src.calibration.temperature_scaling import TemperatureScaling
from src.calibration.mc_dropout import MCDropout
from experiments.utils import get_device, resolve_paths, save_results, set_seed


MODEL_CLASSES = {
    "deeplob": DeepLOB,
    "tcn": TCN,
    "lob_transformer": LOBTransformer,
    "cnn_lstm_attention": CNNLSTMAttention,
}

MODEL_KWARGS = {
    "deeplob": {"input_channels": 40, "cnn_out_channels": 96, "lstm_hidden": 64},
    "tcn": {"input_channels": 40, "num_channels": [64]*6, "kernel_size": 3},
    "lob_transformer": {"input_dim": 40, "d_model": 64, "nhead": 4, "num_encoder_layers": 2, "dim_feedforward": 128},
    "cnn_lstm_attention": {"input_channels": 40, "cnn_filters": [32, 64, 64], "lstm_hidden": 64},
}


def calibrate_model(model_name: str, k: int, device: torch.device,
                    ckpt_dir: Path, results_dir: Path,
                    data_root: Path, mc_samples: int) -> dict:
    print(f"\n--- Calibrating {model_name} k={k} ---")

    # Load saved logits (from training run)
    logits_path = results_dir / f"{model_name}_k{k}_logits.pt"
    if not logits_path.exists():
        print(f"  SKIP: {logits_path} not found (run training first)")
        return {}

    saved = torch.load(logits_path, map_location="cpu")
    val_logits = saved["logits"]
    val_labels = saved["labels"]

    # Temperature scaling
    ts = TemperatureScaling()
    ece_before = ts.compute_ece(val_logits, val_labels, apply_temperature=False)
    T_opt = ts.fit(val_logits, val_labels)
    ece_after = ts.compute_ece(val_logits, val_labels, apply_temperature=True)
    print(f"  T={T_opt:.3f} | ECE before={ece_before:.4f} → after={ece_after:.4f}")

    # MC Dropout: load model checkpoint
    ckpt_path = ckpt_dir / f"{model_name}_k{k}.pt"
    model = MODEL_CLASSES[model_name](**MODEL_KWARGS[model_name])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    # Use a test DataLoader for MC Dropout
    dm = FI2010DataModule(str(data_root), horizon=k, batch_size=64, num_workers=2)
    test_loader = dm.test_dataloader()

    mc = MCDropout(model, n_samples=mc_samples)
    all_entropy, all_epistemic, all_preds, all_labels_test = [], [], [], []

    for i, (x, y) in enumerate(test_loader):
        if i >= 50:  # limit to 50 batches for speed
            break
        x = x.to(device)
        mean_probs, entropy, epistemic = mc.predict(x)
        all_entropy.append(entropy.cpu().numpy())
        all_epistemic.append(epistemic.cpu().numpy())
        all_preds.append(mean_probs.argmax(-1).cpu().numpy())
        all_labels_test.append(y.numpy())

    entropy_arr = np.concatenate(all_entropy)
    epistemic_arr = np.concatenate(all_epistemic)
    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels_test)

    mc_acc = float((preds_arr == labels_arr).mean())
    print(f"  MC Dropout acc={mc_acc:.4f} | mean_entropy={entropy_arr.mean():.4f} "
          f"| mean_epistemic={epistemic_arr.mean():.4f}")

    return {
        "model": model_name,
        "horizon": k,
        "temperature": float(T_opt),
        "ece_before_ts": float(ece_before),
        "ece_after_ts": float(ece_after),
        "ece_reduction_pct": float(100 * (ece_before - ece_after) / (ece_before + 1e-10)),
        "mc_dropout_acc": mc_acc,
        "mc_mean_entropy": float(entropy_arr.mean()),
        "mc_std_entropy": float(entropy_arr.std()),
        "mc_mean_epistemic": float(epistemic_arr.mean()),
        "mc_n_samples": mc_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=list(MODEL_CLASSES.keys()))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    data_root, output_dir = resolve_paths(args.data_root, args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"

    print(f"Device: {device} | MC samples: {args.mc_samples}")

    all_results = {}
    for model_name in args.models:
        r = calibrate_model(
            model_name, args.horizon, device, ckpt_dir, results_dir, data_root, args.mc_samples,
        )
        if r:
            all_results[model_name] = r
            save_results(all_results, results_dir / "calibration_results.json")

    if not all_results:
        print("\nNo results — run training scripts first.")
        return

    print("\n" + "="*70)
    print(f"CALIBRATION SUMMARY (k={args.horizon})")
    print("="*70)
    print(f"{'Model':>22} {'T':>6} {'ECE-raw':>9} {'ECE-ts':>9} {'Δ%':>6} {'MC-acc':>8} {'H[p]':>8}")
    for model_name, r in all_results.items():
        print(f"{model_name:>22} {r['temperature']:>6.3f} {r['ece_before_ts']:>9.4f} "
              f"{r['ece_after_ts']:>9.4f} {r['ece_reduction_pct']:>6.1f} "
              f"{r['mc_dropout_acc']:>8.4f} {r['mc_mean_entropy']:>8.4f}")


if __name__ == "__main__":
    main()
