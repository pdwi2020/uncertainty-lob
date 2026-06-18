"""Exp 1 — DeepLOB baseline: train + evaluate at all 5 prediction horizons.

Usage (local):
    python experiments/run_deeplob.py

Usage (Kaggle):
    python experiments/run_deeplob.py --data-root /kaggle/input/fi-2010-lob \
        --output-dir /kaggle/working

Saves:
    checkpoints/deeplob_k{k}.pt   — best checkpoint per horizon
    results/deeplob_results.json  — accuracy / F1 / Kappa per horizon
    results/deeplob_k{k}_logits.pt — val logits (for calibration in Exp 4)
"""

import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fi2010_loader import FI2010DataModule
from src.models.deeplob import DeepLOB
from src.train import train, evaluate, collect_logits
from experiments.utils import get_device, resolve_paths, save_results, set_seed


HORIZONS = [1, 2, 3, 5, 10]
PRIMARY_HORIZON = 10


def run_horizon(k: int, data_root: Path, ckpt_dir: Path, results_dir: Path,
                device: torch.device, cfg: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"DeepLOB — k={k}")
    print(f"{'='*60}")

    dm = FI2010DataModule(
        str(data_root),
        horizon=k,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    model = DeepLOB(
        input_channels=40,
        cnn_out_channels=96,
        lstm_hidden=64,
        n_classes=3,
        dropout=0.2,
    )

    best_metrics = train(
        model,
        dm.train_dataloader(),
        dm.val_dataloader(),
        cfg,
        device,
        checkpoint_dir=str(ckpt_dir),
        model_name="deeplob",
        horizon=k,
    )

    # Load best checkpoint and evaluate on test set
    ckpt_path = ckpt_dir / f"deeplob_k{k}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    test_loader = dm.test_dataloader()
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1_macro, test_f1_weighted = evaluate(
        model, test_loader, criterion, device,
    )

    # Detailed per-class report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            all_preds.append(out.argmax(dim=-1).cpu().numpy())
            all_labels.append(y.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    kappa = float(cohen_kappa_score(labels, preds))
    report = classification_report(labels, preds, target_names=["down", "stationary", "up"])
    print(report)

    # Save val logits for calibration experiments
    val_logits, val_labels = collect_logits(model, dm.val_dataloader(), device)
    torch.save(
        {"logits": val_logits, "labels": val_labels},
        results_dir / f"deeplob_k{k}_logits.pt",
    )

    result = {
        "model": "deeplob",
        "horizon": k,
        "test_loss": test_loss,
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_weighted": float(test_f1_weighted),
        "test_kappa": kappa,
        "best_val_metrics": best_metrics,
        "label_counts": dm.label_counts,
    }

    print(f"\n[k={k}] test_acc={test_acc:.4f} f1_macro={test_f1_macro:.4f} kappa={kappa:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    data_root, output_dir = resolve_paths(args.data_root, args.output_dir)

    ckpt_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "epochs": args.epochs,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "early_stopping_patience": 10,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    print(f"Device: {device}")
    print(f"Data: {data_root}")
    print(f"Output: {output_dir}")
    print(f"Horizons: {args.horizons}")

    all_results = {}
    for k in args.horizons:
        result = run_horizon(k, data_root, ckpt_dir, results_dir, device, cfg)
        all_results[f"k{k}"] = result
        # Save incrementally in case of crash
        save_results(all_results, results_dir / "deeplob_results.json")

    print("\n" + "="*60)
    print("SUMMARY — DeepLOB")
    print("="*60)
    print(f"{'k':>4} {'Acc':>8} {'F1-mac':>8} {'F1-wt':>8} {'Kappa':>8}")
    for k in args.horizons:
        r = all_results[f"k{k}"]
        print(f"{k:>4} {r['test_acc']:>8.4f} {r['test_f1_macro']:>8.4f} "
              f"{r['test_f1_weighted']:>8.4f} {r['test_kappa']:>8.4f}")


if __name__ == "__main__":
    main()
