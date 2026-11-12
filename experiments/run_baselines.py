"""Exp 2 — TCN, LOBTransformer, CNNLSTMAttention baselines at k=10.

Usage (local):
    python experiments/run_baselines.py

Usage (Kaggle):
    python experiments/run_baselines.py --data-root /kaggle/input/fi-2010-lob \
        --output-dir /kaggle/working --horizons 10

Saves:
    checkpoints/{model}_k{k}.pt
    results/baselines_results.json
    results/{model}_k{k}_logits.pt  — for calibration
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fi2010_loader import FI2010DataModule
from src.models.tcn import TCN
from src.models.lob_transformer import LOBTransformer
from src.models.cnn_lstm_attention import CNNLSTMAttention
from src.train import train, evaluate, collect_logits
from experiments.utils import get_device, resolve_paths, save_results, set_seed


MODEL_REGISTRY = {
    "tcn": lambda: TCN(input_channels=40, num_channels=[64]*6, kernel_size=3, dropout=0.2),
    "lob_transformer": lambda: LOBTransformer(
        input_dim=40, d_model=64, nhead=4, num_encoder_layers=2,
        dim_feedforward=128, dropout=0.1,
    ),
    "cnn_lstm_attention": lambda: CNNLSTMAttention(
        input_channels=40, cnn_filters=[32, 64, 64], cnn_kernel=3,
        lstm_hidden=64, lstm_layers=2, dropout=0.2, use_attention=True,
    ),
}


def run_model(model_name: str, k: int, data_root: Path, ckpt_dir: Path,
              results_dir: Path, device: torch.device, cfg: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"{model_name} — k={k}")
    print(f"{'='*60}")

    dm = FI2010DataModule(
        str(data_root), horizon=k,
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
    )

    model = MODEL_REGISTRY[model_name]()

    best_metrics = train(
        model, dm.train_dataloader(), dm.val_dataloader(), cfg, device,
        checkpoint_dir=str(ckpt_dir), model_name=model_name, horizon=k,
    )

    # Load best and evaluate
    ckpt_path = ckpt_dir / f"{model_name}_k{k}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1_macro, test_f1_weighted = evaluate(
        model, dm.test_dataloader(), criterion, device,
    )

    # Per-class breakdown
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dm.test_dataloader():
            x = x.to(device)
            out = model(x)
            all_preds.append(out.argmax(dim=-1).cpu().numpy())
            all_labels.append(y.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    kappa = float(cohen_kappa_score(labels, preds))
    print(classification_report(labels, preds, target_names=["up", "stationary", "down"]))

    # Save logits for calibration
    val_logits, val_labels = collect_logits(model, dm.val_dataloader(), device)
    torch.save(
        {"logits": val_logits, "labels": val_labels},
        results_dir / f"{model_name}_k{k}_logits.pt",
    )

    result = {
        "model": model_name,
        "horizon": k,
        "test_loss": test_loss,
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_weighted": float(test_f1_weighted),
        "test_kappa": kappa,
        "best_val_metrics": best_metrics,
    }
    print(f"[{model_name} k={k}] acc={test_acc:.4f} f1={test_f1_macro:.4f} kappa={kappa:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--horizons", nargs="+", type=int, default=[10])
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

    print(f"Device: {device} | Data: {data_root}")
    print(f"Models: {args.models} | Horizons: {args.horizons}")

    all_results = {}
    for model_name in args.models:
        for k in args.horizons:
            key = f"{model_name}_k{k}"
            result = run_model(model_name, k, data_root, ckpt_dir, results_dir, device, cfg)
            all_results[key] = result
            save_results(all_results, results_dir / "baselines_results.json")

    print("\n" + "="*60)
    print("SUMMARY — Baselines (k=10)")
    print("="*60)
    print(f"{'Model':>22} {'k':>4} {'Acc':>8} {'F1-mac':>8} {'F1-wt':>8} {'Kappa':>8}")
    for key, r in all_results.items():
        print(f"{r['model']:>22} {r['horizon']:>4} {r['test_acc']:>8.4f} "
              f"{r['test_f1_macro']:>8.4f} {r['test_f1_weighted']:>8.4f} {r['test_kappa']:>8.4f}")


if __name__ == "__main__":
    main()
