"""Exp 3b — NeuralCDE (stabilised). adjoint OFF by default, time-augmented input.

NOTE: superseded by ``compute/runpod_ncde.py`` (the self-contained pod trainer used for the
paper results: batch=128, Hermite splines, clamp(±7), MC-Dropout + H¹ outputs). Shares the
same stabilised ``src/models/neural_cde.py`` (plain-logits forward).

Usage:
    python experiments/run_neural_cde.py
    python experiments/run_neural_cde.py --data-root /workspace/data/fi-2010 --output-dir /workspace/dl_paper

Saves:
    checkpoints/neural_cde_k10.pt
    results/neural_cde_results.json
    results/neural_cde_k10_logits.pt
    results/neural_cde_k10_h1_norms.pt
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

try:
    from src.models.neural_cde import NeuralCDE, compute_h1_norms
    TORCHCDE_AVAILABLE = True
except ImportError as e:
    print(f"torchcde not available: {e}")
    TORCHCDE_AVAILABLE = False

from src.data.fi2010_loader import FI2010DataModule
from experiments.utils import get_device, resolve_paths, save_results, set_seed


def train_neural_cde(model, train_loader, val_loader, cfg, device, ckpt_path):
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],        # 1e-4 — crucial for stability
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_count = 0
    best_metrics = {}

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss, preds, labels = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(y)
                preds.append(logits.argmax(-1).cpu())
                labels.append(y.cpu())
        val_loss /= len(val_loader.dataset)
        preds  = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        acc = float((preds == labels).mean())
        f1  = float(f1_score(labels, preds, average="macro", zero_division=0))

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        print(f"[NeuralCDE] Epoch {epoch:3d} | train={train_loss:.4f} "
              f"val={val_loss:.4f} acc={acc:.4f} f1={f1:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
            best_metrics = {"val_loss": val_loss, "val_acc": acc, "val_f1": f1, "epoch": epoch}
        else:
            patience_count += 1
            if patience_count >= cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (best epoch {best_metrics['epoch']})")
                break

    return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",       default=None)
    parser.add_argument("--output-dir",      default=None)
    parser.add_argument("--horizon",         type=int,   default=10)
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--hidden-channels", type=int,   default=128)  # match paper model
    parser.add_argument("--num-workers",     type=int,   default=2)
    parser.add_argument("--adjoint",         action="store_true")  # default: adjoint OFF (stable)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    if not TORCHCDE_AVAILABLE:
        print("Install torchcde: pip install torchcde torchdiffeq"); return

    set_seed(args.seed)
    device = get_device()
    data_root, output_dir = resolve_paths(args.data_root, args.output_dir)
    ckpt_dir    = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    k = args.horizon
    print(f"Device: {device} | horizon: k={k} | adjoint: {args.adjoint}")
    print(f"hidden_channels: {args.hidden_channels} | lr: 1e-4 | batch: {args.batch_size}")

    dm = FI2010DataModule(
        str(data_root), horizon=k,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = NeuralCDE(
        input_channels=40,
        hidden_channels=args.hidden_channels,
        output_channels=3,
        adjoint=args.adjoint,
        dropout=0.1,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    cfg = {
        "epochs":                 args.epochs,
        "learning_rate":          1e-4,       # KEY FIX: was 1e-3, ODE unstable at higher LR
        "weight_decay":           1e-5,
        "early_stopping_patience": 15,        # more patience than standard models
    }

    ckpt_path = ckpt_dir / f"neural_cde_k{k}.pt"
    best_metrics = train_neural_cde(
        model, dm.train_dataloader(), dm.val_dataloader(), cfg, device, ckpt_path,
    )

    # Evaluate on test set
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for x, y in dm.test_dataloader():
            x, y = x.to(device), y.to(device)
            logits = model(x)
            test_loss += criterion(logits, y).item() * len(y)
            all_preds.append(logits.argmax(-1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
    test_loss /= len(dm.test_dataloader().dataset)
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc    = float((preds == labels).mean())
    f1_mac = float(f1_score(labels, preds, average="macro",    zero_division=0))
    f1_wt  = float(f1_score(labels, preds, average="weighted", zero_division=0))
    kappa  = float(cohen_kappa_score(labels, preds))

    print(classification_report(labels, preds, target_names=["down", "stationary", "up"]))
    print(f"[NeuralCDE k={k}] acc={acc:.4f} f1_mac={f1_mac:.4f} kappa={kappa:.4f}")

    # Save val logits for calibration
    val_logits, val_labels_list = [], []
    with torch.no_grad():
        for x, y in dm.val_dataloader():
            x = x.to(device)
            logits = model(x)
            val_logits.append(logits.cpu())
            val_labels_list.append(y)
    torch.save(
        {"logits": torch.cat(val_logits), "labels": torch.cat(val_labels_list)},
        results_dir / f"neural_cde_k{k}_logits.pt",
    )

    # H¹ trajectory norms (uncertainty proxy)
    print("Computing H¹ norms on test set...")
    h1_norms = compute_h1_norms(model, dm.test_dataloader(), device)
    torch.save(torch.from_numpy(h1_norms), results_dir / f"neural_cde_k{k}_h1_norms.pt")
    print(f"H¹ norm: mean={h1_norms.mean():.4f} std={h1_norms.std():.4f}")

    result = {
        "model": "neural_cde", "horizon": k,
        "test_loss": float(test_loss), "test_acc": acc,
        "test_f1_macro": f1_mac, "test_f1_weighted": f1_wt, "test_kappa": kappa,
        "best_val_metrics": best_metrics,
        "hidden_channels": args.hidden_channels,
        "adjoint": args.adjoint,
        "lr": 1e-4,
    }
    save_results(result, results_dir / "neural_cde_results.json")


if __name__ == "__main__":
    main()
