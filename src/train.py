"""Generic training loop for all LOB models."""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score
from typing import Dict, Optional, Tuple
from pathlib import Path


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Evaluate model. Returns (loss, accuracy, f1_macro, f1_weighted)."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            total_loss += criterion(logits, y).item() * len(y)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    loss = total_loss / len(labels)
    acc = (preds == labels).mean()
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    return loss, acc, f1_macro, f1_weighted


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    checkpoint_dir: str,
    model_name: str = "model",
    horizon: int = 10,
) -> Dict:
    """Train a model with early stopping. Returns dict of best validation metrics.

    config keys used:
        epochs, learning_rate, weight_decay, early_stopping_patience
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    ckpt_path = Path(checkpoint_dir) / f"{model_name}_k{horizon}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = config.get("early_stopping_patience", 10)
    patience_count = 0
    best_metrics: Dict = {}
    epochs = config.get("epochs", 100)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)

        val_loss, val_acc, val_f1_macro, val_f1_weighted = evaluate(
            model, val_loader, criterion, device,
        )
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(
            f"[{model_name}] Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1={val_f1_macro:.4f} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1_macro,
                "val_f1_weighted": val_f1_weighted,
            }
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[{model_name}] Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"[{model_name}] Best: {best_metrics}")
    return best_metrics


def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect all logits and labels from a loader. Used for calibration."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            all_logits.append(logits.cpu())
            all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)
