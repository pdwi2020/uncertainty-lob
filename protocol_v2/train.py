"""Train one immutable Protocol-V2 run with resume and live metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .fi2010 import (
    FI2010SequenceDataset,
    apply_micro_standardization,
    discover_fold_pairs,
    load_fold,
    make_loader,
)
from .io_utils import append_jsonl, atomic_write_json, runtime_metadata, utc_now
from .metrics import classification_metrics
from .models import build_model, forward_for_evaluation, forward_for_training


def configure_eager_optimizer_runtime() -> None:
    if os.name != "nt":
        return

    import torch._dynamo

    # PyTorch 2.5 can access-violate in the Windows Dynamo decorator even when
    # training eagerly. This project never compiles optimizer methods.
    torch._dynamo.disable = lambda fn=None, recursive=True: (
        fn if fn is not None else lambda wrapped: wrapped
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_pair(data_root: Path, fold: int, variant: str = "ZScore"):
    pairs = {pair.fold: pair for pair in discover_fold_pairs(data_root, variant)}
    if fold not in pairs:
        raise ValueError(f"Fold {fold} unavailable for {variant}; found {sorted(pairs)}")
    return pairs[fold]


def batch_to_device(batch, device: torch.device):
    x, y, micro = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    micro_value = micro.to(device, non_blocking=True) if micro.shape[-1] > 0 else None
    return x, y, micro_value


@torch.no_grad()
def evaluate_loader(
    model,
    model_name: str,
    loader,
    device: torch.device,
    use_rollouts: bool = False,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    model.eval()
    all_logits = []
    all_labels = []
    extras: Dict[str, list[torch.Tensor]] = {}
    for batch in loader:
        x, y, micro = batch_to_device(batch, device)
        logits, batch_extras = forward_for_evaluation(
            model, model_name, x, micro, use_rollouts=use_rollouts
        )
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
        for key, value in batch_extras.items():
            extras.setdefault(key, []).append(value.detach().cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    artifact: Dict[str, torch.Tensor] = {"logits": logits, "labels": labels}
    artifact.update({key: torch.cat(values) for key, values in extras.items()})
    return classification_metrics(logits, labels), artifact


def build_datasets(config: Dict[str, Any], device: torch.device):
    data_root = Path(config["data_root"])
    cache_root = Path(config["cache_root"])
    fold = int(config["fold"])
    micro_mode = str(config.get("micro_mode", "none"))
    pair = select_pair(data_root, fold, "ZScore")
    micro_pair = None
    if micro_mode == "corrected_decimal":
        micro_pair = select_pair(data_root, fold, "DecPre")
    loaded = load_fold(
        pair,
        cache_root=cache_root,
        micro_mode=micro_mode,
        micro_pair=micro_pair,
    )

    sequence_length = int(config.get("sequence_length", 100))
    validation_fraction = float(config.get("validation_fraction", 0.10))
    train_stop = int(len(loaded.train_features) * (1.0 - validation_fraction))
    if train_stop <= sequence_length or len(loaded.train_features) - train_stop <= sequence_length:
        raise ValueError("Validation split is too small for the requested sequence length")
    loaded = apply_micro_standardization(loaded, train_stop)

    sample_limit = config.get("sample_limit")
    train_start = 0
    if sample_limit is not None:
        train_start = max(0, train_stop - int(sample_limit))

    train_ds = FI2010SequenceDataset(
        loaded.train_features[train_start:train_stop],
        loaded.train_labels[train_start:train_stop],
        horizon=int(config["horizon"]),
        sequence_length=sequence_length,
        micro=None if loaded.train_micro is None else loaded.train_micro[train_start:train_stop],
    )
    val_ds = FI2010SequenceDataset(
        loaded.train_features[train_stop:],
        loaded.train_labels[train_stop:],
        horizon=int(config["horizon"]),
        sequence_length=sequence_length,
        micro=None if loaded.train_micro is None else loaded.train_micro[train_stop:],
    )
    test_ds = FI2010SequenceDataset(
        loaded.test_features,
        loaded.test_labels,
        horizon=int(config["horizon"]),
        sequence_length=sequence_length,
        micro=loaded.test_micro,
    )

    workers = int(config.get("num_workers", 0))
    train_loader = make_loader(train_ds, int(config["batch_size"]), True, workers, device)
    val_loader = make_loader(val_ds, int(config.get("eval_batch_size", 512)), False, workers, device)
    test_loader = make_loader(test_ds, int(config.get("eval_batch_size", 512)), False, workers, device)
    metadata = {
        "fold": fold,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "test_samples": len(test_ds),
        "train_source": str(pair.train_path),
        "test_source": str(pair.test_path),
        "micro_mode": micro_mode,
        "micro_mean": None if loaded.micro_mean is None else loaded.micro_mean.tolist(),
        "micro_std": None if loaded.micro_std is None else loaded.micro_std.tolist(),
    }
    return train_loader, val_loader, test_loader, metadata


def train_one(config: Dict[str, Any]) -> int:
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.jsonl"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bool(config.get("require_cuda", True)) and device.type != "cuda":
        raise RuntimeError("CUDA is required for this run but is not available")

    atomic_write_json(status_path, {
        "state": "initializing",
        "run_id": config["run_id"],
        "updated_at": utc_now(),
        "config": config,
        "runtime": runtime_metadata(),
    })

    train_loader, val_loader, test_loader, data_metadata = build_datasets(config, device)
    n_micro = 0 if config.get("micro_mode", "none") == "none" else 7
    model = build_model(config, n_micro).to(device)
    model_name = str(config["model"])
    configure_eager_optimizer_runtime()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=int(config.get("scheduler_patience", 5)),
        min_lr=float(config.get("min_learning_rate", 1e-6)),
    )
    use_amp = device.type == "cuda" and bool(config.get("amp", True))
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = -1
    patience_count = 0
    if last_path.exists() and bool(config.get("resume", True)):
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        patience_count = int(checkpoint.get("patience_count", 0))

    epochs = int(config["epochs"])
    early_patience = int(config.get("early_stopping_patience", 12))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    run_started = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_started = time.time()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        skipped_batches = 0
        for batch in train_loader:
            x, y, micro = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits, loss = forward_for_training(
                        model, model_name, x, y, micro, epoch, epochs, config
                    )
            else:
                logits, loss = forward_for_training(
                    model, model_name, x, y, micro, epoch, epochs, config
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
            if not torch.isfinite(grad_norm):
                if scaler is None:
                    raise RuntimeError(f"Non-finite gradient norm at epoch {epoch + 1}")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                continue
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_correct += int(logits.argmax(dim=1).eq(y).sum().item())
            total_samples += len(y)

        val_metrics, _ = evaluate_loader(model, model_name, val_loader, device)
        train_loss = total_loss / max(total_samples, 1)
        scheduler.step(val_metrics["nll"])
        elapsed = time.time() - epoch_started
        learning_rate = float(optimizer.param_groups[0]["lr"])
        improved = val_metrics["nll"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["nll"]
            best_epoch = epoch
            patience_count = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, best_path)
        else:
            patience_count += 1

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "patience_count": patience_count,
            "config": config,
        }
        torch.save(checkpoint, last_path)
        eta = elapsed * max(epochs - epoch - 1, 0)
        event = {
            "type": "epoch",
            "timestamp": utc_now(),
            "epoch": epoch + 1,
            "epochs": epochs,
            "train_loss": train_loss,
            "train_accuracy": total_correct / max(total_samples, 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_kappa": val_metrics["kappa"],
            "val_nll": val_metrics["nll"],
            "val_brier": val_metrics["brier"],
            "val_ece": val_metrics["ece"],
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "best_epoch": best_epoch + 1,
            "best_val_nll": best_val_loss,
            "skipped_batches": skipped_batches,
        }
        append_jsonl(metrics_path, event)
        atomic_write_json(status_path, {
            "state": "running",
            "run_id": config["run_id"],
            "updated_at": utc_now(),
            "started_at": datetime_from_timestamp(run_started),
            "progress": event,
            "data": data_metadata,
            "config": config,
            "runtime": runtime_metadata(),
        })
        print(
            f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.5f} "
            f"val_nll={val_metrics['nll']:.5f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_ece={val_metrics['ece']:.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        if patience_count >= early_patience:
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    val_metrics, val_artifact = evaluate_loader(model, model_name, val_loader, device)
    test_metrics, test_artifact = evaluate_loader(
        model,
        model_name,
        test_loader,
        device,
        use_rollouts=model_name == "lob_sde",
    )
    torch.save(val_artifact, run_dir / "validation_logits.pt")
    torch.save(test_artifact, run_dir / "test_logits.pt")
    summary = {
        "run_id": config["run_id"],
        "state": "completed",
        "completed_at": utc_now(),
        "duration_seconds": time.time() - run_started,
        "best_epoch": int(best["epoch"]) + 1,
        "validation": val_metrics,
        "test": test_metrics,
        "data": data_metadata,
        "config": config,
        "runtime": runtime_metadata(),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(status_path, summary)
    append_jsonl(metrics_path, {"type": "completed", "timestamp": utc_now(), **summary})
    return 0


def datetime_from_timestamp(value: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        raise SystemExit(train_one(config))
    except Exception as exc:
        run_dir = Path(config.get("run_dir", config_path.parent))
        atomic_write_json(run_dir / "status.json", {
            "state": "failed",
            "run_id": config.get("run_id", "unknown"),
            "updated_at": utc_now(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "config": config,
            "runtime": runtime_metadata(),
        })
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
