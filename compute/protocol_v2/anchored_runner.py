"""Train one anchored FI-2010 model/fold/seed run.

The anchored protocol uses exactly one cumulative FI-2010 CF file per run:
Train_Dst_NoAuction_ZScore_CF_N.txt for training/validation and
Test_Dst_NoAuction_ZScore_CF_N.txt for testing. Folds are never concatenated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


PHASE = "anchored_references"
HORIZON = 10
HORIZON_IDX = {1: 0, 2: 1, 3: 2, 5: 3, 10: 4}
HORIZON_COL = HORIZON_IDX[HORIZON]
N_FEATURES = 40
N_CLASSES = 3
LABEL_START = 144
LABEL_STOP = 149

DEFAULT_SEQUENCE_LENGTH = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_EVAL_BATCH_SIZE = 512
DEFAULT_EPOCHS = 80
DEFAULT_LR = 3e-4
DEFAULT_MIN_LR = 1e-6
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_DROPOUT = 0.2
DEFAULT_EARLY_STOPPING_PATIENCE = 12
DEFAULT_SCHEDULER_PATIENCE = 5
DEFAULT_VALIDATION_FRACTION = 0.1
V2_WARMUP_EPOCHS = 5

MODEL_CHOICES = ("tcn", "lob_transformer", "lob_transformer_v2", "neural_cde", "rac_moe")
EXPECTED_COUNTS = {
    1: (35461, 3853, 38298),
    9: (326061, 36141, 31838),
}


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding matching src.models.lob_transformer."""

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class LOBTransformerV2(nn.Module):
    """LOBTransformer with larger width and pre-LN encoder layers."""

    def __init__(
        self,
        input_dim: int = 40,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        n_classes: int = 3,
        sequence_length: int = 100,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=sequence_length + 1,
            dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.classifier = nn.Linear(d_model, n_classes)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        projected = self.input_proj(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, projected], dim=1)
        encoded = self.encoder(self.pos_enc(seq))
        return self.classifier(encoded[:, 0])

    def enable_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


class FI2010SlidingWindowDataset(Dataset):
    """Lazy inclusive sliding-window dataset for one contiguous row block."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int, name: str):
        if features.ndim != 2 or features.shape[1] != N_FEATURES:
            raise ValueError(f"{name}: expected features with shape (rows, 40), got {features.shape}")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(f"{name}: labels shape {labels.shape} does not match features {features.shape}")
        if features.shape[0] < sequence_length:
            raise ValueError(
                f"{name}: need at least {sequence_length} rows to form one window, "
                f"got {features.shape[0]}"
            )
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.name = name

    def __len__(self) -> int:
        return self.features.shape[0] - self.sequence_length + 1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        end = index + self.sequence_length
        x = torch.from_numpy(self.features[index:end])
        y = int(self.labels[end - 1])
        return x, y


@dataclass
class FoldData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_samples: int
    validation_samples: int
    test_samples: int
    train_source: str
    test_source: str


@contextmanager
def cuda_autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast("cuda", enabled=enabled):
            yield
    else:
        with torch.cuda.amp.autocast(enabled=enabled):
            yield


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def torch_load(path: Path, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _bench = os.environ.get("CUDNN_BENCHMARK") == "1"
    torch.backends.cudnn.benchmark = _bench
    torch.backends.cudnn.deterministic = not _bench


def find_repo_root() -> Path:
    candidates = [
        Path(__file__).resolve().parent,   # runner's own directory (works when runner is repo root)
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[2],
        Path.cwd(),
        Path("/kaggle/working/code"),       # Kaggle kernel extraction path
        Path("/kaggle/working/uncertainty-lob"),
        Path("/Users/paritoshdwivedi/uncertainty-lob"),
    ]
    for base in list(candidates):
        candidates.extend(base.parents)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "src" / "models").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repo root containing src/models")


def add_repo_to_path() -> Path:
    root = find_repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def is_zscore_root(path: Path) -> bool:
    return (
        (path / "NoAuction_Zscore_Training").is_dir()
        and (path / "NoAuction_Zscore_Testing").is_dir()
    )


def zscore_candidates(base: Path):
    yield base
    yield base / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore"
    yield base / "NoAuction" / "1.NoAuction_Zscore"
    yield base / "1.NoAuction_Zscore"


def resolve_data_root(cli_data_root: Optional[str]) -> Path:
    env_data_root = os.environ.get("DATA_ROOT")
    bases = []
    if cli_data_root:
        bases.append(Path(cli_data_root).expanduser())
    elif env_data_root:
        bases.append(Path(env_data_root).expanduser())
    else:
        bases.extend(
            [
                Path("/kaggle/input/fi-2010/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore"),
                Path("/kaggle/input/fi-2010"),
                Path("/tmp/fi-2010"),
            ]
        )

    checked = []
    for base in bases:
        for candidate in zscore_candidates(base):
            checked.append(candidate)
            if is_zscore_root(candidate):
                return candidate

    for base in bases:
        if not base.exists():
            continue
        for candidate in base.rglob("1.NoAuction_Zscore"):
            checked.append(candidate)
            if is_zscore_root(candidate):
                return candidate

    checked_text = "\n".join(f"  - {p}" for p in checked[:20])
    raise FileNotFoundError(
        "Could not locate FI-2010 NoAuction/Zscore root. Checked:\n"
        f"{checked_text}\nSet --data-root or DATA_ROOT to the FI-2010 directory."
    )


def fold_file_paths(zscore_root: Path, fold: int) -> Tuple[Path, Path]:
    train_dir = zscore_root / "NoAuction_Zscore_Training"
    test_dir = zscore_root / "NoAuction_Zscore_Testing"
    train_path = train_dir / f"Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
    test_path = test_dir / f"Test_Dst_NoAuction_ZScore_CF_{fold}.txt"

    if not train_path.exists():
        matches = sorted(train_dir.glob(f"Train_Dst_NoAuction_ZScore_CF_{fold}.txt"))
        if matches:
            train_path = matches[0]
    if not test_path.exists():
        matches = sorted(test_dir.glob(f"Test_Dst_NoAuction_ZScore_CF_{fold}.txt"))
        if matches:
            test_path = matches[0]

    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing fold file(s):\n" + "\n".join(missing))
    return train_path, test_path


def default_fallback_cache_dir() -> Path:
    if is_kaggle():
        return Path("/kaggle/working/fi2010_npy_cache")
    return Path("/tmp/fi2010_npy_cache")


def cache_candidates(txt_path: Path, fallback_cache_dir: Path):
    yield txt_path.with_suffix(".npy")
    yield fallback_cache_dir / txt_path.with_suffix(".npy").name


def load_raw_cf(txt_path: Path, fallback_cache_dir: Path) -> np.ndarray:
    fallback_cache_dir.mkdir(parents=True, exist_ok=True)
    txt_mtime = txt_path.stat().st_mtime
    for cache_path in cache_candidates(txt_path, fallback_cache_dir):
        if cache_path.exists() and cache_path.stat().st_mtime >= txt_mtime:
            try:
                print(f"Loading cached {cache_path}", flush=True)
                return np.load(cache_path, allow_pickle=False)
            except Exception as exc:
                print(f"Could not read cache {cache_path}: {exc}; reloading text.", flush=True)

    print(f"Loading {txt_path} with np.loadtxt ...", flush=True)
    raw = np.loadtxt(txt_path, dtype=np.float32)
    for cache_path in cache_candidates(txt_path, fallback_cache_dir):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, raw)
            print(f"Cached {txt_path.name} -> {cache_path}", flush=True)
            break
        except OSError as exc:
            print(f"Could not write cache {cache_path}: {exc}", flush=True)
    return raw


def load_cf_features_labels(txt_path: Path, fallback_cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw = load_raw_cf(txt_path, fallback_cache_dir)
    if raw.ndim != 2 or raw.shape[0] < LABEL_STOP:
        raise ValueError(f"{txt_path}: expected raw shape (149, cols), got {raw.shape}")

    features = np.ascontiguousarray(raw[:N_FEATURES].T, dtype=np.float32)
    labels_raw = np.ascontiguousarray(raw[LABEL_START:LABEL_STOP].T, dtype=np.float32)
    labels = labels_raw[:, HORIZON_COL].astype(np.int64) - 1
    if labels.shape[0] != features.shape[0]:
        raise ValueError(f"{txt_path}: labels/features row mismatch")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= N_CLASSES:
        raise ValueError(f"{txt_path}: labels are not in expected 0..2 range after conversion")
    return features, labels


def num_workers_for_environment() -> int:
    override = os.environ.get("NUM_WORKERS")
    if override is not None:
        return max(0, int(override))
    if is_kaggle() or platform.system() == "Darwin":
        return 0
    cpu_count = os.cpu_count() or 1
    return min(4, max(0, cpu_count - 1))


def make_fold_data(
    zscore_root: Path,
    fold: int,
    sequence_length: int,
    batch_size: int,
    eval_batch_size: int,
    device: torch.device,
) -> FoldData:
    train_path, test_path = fold_file_paths(zscore_root, fold)
    fallback_cache_dir = default_fallback_cache_dir()

    train_features, train_labels = load_cf_features_labels(train_path, fallback_cache_dir)
    total_train_rows = train_features.shape[0]
    split_row = int(total_train_rows * (1.0 - DEFAULT_VALIDATION_FRACTION))
    if split_row < sequence_length:
        raise ValueError(f"Fold {fold}: train split has only {split_row} rows")
    if total_train_rows - split_row < sequence_length:
        raise ValueError(
            f"Fold {fold}: validation split has only {total_train_rows - split_row} rows"
        )

    train_dataset = FI2010SlidingWindowDataset(
        train_features[:split_row],
        train_labels[:split_row],
        sequence_length,
        name=f"fold{fold}/train",
    )
    val_dataset = FI2010SlidingWindowDataset(
        train_features[split_row:],
        train_labels[split_row:],
        sequence_length,
        name=f"fold{fold}/validation",
    )

    test_features, test_labels = load_cf_features_labels(test_path, fallback_cache_dir)
    test_dataset = FI2010SlidingWindowDataset(
        test_features,
        test_labels,
        sequence_length,
        name=f"fold{fold}/test",
    )

    counts = (len(train_dataset), len(val_dataset), len(test_dataset))
    if fold in EXPECTED_COUNTS and counts != EXPECTED_COUNTS[fold]:
        raise ValueError(
            f"Fold {fold}: expected counts {EXPECTED_COUNTS[fold]}, got {counts}. "
            "Check the sequence length, split rule, and FI-2010 source files."
        )

    num_workers = num_workers_for_environment()
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(
        f"Fold {fold} rows: train_total={total_train_rows:,} split_row={split_row:,} "
        f"test_total={test_features.shape[0]:,}",
        flush=True,
    )
    print(
        f"Fold {fold} windows: train={counts[0]:,} validation={counts[1]:,} "
        f"test={counts[2]:,} num_workers={num_workers}",
        flush=True,
    )

    return FoldData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_samples=len(train_dataset),
        validation_samples=len(val_dataset),
        test_samples=len(test_dataset),
        train_source=str(train_path),
        test_source=str(test_path),
    )


def build_model(model_name: str, sequence_length: int) -> Tuple[nn.Module, float]:
    add_repo_to_path()
    if model_name == "tcn":
        from src.models.tcn import TCN

        return (
            TCN(
                input_channels=40,
                num_channels=[64] * 6,
                kernel_size=3,
                dropout=0.2,
                n_classes=3,
            ),
            0.2,
        )

    if model_name == "lob_transformer":
        from src.models.lob_transformer import LOBTransformer

        return (
            LOBTransformer(
                input_dim=40,
                d_model=64,
                nhead=4,
                num_encoder_layers=2,
                dim_feedforward=128,
                dropout=0.2,
                n_classes=3,
            ),
            0.2,
        )

    if model_name == "lob_transformer_v2":
        return (
            LOBTransformerV2(
                input_dim=40,
                d_model=128,
                nhead=8,
                num_encoder_layers=2,
                dim_feedforward=512,
                dropout=0.2,
                n_classes=3,
                sequence_length=sequence_length,
            ),
            0.2,
        )

    if model_name == "neural_cde":
        try:
            import torchcde  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "NeuralCDE requires torchcde. Install it with `pip install torchcde torchdiffeq`."
            ) from exc
        from src.models.neural_cde import NeuralCDE

        return (
            NeuralCDE(
                input_channels=40,
                hidden_channels=128,
                output_channels=3,
                adjoint=False,
                dropout=0.1,
            ),
            0.1,
        )

    if model_name == "rac_moe":
        from src.models.rac_moe import RAC_MoE

        return (
            RAC_MoE(
                input_channels=40,
                lstm_hidden=64,
                n_classes=3,
                n_micro=0,
                n_experts=2,
                gate_hidden=32,
                dropout=0.2,
            ),
            0.2,
        )

    raise ValueError(f"Unknown model: {model_name}")


def extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return extract_logits(model(x))


def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, epochs * max(1, steps_per_epoch))
    warmup_steps = max(1, V2_WARMUP_EPOCHS * max(1, steps_per_epoch))
    min_factor = min_lr / base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(min_factor, float(step + 1) / float(warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
    max_grad_norm: float,
    epoch: int,
    epochs: int,
    run_id: str,
    batch_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    log_every = max(1, len(loader) // 5)

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with cuda_autocast(use_amp):
            logits = forward_logits(model, x)
            loss = criterion(logits, y)
        loss_value = float(loss.detach().float().item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {batch_idx}: {loss_value}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if batch_scheduler is not None:
            batch_scheduler.step()

        batch_size = y.size(0)
        total_loss += loss_value * batch_size
        total_correct += (logits.detach().argmax(dim=1) == y).sum().item()
        total_seen += batch_size

        if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == len(loader):
            running_loss = total_loss / max(1, total_seen)
            running_acc = total_correct / max(1, total_seen)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[{run_id}] epoch {epoch:03d}/{epochs} batch {batch_idx:04d}/{len(loader)} "
                f"train_loss={running_loss:.5f} train_acc={running_acc:.4f} lr={lr:.3e}",
                flush=True,
            )

    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    logits_parts = []
    labels_parts = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        with cuda_autocast(use_amp):
            logits = forward_logits(model, x)
            loss = criterion(logits, y)
        batch_size = y.size(0)
        total_loss += float(loss.detach().float().item()) * batch_size
        total_seen += batch_size
        logits_parts.append(logits.detach().float().cpu())
        labels_parts.append(y.detach().long().cpu())

    return total_loss / max(1, total_seen), torch.cat(logits_parts, dim=0), torch.cat(labels_parts, dim=0)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if idx == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    logits_np = logits.float().numpy().astype(np.float64, copy=False)
    labels_np = labels.long().numpy()
    probs = softmax(logits_np, axis=1)
    preds = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)
    one_hot = np.eye(N_CLASSES, dtype=np.float64)[labels_np]
    errors = (preds != labels_np).astype(np.int64)

    if errors.min() == errors.max():
        error_auroc = float("nan")
    else:
        error_auroc = float(roc_auc_score(errors, 1.0 - max_probs))

    return {
        "accuracy": float((preds == labels_np).mean()),
        "brier": float(((probs - one_hot) ** 2).sum(axis=1).mean()),
        "ece": compute_ece(probs, labels_np, n_bins=15),
        "error_auroc": error_auroc,
        "f1_macro": float(f1_score(labels_np, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels_np, preds, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(labels_np, preds)),
        "nll": float(-np.log(np.clip(probs[np.arange(labels_np.shape[0]), labels_np], 1e-12, 1.0)).mean()),
    }


def save_checkpoint(path: Path, model: nn.Module, epoch: int, val_loss: float, run_id: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": float(val_loss),
            "run_id": run_id,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch_load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    model.load_state_dict(checkpoint)
    return {"epoch": None, "val_loss": None}


def train_model(
    model: nn.Module,
    data: FoldData,
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
    device: torch.device,
    use_amp: bool,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, float]]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DEFAULT_LR,
        betas=(0.9, 0.999),
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )
    if args.model == "lob_transformer_v2":
        plateau_scheduler = None
        batch_scheduler = make_cosine_scheduler(
            optimizer,
            epochs=args.epochs,
            steps_per_epoch=len(data.train_loader),
            base_lr=DEFAULT_LR,
            min_lr=DEFAULT_MIN_LR,
        )
    else:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=DEFAULT_SCHEDULER_PATIENCE,
            min_lr=DEFAULT_MIN_LR,
        )
        batch_scheduler = None

    scaler = make_grad_scaler(use_amp)
    best_path = run_dir / "best.pt"
    best_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=data.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            max_grad_norm=DEFAULT_MAX_GRAD_NORM,
            epoch=epoch,
            epochs=args.epochs,
            run_id=run_id,
            batch_scheduler=batch_scheduler,
        )
        val_loss, val_logits, val_labels = collect_logits(
            model=model,
            loader=data.val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )
        val_epoch_metrics = compute_metrics(val_logits, val_labels)
        if plateau_scheduler is not None:
            plateau_scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, epoch, val_loss, run_id)
        else:
            epochs_without_improvement += 1

        print(
            f"[{run_id}] epoch {epoch:03d}/{args.epochs} done "
            f"train_loss={train_loss:.5f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.5f} val_acc={val_epoch_metrics['accuracy']:.4f} "
            f"val_f1_macro={val_epoch_metrics['f1_macro']:.4f} "
            f"best_epoch={best_epoch} bad_epochs={epochs_without_improvement} "
            f"elapsed={time.time() - epoch_start:.1f}s",
            flush=True,
        )

        if epochs_without_improvement >= DEFAULT_EARLY_STOPPING_PATIENCE:
            print(f"[{run_id}] early stopping at epoch {epoch}; best_epoch={best_epoch}", flush=True)
            break

    if best_epoch == 0:
        raise RuntimeError("Training completed without a finite validation checkpoint")

    load_checkpoint(best_path, model, device)
    val_loss, val_logits, val_labels = collect_logits(
        model=model,
        loader=data.val_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )
    test_loss, test_logits, test_labels = collect_logits(
        model=model,
        loader=data.test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )
    validation_metrics = compute_metrics(val_logits, val_labels)
    test_metrics = compute_metrics(test_logits, test_labels)
    print(
        f"[{run_id}] final best_epoch={best_epoch} "
        f"validation_loss={val_loss:.5f} test_loss={test_loss:.5f} "
        f"test_acc={test_metrics['accuracy']:.4f} test_f1_macro={test_metrics['f1_macro']:.4f}",
        flush=True,
    )
    return best_epoch, val_logits, val_labels, test_logits, test_labels, validation_metrics, test_metrics


def runtime_info() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "cuda_available": bool(cuda_available),
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_summary(
    path: Path,
    *,
    best_epoch: int,
    args: argparse.Namespace,
    run_id: str,
    model_dropout: float,
    data: FoldData,
    duration_seconds: float,
    use_amp: bool,
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    summary = {
        "best_epoch": int(best_epoch),
        "completed_at": iso_utc_now(),
        "config": {
            "amp": bool(use_amp),
            "batch_size": DEFAULT_BATCH_SIZE,
            "dropout": float(model_dropout),
            "early_stopping_patience": DEFAULT_EARLY_STOPPING_PATIENCE,
            "epochs": int(args.epochs),
            "eval_batch_size": DEFAULT_EVAL_BATCH_SIZE,
            "fold": int(args.fold),
            "horizon": HORIZON,
            "learning_rate": DEFAULT_LR,
            "max_grad_norm": DEFAULT_MAX_GRAD_NORM,
            "micro_mode": "none",
            "min_learning_rate": DEFAULT_MIN_LR,
            "model": args.model,
            "phase": PHASE,
            "run_id": run_id,
            "scheduler_patience": DEFAULT_SCHEDULER_PATIENCE,
            "seed": int(args.seed),
            "sequence_length": DEFAULT_SEQUENCE_LENGTH,
            "validation_fraction": DEFAULT_VALIDATION_FRACTION,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
        },
        "data": {
            "fold": int(args.fold),
            "micro_mean": None,
            "micro_mode": "none",
            "micro_std": None,
            "test_samples": int(data.test_samples),
            "test_source": data.test_source,
            "train_samples": int(data.train_samples),
            "train_source": data.train_source,
            "validation_samples": int(data.validation_samples),
        },
        "duration_seconds": float(duration_seconds),
        "run_id": run_id,
        "runtime": runtime_info(),
        "state": "completed",
        "test": test_metrics,
        "validation": validation_metrics,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
        f.write("\n")


def already_completed(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        return summary.get("state") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one anchored FI-2010 model/fold/seed run.")
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 10), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to FI-2010 directory or 1.NoAuction_Zscore root. Defaults to DATA_ROOT, Kaggle, then /tmp/fi-2010.",
    )
    parser.add_argument("--out-dir", type=str, default="./runs")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--smoke", action="store_true", help="Run three epochs for a smoke test.")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 3
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    run_id = f"{PHASE}__{args.model}__f{args.fold}__k{HORIZON}__s{args.seed}__none"
    run_dir = Path(args.out_dir).expanduser() / run_id
    summary_path = run_dir / "summary.json"
    if already_completed(summary_path):
        print("SKIP", flush=True)
        return 0

    start = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    repo_root = add_repo_to_path()
    zscore_root = resolve_data_root(args.data_root)
    print(f"Run: {run_id}", flush=True)
    print(f"Repo root: {repo_root}", flush=True)
    print(f"Data root: {zscore_root}", flush=True)
    print(f"Output dir: {run_dir}", flush=True)
    print(f"Device: {device} AMP={use_amp}", flush=True)

    data = make_fold_data(
        zscore_root=zscore_root,
        fold=args.fold,
        sequence_length=DEFAULT_SEQUENCE_LENGTH,
        batch_size=DEFAULT_BATCH_SIZE,
        eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
        device=device,
    )

    model, model_dropout = build_model(args.model, sequence_length=DEFAULT_SEQUENCE_LENGTH)
    n_params = sum(param.numel() for param in model.parameters())
    print(f"Model: {args.model} params={n_params:,}", flush=True)

    (
        best_epoch,
        val_logits,
        val_labels,
        test_logits,
        test_labels,
        validation_metrics,
        test_metrics,
    ) = train_model(
        model=model,
        data=data,
        args=args,
        run_id=run_id,
        run_dir=run_dir,
        device=device,
        use_amp=use_amp,
    )

    torch.save(
        {"logits": val_logits.float(), "labels": val_labels.long()},
        run_dir / "validation_logits.pt",
    )
    torch.save(
        {"logits": test_logits.float(), "labels": test_labels.long()},
        run_dir / "test_logits.pt",
    )
    duration_seconds = time.time() - start
    write_summary(
        summary_path,
        best_epoch=best_epoch,
        args=args,
        run_id=run_id,
        model_dropout=model_dropout,
        data=data,
        duration_seconds=duration_seconds,
        use_amp=use_amp,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )
    print(f"Completed {run_id} in {duration_seconds:.1f}s", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
