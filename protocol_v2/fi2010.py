"""Fold-aware FI-2010 discovery, loading, and lazy sequence datasets."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


N_FEATURES = 40
LABEL_START = 144
HORIZON_INDEX = {1: 0, 2: 1, 3: 2, 5: 3, 10: 4}
FOLD_RE = re.compile(r"_CF_(\d+)\.(?:txt|npy)$", re.IGNORECASE)


@dataclass(frozen=True)
class FoldPair:
    fold: int
    train_path: Path
    test_path: Path


@dataclass
class LoadedFold:
    train_features: np.ndarray
    train_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    train_micro: Optional[np.ndarray]
    test_micro: Optional[np.ndarray]
    micro_mean: Optional[np.ndarray]
    micro_std: Optional[np.ndarray]


def _fold_number(path: Path) -> int:
    match = FOLD_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse FI-2010 fold from {path}")
    return int(match.group(1))


def _candidate_files(data_root: Path, split: str, variant: str) -> List[Path]:
    prefix = "Train_Dst" if split == "train" else "Test_Dst"
    split_word = "Training" if split == "train" else "Testing"
    candidates: List[Path] = []
    for suffix in ("txt", "npy"):
        for path in data_root.rglob(f"{prefix}*CF_*.{suffix}"):
            if path.name.startswith("._"):
                continue
            text = str(path).lower()
            if split_word.lower() not in text:
                continue
            if variant.lower() not in text:
                continue
            candidates.append(path)
    # Prefer source txt over cache npy when both are discovered in the dataset tree.
    return sorted(set(candidates), key=lambda p: (_fold_number(p), p.suffix != ".txt", str(p)))


def discover_fold_pairs(data_root: Path | str, variant: str = "ZScore") -> List[FoldPair]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"FI-2010 data root does not exist: {root}")
    train: Dict[int, Path] = {}
    test: Dict[int, Path] = {}
    for path in _candidate_files(root, "train", variant):
        train.setdefault(_fold_number(path), path)
    for path in _candidate_files(root, "test", variant):
        test.setdefault(_fold_number(path), path)
    shared = sorted(set(train) & set(test))
    if not shared:
        raise FileNotFoundError(f"No paired FI-2010 {variant} folds found below {root}")
    missing_train = sorted(set(test) - set(train))
    missing_test = sorted(set(train) - set(test))
    if missing_train or missing_test:
        raise ValueError(f"Unpaired folds: missing train={missing_train}, missing test={missing_test}")
    return [FoldPair(fold=fold, train_path=train[fold], test_path=test[fold]) for fold in shared]


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matrix(path: Path, cache_root: Optional[Path] = None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, mmap_mode="r")
    if cache_root is None:
        return np.loadtxt(path, dtype=np.float32)

    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{path.stem}.npy"
    if not cache_path.exists():
        matrix = np.loadtxt(path, dtype=np.float32)
        temp_path = cache_path.with_suffix(".tmp.npy")
        np.save(temp_path, matrix)
        temp_path.replace(cache_path)
    return np.load(cache_path, mmap_mode="r")


def split_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or matrix.shape[0] < LABEL_START + 5:
        raise ValueError(f"Expected FI-2010 matrix shaped (149, N), got {matrix.shape}")
    features = np.asarray(matrix[:N_FEATURES].T, dtype=np.float32)
    labels = np.asarray(matrix[LABEL_START : LABEL_START + 5].T, dtype=np.int64)
    return features, labels


def legacy_micro_features(features: np.ndarray) -> np.ndarray:
    """Reproduce the original gate features for a controlled legacy comparison."""
    snapshot = np.asarray(features, dtype=np.float32)
    bid_p = snapshot[:, 0:40:4]
    bid_v = snapshot[:, 1:40:4]
    ask_p = snapshot[:, 2:40:4]
    ask_v = snapshot[:, 3:40:4]
    eps = 1e-8
    spread = (ask_p[:, 0] - bid_p[:, 0]) / (ask_p[:, 0] + eps)
    depth_imb = (bid_v[:, 0] - ask_v[:, 0]) / (bid_v[:, 0] + ask_v[:, 0] + eps)
    total_bid = bid_v[:, :4].sum(axis=1)
    total_ask = ask_v[:, :4].sum(axis=1)
    vol_ratio = (total_bid - total_ask) / (total_bid + total_ask + eps)
    ofi = bid_v[:, :4] * np.where(bid_p[:, :4] >= 0, 1.0, -1.0)
    ofi -= ask_v[:, :4] * np.where(ask_p[:, :4] <= 0, 1.0, -1.0)
    result = np.column_stack([spread, depth_imb, vol_ratio, np.tanh(ofi)])
    return np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


def corrected_micro_features(features: np.ndarray) -> np.ndarray:
    """Compute finite, temporal microstructure proxies using the documented layout.

    FI-2010 stores each level as ask price, ask volume, bid price, bid volume.
    The values may be independently normalized, so these are treated as stable
    proxies. Decimal-precision inputs should be supplied when available.
    """
    snapshot = np.asarray(features, dtype=np.float32)
    ask_p = snapshot[:, 0:40:4]
    ask_v = snapshot[:, 1:40:4]
    bid_p = snapshot[:, 2:40:4]
    bid_v = snapshot[:, 3:40:4]
    eps = 1e-6

    spread = ask_p[:, 0] - bid_p[:, 0]
    depth_imb = (bid_v[:, 0] - ask_v[:, 0]) / (
        np.abs(bid_v[:, 0]) + np.abs(ask_v[:, 0]) + eps
    )
    total_bid = bid_v[:, :4].sum(axis=1)
    total_ask = ask_v[:, :4].sum(axis=1)
    vol_ratio = (total_bid - total_ask) / (np.abs(total_bid) + np.abs(total_ask) + eps)

    previous_bid_p = np.vstack([bid_p[:1], bid_p[:-1]])
    previous_ask_p = np.vstack([ask_p[:1], ask_p[:-1]])
    previous_bid_v = np.vstack([bid_v[:1], bid_v[:-1]])
    previous_ask_v = np.vstack([ask_v[:1], ask_v[:-1]])
    bid_event = np.where(bid_p[:, :4] >= previous_bid_p[:, :4], bid_v[:, :4], 0.0)
    bid_event -= np.where(bid_p[:, :4] <= previous_bid_p[:, :4], previous_bid_v[:, :4], 0.0)
    ask_event = np.where(ask_p[:, :4] <= previous_ask_p[:, :4], ask_v[:, :4], 0.0)
    ask_event -= np.where(ask_p[:, :4] >= previous_ask_p[:, :4], previous_ask_v[:, :4], 0.0)
    ofi = np.tanh(bid_event - ask_event)

    result = np.column_stack([np.tanh(spread), depth_imb, vol_ratio, ofi])
    return np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


def normalize_micro(
    train_micro: np.ndarray,
    test_micro: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_micro.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_micro.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    train = np.clip((train_micro - mean) / std, -10.0, 10.0).astype(np.float32)
    test = np.clip((test_micro - mean) / std, -10.0, 10.0).astype(np.float32)
    return train, test, mean, std


def load_fold(
    pair: FoldPair,
    cache_root: Optional[Path] = None,
    micro_mode: str = "none",
    micro_pair: Optional[FoldPair] = None,
) -> LoadedFold:
    train_features, train_labels = split_matrix(load_matrix(pair.train_path, cache_root))
    test_features, test_labels = split_matrix(load_matrix(pair.test_path, cache_root))

    if micro_mode == "none":
        return LoadedFold(
            train_features, train_labels, test_features, test_labels,
            None, None, None, None,
        )

    micro_train_features = train_features
    micro_test_features = test_features
    if micro_pair is not None:
        micro_train_features, micro_train_labels = split_matrix(load_matrix(micro_pair.train_path, cache_root))
        micro_test_features, micro_test_labels = split_matrix(load_matrix(micro_pair.test_path, cache_root))
        if micro_train_features.shape[0] != train_features.shape[0] or micro_test_features.shape[0] != test_features.shape[0]:
            raise ValueError("Microstructure variant is not sample-aligned with ZScore data")
        if not np.array_equal(micro_train_labels, train_labels) or not np.array_equal(micro_test_labels, test_labels):
            raise ValueError("Microstructure variant labels do not align with ZScore data")

    if micro_mode == "legacy":
        train_micro = legacy_micro_features(micro_train_features)
        test_micro = legacy_micro_features(micro_test_features)
    elif micro_mode in {"corrected", "corrected_decimal"}:
        train_micro = corrected_micro_features(micro_train_features)
        test_micro = corrected_micro_features(micro_test_features)
    else:
        raise ValueError(f"Unknown micro_mode={micro_mode!r}")

    return LoadedFold(
        train_features, train_labels, test_features, test_labels,
        train_micro, test_micro, None, None,
    )


def apply_micro_standardization(
    loaded: LoadedFold,
    train_stop: int,
) -> LoadedFold:
    """Standardize microstructure features using only the training partition."""
    if loaded.train_micro is None:
        return loaded
    reference = loaded.train_micro[:train_stop]
    mean = reference.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = reference.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def transform(values: np.ndarray) -> np.ndarray:
        return np.clip((values - mean) / std, -10.0, 10.0).astype(np.float32)

    return LoadedFold(
        loaded.train_features,
        loaded.train_labels,
        loaded.test_features,
        loaded.test_labels,
        transform(loaded.train_micro),
        transform(loaded.test_micro),
        mean,
        std,
    )


class FI2010SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        horizon: int,
        sequence_length: int = 100,
        micro: Optional[np.ndarray] = None,
    ) -> None:
        if horizon not in HORIZON_INDEX:
            raise ValueError(f"Unsupported prediction horizon: {horizon}")
        if len(features) != len(labels):
            raise ValueError("Feature and label lengths differ")
        if micro is not None and len(micro) != len(features):
            raise ValueError("Microstructure and feature lengths differ")
        if len(features) <= sequence_length:
            raise ValueError("Not enough samples to build a sequence dataset")
        self.features = features
        self.labels = (labels[:, HORIZON_INDEX[horizon]] - 1).astype(np.int64)
        self.micro = micro
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stop = index + self.sequence_length
        x = torch.from_numpy(np.array(self.features[index:stop], dtype=np.float32, copy=True))
        y = torch.tensor(int(self.labels[stop - 1]), dtype=torch.long)
        if self.micro is None:
            m = torch.empty(0, dtype=torch.float32)
        else:
            m = torch.from_numpy(np.array(self.micro[stop - 1], dtype=np.float32, copy=True))
        return x, y, m


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> DataLoader:
    pin_memory = bool(device is not None and device.type == "cuda")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def audit_fold_sizes(pairs: Sequence[FoldPair], cache_root: Optional[Path] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    previous_train_size: Optional[int] = None
    previous_test_size: Optional[int] = None
    for pair in pairs:
        train = load_matrix(pair.train_path, cache_root)
        test = load_matrix(pair.test_path, cache_root)
        train_size = int(train.shape[1])
        test_size = int(test.shape[1])
        increment = None if previous_train_size is None else train_size - previous_train_size
        rows.append({
            "fold": pair.fold,
            "train_samples": train_size,
            "test_samples": test_size,
            "train_increment": increment,
            "increment_matches_previous_test": None if previous_test_size is None else increment == previous_test_size,
            "train_path": str(pair.train_path),
            "test_path": str(pair.test_path),
        })
        previous_train_size = train_size
        previous_test_size = test_size
    return rows
