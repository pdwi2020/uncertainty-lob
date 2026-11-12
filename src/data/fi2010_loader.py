"""FI-2010 LOB dataset loader following NoAuction/Zscore protocol."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple

# FI-2010: data shape on disk is (149, N_samples).
# Rows 0-39:   40 LOB features (10 levels × {bid_price, bid_vol, ask_price, ask_vol})
# Rows 40-143: additional technical indicators (unused)
# Rows 144-148: labels at k=1,2,3,5,10 horizons; values 1=up, 2=stationary, 3=down (1-indexed).
# Direction checked against the DecPre prices: 76% of label-1 windows move up, 73% of label-3
# windows move down. The crypto/LOBSTER builders use the opposite order (2=up), so their labels
# must be mapped with y -> 2 - y before FI-2010-trained models are scored on them.
_HORIZON_IDX = {1: 0, 2: 1, 3: 2, 5: 3, 10: 4}
_LABEL_ROW_START = 144
_N_FEATURES = 40


def _load_raw_files(file_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load and concatenate FI-2010 txt files. Returns (features, labels) arrays."""
    all_features, all_labels = [], []
    for p in sorted(file_paths):
        data = np.loadtxt(str(p), dtype=np.float32)  # (149, N)
        all_features.append(data[:_N_FEATURES].T)     # (N, 40)
        all_labels.append(data[_label_rows()].T)       # (N, 5)
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def _label_rows():
    return slice(_LABEL_ROW_START, _LABEL_ROW_START + 5)


def _discover_files(zscore_root: Path, split: str) -> List[Path]:
    sub = "NoAuction_Zscore_Training" if split == "train" else "NoAuction_Zscore_Testing"
    prefix = "Train_Dst" if split == "train" else "Test_Dst"
    folder = zscore_root / sub
    files = sorted(folder.glob(f"{prefix}_NoAuction_ZScore_CF_*.txt"))
    files = [f for f in files if not f.name.startswith("._")]
    if not files:
        raise FileNotFoundError(f"No FI-2010 files found in {folder}")
    return files


class FI2010Dataset(Dataset):
    """FI-2010 limit order book dataset.

    Features: 40 (10 LOB levels × 4 fields)
    Sequence length: 100 time steps (sliding window, stride=1)
    Labels: 3-class (0=up, 1=stationary, 2=down) at horizon k
    Protocol: NoAuction/Zscore
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        horizon: int = 10,
        sequence_length: int = 100,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        horizon_idx = _HORIZON_IDX[horizon]
        self.features = features.astype(np.float32)           # (N, 40)
        # Convert 1-indexed labels to 0-indexed (0=up, 1=stationary, 2=down)
        self.labels = (labels[:, horizon_idx] - 1).astype(np.int64)  # (N,)

    def __len__(self) -> int:
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.sequence_length]           # (T, 40)
        y = self.labels[idx + self.sequence_length - 1]               # scalar: label at last step
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class FI2010DataModule:
    """Wraps FI2010Dataset with train/val/test DataLoaders.

    Train: all 9 NoAuction_Zscore training files concatenated (first 80%).
    Val:   last 20% of concatenated training data (temporal split).
    Test:  all 9 NoAuction_Zscore test files concatenated.
    """

    def __init__(
        self,
        data_root: str,
        horizon: int = 10,
        batch_size: int = 64,
        sequence_length: int = 100,
        val_fraction: float = 0.20,
        num_workers: int = 4,
    ):
        self.data_root = Path(data_root)
        self.zscore_root = self.data_root / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore"
        self.horizon = horizon
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self._train_ds: Optional[FI2010Dataset] = None
        self._val_ds: Optional[FI2010Dataset] = None
        self._test_ds: Optional[FI2010Dataset] = None

    def _setup(self):
        if self._train_ds is not None:
            return
        train_files = _discover_files(self.zscore_root, "train")
        test_files = _discover_files(self.zscore_root, "test")

        train_feat, train_labels = _load_raw_files(train_files)
        test_feat, test_labels = _load_raw_files(test_files)

        split_idx = int(len(train_feat) * (1 - self.val_fraction))
        self._train_ds = FI2010Dataset(
            train_feat[:split_idx], train_labels[:split_idx],
            self.horizon, self.sequence_length,
        )
        self._val_ds = FI2010Dataset(
            train_feat[split_idx:], train_labels[split_idx:],
            self.horizon, self.sequence_length,
        )
        self._test_ds = FI2010Dataset(
            test_feat, test_labels, self.horizon, self.sequence_length,
        )

    def train_dataloader(self) -> DataLoader:
        self._setup()
        return DataLoader(
            self._train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        self._setup()
        return DataLoader(
            self._val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        self._setup()
        return DataLoader(
            self._test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    @property
    def label_counts(self) -> dict:
        """Class distribution in training set (0=up, 1=stationary, 2=down)."""
        self._setup()
        counts = np.bincount(self._train_ds.labels, minlength=3)
        return {0: int(counts[0]), 1: int(counts[1]), 2: int(counts[2])}
