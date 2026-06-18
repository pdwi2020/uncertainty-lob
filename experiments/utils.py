"""Shared utilities for experiment scripts."""

import os
import json
import time
from pathlib import Path
import torch
import numpy as np


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_kaggle() -> bool:
    return os.path.exists("/kaggle")


def resolve_paths(args_data_root=None, args_output_dir=None) -> tuple[Path, Path]:
    """Return (data_root, output_dir) based on environment."""
    if is_kaggle():
        # Kaggle: data mounted at /kaggle/input/, outputs to /kaggle/working/
        data_root = Path(args_data_root or "/kaggle/input/fi-2010-lob")
        output_dir = Path(args_output_dir or "/kaggle/working")
    else:
        data_root = Path(
            args_data_root
            or "/Volumes/Crucial X9/data/market_data/equities/kaggle/fi-2010"
        )
        output_dir = Path(
            args_output_dir
            or "/Volumes/Crucial X9/Research Projects/DL_Paper"
        )
    return data_root, output_dir


def save_results(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {path}")


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
