"""Exp 5-6 — HMM regime detection + selective prediction (risk-coverage curves).

Requires: trained checkpoints from run_deeplob.py and run_baselines.py.
Kaggle install: !pip install hmmlearn

Usage:
    python experiments/run_regime.py
    python experiments/run_regime.py --output-dir /kaggle/working

Saves:
    results/regime_results.json       — per-regime F1/accuracy for each model
    results/selective_pred_results.json — coverage-accuracy at thresholds [0.3..0.9]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.regime.hmm_detector import HMMRegimeDetector
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

from src.models.deeplob import DeepLOB
from src.models.tcn import TCN
from src.models.lob_transformer import LOBTransformer
from src.models.cnn_lstm_attention import CNNLSTMAttention
from src.data.fi2010_loader import FI2010DataModule, _discover_files, _load_raw_files
from experiments.utils import get_device, resolve_paths, save_results, set_seed


MODEL_CLASSES = {
    "deeplob": lambda: DeepLOB(input_channels=40, cnn_out_channels=96, lstm_hidden=64),
    "tcn": lambda: TCN(input_channels=40, num_channels=[64]*6, kernel_size=3),
    "lob_transformer": lambda: LOBTransformer(input_dim=40, d_model=64, nhead=4,
                                               num_encoder_layers=2, dim_feedforward=128),
    "cnn_lstm_attention": lambda: CNNLSTMAttention(input_channels=40, cnn_filters=[32,64,64],
                                                   lstm_hidden=64),
}


def build_regime_features(features: np.ndarray) -> np.ndarray:
    """Build HMM features from raw LOB features for regime detection.

    FI-2010 NoAuction/Zscore column order (after _load_raw_files transpose):
      col 0: ask_price_1 (best ask, Z-score normalised)
      col 1: ask_vol_1
      col 2: bid_price_1 (best bid, Z-score normalised)
      col 3: bid_vol_1
      col 4-7: level 2, ...

    Because prices are Z-score normalised (std~1), absolute log-returns are
    uninformative for regime detection. Instead we use:
      - Short/long vol ratio: rolling_std(60) / rolling_std(600)  [scale-invariant]
      - Multi-level OFI: mean OFI across 5 levels
      - Spread proxy: ask_price_1 - bid_price_1 (in normalised units)
    """
    ask_price = features[:, 0]
    ask_vol   = features[:, 1]
    bid_price = features[:, 2]
    bid_vol   = features[:, 3]

    mid = (ask_price + bid_price) / 2
    spread = ask_price - bid_price

    # Mid-price differences (more stable than log-returns for normalised prices)
    mid_diff = np.diff(mid, prepend=mid[0])
    s = pd.Series(mid_diff)
    short_vol = s.rolling(60,  min_periods=1).std().to_numpy()
    long_vol  = s.rolling(600, min_periods=1).std().to_numpy()
    vol_ratio = short_vol / (long_vol + 1e-8)  # >1 = locally volatile regime

    # Multi-level OFI (levels 1-5)
    ofi_levels = []
    for lv in range(5):
        bv = features[:, 4*lv + 3]
        av = features[:, 4*lv + 1]
        denom = bv + av
        ofi_levels.append(np.where(denom > 0, (bv - av) / denom, 0.0))
    ofi = np.mean(ofi_levels, axis=0)

    hmm_features = np.column_stack([vol_ratio, spread, ofi]).astype(np.float32)
    hmm_features = np.nan_to_num(hmm_features, nan=0.0)
    return hmm_features


def collect_predictions(model, loader, device) -> tuple:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=-1)
            all_preds.append(logits.argmax(-1).cpu().numpy())
            all_labels.append(y.numpy())
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_probs)


def selective_prediction_curve(preds, labels, probs, thresholds=None):
    """Compute coverage-accuracy at varying confidence thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.3, 1.0, 0.05)
    confidences = probs.max(axis=-1)
    rows = []
    for theta in thresholds:
        mask = confidences >= theta
        coverage = mask.mean()
        if mask.sum() == 0:
            continue
        acc = accuracy_score(labels[mask], preds[mask])
        f1 = f1_score(labels[mask], preds[mask], average="macro", zero_division=0)
        rows.append({"theta": float(theta), "coverage": float(coverage),
                     "acc": float(acc), "f1_macro": float(f1), "n": int(mask.sum())})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=list(MODEL_CLASSES.keys()))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not HMMLEARN_AVAILABLE:
        print("ERROR: hmmlearn not installed. Run: pip install hmmlearn")
        sys.exit(1)

    set_seed(args.seed)
    device = get_device()
    data_root, output_dir = resolve_paths(args.data_root, args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    k = args.horizon
    print(f"Device: {device} | k={k}")

    # --- Build HMM on raw training features ---
    zscore_root = data_root / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore"
    train_files = _discover_files(zscore_root, "train")
    test_files = _discover_files(zscore_root, "test")
    train_feat, train_labels_raw = _load_raw_files(train_files)
    test_feat, test_labels_raw = _load_raw_files(test_files)

    print("Fitting HMM on training features...")
    hmm_det = HMMRegimeDetector(n_components=2)
    hmm_feat_train = build_regime_features(train_feat)
    hmm_feat_test  = build_regime_features(test_feat)
    hmm_det.fit(hmm_feat_train)
    train_regimes = hmm_det.predict(hmm_feat_train)
    test_regimes  = hmm_det.predict(hmm_feat_test)
    print(f"Train regimes: 0={( train_regimes==0).sum()}, 1={( train_regimes==1).sum()}")
    print(f"Test  regimes: 0={(test_regimes==0).sum()}, 1={(test_regimes==1).sum()}")

    # --- Per-model: regime-conditional performance + selective prediction ---
    regime_results = {}
    selective_results = {}

    # DataLoader for test set (for model predictions)
    dm = FI2010DataModule(str(data_root), horizon=k, batch_size=64, num_workers=args.num_workers)
    test_loader = dm.test_dataloader()

    for model_name in args.models:
        ckpt_path = ckpt_dir / f"{model_name}_k{k}.pt"
        if not ckpt_path.exists():
            print(f"SKIP {model_name}: checkpoint not found")
            continue

        print(f"\n--- {model_name} ---")
        model = MODEL_CLASSES[model_name]()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)

        preds, labels, probs = collect_predictions(model, test_loader, device)

        # Align regime labels to prediction indices
        # predictions start at index sequence_length-1 in the raw test array
        seq_len = 100
        regime_aligned = test_regimes[seq_len - 1 : seq_len - 1 + len(preds)]
        # Truncate if mismatch (multiple test files concatenated vs sliding window count)
        min_len = min(len(preds), len(regime_aligned))
        preds_a = preds[:min_len]
        labels_a = labels[:min_len]
        probs_a = probs[:min_len]
        regimes_a = regime_aligned[:min_len]

        # Regime-conditional performance
        regime_df = hmm_det.regime_conditional_performance(preds_a, labels_a, regimes_a)
        print(regime_df.to_string(index=False))
        regime_results[model_name] = regime_df.to_dict(orient="records")

        # Selective prediction curves
        curve = selective_prediction_curve(preds_a, labels_a, probs_a)
        selective_results[model_name] = curve
        print(f"  Selective@θ=0.7: "
              f"coverage={next((r['coverage'] for r in curve if r['theta']>=0.7), 'n/a'):.3f}")

    save_results(regime_results, results_dir / "regime_results.json")
    save_results(selective_results, results_dir / "selective_pred_results.json")

    print("\n=== Regime detection complete ===")


if __name__ == "__main__":
    main()
