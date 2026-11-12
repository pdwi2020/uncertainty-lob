"""Exp 9 — Market-making backtest: spread-capture Sharpe + adverse-selection for all models.

Requires: trained checkpoints from run_deeplob.py and run_baselines.py.

Usage:
    python experiments/run_backtest.py
    python experiments/run_backtest.py --output-dir /kaggle/working

Saves:
    results/backtest_results.json — Sharpe, fill rate, adverse rate per model
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.deeplob import DeepLOB
from src.models.tcn import TCN
from src.models.lob_transformer import LOBTransformer
from src.models.cnn_lstm_attention import CNNLSTMAttention
from src.data.fi2010_loader import FI2010DataModule, _discover_files, _load_raw_files
from src.backtest.market_making import MarketMakingSimulator
from experiments.utils import get_device, resolve_paths, save_results, set_seed


MODEL_CLASSES = {
    "no_signal": None,  # baseline: no directional signal
    "deeplob": lambda: DeepLOB(input_channels=40, cnn_out_channels=96, lstm_hidden=64),
    "tcn": lambda: TCN(input_channels=40, num_channels=[64]*6, kernel_size=3),
    "lob_transformer": lambda: LOBTransformer(input_dim=40, d_model=64, nhead=4,
                                               num_encoder_layers=2, dim_feedforward=128),
    "cnn_lstm_attention": lambda: CNNLSTMAttention(input_channels=40, cnn_filters=[32,64,64],
                                                   lstm_hidden=64),
}


def extract_mid_and_ofi(raw_features: np.ndarray) -> tuple:
    """Extract mid-price and OFI from raw LOB features (FI-2010 layout).

    FI-2010 NoAuction/Zscore column order (after _load_raw_files transpose):
      col 0: ask_price_1 (best ask)
      col 1: ask_vol_1
      col 2: bid_price_1 (best bid)
      col 3: bid_vol_1
      ... repeating for 10 levels
    """
    ask_price = raw_features[:, 0]   # best ask (Z-score normalised)
    ask_vol   = raw_features[:, 1]
    bid_price = raw_features[:, 2]   # best bid (Z-score normalised)
    bid_vol   = raw_features[:, 3]
    mid = (ask_price + bid_price) / 2
    ofi = np.where((bid_vol + ask_vol) > 0,
                   (bid_vol - ask_vol) / (bid_vol + ask_vol), 0.0)
    return mid, ofi


def get_model_signals(model, test_loader, device) -> np.ndarray:
    """Run model on test loader; return prediction array aligned to time steps."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            all_preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=list(MODEL_CLASSES.keys()))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--spread-bps", type=float, default=2.0)
    parser.add_argument("--skew-bps", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    data_root, output_dir = resolve_paths(args.data_root, args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    k = args.horizon
    print(f"Device: {device} | k={k} | spread={args.spread_bps}bps skew={args.skew_bps}bps")

    # Load raw test features to extract mid-price and OFI
    zscore_root = data_root / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore"
    test_files = _discover_files(zscore_root, "test")
    test_feat, _ = _load_raw_files(test_files)
    mid_prices, ofi = extract_mid_and_ofi(test_feat)
    print(f"Test series length: {len(mid_prices)}")

    dm = FI2010DataModule(str(data_root), horizon=k, batch_size=64, num_workers=args.num_workers)
    test_loader = dm.test_dataloader()

    # Price series is Z-score normalised: scale spread/skew by price std
    price_std = float(np.std(mid_prices))
    print(f"Mid-price std (Z-score normalised): {price_std:.4f}")
    sim = MarketMakingSimulator(
        spread_bps=args.spread_bps,
        skew_bps=args.skew_bps,
        price_std=price_std * 1e-4,  # 2 BPS of 1 std-unit
    )
    all_results = {}

    for model_name in args.models:
        print(f"\n--- Backtest: {model_name} ---")

        if model_name == "no_signal":
            # Naive market-making: no directional signal (all flat)
            seq_len = 100
            n_preds = len(test_feat) - seq_len
            signals = np.ones(len(test_feat), dtype=int)  # all stationary
        else:
            ckpt_path = ckpt_dir / f"{model_name}_k{k}.pt"
            if not ckpt_path.exists():
                print(f"  SKIP: checkpoint not found")
                continue
            model = MODEL_CLASSES[model_name]()
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model = model.to(device)
            preds = get_model_signals(model, test_loader, device)
            # Align: predictions start at index seq_len-1 in raw test array
            seq_len = 100
            signals = np.ones(len(test_feat), dtype=int)  # default: flat
            # FI-2010 classes are 0=up, 2=down; the simulator reads 0=down, 2=up
            signals[seq_len - 1 : seq_len - 1 + len(preds)] = 2 - preds

        results_df = sim.simulate(mid_prices, signals, ofi)
        metrics = sim.compute_metrics(results_df)
        all_results[model_name] = metrics

        print(f"  Sharpe={metrics['sharpe']:.3f} | PnL={metrics['total_pnl']:.6f} "
              f"| fills={metrics['n_fills']} | adverse_rate={metrics['adverse_rate']:.3f}")

    save_results(all_results, results_dir / "backtest_results.json")

    print("\n" + "="*70)
    print(f"BACKTEST SUMMARY (k={k}, spread={args.spread_bps}bps)")
    print("="*70)
    print(f"{'Model':>22} {'Sharpe':>8} {'PnL':>12} {'FillRate':>10} {'AdvRate':>10}")
    for name, r in all_results.items():
        print(f"{name:>22} {r['sharpe']:>8.3f} {r['total_pnl']:>12.6f} "
              f"{r['fill_rate']:>10.4f} {r['adverse_rate']:>10.4f}")


if __name__ == "__main__":
    main()
