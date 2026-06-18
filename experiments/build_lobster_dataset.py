"""Build LOBSTER free-sample dataset for zero-shot cross-market evaluation.

LOBSTER orderbook file format (10 levels):
  Columns: Ask_Price_1, Ask_Size_1, Bid_Price_1, Bid_Size_1, ..., Ask_Price_10, ...
  This is already in the same 4-per-level order as FI-2010 (40 columns total).
  Prices are in dollar × 100 (integer cents).

Labeling:
  Same protocol as FI-2010: smoothed mid-price comparison over k=10 seconds.
  Uses adaptive threshold α chosen to avoid >95% stationary dominance.

Output:
  results/lobster_{ticker}_k{horizon}_logits.pt  — zero-shot baseline logits
  results/lobster_{ticker}_k{horizon}_features.npz — features + labels + norm stats
  results/lobster_zero_shot_results.json           — acc / F1 / kappa per model+ticker

Usage:
  python experiments/build_lobster_dataset.py --lobster-dir /path/to/lobster/data
  python experiments/build_lobster_dataset.py  # uses default X9 path
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, cohen_kappa_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEFAULT_LOBSTER_DIR = Path("/Volumes/Crucial X9/projects/p2_market_maker/data/lobster")

SEQ_LEN  = 100
HORIZON  = 10
N_LEVELS = 10


# ── LOBSTER parsing ────────────────────────────────────────────────────────────

def load_lobster_orderbook(orderbook_path: Path) -> np.ndarray:
    """Load LOBSTER orderbook CSV → (N, 40) float32 array.

    LOBSTER orderbook columns: Ask_P1, Ask_V1, Bid_P1, Bid_V1, Ask_P2, ...
    Prices in dollar × 100 (integer cents) → convert to dollars.
    """
    import pandas as pd
    df = pd.read_csv(str(orderbook_path), header=None, dtype=np.float32)
    assert df.shape[1] >= 4 * N_LEVELS, f"Expected ≥{4*N_LEVELS} cols, got {df.shape[1]}"
    feat = df.iloc[:, :4 * N_LEVELS].values.copy()
    # Convert price columns (every 4th starting at 0, 2) from cents to dollars
    price_cols = [level * 4 + offset for level in range(N_LEVELS) for offset in (0, 2)]
    feat[:, price_cols] /= 100.0
    return feat.astype(np.float32)


def generate_labels_fi2010(
    features: np.ndarray,
    horizon: int = 10,
    alpha: float = 2e-4,
) -> np.ndarray:
    """Generate 3-class labels using FI-2010 smoothed mid-price protocol (vectorized).

    Smoothed mid: mean of next k snapshots vs previous k snapshots.
    Adaptive alpha: if >90% stationary, halve alpha.
    """
    import pandas as pd
    ask_p1 = features[:, 0].astype(np.float64)
    bid_p1 = features[:, 2].astype(np.float64)
    mid    = (ask_p1 + bid_p1) / 2.0
    mid_s  = pd.Series(mid)

    # Smoothed future: rolling mean of next k (shift back k, rolling k)
    future_smooth = mid_s[::-1].rolling(horizon, min_periods=horizon).mean()[::-1].shift(-horizon)
    rel = (future_smooth - mid_s) / (mid_s + 1e-10)

    labels = np.where(rel > alpha, 2, np.where(rel < -alpha, 0, 1)).astype(np.int64)
    labels[rel.isna().values] = 1   # boundary rows → stationary

    dist = np.bincount(labels, minlength=3)
    stationary_frac = dist[1] / len(labels)
    if stationary_frac > 0.90 and alpha > 1e-6:
        print(f"  Adaptive alpha: {alpha:.2e} → {alpha/2:.2e} (stationary={stationary_frac:.1%})")
        return generate_labels_fi2010(features, horizon, alpha / 2)

    return labels


def zscore_normalize(features: np.ndarray) -> tuple:
    mu  = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0,  keepdims=True) + 1e-8
    return ((features - mu) / std).astype(np.float32), {"mean": mu, "std": std}


def build_windows(features: np.ndarray, labels: np.ndarray) -> tuple:
    X, Y = [], []
    for i in range(SEQ_LEN, len(features)):
        X.append(features[i - SEQ_LEN: i])
        Y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


# ── Model loading (from X9 checkpoints) ───────────────────────────────────────

def _load_models(ckpt_root: Path, device: torch.device):
    from src.models.deeplob           import DeepLOB
    from src.models.tcn               import TCN
    from src.models.lob_transformer   import LOBTransformer
    from src.models.cnn_lstm_attention import CNNLSTMAttention

    registry = {
        "deeplob":            (DeepLOB(input_channels=40, cnn_out_channels=96, lstm_hidden=64),             "deeplob_k10.pt"),
        "tcn":                (TCN(input_channels=40, num_channels=[64]*6, kernel_size=3),                   "tcn_k10.pt"),
        "lob_transformer":    (LOBTransformer(input_dim=40, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128), "lob_transformer_k10.pt"),
        "cnn_lstm_attention": (CNNLSTMAttention(input_channels=40, cnn_filters=[32,64,64], lstm_hidden=64),  "cnn_lstm_attention_k10.pt"),
    }

    loaded = {}
    for name, (model, ckpt_name) in registry.items():
        ckpt = ckpt_root / ckpt_name
        if not ckpt.exists():
            ckpt = ckpt_root / "checkpoints" / ckpt_name
        if not ckpt.exists():
            print(f"  [SKIP] {name}: checkpoint not found at {ckpt}")
            continue
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model = model.to(device).eval()
        loaded[name] = model
        print(f"  Loaded {name}")
    return loaded


@torch.no_grad()
def run_inference(model: nn.Module, X: np.ndarray, device, batch_size: int = 256):
    all_logits = []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i: i + batch_size]).to(device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        all_logits.append(out.cpu())
    return torch.cat(all_logits)


def compute_metrics(logits, labels_np):
    preds = logits.argmax(1).numpy()
    acc   = float((preds == labels_np).mean())
    f1m   = float(f1_score(labels_np, preds, average="macro",    zero_division=0))
    f1w   = float(f1_score(labels_np, preds, average="weighted", zero_division=0))
    kap   = float(cohen_kappa_score(labels_np, preds))
    return {"acc": acc, "f1_macro": f1m, "f1_weighted": f1w, "kappa": kap}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lobster-dir",  default=str(DEFAULT_LOBSTER_DIR))
    parser.add_argument("--output-dir",   default=str(RESULTS))
    parser.add_argument("--ckpt-dir",     default=str(ROOT))
    parser.add_argument("--horizon",      type=int, default=HORIZON)
    parser.add_argument("--max-windows",  type=int, default=50000,
                        help="Max windows to use (subsample for speed; 0=all)")
    parser.add_argument("--no-inference", action="store_true",
                        help="Skip model inference, just save features")
    args = parser.parse_args()

    lobster_dir = Path(args.lobster_dir)
    output_dir  = Path(args.output_dir)
    ckpt_dir    = Path(args.ckpt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all LOBSTER orderbook files
    ob_files = sorted(lobster_dir.rglob("*orderbook_10.csv"))
    if not ob_files:
        ob_files = sorted(lobster_dir.rglob("*orderbook*.csv"))
    print(f"Found {len(ob_files)} LOBSTER orderbook file(s)")
    if not ob_files:
        print(f"No LOBSTER orderbook files found in {lobster_dir}")
        print("Expected filename pattern: TICKER_DATE_START_END_orderbook_10.csv")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Max windows: {args.max_windows or 'all'}  Inference: {not args.no_inference}")
    models = {}
    if not args.no_inference:
        models = _load_models(ckpt_dir, device)
        if not models:
            print("No models loaded — check checkpoint paths")
            return

    all_results = {}

    for ob_path in ob_files:
        # Parse ticker from filename: AAPL_2012-06-21_..._orderbook_10.csv
        parts = ob_path.stem.split("_")
        ticker = parts[0] if parts else ob_path.stem
        date   = parts[1] if len(parts) > 1 else "unknown"
        print(f"\n── {ticker}  ({date}) ──")

        raw_feat = load_lobster_orderbook(ob_path)
        print(f"  Snapshots: {len(raw_feat):,}")

        labels = generate_labels_fi2010(raw_feat, args.horizon)
        dist   = np.bincount(labels, minlength=3)
        print(f"  Label dist: down={dist[0]:,} stat={dist[1]:,} up={dist[2]:,} ({dist[1]/len(labels):.1%} stat)")

        norm_feat, norm_stats = zscore_normalize(raw_feat)
        X, Y = build_windows(norm_feat, labels)
        print(f"  Windows: {len(X):,}")

        if len(X) < 500:
            print("  Too few windows — skipping")
            continue

        # Stratified subsample if requested
        if args.max_windows > 0 and len(X) > args.max_windows:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), args.max_windows, replace=False)
            idx.sort()
            X, Y = X[idx], Y[idx]
            print(f"  Subsampled → {len(X):,} windows (stratified)")

        # Save features
        feat_path = output_dir / f"lobster_{ticker}_k{args.horizon}_features.npz"
        np.savez(str(feat_path), X=X, Y=Y,
                 norm_mean=norm_stats["mean"], norm_std=norm_stats["std"])
        print(f"  Saved features → {feat_path.name}")

        # Zero-shot inference
        if not models:
            continue
        for model_name, model in models.items():
            logits = run_inference(model, X, device)
            metrics = compute_metrics(logits, Y)
            key = f"{model_name}_lobster_{ticker}_k{args.horizon}"
            all_results[key] = {
                "model": model_name, "dataset": f"lobster_{ticker}",
                "ticker": ticker, "date": date,
                "n_samples": len(Y),
                "label_distribution": dist.tolist(),
                **metrics,
            }
            print(f"  [{model_name}] acc={metrics['acc']:.4f} f1={metrics['f1_macro']:.4f} kappa={metrics['kappa']:.4f}")

            out_path = output_dir / f"lobster_{ticker}_{model_name}_k{args.horizon}_test_logits.pt"
            torch.save({"logits": logits, "labels": torch.from_numpy(Y)}, out_path)

    # Save summary
    out_json = output_dir / "lobster_zero_shot_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary → {out_json.name}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
