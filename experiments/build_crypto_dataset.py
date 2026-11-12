"""Build vol-normalized crypto LOB datasets for cross-market recalibration.

Symbols: BTCUSDT, ETHUSDT, SOLUSDT (Bybit L2 orderbook parquet files on X9).

Vol-normalized labeling (de-degenerates 99.97%-stationary problem):
  σ_t = rolling std of log-returns over past 60 seconds
  z_{t,k} = (m_{t+k} − m_t) / (σ_t · m_t + ε)
  label = 2 (up)   if z > +c
          0 (down)  if z < -c
          1 (stat)  otherwise
  Threshold c = 1.0 (1 realized-vol unit); gives ~25-30% up/down split empirically.

Output:
  results/crypto_{symbol}_k{horizon}_logits.pt  — zero-shot baseline logits (all 4 models)
  results/crypto_{symbol}_k{horizon}_features.npz — norm features + labels + norm_stats
  results/crypto_zero_shot_results.json

Usage:
  python experiments/build_crypto_dataset.py
  python experiments/build_crypto_dataset.py --symbols BTCUSDT,ETHUSDT --days 7
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, cohen_kappa_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

BYBIT_BASE = Path("/Volumes/Crucial X9/data/market_data/exchange=bybit/instrument_type=orderbook_l2")

SEQ_LEN   = 100
HORIZON   = 10
N_LEVELS  = 10
VOL_WINDOW = 60   # seconds for rolling realized-vol estimation
VOL_THRESH  = 1.0  # z-score threshold for up/down label


# ── Orderbook reconstruction (reuses bybit_loader logic) ──────────────────────

class _OrderBook:
    def __init__(self):
        self.bids: dict = {}
        self.asks: dict = {}

    def apply(self, bids_raw, asks_raw, is_snapshot=False):
        if is_snapshot:
            self.bids.clear()
            self.asks.clear()
        for p_s, v_s in bids_raw:
            p, v = float(p_s), float(v_s)
            if v == 0.0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = v
        for p_s, v_s in asks_raw:
            p, v = float(p_s), float(v_s)
            if v == 0.0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = v

    def top_n(self, n=10):
        if len(self.bids) < n or len(self.asks) < n:
            return None
        top_bids = sorted(self.bids.items(), reverse=True)[:n]
        top_asks = sorted(self.asks.items())[:n]
        row = []
        for i in range(n):
            ap, av = top_asks[i]
            bp, bv = top_bids[i]
            row.extend([ap, av, bp, bv])
        return row


import json as _json_mod

def _parse_levels(raw):
    if isinstance(raw, list):
        return raw
    try:
        return _json_mod.loads(raw)
    except Exception:
        return ast.literal_eval(raw)


def reconstruct_1s_snapshots(df: pd.DataFrame, n_levels=10) -> pd.DataFrame:
    """Reconstruct 1-second LOB snapshots from snapshot+delta stream.

    Optimized: top_n() called only at second boundaries (~86k/day) not per event.
    """
    book = _OrderBook()
    records = []
    current_sec = None

    # itertuples is ~10x faster than iterrows
    for row in df.itertuples(index=False):
        ts_s = int(row.ts) // 1000
        is_snap = row.type == 'snapshot'
        book.apply(_parse_levels(row.bids), _parse_levels(row.asks), is_snap)

        if current_sec is None:
            current_sec = ts_s
            continue

        # Emit one snapshot per second at each second boundary
        while ts_s > current_sec:
            snap = book.top_n(n_levels)
            if snap is not None:
                records.append((current_sec, snap))
            current_sec += 1

    if not records:
        return pd.DataFrame()

    cols = []
    for i in range(1, n_levels + 1):
        cols += [f'ask_p{i}', f'ask_v{i}', f'bid_p{i}', f'bid_v{i}']

    timestamps, rows = zip(*records)
    result = pd.DataFrame(rows, index=pd.to_datetime(timestamps, unit='s', utc=True), columns=cols)
    result.index.name = 'timestamp'
    return result


# ── Vol-normalized labeling ────────────────────────────────────────────────────

def vol_normalized_labels(
    snapshots: pd.DataFrame,
    horizon: int = HORIZON,
    vol_window: int = VOL_WINDOW,
    thresh: float = VOL_THRESH,
) -> np.ndarray:
    """Vol-normalized mid-price movement labels.

    z_{t,k} = (m_{t+k} − m_t) / (σ_t · m_t + ε)
    σ_t = rolling std of log-returns over past vol_window seconds.
    label: 2=up if z > thresh; 0=down if z < -thresh; 1=stationary otherwise.
    """
    mid = (snapshots['ask_p1'] + snapshots['bid_p1']) / 2.0
    log_ret = np.log(mid / mid.shift(1)).fillna(0)
    sigma_t = log_ret.rolling(vol_window, min_periods=10).std().fillna(log_ret.std())

    future_mid = mid.shift(-horizon)
    # z_{t,k} = (m_{t+k} - m_t) / (σ_t * m_t + ε)
    eps = 1e-8
    z = (future_mid - mid) / (sigma_t * mid + eps)

    labels = np.where(z > thresh, 2, np.where(z < -thresh, 0, 1)).astype(np.int64)
    labels[mid.isna().values | future_mid.isna().values] = 1   # fill NaN boundary with stationary

    dist = np.bincount(labels, minlength=3)
    print(f"  Vol-norm labels (thresh={thresh:.1f}): down={dist[0]:,} stat={dist[1]:,} up={dist[2]:,} ({dist[1]/len(labels):.1%} stat)")

    # Adaptive threshold: if still >85% stationary, halve thresh
    if dist[1] / len(labels) > 0.85 and thresh > 0.1:
        print(f"  Adaptive: thresh {thresh:.2f} → {thresh/2:.2f}")
        return vol_normalized_labels(snapshots, horizon, vol_window, thresh / 2)

    return labels


def zscore_normalize(features: np.ndarray, stats=None):
    if stats is None:
        mu  = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0,  keepdims=True) + 1e-8
        stats = {"mean": mu, "std": std}
    return ((features - stats["mean"]) / stats["std"]).astype(np.float32), stats


def build_windows(features: np.ndarray, labels: np.ndarray):
    X, Y = [], []
    for i in range(SEQ_LEN, len(features)):
        X.append(features[i - SEQ_LEN: i])
        Y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_baseline_models(ckpt_root: Path, device):
    from src.models.deeplob            import DeepLOB
    from src.models.tcn                import TCN
    from src.models.lob_transformer    import LOBTransformer
    from src.models.cnn_lstm_attention import CNNLSTMAttention

    registry = {
        "deeplob":            (DeepLOB(40, 96, 64),
                               "deeplob_k10.pt"),
        "tcn":                (TCN(40, [64]*6, 3),
                               "tcn_k10.pt"),
        "lob_transformer":    (LOBTransformer(40, 64, 4, 2, 128),
                               "lob_transformer_k10.pt"),
        "cnn_lstm_attention": (CNNLSTMAttention(40, [32,64,64], lstm_hidden=64),
                               "cnn_lstm_attention_k10.pt"),
    }

    loaded = {}
    for name, (model, ckpt_name) in registry.items():
        for search_path in [ckpt_root / ckpt_name, ckpt_root / "checkpoints" / ckpt_name]:
            if search_path.exists():
                state = torch.load(search_path, map_location=device, weights_only=False)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state)
                model = model.to(device).eval()
                loaded[name] = model
                print(f"  Loaded {name}")
                break
        else:
            print(f"  [SKIP] {name}: checkpoint not found")
    return loaded


@torch.no_grad()
def run_inference(model: nn.Module, X: np.ndarray, device, batch_size=256):
    logits = []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i: i + batch_size]).to(device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        logits.append(out.cpu())
    return torch.cat(logits)


def compute_metrics(logits, Y):
    Y = 2 - np.asarray(Y)   # targets use 2=up; FI-2010 models use 0=up
    preds = logits.argmax(1).numpy()
    return {
        "acc":         float((preds == Y).mean()),
        "f1_macro":    float(f1_score(Y, preds, average="macro",    zero_division=0)),
        "f1_weighted": float(f1_score(Y, preds, average="weighted", zero_division=0)),
        "kappa":       float(cohen_kappa_score(Y, preds)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols",     default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--bybit-base",  default=str(BYBIT_BASE))
    parser.add_argument("--output-dir",  default=str(RESULTS))
    parser.add_argument("--ckpt-dir",    default=str(ROOT))
    parser.add_argument("--horizon",     type=int, default=HORIZON)
    parser.add_argument("--days",        type=int, default=3,
                        help="Number of recent days to process per symbol")
    parser.add_argument("--max-windows", type=int, default=50000,
                        help="Max windows per symbol (0=all)")
    args = parser.parse_args()

    symbols    = [s.strip() for s in args.symbols.split(",")]
    bybit_base = Path(args.bybit_base)
    output_dir = Path(args.output_dir)
    ckpt_dir   = Path(args.ckpt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Symbols: {symbols}  Days per symbol: {args.days}")

    models = load_baseline_models(ckpt_dir, device)
    if not models:
        print("No models loaded — check checkpoint paths")
        return

    all_results = {}

    for symbol in symbols:
        sym_dir = bybit_base / symbol
        if not sym_dir.exists():
            print(f"\n[SKIP] {symbol}: {sym_dir} not found")
            continue

        # Load most recent N days
        date_dirs = sorted([d for d in sym_dir.iterdir() if d.is_dir() and d.name.startswith("date=")])
        selected  = date_dirs[-args.days:]
        print(f"\n── {symbol} ──  ({len(selected)} days: {selected[0].name} → {selected[-1].name})")

        day_snaps = []
        for day_dir in selected:
            pq = day_dir / "orderbook.parquet"
            if not pq.exists():
                print(f"  [skip] {pq} not found")
                continue
            print(f"  Reconstructing {day_dir.name}...", end=" ", flush=True)
            raw = pd.read_parquet(pq, columns=["ts", "type", "bids", "asks"])
            snaps = reconstruct_1s_snapshots(raw, N_LEVELS)
            print(f"{len(snaps):,} 1s snapshots")
            if not snaps.empty:
                day_snaps.append(snaps)

        if not day_snaps:
            print(f"  No data reconstructed for {symbol}")
            continue

        snapshots = pd.concat(day_snaps).sort_index()
        snapshots = snapshots[~snapshots.index.duplicated(keep="last")]
        print(f"  Total snapshots: {len(snapshots):,}")

        # Vol-normalized labels
        labels = vol_normalized_labels(snapshots, args.horizon)
        dist   = np.bincount(labels, minlength=3)

        # Z-score normalize features
        feat_raw = snapshots.values.astype(np.float32)
        norm_feat, norm_stats = zscore_normalize(feat_raw)

        # Build sliding windows
        X, Y = build_windows(norm_feat, labels)
        print(f"  Windows: {len(X):,}")
        if len(X) < 500:
            print(f"  Too few windows — skipping")
            continue

        # Subsample for CPU speed
        if args.max_windows > 0 and len(X) > args.max_windows:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), args.max_windows, replace=False)
            idx.sort()
            X, Y = X[idx], Y[idx]
            print(f"  Subsampled → {len(X):,} windows")

        # Save features
        feat_path = output_dir / f"crypto_{symbol}_k{args.horizon}_features.npz"
        np.savez(str(feat_path), X=X, Y=Y,
                 norm_mean=norm_stats["mean"], norm_std=norm_stats["std"])
        print(f"  Saved features → {feat_path.name}")

        # Zero-shot inference for each model
        for model_name, model in models.items():
            logits  = run_inference(model, X, device)
            metrics = compute_metrics(logits, Y)
            key = f"{model_name}_{symbol}_k{args.horizon}"
            all_results[key] = {
                "model": model_name, "symbol": symbol,
                "horizon": args.horizon, "n_days": len(selected),
                "n_samples": len(Y),
                "label_distribution": dist.tolist(),
                "label_frac_stat": round(dist[1] / len(Y), 4),
                **{k: round(v, 4) for k, v in metrics.items()},
            }
            print(f"  [{model_name}] acc={metrics['acc']:.4f} f1={metrics['f1_macro']:.4f} kappa={metrics['kappa']:.4f}")

            out_path = output_dir / f"crypto_{symbol}_{model_name}_k{args.horizon}_logits.pt"
            torch.save({"logits": logits, "labels": torch.from_numpy(Y)}, out_path)

    # Summary
    out_json = output_dir / "crypto_zero_shot_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary → {out_json.name}")

    if all_results:
        print("\n── Zero-shot Summary (vol-normalized labels) ──")
        print(f"{'Key':<45} {'acc':>7} {'f1_mac':>7} {'kappa':>7} {'%stat':>7}")
        print("-" * 75)
        for k, v in sorted(all_results.items()):
            print(f"{k:<45} {v['acc']:>7.4f} {v['f1_macro']:>7.4f} {v['kappa']:>7.4f} {v['label_frac_stat']:>7.1%}")


if __name__ == "__main__":
    main()
