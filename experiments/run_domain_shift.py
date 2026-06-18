"""Exp 7 — Domain shift: FI-2010 trained models evaluated on Bybit BTC/USDT LOB.

Reconstructs order book from snapshots+deltas, extracts 40 features in FI-2010
format (10 levels × {ask_price, ask_vol, bid_price, bid_vol}), applies Z-score
normalization, generates mid-price movement labels, runs inference.

Usage:
    python experiments/run_domain_shift.py
    python experiments/run_domain_shift.py --days 7 --output-dir /workspace/dl_paper

Saves:
    results/domain_shift_results.json
"""

import sys
import ast
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.deeplob          import DeepLOB
from src.models.tcn              import TCN
from src.models.lob_transformer  import LOBTransformer
from src.models.cnn_lstm_attention import CNNLSTMAttention
from experiments.utils import get_device, resolve_paths, save_results, set_seed

N_LEVELS    = 10    # match FI-2010
SEQ_LEN     = 100
HORIZON     = 10    # k=10 (match trained checkpoints)
LABEL_ALPHA = 0.001 # relative threshold for up/down label (0.1% mid-price move)
SAMPLE_EVERY_MS = 500  # resample orderbook every 500ms

BYBIT_ROOT = Path("/Volumes/Crucial X9/Research Projects/DL_Paper/datasets/bybit_btcusdt_l2")

MODEL_REGISTRY = {
    "deeplob": (
        lambda: DeepLOB(input_channels=40, cnn_out_channels=96, lstm_hidden=64),
        "deeplob_k10.pt",
    ),
    "tcn": (
        lambda: TCN(input_channels=40, num_channels=[64]*6, kernel_size=3),
        "tcn_k10.pt",
    ),
    "lob_transformer": (
        lambda: LOBTransformer(input_dim=40, d_model=64, nhead=4,
                               num_encoder_layers=2, dim_feedforward=128),
        "lob_transformer_k10.pt",
    ),
    "cnn_lstm_attention": (
        lambda: CNNLSTMAttention(input_channels=40, cnn_filters=[32, 64, 64],
                                 lstm_hidden=64),
        "cnn_lstm_attention_k10.pt",
    ),
}


# ── Order book reconstruction ──────────────────────────────────────────────────

def parse_levels(raw) -> dict:
    """Parse bids or asks field (list of [price, size] pairs) → {price: size}."""
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    return {float(p): float(s) for p, s in raw}


def reconstruct_book(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct LOB snapshots at SAMPLE_EVERY_MS intervals.

    Returns:
        features: (N, 40) float32 — 10 levels × [ap, av, bp, bv]
        mid_prices: (N,) float64
    """
    df = df.sort_values("ts").reset_index(drop=True)

    bids: dict = {}
    asks: dict = {}
    snapshots_feat = []
    snapshots_mid  = []

    last_sample_ts = None

    for _, row in df.iterrows():
        ts   = row["ts"]
        rtype = row["type"]
        raw_bids = row["bids"]
        raw_asks = row["asks"]

        if rtype == "snapshot":
            bids = parse_levels(raw_bids)
            asks = parse_levels(raw_asks)
        else:  # delta
            updates = parse_levels(raw_bids)
            for p, s in updates.items():
                if s == 0:
                    bids.pop(p, None)
                else:
                    bids[p] = s
            updates = parse_levels(raw_asks)
            for p, s in updates.items():
                if s == 0:
                    asks.pop(p, None)
                else:
                    asks[p] = s

        if not bids or not asks:
            continue

        # Sample at regular intervals
        if last_sample_ts is None or (ts - last_sample_ts) >= SAMPLE_EVERY_MS:
            last_sample_ts = ts

            top_bids = sorted(bids.items(), reverse=True)[:N_LEVELS]
            top_asks = sorted(asks.items())[:N_LEVELS]

            if len(top_bids) < N_LEVELS or len(top_asks) < N_LEVELS:
                continue

            # FI-2010 feature order: ask_p, ask_v, bid_p, bid_v per level
            feat = []
            for i in range(N_LEVELS):
                ap, av = top_asks[i]
                bp, bv = top_bids[i]
                feat.extend([ap, av, bp, bv])

            mid = (top_bids[0][0] + top_asks[0][0]) / 2
            snapshots_feat.append(feat)
            snapshots_mid.append(mid)

    if len(snapshots_feat) < SEQ_LEN + HORIZON + 10:
        return np.empty((0, 40)), np.empty(0)

    return np.array(snapshots_feat, dtype=np.float64), np.array(snapshots_mid)


def generate_labels(mid_prices: np.ndarray, k: int = HORIZON,
                    alpha: float = LABEL_ALPHA) -> np.ndarray:
    """Smoothed mid-price labels matching FI-2010 protocol.

    Label t: compare mean of next-k mid-prices vs mean of prev-k mid-prices.
    up=2, stationary=1, down=0
    """
    N = len(mid_prices)
    labels = np.ones(N, dtype=np.int64)  # default stationary

    for t in range(k, N - k):
        m_prev = mid_prices[t - k: t].mean()
        m_next = mid_prices[t: t + k].mean()
        if m_next > m_prev * (1 + alpha):
            labels[t] = 2  # up
        elif m_next < m_prev * (1 - alpha):
            labels[t] = 0  # down

    return labels


def zscore_normalize(features: np.ndarray) -> np.ndarray:
    """Z-score normalize each of the 40 features across time."""
    mu  = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return ((features - mu) / std).astype(np.float32)


def build_windows(features: np.ndarray, labels: np.ndarray,
                  seq_len: int = SEQ_LEN, k: int = HORIZON):
    """Build sliding windows. Label at window end (offset by k from label gen)."""
    X, Y = [], []
    for i in range(seq_len, len(features) - k):
        X.append(features[i - seq_len: i])
        Y.append(labels[i + k - 1])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model: nn.Module, X: np.ndarray, batch_size: int,
                  device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i: i + batch_size]).to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",  default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bybit-root", default=str(BYBIT_ROOT))
    parser.add_argument("--days",       type=int, default=7,
                        help="Number of Bybit days to use for evaluation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    _, output_dir = resolve_paths(args.data_root, args.output_dir)
    ckpt_dir    = output_dir / "checkpoints"
    results_dir = output_dir / "results"

    bybit_root = Path(args.bybit_root)
    parquet_days = sorted([
        d for d in bybit_root.iterdir()
        if d.is_dir() and d.name.startswith("date=")
    ])

    if not parquet_days:
        print(f"No parquet date partitions found in {bybit_root}")
        return

    # Use the most recent N days
    selected = parquet_days[-args.days:]
    print(f"Using {len(selected)} days: {selected[0].name} → {selected[-1].name}")

    # ── Load and reconstruct LOB ──
    all_feat, all_mid = [], []
    for day_dir in selected:
        pq = day_dir / "orderbook.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        feat, mid = reconstruct_book(df)
        if len(feat) > 0:
            all_feat.append(feat)
            all_mid.append(mid)
            print(f"  {day_dir.name}: {len(feat)} snapshots")

    if not all_feat:
        print("No data reconstructed.")
        return

    features  = np.concatenate(all_feat, axis=0)
    mid_prices = np.concatenate(all_mid, axis=0)
    print(f"\nTotal snapshots: {len(features)}")

    # ── Normalize + labels + windows ──
    features_norm = zscore_normalize(features)
    labels = generate_labels(mid_prices, k=HORIZON, alpha=LABEL_ALPHA)
    X, Y   = build_windows(features_norm, labels)
    print(f"Windows: {len(X)} | Label dist: {np.bincount(Y)}")

    if len(X) < 100:
        print("Too few windows — increase --days")
        return

    # ── FI-2010 baseline (from saved results) ──
    fi2010_ref = {}
    ref_path = results_dir / "baselines_results.json"
    if ref_path.exists():
        import json
        ref = json.load(open(ref_path))
        for m in MODEL_REGISTRY:
            key = f"{m}_k10" if m != "deeplob" else None
            if m == "deeplob":
                dl_ref = json.load(open(results_dir / "deeplob_results.json"))
                fi2010_ref[m] = dl_ref["k10"]["test_acc"]
            elif key in ref:
                fi2010_ref[m] = ref[key]["test_acc"]

    # ── Load models and run inference ──
    all_results = {}
    for model_name, (model_fn, ckpt_name) in MODEL_REGISTRY.items():
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"[SKIP] {model_name}: checkpoint not found ({ckpt_path})")
            continue

        model = model_fn()
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                          weights_only=False))
        model = model.to(device)

        preds = run_inference(model, X, args.batch_size, device)
        acc   = float((preds == Y).mean())
        f1_mac = float(f1_score(Y, preds, average="macro",    zero_division=0))
        f1_wt  = float(f1_score(Y, preds, average="weighted", zero_division=0))
        kappa  = float(cohen_kappa_score(Y, preds))

        fi_acc = fi2010_ref.get(model_name, None)
        drop   = round((fi_acc - acc) * 100, 2) if fi_acc else None

        result = {
            "model":              model_name,
            "dataset":            "bybit_btcusdt",
            "n_samples":          len(Y),
            "n_days":             len(selected),
            "acc":                round(acc,    4),
            "f1_macro":           round(f1_mac, 4),
            "f1_weighted":        round(f1_wt,  4),
            "kappa":              round(kappa,  4),
            "fi2010_test_acc":    round(fi_acc, 4) if fi_acc else None,
            "acc_drop_pct":       drop,
            "label_distribution": np.bincount(Y).tolist(),
        }
        all_results[model_name] = result
        print(f"[{model_name}] acc={acc:.4f} f1={f1_mac:.4f} "
              f"(FI-2010: {fi_acc:.4f}, drop: {drop:+.2f}%)" if fi_acc else
              f"[{model_name}] acc={acc:.4f} f1={f1_mac:.4f}")

    if not all_results:
        print("No models evaluated.")
        return

    save_results(all_results, results_dir / "domain_shift_results.json")

    print("\n" + "=" * 65)
    print("DOMAIN SHIFT SUMMARY (FI-2010 → Bybit BTC/USDT)")
    print("=" * 65)
    print(f"{'Model':<24} {'FI-2010':>8} {'Bybit':>8} {'Drop':>7} {'F1':>7}")
    print("-" * 65)
    for name, r in all_results.items():
        fi = f"{r['fi2010_test_acc']:.4f}" if r["fi2010_test_acc"] else "  —   "
        drop = f"{r['acc_drop_pct']:+.2f}%" if r["acc_drop_pct"] is not None else "  —  "
        print(f"{name:<24} {fi:>8} {r['acc']:>8.4f} {drop:>7} {r['f1_macro']:>7.4f}")


if __name__ == "__main__":
    main()
