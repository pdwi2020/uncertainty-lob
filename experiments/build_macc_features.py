"""FEAT — Build macc_features.npz: per-sample regime + microstructure arrays.

Aligns to the saved val logits (337,974 samples) and, when available, to test
logits from C-stat (#36). Both val and test features use the same HMM fitted on
the training portion (first 80%) of the FI-2010 training data.

Output: results/macc_features.npz  (and macc_features_test.npz when test available)

Arrays in the .npz:
  r_val     (N_val,)     — HMM regime label {0=low-vol, 1=high-vol}
  m_val     (N_val, 7)   — microstructure vector [spread, depth_imb, vol_ratio,
                            ofi_l1, ofi_l2, ofi_l3, ofi_l4]
  psi_val   (N_val,)     — toxicity score ψ ∈ [0,1]
  entropy_val (N_val,)   — softmax entropy from DeepLOB val logits
  r_test    (N_test,)    — same, test partition (if test logits present)
  m_test    (N_test, 7)
  psi_test  (N_test,)
  entropy_test (N_test,)

Feature definitions:
  spread    = (ask_p1 - bid_p1) / (|mid| + 1e-8)            [normalized spread]
  depth_imb = (bid_v1 - ask_v1) / (bid_v1 + ask_v1 + 1e-8)  [L1 depth imbalance]
  vol_ratio = short_vol_60 / (long_vol_600 + 1e-8)           [vol regime signal]
  ofi_lk    = (bid_vk - ask_vk) / (bid_vk + ask_vk + 1e-8)  [per-level OFI proxy]
  toxicity ψ = sigmoid(3|ofi_mean| + 2|spread_norm| + |depth_imb|)  [in (0,1)]

Usage:
  python experiments/build_macc_features.py
  python experiments/build_macc_features.py --data-root /path/to/fi-2010
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# FI-2010 layout: ask_p1=0, ask_v1=1, bid_p1=2, bid_v1=3, ask_p2=4, ...
def _col(level: int, field: str) -> int:
    return (level - 1) * 4 + {"ask_p": 0, "ask_v": 1, "bid_p": 2, "bid_v": 3}[field]

SEQ_LEN     = 100
VAL_FRAC    = 0.20


# ── Microstructure features ────────────────────────────────────────────────────

def extract_microstructure(feat: np.ndarray) -> np.ndarray:
    """Extract 7-dim microstructure vector from (N, 40) raw LOB features.

    Returns (N, 7) float32: [spread_norm, depth_imb_l1, vol_ratio, ofi_l1, ofi_l2, ofi_l3, ofi_l4]
    vol_ratio is computed as rolling-window signal (needs time series).
    """
    ask_p1 = feat[:, _col(1, "ask_p")]
    bid_p1 = feat[:, _col(1, "bid_p")]
    ask_v1 = feat[:, _col(1, "ask_v")]
    bid_v1 = feat[:, _col(1, "bid_v")]
    mid    = (ask_p1 + bid_p1) / 2.0

    spread_norm = (ask_p1 - bid_p1) / (np.abs(mid) + 1e-8)
    depth_imb   = (bid_v1 - ask_v1) / (bid_v1 + ask_v1 + 1e-8)

    # Vol ratio (rolling): use mid-price differences (Z-scored, so diffs are fine)
    mid_diff = np.diff(mid, prepend=mid[0])
    s        = pd.Series(mid_diff)
    short_v  = s.rolling(60,  min_periods=1).std().to_numpy()
    long_v   = s.rolling(600, min_periods=1).std().to_numpy()
    vol_ratio = short_v / (long_v + 1e-8)

    # Per-level OFI proxy: (bid_v - ask_v) / (bid_v + ask_v)
    ofi = []
    for lv in range(1, 5):
        bv = feat[:, _col(lv, "bid_v")]
        av = feat[:, _col(lv, "ask_v")]
        ofi.append(np.clip((bv - av) / (bv + av + 1e-8), -1, 1))

    return np.column_stack([spread_norm, depth_imb, vol_ratio] + ofi).astype(np.float32)


def compute_toxicity(m: np.ndarray) -> np.ndarray:
    """Toxicity ψ(x) ∈ (0,1) — sigmoid of weighted microstructure signals.

    Higher ψ = more toxic/adverse-selection risk.
    """
    ofi_mean = m[:, 3:7].mean(axis=1)   # mean OFI across 4 levels
    spread   = m[:, 0]
    depth_imb = m[:, 1]
    raw = np.abs(ofi_mean) * 3.0 + np.abs(spread) * 2.0 + np.abs(depth_imb) * 1.0
    return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)   # sigmoid


# ── HMM regime features (same as run_regime.py) ───────────────────────────────

def build_hmm_features(feat: np.ndarray) -> np.ndarray:
    """Build 3-dim HMM feature vector [vol_ratio, spread, ofi_mean].

    Matches the exact definition in run_regime.py / HMMRegimeDetector.
    """
    ask_p1 = feat[:, 0]
    bid_p1 = feat[:, 2]
    mid    = (ask_p1 + bid_p1) / 2.0
    spread = ask_p1 - bid_p1

    mid_diff = np.diff(mid, prepend=mid[0])
    s  = pd.Series(mid_diff)
    sv = s.rolling(60,  min_periods=1).std().to_numpy()
    lv = s.rolling(600, min_periods=1).std().to_numpy()
    vol_ratio = sv / (lv + 1e-8)

    ofi_levels = []
    for lv_idx in range(5):
        bv = feat[:, 4 * lv_idx + 3]
        av = feat[:, 4 * lv_idx + 1]
        denom = bv + av
        ofi_levels.append(np.where(denom > 0, (bv - av) / denom, 0.0))
    ofi_mean = np.mean(ofi_levels, axis=0)

    out = np.column_stack([vol_ratio, spread, ofi_mean]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",  default=str(ROOT / "datasets" / "fi_2010"))
    parser.add_argument("--output-dir", default=str(RESULTS))
    args = parser.parse_args()

    from src.data.fi2010_loader import _discover_files, _load_raw_files
    try:
        from hmmlearn import hmm as hmmlearn_hmm
    except ImportError:
        print("ERROR: hmmlearn not installed. Run: pip install hmmlearn")
        sys.exit(1)

    data_root   = Path(args.data_root)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zscore_root = data_root / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore"

    print("Loading FI-2010 training data...")
    train_files = _discover_files(zscore_root, "train")
    train_feat_all, train_labs_all = _load_raw_files(train_files)
    N_train_all = len(train_feat_all)
    split_idx   = int(N_train_all * (1 - VAL_FRAC))
    print(f"  Total train rows: {N_train_all:,}  split_idx: {split_idx:,}")

    train_feat = train_feat_all[:split_idx]
    val_feat   = train_feat_all[split_idx:]
    N_val_logits = len(val_feat) - SEQ_LEN   # should be 337,974
    print(f"  Val raw rows: {len(val_feat):,}  Val logits: {N_val_logits:,}")

    # ── Fit HMM ─────────────────────────────────────────────────────────────
    print("Fitting HMM on training portion...")
    hmm_train = build_hmm_features(train_feat)
    model = hmmlearn_hmm.GaussianHMM(n_components=2, covariance_type="full",
                                      n_iter=100, random_state=42)
    model.fit(hmm_train)
    means_vol = model.means_[:, 0]
    high_vol_comp = int(np.argmax(means_vol))
    print(f"  High-vol component: {high_vol_comp}  means: {means_vol}")

    def predict_regimes(feat_arr):
        hmm_f = build_hmm_features(feat_arr)
        raw   = model.predict(hmm_f)
        return np.where(raw == high_vol_comp, 1, 0).astype(np.int8)

    # ── Val features ─────────────────────────────────────────────────────────
    print("Extracting val microstructure features...")
    m_all_val   = extract_microstructure(val_feat)
    r_all_val   = predict_regimes(val_feat)
    psi_all_val = compute_toxicity(m_all_val)

    # Align to logit index: logit i uses window ending at val_feat[i + SEQ_LEN - 1]
    m_val   = m_all_val  [SEQ_LEN - 1 : SEQ_LEN - 1 + N_val_logits]
    r_val   = r_all_val  [SEQ_LEN - 1 : SEQ_LEN - 1 + N_val_logits]
    psi_val = psi_all_val[SEQ_LEN - 1 : SEQ_LEN - 1 + N_val_logits]

    # Entropy from DeepLOB val logits
    entropy_val = None
    for ckpt_name in ["deeplob_k10_logits.pt", "cnn_lstm_attention_k10_logits.pt"]:
        ckpt = output_dir / ckpt_name
        if ckpt.exists():
            saved = torch.load(ckpt, map_location="cpu", weights_only=False)
            logits = saved["logits"].float()
            probs  = torch.softmax(logits, dim=-1).numpy()
            entropy_val = -(probs * np.log(probs + 1e-12)).sum(axis=1).astype(np.float32)
            print(f"  Entropy from {ckpt_name}: shape={entropy_val.shape}")
            break
    if entropy_val is None:
        entropy_val = np.zeros(N_val_logits, dtype=np.float32)
        print("  WARNING: no val logits found for entropy computation")

    r_val_dist = np.bincount(r_val, minlength=2)
    print(f"  Val regime dist: low={r_val_dist[0]:,} ({r_val_dist[0]/len(r_val):.1%}) "
          f"high={r_val_dist[1]:,} ({r_val_dist[1]/len(r_val):.1%})")
    print(f"  Toxicity: mean={psi_val.mean():.4f} std={psi_val.std():.4f}")

    # ── Test features (if test logits available) ──────────────────────────────
    has_test = (output_dir / "deeplob_k10_test_logits.pt").exists()
    if has_test:
        print("\nTest logits found — extracting test features...")
        test_files = _discover_files(zscore_root, "test")
        test_feat, _ = _load_raw_files(test_files)
        N_test_logits = len(test_feat) - SEQ_LEN
        print(f"  Test raw rows: {len(test_feat):,}  Test logits: {N_test_logits:,}")

        m_all_test   = extract_microstructure(test_feat)
        r_all_test   = predict_regimes(test_feat)
        psi_all_test = compute_toxicity(m_all_test)

        m_test   = m_all_test  [SEQ_LEN - 1 : SEQ_LEN - 1 + N_test_logits]
        r_test   = r_all_test  [SEQ_LEN - 1 : SEQ_LEN - 1 + N_test_logits]
        psi_test = psi_all_test[SEQ_LEN - 1 : SEQ_LEN - 1 + N_test_logits]

        saved_test = torch.load(output_dir / "deeplob_k10_test_logits.pt",
                                map_location="cpu", weights_only=False)
        probs_test = torch.softmax(saved_test["logits"].float(), dim=-1).numpy()
        entropy_test = -(probs_test * np.log(probs_test + 1e-12)).sum(axis=1).astype(np.float32)

        r_test_dist = np.bincount(r_test, minlength=2)
        print(f"  Test regime dist: low={r_test_dist[0]:,} ({r_test_dist[0]/len(r_test):.1%}) "
              f"high={r_test_dist[1]:,} ({r_test_dist[1]/len(r_test):.1%})")
    else:
        print("\nTest logits not yet available (C-stat pending) — saving val features only")
        m_test = r_test = psi_test = entropy_test = None

    # ── Save ─────────────────────────────────────────────────────────────────
    save_dict = dict(
        r_val=r_val, m_val=m_val, psi_val=psi_val, entropy_val=entropy_val,
        split_idx=np.array([split_idx]),
        seq_len=np.array([SEQ_LEN]),
    )
    if m_test is not None:
        save_dict.update(dict(
            r_test=r_test, m_test=m_test, psi_test=psi_test, entropy_test=entropy_test,
        ))

    out_path = output_dir / "macc_features.npz"
    np.savez(str(out_path), **save_dict)
    print(f"\nSaved → {out_path}")
    print(f"  r_val: {r_val.shape}  m_val: {m_val.shape}  psi_val: {psi_val.shape}")
    if m_test is not None:
        print(f"  r_test: {r_test.shape}  m_test: {m_test.shape}")

    # Feature stats
    print("\nMicrostructure feature statistics (val):")
    feat_names = ["spread", "depth_imb", "vol_ratio", "ofi_l1", "ofi_l2", "ofi_l3", "ofi_l4"]
    for i, name in enumerate(feat_names):
        col = m_val[:, i]
        print(f"  {name:12s}: mean={col.mean():.4f} std={col.std():.4f} "
              f"p25={np.percentile(col,25):.4f} p75={np.percentile(col,75):.4f}")


if __name__ == "__main__":
    main()
