"""Week 2: LOB feature extraction — OFI, spread, queue imbalance, depth-weighted mid.

Generates:
    figures/feature_timeseries.png   — 5 engineered features over time
    figures/feature_correlation.png  — cross-feature correlation heatmap
    figures/ofi_vs_labels.png        — OFI signal vs k=10 direction
    figures/feature_stats.json       — summary statistics

Run locally:
    python experiments/run_feature_extraction.py

Run on Kaggle:
    - DATA_ROOT auto-detected from /kaggle path
"""

import os
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
if os.path.exists("/kaggle"):
    DATA_ROOT = "/kaggle/input/datasets/ulfricirons/fi-2010"
    OUT_DIR = Path("/kaggle/working/figures")
    sys.path.insert(0, "/kaggle/working/dl_paper")
else:
    DATA_ROOT = "/Volumes/Crucial X9/data/market_data/equities/kaggle/fi-2010"
    OUT_DIR = Path(__file__).parent / "figures"
    sys.path.insert(0, str(Path(__file__).parent.parent))

ZSCORE_ROOT = Path(DATA_ROOT) / "BenchmarkDatasets/NoAuction/1.NoAuction_Zscore"
TRAIN_DIR = ZSCORE_ROOT / "NoAuction_Zscore_Training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from src.features.lob_features import (
    extract_all, order_flow_imbalance, realized_volatility, mid_price,
)

# FI-2010 column layout
def _col(level: int, field: str) -> int:
    return (level - 1) * 4 + ["ask_price", "ask_vol", "bid_price", "bid_vol"].index(field)


# ── Load CF_7 ─────────────────────────────────────────────────────────────────
train_files = sorted([f for f in TRAIN_DIR.glob("Train_Dst_NoAuction_ZScore_CF_*.txt")
                      if not f.name.startswith("._")])
viz_file = train_files[-1]
raw = np.loadtxt(str(viz_file), dtype=np.float32)
features = raw[:40].T        # (N, 40)
labels_k10 = (raw[148] - 1).astype(int)  # k=10, 0-indexed
N = len(features)
print(f"Loaded {viz_file.name}: N={N}")

eng = extract_all(features)
ofi = order_flow_imbalance(features)  # (N-1,)
mid = mid_price(features)
rv = realized_volatility(mid, window=3600)


# ── Figure 1: Feature time series (first 5000 steps) ─────────────────────────
T = 5000
ofi_padded = np.concatenate([[0.0], ofi])  # (N,) — pad first step with 0

plot_specs = [
    ("mid_price",        eng["mid_price"][:T],          "#1976D2"),
    ("spread",           eng["spread"][:T],              "#E53935"),
    ("queue_imbalance",  eng["queue_imbalance"][:T],     "#43A047"),
    ("depth_wtd_mid",    eng["depth_weighted_mid"][:T],  "#FB8C00"),
    ("OFI (cumsum)",     ofi_padded[:T].cumsum(),        "#8E24AA"),
]
fig, axes = plt.subplots(len(plot_specs), 1, figsize=(14, 12), sharex=True)
for ax, (name, vals, color) in zip(axes, plot_specs):
    ax.plot(vals, color=color, linewidth=0.6)
    ax.set_ylabel(name, fontsize=8)
    ax.grid(alpha=0.25)
axes[-1].set_xlabel("Time step")
fig.suptitle(f"Engineered LOB Features — {viz_file.stem} (first {T} steps)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_timeseries.png", dpi=150)
plt.close()
print("Saved feature_timeseries.png")


# ── Figure 2: Cross-feature correlation heatmap ───────────────────────────────
feat_matrix = np.column_stack([
    eng["mid_price"],
    eng["spread"],
    eng["relative_spread"],
    eng["queue_imbalance"],
    eng["depth_weighted_mid"],
    ofi_padded,
])
feat_names = ["mid_price", "spread", "rel_spread", "queue_imb", "depth_wtd_mid", "OFI"]
corr = np.corrcoef(feat_matrix.T)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(feat_names)))
ax.set_xticklabels(feat_names, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels(feat_names, fontsize=9)
for i in range(len(feat_names)):
    for j in range(len(feat_names)):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(corr[i, j]) > 0.65 else "black")
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title(f"Feature Correlation — {viz_file.stem}")
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_correlation.png", dpi=150)
plt.close()
print("Saved feature_correlation.png")


# ── Figure 3: OFI signal vs direction labels ──────────────────────────────────
T2 = min(10000, N - 1)
window = 50
ofi_smooth = np.convolve(ofi[:T2], np.ones(window) / window, mode="same")
labels_aligned = labels_k10[1:T2 + 1]  # labels aligned to OFI (starts at t=1)

cls_colors = {0: "#4CAF50", 1: "#9E9E9E", 2: "#F44336"}   # FI-2010: 0=up, 2=down
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(ofi_smooth, color="#8E24AA", linewidth=0.8)
ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax1.set_ylabel(f"OFI ({window}-step MA)")
ax1.set_title("Order Flow Imbalance vs Mid-price Direction (k=10)")
ax1.grid(alpha=0.25)

for cls, color in cls_colors.items():
    mask = labels_aligned == cls
    ax2.scatter(np.where(mask)[0], np.full(mask.sum(), cls), c=color, s=1, alpha=0.25)
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(["Up", "Stat.", "Down"], fontsize=8)
ax2.set_xlabel("Time step")
ax2.set_title("k=10 label", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "ofi_vs_labels.png", dpi=150)
plt.close()
print("Saved ofi_vs_labels.png")


# ── Feature statistics JSON ───────────────────────────────────────────────────
stats = {}
for name, arr in eng.items():
    valid = arr[~np.isnan(arr)]
    stats[name] = {
        "mean": float(valid.mean()), "std": float(valid.std()),
        "min": float(valid.min()),   "max": float(valid.max()),
    }
stats["ofi"] = {
    "mean": float(ofi.mean()), "std": float(ofi.std()),
    "min": float(ofi.min()),   "max": float(ofi.max()),
}
with open(OUT_DIR / "feature_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("Saved feature_stats.json")
print(json.dumps(stats, indent=2))
print("\nWeek 2 feature extraction complete.")
