"""Week 1: FI-2010 data audit and LOB visualisation.

Generates:
    figures/lob_snapshot.png        — LOB depth chart at t=1000
    figures/midprice_series.png     — mid-price + label overlay (2000 steps)
    figures/label_distribution.png  — class counts per horizon (all 9 stocks)
    figures/audit_summary.json      — data statistics

Run locally:
    python experiments/run_lob_audit.py

Run on Kaggle:
    - DATA_ROOT auto-detected from /kaggle path
    - figures saved to /kaggle/working/figures/
"""

import os
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
if os.path.exists("/kaggle"):
    DATA_ROOT = "/kaggle/input/datasets/ulfricirons/fi-2010"
    OUT_DIR = Path("/kaggle/working/figures")
else:
    DATA_ROOT = "/Volumes/Crucial X9/data/market_data/equities/kaggle/fi-2010"
    OUT_DIR = Path(__file__).parent / "figures"

ZSCORE_ROOT = Path(DATA_ROOT) / "BenchmarkDatasets/NoAuction/1.NoAuction_Zscore"
TRAIN_DIR = ZSCORE_ROOT / "NoAuction_Zscore_Training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FI-2010 column layout per level: ask_price, ask_vol, bid_price, bid_vol
def _col(level: int, field: str) -> int:
    return (level - 1) * 4 + ["ask_price", "ask_vol", "bid_price", "bid_vol"].index(field)


# ── Load one file for visualisation (CF_7 — largest training stock) ───────────
train_files = sorted([f for f in TRAIN_DIR.glob("Train_Dst_NoAuction_ZScore_CF_*.txt")
                      if not f.name.startswith("._")])
print(f"Found {len(train_files)} training files")

viz_file = train_files[-1]  # CF_7 or last numerically
raw = np.loadtxt(str(viz_file), dtype=np.float32)  # (149, N)
features = raw[:40].T        # (N, 40)
labels_all = raw[144:149].T  # (N, 5)
N = len(features)
print(f"Loaded {viz_file.name}: N={N} steps")


# ── Figure 1: LOB Depth Snapshot ──────────────────────────────────────────────
t = 1000
bid_prices = [features[t, _col(i, "bid_price")] for i in range(1, 11)]
ask_prices = [features[t, _col(i, "ask_price")] for i in range(1, 11)]
bid_vols   = [features[t, _col(i, "bid_vol")]   for i in range(1, 11)]
ask_vols   = [features[t, _col(i, "ask_vol")]   for i in range(1, 11)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(10), bid_vols, color="#2196F3", alpha=0.85, label="Bid")
ax.barh(range(10), [-v for v in ask_vols], color="#F44336", alpha=0.85, label="Ask")
ax.set_yticks(range(10))
ax.set_yticklabels(
    [f"L{i+1}  B={bid_prices[i]:.2f} | A={ask_prices[i]:.2f}" for i in range(10)],
    fontsize=8,
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Volume (Z-score normalised)")
ax.set_title(f"FI-2010 LOB Depth Snapshot  (t={t}, {viz_file.stem})")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "lob_snapshot.png", dpi=150)
plt.close()
print("Saved lob_snapshot.png")


# ── Figure 2: Mid-price time series + k=10 label overlay ─────────────────────
T_PLOT = 2000
mid = (features[:T_PLOT, _col(1, "bid_price")] + features[:T_PLOT, _col(1, "ask_price")]) / 2
labels_k10 = (labels_all[:T_PLOT, 4] - 1).astype(int)  # 0-indexed

cls_colors = {0: "#4CAF50", 1: "#9E9E9E", 2: "#F44336"}   # FI-2010: 0=up, 2=down

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(mid, color="#1976D2", linewidth=0.8)
ax1.set_ylabel("Mid-price (Z-score)")
ax1.set_title(f"FI-2010 Mid-price and k=10 Labels — {viz_file.stem}")
ax1.grid(alpha=0.3)

for i in range(T_PLOT - 1):
    ax2.axvspan(i, i + 1, alpha=0.7, color=cls_colors[labels_k10[i]], linewidth=0)
ax2.set_yticks([])
ax2.legend(
    handles=[Patch(color=cls_colors[c], label=n) for c, n in zip([0, 1, 2], ["Up", "Stationary", "Down"])],
    loc="upper right", fontsize=8,
)
ax2.set_xlabel("Time step")
ax2.set_title("Direction labels (k=10)", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "midprice_series.png", dpi=150)
plt.close()
print("Saved midprice_series.png")


# ── Figure 3: Label distribution across all 9 stocks × 5 horizons ────────────
horizons = [1, 2, 3, 5, 10]
counts = np.zeros((5, 3), dtype=np.int64)

print("Loading all training files for label counts...")
for tf in train_files:
    d = np.loadtxt(str(tf), dtype=np.float32)
    lbl = d[144:149].T  # (N, 5)
    for hi in range(5):
        for ci in range(3):
            counts[hi, ci] += int((lbl[:, hi] == ci + 1).sum())

fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)
bar_colors = ["#4CAF50", "#9E9E9E", "#F44336"]
bar_labels = ["Up", "Stationary", "Down"]   # raw labels 1, 2, 3
for i, (ax, h) in enumerate(zip(axes, horizons)):
    total = counts[i].sum()
    pcts = counts[i] / total * 100
    bars = ax.bar(bar_labels, pcts, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_title(f"k = {h}", fontsize=11)
    ax.set_ylim(0, 60)
    if i == 0:
        ax.set_ylabel("% samples")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
fig.suptitle("FI-2010 Class Distribution per Prediction Horizon (all 9 training stocks)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "label_distribution.png", dpi=150)
plt.close()
print("Saved label_distribution.png")


# ── Audit summary JSON ────────────────────────────────────────────────────────
summary = {
    "dataset": "FI-2010 NoAuction Zscore",
    "n_train_files": len(train_files),
    "n_timesteps_viz_file": int(N),
    "n_features": 40,
    "sequence_length": 100,
    "label_horizons": horizons,
    "class_counts_per_horizon": {
        f"k={h}": {
            "down": int(counts[i, 0]),
            "stationary": int(counts[i, 1]),
            "up": int(counts[i, 2]),
            "imbalance_ratio": float(min(counts[i, 0], counts[i, 2]) / (counts[i, 1] + 1e-8)),
        }
        for i, h in enumerate(horizons)
    },
    "feature_stats": {
        "mean": float(features.mean()),
        "std": float(features.std()),
        "min": float(features.min()),
        "max": float(features.max()),
    },
}
out_json = OUT_DIR / "audit_summary.json"
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print("Saved audit_summary.json")
print(json.dumps(summary, indent=2))
print("\nWeek 1 audit complete.")
