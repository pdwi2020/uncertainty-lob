#!/usr/bin/env python3
"""Generate all publication-ready figures for the DL Paper.

Reads result JSONs from ./results/ and saves figures to ./figures/.
Run: python3 generate_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS = Path("results")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

MODEL_LABELS = {
    "deeplob":             "DeepLOB",
    "tcn":                 "TCN",
    "lob_transformer":     "LOBTransformer",
    "cnn_lstm_attention":  "CNN-LSTM-Attn",
}
MODEL_COLORS = {
    "deeplob":             "#1f77b4",
    "tcn":                 "#ff7f0e",
    "lob_transformer":     "#2ca02c",
    "cnn_lstm_attention":  "#d62728",
}
MODELS = list(MODEL_LABELS.keys())


def load(fname):
    return json.load(open(RESULTS / fname))


# ── Fig 1: Horizon analysis (DeepLOB k=1,2,3,5,10) ───────────────────────────
def fig_horizon_analysis():
    d = load("deeplob_results.json")
    horizons = [1, 2, 3, 5, 10]
    acc  = [d[f"k{k}"]["test_acc"]      * 100 for k in horizons]
    f1   = [d[f"k{k}"]["test_f1_macro"] * 100 for k in horizons]
    kappa = [d[f"k{k}"]["test_kappa"]           for k in horizons]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.plot(horizons, acc,   "o-", color="#1f77b4", label="Accuracy (%)", lw=2)
    ax1.plot(horizons, f1,    "s--", color="#ff7f0e", label="F1-macro (%)", lw=2)
    ax2.plot(horizons, kappa, "^:", color="#2ca02c", label="Cohen's κ", lw=2)
    ax1.set_xlabel("Prediction Horizon k")
    ax1.set_ylabel("Accuracy / F1-macro (%)")
    ax2.set_ylabel("Cohen's κ", color="#2ca02c")
    ax1.set_xticks(horizons)
    ax1.set_ylim(70, 95)
    ax2.set_ylim(0.55, 0.90)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.set_title("DeepLOB Performance across Prediction Horizons")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_horizon_analysis.pdf")
    fig.savefig(FIGURES / "fig1_horizon_analysis.png")
    plt.close()
    print("✓ fig1_horizon_analysis")


# ── Fig 2: Model comparison bar chart (k=10) ─────────────────────────────────
def fig_model_comparison():
    dl = load("deeplob_results.json")
    bs = load("baselines_results.json")

    rows = {
        "deeplob":            dl["k10"],
        "tcn":                bs["tcn_k10"],
        "lob_transformer":    bs["lob_transformer_k10"],
        "cnn_lstm_attention": bs["cnn_lstm_attention_k10"],
    }
    labels = [MODEL_LABELS[m] for m in MODELS]
    acc    = [rows[m]["test_acc"]      * 100 for m in MODELS]
    f1_mac = [rows[m]["test_f1_macro"] * 100 for m in MODELS]
    f1_wt  = [rows[m]["test_f1_weighted"] * 100 for m in MODELS]

    x = np.arange(len(MODELS))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, acc,    w, label="Accuracy",   color=[MODEL_COLORS[m] for m in MODELS], alpha=0.9)
    ax.bar(x,     f1_mac, w, label="F1-macro",   color=[MODEL_COLORS[m] for m in MODELS], alpha=0.6, hatch="//")
    ax.bar(x + w, f1_wt,  w, label="F1-weighted",color=[MODEL_COLORS[m] for m in MODELS], alpha=0.4, hatch="xx")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(78, 92)
    ax.set_title("Model Comparison at k=10 (FI-2010 NoAuction/Zscore)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    for i, m in enumerate(MODELS):
        ax.text(x[i] - w, acc[i] + 0.1, f"{acc[i]:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_model_comparison.pdf")
    fig.savefig(FIGURES / "fig2_model_comparison.png")
    plt.close()
    print("✓ fig2_model_comparison")


# ── Fig 3: ECE before/after temperature scaling ───────────────────────────────
def fig_calibration_ece():
    cal = load("calibration_results.json")
    labels   = [MODEL_LABELS[m] for m in MODELS]
    ece_pre  = [cal[m]["ece_before_ts"] for m in MODELS]
    ece_post = [cal[m]["ece_after_ts"]  for m in MODELS]
    temps    = [cal[m]["temperature"]   for m in MODELS]

    x = np.arange(len(MODELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - w/2, ece_pre,  w, label="Before TS", color="#d62728", alpha=0.8)
    bars2 = ax.bar(x + w/2, ece_post, w, label="After TS",  color="#2ca02c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Expected Calibration Error (ECE ↓)")
    ax.set_title("Calibration: ECE Before vs After Temperature Scaling")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars1, ece_pre):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val, T in zip(bars2, ece_post, temps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.3f}\n(T={T:.2f})", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_ece_calibration.pdf")
    fig.savefig(FIGURES / "fig3_ece_calibration.png")
    plt.close()
    print("✓ fig3_ece_calibration")


# ── Fig 4: MC Dropout entropy distributions ───────────────────────────────────
def fig_mc_entropy():
    cal = load("calibration_results.json")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(MODELS))
    means = [cal[m]["mc_mean_entropy"] for m in MODELS]
    stds  = [cal[m]["mc_std_entropy"]  for m in MODELS]
    colors = [MODEL_COLORS[m] for m in MODELS]
    bars = ax.bar(x, means, color=colors, alpha=0.8, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Predictive Entropy H[p] (MC Dropout, 30 samples)")
    ax.set_title("MC Dropout Uncertainty — Mean Predictive Entropy at k=10")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_mc_entropy.pdf")
    fig.savefig(FIGURES / "fig4_mc_entropy.png")
    plt.close()
    print("✓ fig4_mc_entropy")


# ── Fig 5: Conformal prediction — coverage vs alpha ───────────────────────────
def fig_conformal_coverage():
    conf = load("conformal_results.json")
    alphas = [0.05, 0.10, 0.15, 0.20]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: coverage (should be ≥ 1-alpha)
    ax = axes[0]
    for m in MODELS:
        key = f"{m}_k10"
        rows = conf.get(key, [])
        if not rows:
            continue
        cov = [r["coverage"] for r in rows]
        ax.plot(alphas, cov, "o-", color=MODEL_COLORS[m], label=MODEL_LABELS[m], lw=2)
    ax.plot(alphas, [1 - a for a in alphas], "k--", lw=1.5, label="Nominal 1−α")
    ax.set_xlabel("Significance level α")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("(a) Conformal Coverage Guarantee")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: avg set size
    ax = axes[1]
    for m in MODELS:
        key = f"{m}_k10"
        rows = conf.get(key, [])
        if not rows:
            continue
        sizes = [r["avg_set_size"] for r in rows]
        ax.plot(alphas, sizes, "s--", color=MODEL_COLORS[m], label=MODEL_LABELS[m], lw=2)
    ax.axhline(1.0, color="k", ls=":", lw=1.5, label="Singleton (size=1)")
    ax.set_xlabel("Significance level α")
    ax.set_ylabel("Average prediction set size")
    ax.set_title("(b) Prediction Set Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("LAC Conformal Prediction — FI-2010 LOB (k=10)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_conformal.pdf")
    fig.savefig(FIGURES / "fig5_conformal.png")
    plt.close()
    print("✓ fig5_conformal")


# ── Fig 6: Selective prediction (coverage-accuracy trade-off) ─────────────────
def fig_selective_prediction():
    sel = load("selective_pred_results.json")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in MODELS:
        rows = sel.get(m, [])
        if not rows:
            continue
        cov = [r["coverage"] for r in rows]
        acc = [r["acc"] * 100 for r in rows]
        ax.plot(cov, acc, "o-", color=MODEL_COLORS[m], label=MODEL_LABELS[m], lw=2, ms=5)
    ax.set_xlabel("Coverage (fraction of samples predicted)")
    ax.set_ylabel("Accuracy on predicted samples (%)")
    ax.set_title("Selective Prediction: Coverage–Accuracy Trade-off (k=10)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_selective_prediction.pdf")
    fig.savefig(FIGURES / "fig6_selective_prediction.png")
    plt.close()
    print("✓ fig6_selective_prediction")


# ── Fig 7: Regime-conditional performance ────────────────────────────────────
def fig_regime_performance():
    reg = load("regime_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax_idx, metric in enumerate(["accuracy", "f1_macro"]):
        ax = axes[ax_idx]
        x = np.arange(len(MODELS))
        low_vol  = [reg[m][0][metric] * 100 if isinstance(reg[m], list) else reg[m].get(metric, 0) * 100 for m in MODELS]
        high_vol = [reg[m][1][metric] * 100 if isinstance(reg[m], list) and len(reg[m]) > 1 else 0 for m in MODELS]

        w = 0.35
        bars1 = ax.bar(x - w/2, low_vol,  w, label="Low-vol regime",  color="#2ca02c", alpha=0.8)
        if any(v > 0 for v in high_vol):
            bars2 = ax.bar(x + w/2, high_vol, w, label="High-vol regime", color="#d62728", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=10, ha="right")
        ax.set_ylabel(f"{'Accuracy' if metric == 'accuracy' else 'F1-macro'} (%)")
        ax.set_title(f"({'a' if ax_idx == 0 else 'b'}) Regime-Conditional {'Accuracy' if metric == 'accuracy' else 'F1-macro'}")
        ax.set_ylim(78, 95)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars1, low_vol):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("HMM Regime-Conditional Performance (k=10)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_regime_performance.pdf")
    fig.savefig(FIGURES / "fig7_regime_performance.png")
    plt.close()
    print("✓ fig7_regime_performance")


# ── Fig 8: Backtest Sharpe ratio ──────────────────────────────────────────────
def fig_backtest():
    bt = load("backtest_results.json")
    models_bt = ["no_signal", "deeplob", "tcn", "lob_transformer", "cnn_lstm_attention"]
    labels = ["No Signal", "DeepLOB", "TCN", "LOBTransformer", "CNN-LSTM-Attn"]
    sharpes = [bt[m]["sharpe"] for m in models_bt]
    fills   = [bt[m]["fill_rate"] * 100 for m in models_bt]
    adverse = [bt[m]["adverse_rate"] * 100 for m in models_bt]
    colors  = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    bars = ax.bar(range(len(models_bt)), sharpes, color=colors, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(models_bt)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.set_title("(a) Market-Making Sharpe Ratio")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.5 if v >= 0 else -1.5),
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    x = np.arange(len(models_bt))
    w = 0.35
    ax.bar(x - w/2, fills,   w, label="Fill rate (%)",   color="#1f77b4", alpha=0.8)
    ax.bar(x + w/2, adverse, w, label="Adverse rate (%)", color="#d62728", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("(b) Fill Rate vs Adverse Selection Rate")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Market-Making Backtest Results (k=10 signal)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_backtest.pdf")
    fig.savefig(FIGURES / "fig8_backtest.png")
    plt.close()
    print("✓ fig8_backtest")


# ── Fig 9: Reliability diagram (calibration curves) ──────────────────────────
def fig_reliability_diagrams():
    """Plot reliability diagrams from saved val logits."""
    import torch
    import torch.nn.functional as F

    logit_files = {
        "deeplob":            "deeplob_k10_logits.pt",
        "tcn":                "tcn_k10_logits.pt",
        "lob_transformer":    "lob_transformer_k10_logits.pt",
        "cnn_lstm_attention": "cnn_lstm_attention_k10_logits.pt",
    }
    n_bins = 15
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for ax_i, (m, fname) in enumerate(logit_files.items()):
        path = RESULTS / fname
        if not path.exists():
            print(f"  SKIP reliability {m} (no logit file)")
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False)
        logits = saved["logits"].float()
        labels = saved["labels"].long()
        probs  = F.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)
        correct = (preds == labels).float()

        bins = np.linspace(0, 1, n_bins + 1)
        bin_acc, bin_conf, bin_n = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confs.numpy() >= lo) & (confs.numpy() < hi)
            if mask.sum() > 0:
                bin_acc.append(correct.numpy()[mask].mean())
                bin_conf.append(confs.numpy()[mask].mean())
                bin_n.append(mask.sum())

        ax = axes[ax_i]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.bar(bin_conf, bin_acc, width=1/n_bins, alpha=0.6,
               color=MODEL_COLORS[m], label="Model", align="center")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability Diagram — {MODEL_LABELS[m]}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Reliability Diagrams — Before Temperature Scaling (k=10)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_reliability_diagrams.pdf")
    fig.savefig(FIGURES / "fig9_reliability_diagrams.png")
    plt.close()
    print("✓ fig9_reliability_diagrams")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)

    print(f"Saving figures to {FIGURES.resolve()}\n")
    fig_horizon_analysis()
    fig_model_comparison()
    fig_calibration_ece()
    fig_mc_entropy()
    fig_conformal_coverage()
    fig_selective_prediction()
    fig_regime_performance()
    fig_backtest()
    fig_reliability_diagrams()
    print(f"\nAll figures saved to {FIGURES.resolve()}")
