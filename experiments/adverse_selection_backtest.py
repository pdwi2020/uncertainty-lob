"""E4 — Conformal adverse-selection filter: backtest ladder.

MACC-LOB Component 4: gated trading strategy using conformal prediction sets
and OFI-sign agreement as a microstructure toxicity filter.

Ladder evaluated on the same eval split as E2a (val 50/50 calib→eval):
  1. raw          — trade whenever argmax ≠ stationary (label 1)
  2. confidence   — + require max softmax ≥ 0.60
  3. conformal    — + require |C(x)| = 1 (singleton set, toxicity_closed q̂/λ from E2a)
  4. ofi_gate     — + require sign(OFI_l1) agrees with predicted direction  ← PROPOSED

Labels: FI-2010 order, 0=up, 1=stationary, 2=down.
Directions: up prediction → long (OFI_l1 > 0 agrees); down prediction → short (OFI_l1 < 0 agrees).

Metrics per filter per model:
  trade_coverage:    fraction of eval samples with a trade
  hit_rate:          fraction of trades that are correct
  pnl_per_trade:     mean binary PnL (+1 correct, -1 wrong)
  sharpe:            annualized Sharpe (252 days, 16,000 steps/day)
  adverse_sel_rate:  P(wrong | ψ > ψ_median, trade emitted)
  high_vol_trade_frac: fraction of emitted trades in high-vol regime

Claim: the OFI gate reduces adverse-selection rate and improves risk-adjusted metrics
without claiming profitability.

Split convention: val 50/50 (calib → eval), matching E2a.
  With --use-test: val = calib, test = eval (requires test logits + macc_features r_test/psi_test).

Saves: results/adverse_selection_results.json

Usage:
  python experiments/adverse_selection_backtest.py
  python experiments/adverse_selection_backtest.py --alpha 0.10 --use-test
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

SAMPLES_PER_DAY = 16_000   # FI-2010 active trading period estimate (annualization anchor)
CONF_THRESHOLD  = 0.60     # softmax confidence threshold for filter 2
STAT_CLASS      = 1        # label for "stationary" (no trade)
UP_CLASS        = 0        # FI-2010: raw label 1 = up   -> class 0
DOWN_CLASS      = 2        # FI-2010: raw label 3 = down -> class 2
OFI_L1_IDX      = 3        # column index of ofi_l1 in 7-dim microstructure vector


# ── Conformal utilities (mirrors E2a) ─────────────────────────────────────────

def get_q_hat(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(scores, min(level, 1.0), method="higher"))


def prediction_sets_toxicity(
    probs: np.ndarray,
    psi: np.ndarray,
    q_hat: float,
    lam: float,
) -> np.ndarray:
    """Bool (N, K): (1 − p̂_k + λ·ψ) ≤ q̂."""
    return (1.0 - probs + lam * psi[:, None]) <= q_hat


# ── Per-trade PnL + metrics ────────────────────────────────────────────────────

def trade_returns(signal: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Binary return at each timestep: +1 correct, -1 wrong, 0 no trade.

    signal: array of predicted class (0 or 2) where trade emitted, -1 = no trade.
    """
    ret = np.zeros(len(labels), dtype=np.float32)
    mask = signal >= 0
    ret[mask] = np.where(signal[mask] == labels[mask], 1.0, -1.0)
    return ret


def annualized_sharpe(returns: np.ndarray, samples_per_day: int = SAMPLES_PER_DAY) -> float:
    """Annualized Sharpe from per-step return series (including 0s for no-trade steps)."""
    if returns.std() < 1e-8:
        return 0.0
    steps_per_year = 252 * samples_per_day
    return float(returns.mean() / returns.std() * math.sqrt(steps_per_year))


def compute_metrics(
    signal: np.ndarray,
    labels: np.ndarray,
    psi: np.ndarray,
    regimes: np.ndarray,
) -> dict:
    """Compute all backtest metrics for a given signal vector."""
    N = len(labels)
    trade_mask = signal >= 0

    if trade_mask.sum() == 0:
        return {
            "trade_coverage":     0.0,
            "hit_rate":           float("nan"),
            "pnl_per_trade":      float("nan"),
            "sharpe":             0.0,
            "adverse_sel_rate":   float("nan"),
            "high_vol_trade_frac": float("nan"),
            "n_trades":           0,
        }

    ret = trade_returns(signal, labels)
    n_trades = int(trade_mask.sum())

    hit_rate = float((ret[trade_mask] == 1).mean())
    pnl_per_trade = float(ret[trade_mask].mean())
    sharpe = annualized_sharpe(ret)

    # Adverse selection rate: P(wrong | ψ > median, trade)
    psi_med = float(np.median(psi))
    adv_mask = trade_mask & (psi > psi_med)
    adverse_sel_rate = float(
        (ret[adv_mask] == -1).mean()
    ) if adv_mask.sum() > 0 else float("nan")

    # High-vol trade fraction
    if regimes is not None:
        high_mask = trade_mask & (regimes == 1)
        high_vol_frac = float(high_mask.sum() / n_trades)
    else:
        high_vol_frac = float("nan")

    return {
        "trade_coverage":      round(float(trade_mask.mean()), 4),
        "hit_rate":            round(hit_rate, 4),
        "pnl_per_trade":       round(pnl_per_trade, 4),
        "sharpe":              round(sharpe, 4),
        "adverse_sel_rate":    round(adverse_sel_rate, 4),
        "high_vol_trade_frac": round(high_vol_frac, 4),
        "n_trades":            n_trades,
    }


# ── Filter implementations ────────────────────────────────────────────────────

def filter_raw(probs: np.ndarray) -> np.ndarray:
    """Trade whenever argmax ≠ stationary."""
    preds = probs.argmax(axis=1)
    signal = np.where(preds != STAT_CLASS, preds, -1)
    return signal


def filter_confidence(probs: np.ndarray, threshold: float = CONF_THRESHOLD) -> np.ndarray:
    """Trade when argmax ≠ stationary AND max softmax ≥ threshold."""
    preds = probs.argmax(axis=1)
    max_p = probs.max(axis=1)
    signal = np.where((preds != STAT_CLASS) & (max_p >= threshold), preds, -1)
    return signal


def filter_conformal(
    probs: np.ndarray,
    psi: np.ndarray,
    q_hat: float,
    lam: float,
) -> np.ndarray:
    """Trade when |C(x)| = 1 AND singleton class ≠ stationary."""
    sets = prediction_sets_toxicity(probs, psi, q_hat, lam)
    set_sizes = sets.sum(axis=1)
    # Singleton: exactly one class included
    singleton_mask = set_sizes == 1
    # Which class is the singleton?
    singleton_class = np.where(singleton_mask, sets.argmax(axis=1), -1)
    # Reject stationary singletons
    signal = np.where(
        singleton_mask & (singleton_class != STAT_CLASS),
        singleton_class,
        -1,
    )
    return signal


def filter_ofi_gate(
    probs: np.ndarray,
    psi: np.ndarray,
    m: np.ndarray,
    q_hat: float,
    lam: float,
) -> np.ndarray:
    """Conformal singleton PLUS sign(OFI_l1) agrees with predicted direction."""
    base_signal = filter_conformal(probs, psi, q_hat, lam)
    ofi_l1 = m[:, OFI_L1_IDX]

    # OFI agreement: an up prediction needs bid-heavy flow (ofi_l1 > 0),
    #                a down prediction needs ask-heavy flow (ofi_l1 < 0)
    up_agree   = (base_signal == UP_CLASS) & (ofi_l1 > 0)
    down_agree = (base_signal == DOWN_CLASS) & (ofi_l1 < 0)
    signal = np.where(up_agree | down_agree, base_signal, -1)
    return signal


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",    default=str(RESULTS))
    parser.add_argument("--alpha",          type=float, default=0.10)
    parser.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD,
                        help="Max softmax threshold for filter 2 (default 0.60)")
    parser.add_argument("--use-test",       action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    alpha = args.alpha
    conf_thr = args.conf_threshold
    print(f"Target coverage: {1-alpha:.0%}  α={alpha}  conf_threshold={conf_thr}")

    # Load MACC features
    macc    = np.load(results_dir / "macc_features.npz")
    psi_val = macc["psi_val"].astype(np.float32)
    m_val   = macc["m_val"].astype(np.float32)
    r_val   = macc["r_val"].astype(np.int64)

    # Load E2a conformal results (q̂ and λ* per model)
    conformal_path = results_dir / "micro_conformal_results.json"
    if not conformal_path.exists():
        print("ERROR: micro_conformal_results.json missing — run E2a first")
        sys.exit(1)
    with open(conformal_path) as f:
        conformal_results = json.load(f)

    model_names = ["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention"]
    all_results = {}

    for model_name in model_names:
        if model_name not in conformal_results:
            print(f"[SKIP] {model_name}: not in conformal results")
            continue

        if args.use_test:
            ckpt_val  = results_dir / f"{model_name}_k10_logits.pt"
            ckpt_test = results_dir / f"{model_name}_k10_test_logits.pt"
            if not ckpt_test.exists():
                print(f"[SKIP] {model_name}: test logits missing")
                continue
            val_d  = torch.load(ckpt_val,  map_location="cpu", weights_only=False)
            test_d = torch.load(ckpt_test, map_location="cpu", weights_only=False)
            calib_logits = val_d["logits"].float()
            calib_labels = val_d["labels"].long().numpy()
            eval_logits  = test_d["logits"].float()
            eval_labels  = test_d["labels"].long().numpy()
            calib_probs  = F.softmax(calib_logits, dim=-1).numpy()
            eval_probs   = F.softmax(eval_logits,  dim=-1).numpy()
            calib_psi    = psi_val
            eval_psi     = macc["psi_test"].astype(np.float32) if "psi_test" in macc else np.zeros(len(eval_labels), dtype=np.float32)
            eval_m       = macc["m_test"].astype(np.float32)   if "m_test"  in macc else np.zeros((len(eval_labels), 7), dtype=np.float32)
            eval_reg     = macc["r_test"].astype(np.int64)     if "r_test"  in macc else None
            calib_labels_arr = calib_labels
            split = "val→test"
        else:
            ckpt = results_dir / f"{model_name}_k10_logits.pt"
            if not ckpt.exists():
                print(f"[SKIP] {model_name}: {ckpt.name} missing")
                continue
            d      = torch.load(ckpt, map_location="cpu", weights_only=False)
            logits = d["logits"].float()
            labels = d["labels"].long().numpy()
            probs  = F.softmax(logits, dim=-1).numpy()
            n_c    = len(labels) // 2
            calib_probs, eval_probs    = probs[:n_c], probs[n_c:]
            calib_labels_arr, eval_labels = labels[:n_c], labels[n_c:]
            calib_psi, eval_psi        = psi_val[:n_c], psi_val[n_c:]
            _calib_m, eval_m           = m_val[:n_c],   m_val[n_c:]
            _calib_reg, eval_reg       = r_val[:n_c],   r_val[n_c:]
            split = "val_calib→val_eval"

        print(f"\n── {model_name} ({split}) ──")
        print(f"  Eval: {len(eval_labels):,}  ψ_median={np.median(eval_psi):.4f}")

        # ── Reconstruct conformal threshold from E2a ──────────────────────────
        model_conf = conformal_results[model_name]
        lam_star   = float(model_conf.get("lambda_star", 0.0))

        tc_entry = model_conf["methods"].get("toxicity_closed", {})
        if tc_entry:
            q_hat = float(tc_entry.get("q_hat", 0.9))
        else:
            # Fallback: recompute from calib (same as E2a logic)
            from micro_conformal import lac_scores, toxicity_scores, get_q_hat as gq
            calib_scores = toxicity_scores(calib_probs, calib_labels_arr, calib_psi, lam_star)
            q_hat = gq(calib_scores, alpha)

        print(f"  λ*={lam_star:.4f}  q̂={q_hat:.4f}")

        # ── Run ladder ────────────────────────────────────────────────────────
        filters = {
            "raw":        filter_raw(eval_probs),
            "confidence": filter_confidence(eval_probs, conf_thr),
            "conformal":  filter_conformal(eval_probs, eval_psi, q_hat, lam_star),
            "ofi_gate":   filter_ofi_gate(eval_probs, eval_psi, eval_m, q_hat, lam_star),
        }

        model_res = {
            "split": split, "alpha": alpha,
            "lambda_star": lam_star, "q_hat": round(q_hat, 4),
            "conf_threshold": conf_thr,
            "filters": {},
        }

        print(f"  {'Filter':<14} {'cov':>7} {'hit':>7} {'pnl/trade':>10} {'sharpe':>8} {'adv_sel':>8} {'n_trades':>9}")
        print(f"  {'-'*67}")

        for fname, sig in filters.items():
            stats = compute_metrics(sig, eval_labels, eval_psi, eval_reg)
            model_res["filters"][fname] = stats
            print(
                f"  {fname:<14}"
                f" {stats['trade_coverage']:>7.4f}"
                f" {stats['hit_rate']:>7.4f}"
                f" {stats['pnl_per_trade']:>10.4f}"
                f" {stats['sharpe']:>8.4f}"
                f" {stats['adverse_sel_rate']:>8.4f}"
                f" {stats['n_trades']:>9,}"
            )

        # Delta analysis (ofi_gate vs raw)
        raw_stats = model_res["filters"]["raw"]
        ofi_stats = model_res["filters"]["ofi_gate"]
        if raw_stats["n_trades"] > 0 and ofi_stats["n_trades"] > 0:
            hit_delta  = ofi_stats["hit_rate"] - raw_stats["hit_rate"]
            adv_delta  = ofi_stats["adverse_sel_rate"] - raw_stats["adverse_sel_rate"]
            sharpe_delta = ofi_stats["sharpe"] - raw_stats["sharpe"]
            print(f"  ofi_gate vs raw: Δhit={hit_delta:+.4f}  Δadv_sel={adv_delta:+.4f}  ΔSharpe={sharpe_delta:+.4f}")
            model_res["ofi_vs_raw_delta"] = {
                "delta_hit_rate":       round(hit_delta, 4),
                "delta_adverse_sel":    round(adv_delta, 4),
                "delta_sharpe":         round(sharpe_delta, 4),
                "coverage_reduction":   round(raw_stats["trade_coverage"] - ofi_stats["trade_coverage"], 4),
            }

        all_results[model_name] = model_res

    out_path = results_dir / "adverse_selection_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path.name}")

    # Summary table across models
    print("\n── Summary: ofi_gate vs raw (Δadverse-sel) ──")
    for mn, res in all_results.items():
        if "ofi_vs_raw_delta" in res:
            d = res["ofi_vs_raw_delta"]
            print(f"  {mn:<26}: Δhit={d['delta_hit_rate']:+.4f}  Δadv={d['delta_adverse_sel']:+.4f}  ΔSharpe={d['delta_sharpe']:+.4f}")


if __name__ == "__main__":
    main()
