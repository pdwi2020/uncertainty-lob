"""Regime-adaptive temperature ladder (seed 0, fold-9 validation, temporal halves, 10-bin ECE)
and regime-conditional accuracy (fold-9 test, median split of 3,600-tick rolling volatility of
mid-price changes, three seeds).

Usage: python regime_lf.py {orig|new}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score

from common import BASE, ece, fit_temperature, load_logits
from fold9_features import FI, SEQ, load


def regime_ts(new):
    out = {}
    for m in BASE:
        z, y = load_logits(m, 0, "validation", new)
        h = len(y) // 2
        t, tl, th = fit_temperature(z, y), fit_temperature(z[:h], y[:h]), fit_temperature(z[h:], y[h:])
        pr = np.vstack([softmax(z[:h] / tl, axis=1), softmax(z[h:] / th, axis=1)])
        out[m] = {"T_global": t, "T_low": tl, "T_high": th, "ECE_raw": ece(softmax(z, axis=1), y, 10),
                  "ECE_gT": ece(softmax(z / t, axis=1), y, 10), "ECE_rT": ece(pr, y, 10), "n_val": int(len(y))}
    return out


def regime_conditional(new):
    feat, _ = load(FI / "NoAuction_Zscore_Testing" / "Test_Dst_NoAuction_ZScore_CF_9.txt")
    mid = (feat[:, 0] + feat[:, 2]) / 2
    rv = pd.Series(np.diff(mid, prepend=mid[0])).rolling(3600, min_periods=1).std().to_numpy()[SEQ - 1:]
    high = rv > np.median(rv)
    out = {}
    for m in BASE:
        acc, f1 = {0: [], 1: []}, {0: [], 1: []}
        for s in range(3):
            z, y = load_logits(m, s, "test", new)
            pred = z.argmax(1)
            for r, mask in ((0, ~high), (1, high)):
                acc[r].append((pred[mask] == y[mask]).mean())
                f1[r].append(f1_score(y[mask], pred[mask], average="macro"))
        out[m] = {name: {"acc_mean": float(np.mean(acc[r])), "acc_std": float(np.std(acc[r])),
                         "f1_macro_mean": float(np.mean(f1[r])), "f1_macro_std": float(np.std(f1[r])),
                         "n_samples": int((high if r else ~high).sum())}
                  for r, name in ((0, "low_vol"), (1, "high_vol"))}
    return out


if __name__ == "__main__":
    new = sys.argv[1] == "new"
    res = {"regime_ts": regime_ts(new), "regime_conditional": regime_conditional(new)}
    (Path(__file__).resolve().parent / f"regime_{sys.argv[1]}.json").write_text(json.dumps(res, indent=1))
    for m in BASE:
        r, c = res["regime_ts"][m], res["regime_conditional"][m]
        print(f"{m:20s} T={r['T_global']:.3f} Tl={r['T_low']:.3f} Th={r['T_high']:.3f} ECE raw/gT/rT="
              f"{r['ECE_raw']:.4f}/{r['ECE_gT']:.4f}/{r['ECE_rT']:.4f} | acc low/high="
              f"{c['low_vol']['acc_mean']:.4f}/{c['high_vol']['acc_mean']:.4f}")
