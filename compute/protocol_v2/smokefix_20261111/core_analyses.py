"""Table I aggregates, temperature scaling, LAC, selective prediction and the deep ensemble.

Usage: python core_analyses.py {orig|new}
  orig  reproduces the published JSONs from the original runs (regression check)
  new   recomputes with the four smoke-test runs replaced by their retrains
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score

from common import (ALL6, BASE, ci95, ece, fit_temperature, lac_eval, lac_qhat, load_logits,
                    run_dir)

METRICS = ["accuracy", "f1_macro", "f1_weighted", "kappa", "ece", "nll", "brier", "error_auroc"]


def agg(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "ci95": float(ci95(v)), "std": float(v.std()), "n": len(v)}


def baseline_table(new):
    out = {"all_folds": {}, "fold9": {}}
    for m in ALL6:
        rows = {(f, s): json.load(open(run_dir(m, f, s, new) / "summary.json"))
                for f in range(1, 10) for s in range(3)}
        for r in rows.values():
            assert r["config"]["epochs"] == 80, (m, r["run_id"], r["config"]["epochs"])
        out["all_folds"][m] = {k: agg([r["test"][k] for r in rows.values()]) for k in METRICS}
        out["fold9"][m] = {k: agg([rows[(9, s)]["test"][k] for s in range(3)]) for k in METRICS}
    return out


def ts_and_lac(new):
    ts, lac = {}, {}
    for m in BASE + ["lob_transformer_v2"]:
        per = []
        for s in range(3):
            zv, yv = load_logits(m, s, "validation", new)
            zt, yt = load_logits(m, s, "test", new)
            t = fit_temperature(zv, yv)
            pt = softmax(zt, axis=1)
            per.append({"T": t, "ece_raw": ece(pt, yt), "ece_scaled": ece(softmax(zt / t, axis=1), yt),
                        "acc": float((pt.argmax(1) == yt).mean())})
            if s == 0:
                q = lac_qhat(softmax(zv, axis=1), yv)
                lac[m] = {**lac_eval(pt, yt, q), "qhat": q}
        ts[m] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        ts[m]["per_seed"] = per
    return {"temp_scaling": ts, "lac": lac}


def selective(new):
    out = {}
    for m in BASE:
        z, y = load_logits(m, 0, "test", new)
        p = softmax(z, axis=1)
        order = np.argsort(-p.max(1), kind="stable")
        correct = (p.argmax(1) == y)[order]
        out[m] = {f"cov{c}": float(correct[: int(round(len(y) * c / 100))].mean()) for c in (100, 80, 60, 40)}
    return out


def ensemble(new):
    probs, y0 = [], None
    for m in BASE:
        for s in range(3):
            z, y = load_logits(m, s, "test", new)
            assert y0 is None or np.array_equal(y, y0)
            y0 = y
            probs.append(softmax(z, axis=1))
    p = np.mean(probs, axis=0)
    pred = p.argmax(1)
    return {"accuracy": float((pred == y0).mean()), "ece": ece(p, y0),
            "kappa": float(cohen_kappa_score(y0, pred)), "n_test": int(len(y0))}


if __name__ == "__main__":
    new = sys.argv[1] == "new"
    res = {"table2_baseline": baseline_table(new), "conformal_logit_only": ts_and_lac(new),
           "selective_prediction": selective(new), "deep_ensemble_4model_3seeds": ensemble(new)}
    out = Path(__file__).resolve().parent / f"core_{sys.argv[1]}.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)
