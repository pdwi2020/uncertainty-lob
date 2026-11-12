"""Leakage-free ACI / majority-vote AgACI on the fold-9 test stream, per seed (as in aci.json).
Reuses the update rules of experiments/aci_recalibration.py; calibration = fold-9 validation logits."""
import json, sys
import numpy as np, torch
import torch.nn.functional as F
from common import run_dir
ALPHA, ETAS = 0.10, [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

def probs(model, seed, split, new):
    d = torch.load(run_dir(model, 9, seed, new) / f"{split}_logits.pt", map_location="cpu", weights_only=False)
    return F.softmax(d["logits"].float(), dim=-1).numpy(), d["labels"].long().numpy()

def lac_q(scores, alpha):
    n = len(scores); return float(np.quantile(scores, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)))

def sets_stats(sets, y):
    return float(sets[np.arange(len(y)), y].mean()), float(sets.sum(1).mean()), float((sets.sum(1) == 1).mean())

def aci(p, y, q0, eta):
    q, out = q0, np.zeros_like(p, dtype=bool)
    for t in range(len(y)):
        out[t] = (1.0 - p[t]) <= q
        q = float(np.clip(q + eta * ((0.0 if out[t][y[t]] else 1.0) - ALPHA), 0.0, 1.0))
    return sets_stats(out, y)

def agaci(p, y, q0):
    qs, out = [q0] * len(ETAS), np.zeros_like(p, dtype=bool)
    for t in range(len(y)):
        votes = sum(((1.0 - p[t]) <= q).astype(float) for q in qs)
        out[t] = votes >= len(ETAS) / 2.0
        for k, eta in enumerate(ETAS):
            qs[k] = float(np.clip(qs[k] + eta * ((0.0 if (1.0 - p[t][y[t]]) <= qs[k] else 1.0) - ALPHA), 0.0, 1.0))
    return sets_stats(out, y)

def run(model, new):
    res = {"standard_lac": [], "aci_eta005": [], "agaci_K6": []}
    for s in range(3):
        pv, yv = probs(model, s, "validation", new); pt, yt = probs(model, s, "test", new)
        q = lac_q(1.0 - pv[np.arange(len(yv)), yv], ALPHA)
        res["standard_lac"].append(sets_stats((1.0 - pt) <= q, yt))
        res["aci_eta005"].append(aci(pt, yt, q, 0.05))
        res["agaci_K6"].append(agaci(pt, yt, q))
    return {k: {"coverage_per_seed": [v[0] for v in vals], "coverage_mean": float(np.mean([v[0] for v in vals])),
                "mean_set_size_mean": float(np.mean([v[1] for v in vals])),
                "singleton_rate_mean": float(np.mean([v[2] for v in vals]))} for k, vals in res.items()}

if __name__ == "__main__":
    new = sys.argv[1] == "new"; models = sys.argv[2:] or ["deeplob", "tcn", "lob_transformer_v2", "cnn_lstm_attention"]
    out = {m: run(m, new) for m in models}
    print(json.dumps({m: {k: [round(c, 4) for c in v["coverage_per_seed"]] for k, v in r.items()} for m, r in out.items()}))
    json.dump(out, open(f"aci_{sys.argv[1]}.json", "w"), indent=1)
