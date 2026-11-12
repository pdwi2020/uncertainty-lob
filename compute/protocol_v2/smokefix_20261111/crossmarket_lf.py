"""Cross-market zero-shot transfer and tiny-budget recalibration, fold-9 seed-0 source models.

Reuses the builders, inference and ECE of compute/protocol_v2/crossmarket_infer.py; the capped
temperature fit is the grid T in [0.5, 10] (step 0.25) recorded in crossmarket.json, and the
uncapped optimum is LBFGS on log T.

Usage: python crossmarket_lf.py {orig|new} [model ...] --flip
  --flip maps the target labels (2=up) to the FI-2010 order (0=up). Leave it out only to reproduce
  the earlier, label-swapped numbers.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "compute" / "protocol_v2"))
import crossmarket_infer as cm  # noqa: E402
from common import run_dir  # noqa: E402

GRID = np.arange(0.5, 10.0 + 1e-9, 0.25)
K_BUDGETS = [50, 200, 1000, 5000]


def grid_temperature(logits, labels):
    lg, ly = torch.from_numpy(logits).float(), torch.from_numpy(labels).long()
    losses = [F.cross_entropy(lg / t, ly).item() for t in GRID]
    return float(GRID[int(np.argmin(losses))])


def recalibrate(logits, labels):
    res = {"ece_zs": cm.ece_score(F.softmax(torch.from_numpy(logits).float(), -1).numpy(), labels)}
    for k in K_BUDGETS:
        t = grid_temperature(logits[:k], labels[:k])
        probs = F.softmax(torch.from_numpy(logits[k:] / t).float(), -1).numpy()
        res[f"K{k}"] = {"ece": cm.ece_score(probs, labels[k:]), "T": t}
    res["T_unbounded_K200"] = cm.fit_temperature(logits[:200], labels[:200])
    return res


FLIP = "--flip" in sys.argv


def main():
    sys.argv = [a for a in sys.argv if a != "--flip"]
    new = sys.argv[1] == "new"
    models = sys.argv[2:] or ["deeplob", "tcn"]
    out = {"markets": {}, "recalibration": {}}
    for mkt, fname in cm.MARKETS.items():
        d = np.load(cm.FEAT_DIR / fname, mmap_mode="r")
        x, y = np.asarray(d["X"], dtype=np.float32), np.asarray(d["Y"]).astype(np.int64)
        if FLIP:
            y = 2 - y   # target sets use 2=up/0=down; FI-2010 models use 0=up/2=down
        out["markets"][mkt] = {"n": int(len(y)), "stat_pct": float((y == 1).mean()), "models": {}}
        out["recalibration"][mkt] = {}
        for name in models:
            model = cm._build(name)
            ck = torch.load(run_dir(name, 9, 0, new) / "best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(ck.get("model", ck.get("model_state_dict", ck)))
            model.eval()
            with torch.no_grad():
                logits = cm.infer_logits(model, x).numpy()
            out["markets"][mkt]["models"][name] = cm.metrics_from_logits(logits, y)
            out["recalibration"][mkt][name] = recalibrate(logits, y)
            m = out["markets"][mkt]["models"][name]
            print(f"{mkt:5s} {name:20s} acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
                  f"kappa={m['kappa']:.4f} ece={m['ece']:.4f} | "
                  f"K200 ece={out['recalibration'][mkt][name]['K200']['ece']:.4f} "
                  f"Tunb={out['recalibration'][mkt][name]['T_unbounded_K200']:.4g}", flush=True)
    json.dump(out, open(Path(__file__).resolve().parent / f"crossmarket_{sys.argv[1]}{'_flip' if FLIP else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
