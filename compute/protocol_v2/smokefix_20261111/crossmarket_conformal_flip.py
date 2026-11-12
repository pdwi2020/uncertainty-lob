"""Tiny-budget split conformal (LAC, alpha=0.10) on each target market: q-hat from the first
n_cal target windows, coverage and mean set size on the remainder. Fold-9 seed-0 source models
(TCN retrained). Same inference path as crossmarket_lf.py."""
import json, math, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "compute" / "protocol_v2"))
import crossmarket_infer as cm
from common import run_dir

out = {}
for name in ["deeplob", "tcn"]:
    model = cm._build(name)
    ck = torch.load(run_dir(name, 9, 0, True) / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck.get("model_state_dict", ck))); model.eval()
    for mkt, fname in cm.MARKETS.items():
        d = np.load(cm.FEAT_DIR / fname, mmap_mode="r")
        x, y = np.asarray(d["X"], dtype=np.float32), np.asarray(d["Y"]).astype(np.int64)
        y = 2 - y   # map target labels to the FI-2010 convention (0=up, 2=down)
        with torch.no_grad():
            p = F.softmax(cm.infer_logits(model, x), -1).numpy().astype(np.float64)
        for n in (200, 1000):
            s = 1 - p[np.arange(n), y[:n]]
            q = float(np.quantile(s, min(1, math.ceil((n + 1) * 0.9) / n), method="higher"))
            sets = (1 - p[n:]) <= q
            cov = float(sets[np.arange(len(y) - n), y[n:]].mean()); size = float(sets.sum(1).mean())
            out[f"{mkt}_{name}_n{n}"] = {"coverage": cov, "mean_size": size, "qhat": q}
            print(f"{mkt:5s} {name:8s} n_cal={n:5d} coverage={cov:.4f} size={size:.3f}", flush=True)
json.dump(out, open(Path(__file__).resolve().parent / "crossmarket_conformal_new_flip.json", "w"), indent=1)
