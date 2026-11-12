"""LAC per-regime coverage gap on the LEAKY concatenated-protocol validation logits (337,974),
under the old HMM labels and under the v_ratio calibration-median split, calib/eval = val halves."""
import math, numpy as np, torch
import torch.nn.functional as F
X9 = "/Volumes/Crucial X9/Research Projects/DL_Paper/results/"
f = np.load(X9 + "macc_features.npz")
def gap(probs, y, r):
    n = len(y) // 2
    s = 1 - probs[np.arange(n), y[:n]]
    q = float(np.quantile(s, min(math.ceil((n + 1) * 0.9) / n, 1), method="higher"))
    cov = ((1 - probs[n:]) <= q)[np.arange(len(y) - n), y[n:]]
    re = r[n:]
    return abs(cov[re == 0].mean() - cov[re == 1].mean()), np.bincount(re)
vr = f["m_val"][:, 2]; n = len(vr) // 2
rules = {"hmm (old)": f["r_val"].astype(int), "vr_median": (vr > np.median(vr[:n])).astype(int)}
for m in ["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention"]:
    try:
        d = torch.load(X9 + f"{m}_k10_logits.pt", map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(m, "missing"); continue
    p = F.softmax(d["logits"].float(), -1).numpy(); y = d["labels"].long().numpy()
    print(m, len(y), {k: (round(float(g), 4), c.tolist()) for k, (g, c) in ((k, gap(p, y, r)) for k, r in rules.items())})
