"""Cross-market zero-shot transfer with LEAKAGE-FREE FI-2010 source models.

Forward-passes the four anchored (leakage-free) fold-9 checkpoints --- DeepLOB,
TCN, LOBTransformer, CNN-LSTM-Attention --- on prebuilt crypto (Bybit
BTC/ETH/SOL) and equity (LOBSTER AAPL) feature windows, then:

  1. Zero-shot metrics per market x model: accuracy, F1-macro, Cohen kappa,
     15-bin ECE (matches Table `tab:domainshift`).
  2. Tiny-budget temperature recalibration: fit global T* on the first K target
     samples (K in {50,200,1000,5000}), evaluate ECE on the remainder
     (matches Table `tab:recalib`).

Pure post-hoc inference, CPU. The crypto/equity windows are already in the same
Z-scored input space the FI-2010 models were trained on (norm_mean/norm_std are
stored in each .npz for provenance).

Source checkpoints (fold 9, seed 0, micro=none):
  deeplob, cnn_lstm_attention, tcn  -> X9 protocol_v2_long campaign (Arti)
  lob_transformer                   -> /tmp/lobt_full (local MPS re-run)

Outputs:
  results/leakage_free/crossmarket.json   (paper numbers; tracked in repo)
  /tmp/dl_paper_crossmarket/<MKT>_<model>_logits.pt   (for optional conformal driver)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, f1_score

REPO = Path("/Users/paritoshdwivedi/uncertainty-lob")
sys.path.insert(0, str(REPO))

from src.models.deeplob import DeepLOB                         # noqa: E402
from src.models.tcn import TCN                                 # noqa: E402
from src.models.lob_transformer import LOBTransformer          # noqa: E402
from src.models.cnn_lstm_attention import CNNLSTMAttention     # noqa: E402

X9 = Path("/Volumes/Crucial X9/Research Projects/DL_Paper")
FEAT_DIR = X9 / "results"
CAMPAIGN = X9 / "experiments/new/DL_Paper_v2_complete/runs/protocol_v2_long"

OUT_JSON = REPO / "results/leakage_free/crossmarket.json"
LOGIT_DIR = Path("/tmp/dl_paper_crossmarket")
LOGIT_DIR.mkdir(parents=True, exist_ok=True)

# market label -> feature npz
MARKETS = {
    "BTC": "crypto_BTCUSDT_k10_features.npz",
    "ETH": "crypto_ETHUSDT_k10_features.npz",
    "SOL": "crypto_SOLUSDT_k10_features.npz",
    "AAPL": "lobster_AAPL_k10_features.npz",
}

# model name -> (builder, checkpoint path, state_dict key)
def _build(name):
    if name == "deeplob":
        return DeepLOB(input_channels=40, cnn_out_channels=96, lstm_hidden=64,
                       n_classes=3, dropout=0.2)
    if name == "tcn":
        return TCN(input_channels=40, num_channels=[64] * 6, kernel_size=3,
                   dropout=0.2, n_classes=3)
    if name == "lob_transformer":
        return LOBTransformer(input_dim=40, d_model=64, nhead=4,
                              num_encoder_layers=2, dim_feedforward=128,
                              dropout=0.2, n_classes=3)
    if name == "cnn_lstm_attention":
        return CNNLSTMAttention(input_channels=40, cnn_filters=[32, 64, 64],
                                cnn_kernel=3, lstm_hidden=64, lstm_layers=2,
                                dropout=0.2, n_classes=3, use_attention=True)
    raise ValueError(name)


CKPTS = {
    "deeplob": CAMPAIGN / "anchored_references__deeplob__f9__k10__s0__none/best.pt",
    "tcn": CAMPAIGN / "anchored_references__tcn__f9__k10__s0__none/best.pt",
    "cnn_lstm_attention": CAMPAIGN / "anchored_references__cnn_lstm_attention__f9__k10__s0__none/best.pt",
    "lob_transformer": Path("/tmp/lobt_full/anchored_references__lob_transformer__f9__k10__s0__none/best.pt"),
}
MODELS = ["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention"]
K_BUDGETS = [50, 200, 1000, 5000]


def load_model(name):
    model = _build(name)
    ck = torch.load(CKPTS[name], map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        sd = ck["model"]
    elif isinstance(ck, dict) and "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    else:
        sd = ck
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def infer_logits(model, X, batch=512):
    """X: (N,100,40) float32 -> logits (N,3)."""
    out = []
    xt = torch.from_numpy(np.ascontiguousarray(X)).float()
    for i in range(0, len(xt), batch):
        o = model(xt[i:i + batch])
        if isinstance(o, tuple):
            o = o[0]
        out.append(o.float())
    return torch.cat(out, 0)


def ece_score(probs, labels, n_bins=15):
    """15-bin expected calibration error on top-1 confidence."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits, labels, max_iter=200):
    """Fit scalar T>0 minimizing NLL on (logits, labels). Returns T*."""
    lg = torch.from_numpy(logits).float()
    ly = torch.from_numpy(labels).long()
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T), T0=1
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_T.exp(), ly)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().item())


def metrics_from_logits(logits, labels):
    probs = F.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()
    pred = probs.argmax(1)
    return {
        "accuracy": float((pred == labels).mean()),
        "f1_macro": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(labels, pred)),
        "ece": ece_score(probs, labels),
    }


def recalibrate(logits, labels):
    """Tiny-budget temperature recalibration. Fit T on first K, eval ECE on rest."""
    res = {"ece_zs": ece_score(F.softmax(torch.from_numpy(logits).float(), -1).numpy(), labels)}
    for K in K_BUDGETS:
        if K >= len(labels):
            continue
        T = fit_temperature(logits[:K], labels[:K])
        ev_logits, ev_labels = logits[K:], labels[K:]
        ev_probs = F.softmax(torch.from_numpy(ev_logits / T).float(), -1).numpy()
        res[f"K{K}"] = {"ece": ece_score(ev_probs, ev_labels), "T": round(T, 4)}
    return res


def main():
    summary = {
        "protocol": "leakage_free_anchored_fold9_seed0",
        "source_note": "FI-2010 anchored walk-forward fold-9 checkpoints; no concatenation leakage",
        "n_windows_per_market": 50000,
        "markets": {},
        "recalibration": {},
    }

    # cache models once
    models = {}
    for name in MODELS:
        if not CKPTS[name].exists():
            print(f"[WARN] checkpoint missing for {name}: {CKPTS[name]}")
            continue
        models[name] = load_model(name)
        print(f"[OK] loaded {name}")

    for mkt, fname in MARKETS.items():
        d = np.load(FEAT_DIR / fname, mmap_mode="r")
        X = np.asarray(d["X"], dtype=np.float32)
        Y = np.asarray(d["Y"]).astype(np.int64)
        stat_pct = float((Y == 1).mean())
        summary["markets"][mkt] = {"n": int(len(Y)), "stat_pct": round(stat_pct, 4), "models": {}}
        summary["recalibration"][mkt] = {}
        print(f"\n### {mkt}  N={len(Y):,}  stat%={stat_pct:.3f}")

        for name in MODELS:
            if name not in models:
                continue
            logits = infer_logits(models[name], X).numpy()
            torch.save({"logits": torch.from_numpy(logits), "labels": torch.from_numpy(Y)},
                       LOGIT_DIR / f"{mkt}_{name}_logits.pt")
            m = metrics_from_logits(logits, Y)
            summary["markets"][mkt]["models"][name] = {k: round(v, 4) for k, v in m.items()}
            rc = recalibrate(logits, Y)
            summary["recalibration"][mkt][name] = rc
            kbest = rc.get("K200", {})
            print(f"  {name:20s} acc={m['accuracy']:.3f} F1={m['f1_macro']:.3f} "
                  f"kappa={m['kappa']:.3f} ECE={m['ece']:.3f} | "
                  f"ECE-zs={rc['ece_zs']:.3f} ->K200={kbest.get('ece', float('nan')):.3f} "
                  f"(T={kbest.get('T', float('nan'))})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")
    print("CROSSMARKET_INFER_DONE")


if __name__ == "__main__":
    main()
