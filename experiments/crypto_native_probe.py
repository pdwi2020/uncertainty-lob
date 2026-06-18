"""Phase 3 (go/no-go) -- Native-trained crypto LOB feasibility probe.

The paper's existing crypto results are ZERO-SHOT (FI-2010 models applied to crypto),
which is near-chance. This probe asks the different question: if we TRAIN a model
*natively* on crypto LOB, does it beat chance meaningfully? If yes, a genuine
second-market UQ validation is worth running; if near-chance, we document an honest
negative and rely on FI-2010 + the leave-one-stock-out generalization result.

Data (already on disk, FI-2010 40-feature schema, 3-class vol-normalized labels):
  results/crypto_{BTCUSDT,ETHUSDT,SOLUSDT}_k10_features.npz  -> X (50000,100,40), Y (50000,)

Design: pool the three symbols; per-symbol temporal 80/20 split (no leakage);
per-channel standardization on TRAIN; compact dilated-TCN; ~12 epochs.

GO if test accuracy > 0.42 AND Cohen's kappa > 0.08 on the held-out split
(meaningfully above the 33.3% / kappa=0 chance floor). Saves native test logits.

Output:
  results/crypto_native_probe.json
  results/crypto_native_pooled_tcn_test_logits.pt   (logits, labels, symbol)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/Volumes/Crucial X9/Research Projects/DL_Paper")
RESULTS = ROOT / "results"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
PER_SYM = 30_000          # subsample per symbol for a fast probe
TRAIN_FRAC = 0.8
EPOCHS = 12
BATCH = 256
SEED = 42


def cohen_kappa(y_true, y_pred, n_cls=3):
    po = float((y_true == y_pred).mean())
    pe = 0.0
    n = len(y_true)
    for c in range(n_cls):
        pe += (np.mean(y_true == c) * np.mean(y_pred == c))
    return (po - pe) / (1 - pe + 1e-12)


def macro_f1(y_true, y_pred, n_cls=3):
    f1s = []
    for c in range(n_cls):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(2 * prec * rec / (prec + rec + 1e-12))
    return float(np.mean(f1s))


class TinyTCN(nn.Module):
    def __init__(self, in_ch=40, hid=64, n_cls=3):
        super().__init__()

        def block(ci, co, d):
            return nn.Sequential(
                nn.Conv1d(ci, co, 3, padding=d, dilation=d),
                nn.BatchNorm1d(co), nn.ReLU(),
            )

        self.net = nn.Sequential(block(in_ch, hid, 1), block(hid, hid, 2),
                                 block(hid, hid, 4))
        self.head = nn.Linear(hid, n_cls)

    def forward(self, x):            # (B,100,40)
        h = self.net(x.transpose(1, 2)).mean(dim=2)
        return self.head(h)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")

    Xtr, Ytr, Xte, Yte, Ste = [], [], [], [], []
    for sym in SYMBOLS:
        fp = RESULTS / f"crypto_{sym}_k10_features.npz"
        if not fp.exists():
            print(f"[SKIP] {sym} missing"); continue
        d = np.load(fp)
        X = d["X"][:PER_SYM].astype(np.float32)
        Y = d["Y"][:PER_SYM].astype(np.int64)
        n_tr = int(len(Y) * TRAIN_FRAC)          # temporal split (no leakage)
        Xtr.append(X[:n_tr]); Ytr.append(Y[:n_tr])
        Xte.append(X[n_tr:]); Yte.append(Y[n_tr:])
        Ste.append(np.full(len(Y) - n_tr, sym))
        cls, cnt = np.unique(Y, return_counts=True)
        print(f"  {sym}: {len(Y):,} windows  class dist={dict(zip(cls.tolist(), cnt.tolist()))}")

    Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)
    Xte = np.concatenate(Xte); Yte = np.concatenate(Yte); Ste = np.concatenate(Ste)

    # per-channel standardization on TRAIN
    mu = Xtr.reshape(-1, 40).mean(0).astype(np.float32)
    sd = Xtr.reshape(-1, 40).std(0).astype(np.float32) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    print(f"train={len(Ytr):,}  test={len(Yte):,}  chance acc=33.3%")

    model = TinyTCN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    Xtr_t = torch.from_numpy(Xtr); Ytr_t = torch.from_numpy(Ytr)
    n = len(Ytr_t)

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr_t[idx].to(dev); yb = Ytr_t[idx].to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        # quick test eval
        model.eval()
        with torch.no_grad():
            logits = []
            for i in range(0, len(Yte), 1024):
                xb = torch.from_numpy(Xte[i:i + 1024]).to(dev)
                logits.append(model(xb).cpu())
            logits = torch.cat(logits)
            pred = logits.argmax(1).numpy()
        acc = float((pred == Yte).mean())
        print(f"  ep {ep+1:2d}  train_loss={tot/n:.4f}  test_acc={acc:.4f}")

    kap = cohen_kappa(Yte, pred); f1 = macro_f1(Yte, pred)
    # per-symbol breakdown
    per_sym = {}
    for sym in SYMBOLS:
        m = Ste == sym
        if m.sum():
            per_sym[sym] = {
                "n": int(m.sum()),
                "acc": round(float((pred[m] == Yte[m]).mean()), 4),
                "kappa": round(cohen_kappa(Yte[m], pred[m]), 4),
            }

    go = (acc > 0.42) and (kap > 0.08)
    out = {
        "device": dev, "train_n": int(len(Ytr)), "test_n": int(len(Yte)),
        "epochs": EPOCHS, "test_acc": round(acc, 4), "test_kappa": round(kap, 4),
        "test_macro_f1": round(f1, 4), "chance_acc": 0.3333,
        "per_symbol": per_sym,
        "decision": "GO" if go else "NO-GO",
        "criterion": "acc>0.42 and kappa>0.08",
    }
    with open(RESULTS / "crypto_native_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    torch.save({"logits": logits, "labels": torch.from_numpy(Yte),
                "symbol": Ste.tolist()},
               RESULTS / "crypto_native_pooled_tcn_test_logits.pt")

    print(f"\nPOOLED  acc={acc:.4f}  kappa={kap:.4f}  macroF1={f1:.4f}")
    for s, v in per_sym.items():
        print(f"  {s}: acc={v['acc']:.4f}  kappa={v['kappa']:.4f}  (n={v['n']:,})")
    print(f"\nDECISION: {out['decision']}  ({out['criterion']})")
    print("CRYPTO_NATIVE_PROBE_DONE")


if __name__ == "__main__":
    main()
