"""
LOBTransformer v2 — self-contained RunPod training script.
d_model=128, n_heads=8, n_layers=2, pre-LN, warmup+cosine LR.

Expects:
  - /workspace/data/fi-2010/   (downloaded by setup.sh)
  - scikit-learn installed (pip install scikit-learn -q)
Saves:
  - /workspace/lob_transformer_v2_results.json
  - /workspace/lob_transformer_v2_k10.pt
  - /workspace/lob_transformer_v2_k10_logits.pt
"""

import os, sys, json, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report

# ─────────────────────────────── Config ──────────────────────────────────────
HORIZON       = 10
BATCH_SIZE    = 64
D_MODEL       = 128
N_HEADS       = 8
N_LAYERS      = 2
DIM_FF        = 512
DROPOUT       = 0.1
T             = 100
LR            = 1e-4
WARMUP_EPOCHS = 5
EPOCHS        = 100
PATIENCE      = 15
SEED          = 42
DATA_ROOT     = Path("/workspace/data/fi-2010")
OUTPUT_DIR    = Path("/workspace")

# ─────────────────────────────── Data ────────────────────────────────────────
_HORIZON_IDX = {1: 0, 2: 1, 3: 2, 5: 3, 10: 4}
_LABEL_START = 144
_N_FEAT      = 40

def _load_files(paths):
    feats, labs = [], []
    for p in sorted(paths):
        npy = Path(str(p).replace(".txt", ".npy"))
        if npy.exists():
            data = np.load(str(npy))
        else:
            print(f"  Loading {p.name} via np.loadtxt ...")
            data = np.loadtxt(str(p), dtype=np.float32)
            np.save(str(npy), data)
            print(f"  Cached → {npy.name}")
        feats.append(data[:_N_FEAT].T)
        labs.append(data[_LABEL_START:_LABEL_START + 5].T)
    return np.concatenate(feats), np.concatenate(labs)

def _find_files(zscore_root, split):
    sub    = "NoAuction_Zscore_Training" if split == "train" else "NoAuction_Zscore_Testing"
    prefix = "Train_Dst" if split == "train" else "Test_Dst"
    files  = sorted((zscore_root / sub).glob(f"{prefix}_NoAuction_ZScore_CF_*.txt"))
    files  = [f for f in files if not f.name.startswith("._")]
    if not files:
        raise FileNotFoundError(f"No FI-2010 files in {zscore_root / sub}")
    return files

class LOBDataset(Dataset):
    def __init__(self, feat, lab, horizon=10, seq_len=100):
        self.feat = feat.astype(np.float32)
        self.lab  = (lab[:, _HORIZON_IDX[horizon]] - 1).astype(np.int64)
        self.T    = seq_len

    def __len__(self):
        return len(self.feat) - self.T

    def __getitem__(self, i):
        return torch.from_numpy(self.feat[i:i + self.T]), torch.tensor(self.lab[i + self.T - 1])

def make_loaders():
    zscore   = DATA_ROOT / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore"
    tr_f, tr_l = _load_files(_find_files(zscore, "train"))
    te_f, te_l = _load_files(_find_files(zscore, "test"))
    split    = int(len(tr_f) * 0.80)
    kw       = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    train_dl = DataLoader(LOBDataset(tr_f[:split], tr_l[:split], HORIZON, T), shuffle=True,  **kw)
    val_dl   = DataLoader(LOBDataset(tr_f[split:], tr_l[split:], HORIZON, T), shuffle=False, **kw)
    test_dl  = DataLoader(LOBDataset(te_f,         te_l,         HORIZON, T), shuffle=False, **kw)
    print(f"Train: {len(train_dl.dataset):,}  Val: {len(val_dl.dataset):,}  Test: {len(test_dl.dataset):,}")
    return train_dl, val_dl, test_dl

# ─────────────────────────────── Model ───────────────────────────────────────
class LOBTransformerV2(nn.Module):
    def __init__(self, input_dim=40, d_model=128, n_heads=8,
                 n_layers=2, dim_ff=512, dropout=0.1, n_classes=3, seq_len=100):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb    = nn.Embedding(seq_len, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):                               # x: (B, T, 40)
        B, S, _ = x.shape
        pos = torch.arange(S, device=x.device)
        x   = self.input_proj(x) + self.pos_emb(pos)   # (B, T, d_model)
        x   = self.encoder(x)                           # (B, T, d_model)
        x   = self.norm(x[:, -1])                       # last timestep
        return self.head(self.drop(x))

# ─────────────────────────────── LR Schedule ─────────────────────────────────
def make_scheduler(opt, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# ─────────────────────────────── Training ────────────────────────────────────
def train(model, train_dl, val_dl, device, ckpt):
    model  = model.to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    total_steps  = EPOCHS * len(train_dl)
    warmup_steps = WARMUP_EPOCHS * len(train_dl)
    sched  = make_scheduler(opt, warmup_steps, total_steps)
    crit   = nn.CrossEntropyLoss()
    best_val, patience_count, best_meta, step = float("inf"), 0, {}, 0

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tr_loss += loss.item() * len(y)
        tr_loss /= len(train_dl.dataset)

        model.eval()
        vl, ps, ls = 0.0, [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vl += crit(logits, y).item() * len(y)
                ps.append(logits.argmax(-1).cpu())
                ls.append(y.cpu())
        vl /= len(val_dl.dataset)
        ps, ls = torch.cat(ps).numpy(), torch.cat(ls).numpy()
        acc = float((ps == ls).mean())
        f1  = float(f1_score(ls, ps, average="macro", zero_division=0))
        print(f"[Ep {ep:3d}] train={tr_loss:.4f} val={vl:.4f} acc={acc:.4f} "
              f"f1={f1:.4f} lr={opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s")
        sys.stdout.flush()

        if vl < best_val:
            best_val, patience_count = vl, 0
            torch.save(model.state_dict(), ckpt)
            best_meta = {"epoch": ep, "val_loss": vl, "val_acc": acc, "val_f1": f1}
        else:
            patience_count += 1
            if patience_count >= PATIENCE and ep > WARMUP_EPOCHS:
                print(f"Early stop (best ep {best_meta['epoch']})")
                break
    return best_meta

# ─────────────────────────────── Evaluation ──────────────────────────────────
def evaluate(model, loader, device):
    crit = nn.CrossEntropyLoss()
    model.eval()
    loss, ps, ls, logits_all = 0.0, [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss  += crit(logits, y).item() * len(y)
            ps.append(logits.argmax(-1).cpu())
            ls.append(y.cpu())
            logits_all.append(logits.cpu())
    loss /= len(loader.dataset)
    ps = np.concatenate([p.numpy() for p in ps])
    ls = np.concatenate([l.numpy() for l in ls])
    return loss, ps, ls, torch.cat(logits_all)

# ─────────────────────────────── Main ────────────────────────────────────────
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  GPU: {gpu_name}")
    print(f"Config: d_model={D_MODEL} n_heads={N_HEADS} n_layers={N_LAYERS} "
          f"dim_ff={DIM_FF} T={T} batch={BATCH_SIZE} lr={LR} warmup={WARMUP_EPOCHS}ep")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = OUTPUT_DIR / "lob_transformer_v2_k10.pt"

    print("\n=== Loading data ===")
    train_dl, val_dl, test_dl = make_loaders()

    print("\n=== Building model ===")
    model    = LOBTransformerV2(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                                dim_ff=DIM_FF, dropout=DROPOUT, seq_len=T)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LOBTransformerV2 params: {n_params:,}")

    print("\n=== Training ===")
    best_meta = train(model, train_dl, val_dl, device, ckpt)

    print("\n=== Test evaluation ===")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model = model.to(device)
    test_loss, preds, labels, test_logits = evaluate(model, test_dl, device)
    acc    = float((preds == labels).mean())
    f1_mac = float(f1_score(labels, preds, average="macro",    zero_division=0))
    f1_wt  = float(f1_score(labels, preds, average="weighted", zero_division=0))
    kappa  = float(cohen_kappa_score(labels, preds))
    print(classification_report(labels, preds, target_names=["down", "stationary", "up"]))
    print(f"\n[LOBTransformerV2 k=10]  acc={acc:.4f}  f1_mac={f1_mac:.4f}  kappa={kappa:.4f}")

    # Val logits for calibration pipeline
    _, _, _, val_logits_t = evaluate(model, val_dl, device)
    val_labels = torch.tensor(val_dl.dataset.lab[val_dl.dataset.T - 1:])
    torch.save({"logits": val_logits_t, "labels": val_labels},
               OUTPUT_DIR / "lob_transformer_v2_k10_logits.pt")

    result = {
        "model": "lob_transformer_v2", "horizon": HORIZON,
        "test_loss": float(test_loss), "test_acc": acc,
        "test_f1_macro": f1_mac, "test_f1_weighted": float(f1_wt),
        "test_kappa": kappa, "best_val_metrics": best_meta,
        "config": {
            "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
            "dim_ff": DIM_FF, "dropout": DROPOUT, "T": T,
            "batch_size": BATCH_SIZE, "lr": LR, "warmup_epochs": WARMUP_EPOCHS,
        },
        "gpu": gpu_name,
    }
    out_path = OUTPUT_DIR / "lob_transformer_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults → {out_path}")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
