"""
G4: RAC-MoE 3-seed training on Google Colab (T4/L4).

HOW TO USE:
  1. Open a Colab notebook (GPU runtime, T4 or better)
  2. Upload this file, then run:  exec(open('colab_rac_moe_g4.py').read())
     OR paste the entire contents into a single code cell and run.
  3. BEFORE running, add your kaggle.json key:
       from google.colab import files
       files.upload()   # upload kaggle.json
       !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  4. Optionally mount Google Drive first for persistent output:
       from google.colab import drive; drive.mount('/content/drive')
       # then set OUTPUT = Path('/content/drive/MyDrive/rac_moe_results')

Outputs (saved to OUTPUT dir, default /content/rac_moe_results/):
  rac_moe_gate_k10_seed{0,1,2}.pt                    — best checkpoint
  rac_moe_gate_k10_seed{0,1,2}_test_logits.pt        — {logits, labels}
  rac_moe_gate_k10_seed{0,1,2}_val_logits.pt         — {logits, labels}
  rac_moe_gate_k10_seed{0,1,2}_train_log.json        — epoch log
  rac_moe_summary.json                               — 3-seed results
  rac_moe_artifacts.tar.gz                           — tar of all above

After training, download the tar.gz from the Files panel (or from Drive).
"""

# ── Install / imports ─────────────────────────────────────────────────────────
import subprocess, sys
for pkg in ["scikit-learn"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import json, math, os, time, tarfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT    = Path("/content/rac_moe_results"); OUTPUT.mkdir(exist_ok=True)
DATA_ROOT = Path("/content/fi_2010_data")
HORIZON   = 10
EPOCHS    = 50
BATCH     = 256
LR        = 1e-4
SEEDS     = [0, 1, 2]
SEQ_LEN   = 100
_N_FEAT   = 40
_LABEL_START = 144
_HOR_IDX  = {1:0, 2:1, 3:2, 5:3, 10:4}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ── Download FI-2010 from Kaggle ──────────────────────────────────────────────
def download_fi2010():
    if DATA_ROOT.exists() and any(DATA_ROOT.rglob("*Training*")):
        print("FI-2010 already downloaded.")
        return
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print("Downloading FI-2010 from Kaggle ...", flush=True)
    import subprocess
    r = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "ulfricirons/fi-2010",
         "--unzip", "-p", str(DATA_ROOT)],
        capture_output=True, text=True
    )
    print(r.stdout); print(r.stderr)
    if r.returncode != 0:
        raise RuntimeError("Kaggle download failed — check your API key and dataset slug.")

download_fi2010()

# ── Data loading (lazy, avoids OOM) ───────────────────────────────────────────
def _find_folder(root: Path, pattern: str) -> Path:
    for p in root.rglob(pattern):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Folder matching '{pattern}' not found under {root}")

def _load_raw(paths, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    feats_list, labs_list = [], []
    for p in sorted(paths):
        npy = cache / (p.name.replace(".txt", ".npy"))
        if npy.exists():
            data = np.load(str(npy))
        else:
            print(f"  Parsing {p.name} ...", flush=True)
            data = np.loadtxt(str(p), dtype=np.float32)
            np.save(str(npy), data)
        feats_list.append(data[:_N_FEAT].T)
        labs_list.append(data[_LABEL_START: _LABEL_START + 5].T)
    feats = np.concatenate(feats_list, axis=0)
    labs  = np.concatenate(labs_list, axis=0)
    return feats.astype(np.float32), labs.astype(np.float32)


def _compute_micro(snap: np.ndarray) -> np.ndarray:
    """7-dim microstructure vector from last LOB snapshot (shape: 40,)."""
    def bid_p(l): return snap[4*l]
    def bid_v(l): return snap[4*l+1]
    def ask_p(l): return snap[4*l+2]
    def ask_v(l): return snap[4*l+3]
    spread    = (ask_p(0) - bid_p(0)) / (ask_p(0) + 1e-8)
    depth_imb = (bid_v(0) - ask_v(0)) / (bid_v(0) + ask_v(0) + 1e-8)
    total_bid = sum(bid_v(l) for l in range(4))
    total_ask = sum(ask_v(l) for l in range(4))
    vol_ratio = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)
    def ofi(l):
        return bid_v(l)*(1 if bid_p(l) >= 0 else -1) - ask_v(l)*(1 if ask_p(l) <= 0 else -1)
    ofis = [float(np.tanh(ofi(l))) for l in range(4)]
    return np.array([spread, depth_imb, vol_ratio] + ofis, dtype=np.float32)


class LazyLOBDataset(Dataset):
    """Lazy-loading LOB dataset: keeps raw arrays, slices windows in __getitem__."""
    def __init__(self, feats, labs, horizon=10, seq_len=100, compute_micro=True):
        hi = _HOR_IDX[horizon]
        self.feats = feats
        self.labels = (labs[:, hi] - 1).astype(np.int64)
        self.seq_len = seq_len
        self.compute_micro = compute_micro
        self.n = len(feats) - seq_len

    def __len__(self): return self.n

    def __getitem__(self, i):
        window = self.feats[i: i + self.seq_len]   # (seq_len, 40)
        x = torch.from_numpy(window)
        y = int(self.labels[i + self.seq_len])
        if self.compute_micro:
            m = torch.from_numpy(_compute_micro(window[-1]))
        else:
            m = torch.zeros(0)
        return x, y, m


# ── Model definitions ─────────────────────────────────────────────────────────
class BahdanauAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.W = nn.Linear(h, h, bias=False)
        self.v = nn.Linear(h, 1, bias=False)
    def forward(self, o):
        return (F.softmax(self.v(torch.tanh(self.W(o))), 1) * o).sum(1)


class CNNLSTMAttention(nn.Module):
    def __init__(self, input_channels=40, cnn_filters=None, cnn_kernel=3,
                 lstm_hidden=64, lstm_layers=2, dropout=0.2, n_classes=3,
                 use_attention=True):
        super().__init__()
        if cnn_filters is None: cnn_filters = [32, 64, 64]
        layers = []; ic = input_channels
        for oc in cnn_filters:
            layers += [nn.Conv1d(ic, oc, cnn_kernel, padding=cnn_kernel//2),
                       nn.BatchNorm1d(oc), nn.ReLU()]
            ic = oc
        self.cnn  = nn.Sequential(*layers)
        self.lstm = nn.LSTM(cnn_filters[-1], lstm_hidden, lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.)
        self.attn = BahdanauAttention(lstm_hidden) if use_attention else None
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        o, (h, _) = self.lstm(x)
        ctx = self.attn(o) if self.attn else h[-1]
        return self.fc(self.drop(ctx))


class ExpertHead(nn.Module):
    def __init__(self, feat_dim, n_classes=3, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, n_classes))
    def forward(self, feat): return self.fc(feat)


class RegimeGate(nn.Module):
    def __init__(self, n_micro, n_experts=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_micro, hidden), nn.ReLU(), nn.Linear(hidden, n_experts))
    def forward(self, m): return F.softmax(self.net(m), dim=-1)


class RAC_MoE(nn.Module):
    def __init__(self, input_channels=40, lstm_hidden=64, n_classes=3,
                 n_micro=7, n_experts=2, gate_hidden=32, dropout=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.n_experts = n_experts
        self.backbone = CNNLSTMAttention(
            input_channels=input_channels, cnn_filters=[32, 64, 64],
            lstm_hidden=lstm_hidden, lstm_layers=2, dropout=dropout,
            n_classes=n_classes, use_attention=True)
        self.backbone.fc = nn.Identity()
        self.experts = nn.ModuleList([
            ExpertHead(lstm_hidden, n_classes, dropout) for _ in range(n_experts)])
        self.log_temps = nn.Parameter(torch.zeros(n_experts))
        self.gate = RegimeGate(n_micro, n_experts, gate_hidden)

    def forward(self, x, m=None):
        feat = self.backbone(x)
        temps = torch.exp(self.log_temps)
        expert_logits = torch.stack(
            [e(feat) / temps[i] for i, e in enumerate(self.experts)], dim=1)
        if m is not None:
            gate_w = self.gate(m)
        else:
            gate_w = torch.ones(feat.size(0), self.n_experts, device=feat.device) / self.n_experts
        logits = (gate_w.unsqueeze(-1) * expert_logits).sum(dim=1)
        return logits, gate_w


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds, labels):
    acc = float((preds == labels).mean())
    f1m = float(f1_score(labels, preds, average="macro", zero_division=0))
    kap = float(cohen_kappa_score(labels, preds))
    return {"acc": acc, "f1_macro": f1m, "kappa": kap}


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    all_logits, all_labels = [], []
    for x, y, m in loader:
        x = x.to(dev)
        m_dev = m.to(dev) if m.size(-1) > 0 else None
        logits, _ = model(x, m_dev)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels)
    preds = logits_t.argmax(1).numpy()
    return compute_metrics(preds, labels_t.numpy()), logits_t, labels_t


# ── Load data ─────────────────────────────────────────────────────────────────
cache = OUTPUT / "data_cache"
train_folder = _find_folder(DATA_ROOT, "NoAuction_Zscore_Training")
test_folder  = _find_folder(DATA_ROOT, "NoAuction_Zscore_Testing")

all_train_files = sorted(f for f in train_folder.glob("Train_Dst_NoAuction_ZScore_CF_*.txt")
                         if not f.name.startswith("._"))
test_files      = sorted(f for f in test_folder.glob("Test_Dst_NoAuction_ZScore_CF_*.txt")
                         if not f.name.startswith("._"))
split_idx = int(len(all_train_files) * 0.8)
train_files = all_train_files[:split_idx]
val_files   = all_train_files[split_idx:]

print(f"Train files: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}", flush=True)

print("Loading raw data ...", flush=True)
tr_feats, tr_labs = _load_raw(train_files, cache)
va_feats, va_labs = _load_raw(val_files,   cache)
te_feats, te_labs = _load_raw(test_files,  cache)

train_ds = LazyLOBDataset(tr_feats, tr_labs, HORIZON, SEQ_LEN, compute_micro=True)
val_ds   = LazyLOBDataset(va_feats, va_labs, HORIZON, SEQ_LEN, compute_micro=True)
test_ds  = LazyLOBDataset(te_feats, te_labs, HORIZON, SEQ_LEN, compute_micro=True)
print(f"Dataset sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

# ── Train 3 seeds ─────────────────────────────────────────────────────────────
all_results = {}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed); np.random.seed(seed)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=512,   shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512,   shuffle=False, num_workers=2, pin_memory=True)

    model = RAC_MoE(input_channels=40, lstm_hidden=64, n_classes=3,
                    n_micro=7, n_experts=2, gate_hidden=32, dropout=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6)

    best_val_acc = 0.0
    ckpt_path = OUTPUT / f"rac_moe_gate_k{HORIZON}_seed{seed}.pt"
    train_log = []

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        total_loss = 0.0; n_correct = 0; n_total = 0

        for x, y, m in train_loader:
            x, y = x.to(device), y.to(device)
            m_dev = m.to(device) if m.size(-1) > 0 else None
            optimizer.zero_grad()
            logits, _ = model(x, m_dev)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n_correct  += (logits.argmax(1) == y).sum().item()
            n_total    += x.size(0)

        tr_loss = total_loss / max(n_total, 1)
        tr_acc  = n_correct / max(n_total, 1)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["acc"])
        elapsed = time.time() - t0

        entry = {"epoch": epoch, "train_loss": round(tr_loss, 6),
                 "train_acc": round(tr_acc, 4), "val_acc": round(val_metrics["acc"], 4),
                 "val_f1_mac": round(val_metrics["f1_macro"], 4),
                 "val_kappa": round(val_metrics["kappa"], 4), "elapsed_s": round(elapsed, 1)}
        train_log.append(entry)
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  loss={tr_loss:.4f}  "
              f"tr_acc={tr_acc:.4f}  val_acc={val_metrics['acc']:.4f}  "
              f"kappa={val_metrics['kappa']:.4f}  ({elapsed:.1f}s)", flush=True)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_acc": best_val_acc, "seed": seed}, str(ckpt_path))
            print(f"    ↑ Best val_acc={best_val_acc:.4f} → {ckpt_path.name}", flush=True)

    print(f"\nBest val acc (seed {seed}): {best_val_acc:.4f}", flush=True)

    # Save train log
    log_path = OUTPUT / f"rac_moe_gate_k{HORIZON}_seed{seed}_train_log.json"
    with open(str(log_path), "w") as f:
        json.dump(train_log, f, indent=2)

    # Load best checkpoint and generate logits
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    for split, loader, tag in [("val", val_loader, "val"), ("test", test_loader, "test")]:
        metrics, logits_t, labels_t = evaluate(model, loader, device)
        out_pt = OUTPUT / f"rac_moe_gate_k{HORIZON}_seed{seed}_{tag}_logits.pt"
        torch.save({"logits": logits_t, "labels": labels_t}, str(out_pt))
        print(f"  {split}: acc={metrics['acc']:.4f}  f1={metrics['f1_macro']:.4f}  "
              f"kappa={metrics['kappa']:.4f}", flush=True)
        print(f"  Saved → {out_pt.name}", flush=True)
        if tag == "test":
            all_results[f"seed{seed}"] = {"best_val_acc": best_val_acc, **metrics}

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("3-SEED SUMMARY")
print(f"{'='*60}")
accs = [all_results[f"seed{s}"]["acc"] for s in SEEDS]
f1s  = [all_results[f"seed{s}"]["f1_macro"] for s in SEEDS]
kaps = [all_results[f"seed{s}"]["kappa"] for s in SEEDS]
summary = {
    "seeds": all_results,
    "mean_acc":   float(np.mean(accs)),
    "std_acc":    float(np.std(accs)),
    "mean_f1":    float(np.mean(f1s)),
    "mean_kappa": float(np.mean(kaps)),
}
print(json.dumps(summary, indent=2))
with open(str(OUTPUT / "rac_moe_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ── Bundle artifacts ──────────────────────────────────────────────────────────
tar_path = "/content/rac_moe_artifacts.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    for f in OUTPUT.iterdir():
        if f.suffix in (".pt", ".json"):
            tar.add(str(f), arcname=f.name)
print(f"\nArtifacts bundled → {tar_path}")
print("Download from the Files panel (left sidebar) or Google Drive.")
