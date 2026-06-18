"""
Kaggle kernel: generate test-partition logits for all 4 baseline models.

SETUP:
1. Add dataset: ulfricirons/fi-2010
2. Add your custom dataset containing the 4 checkpoint files:
      deeplob_k10.pt, tcn_k10.pt, lob_transformer_k10.pt, cnn_lstm_attention_k10.pt
   Adjust CKPT_DIR below to match the dataset mount path.

Outputs (in /kaggle/working/):
   deeplob_k10_test_logits.pt
   tcn_k10_test_logits.pt
   lob_transformer_k10_test_logits.pt
   cnn_lstm_attention_k10_test_logits.pt
   baseline_test_results.json   (test acc / F1 / kappa for each model)
"""

import os, sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/kaggle/input/fi-2010")
# Update this to match your checkpoint dataset name on Kaggle
CKPT_DIR  = Path("/kaggle/input/lob-checkpoints")
OUTPUT    = Path("/kaggle/working")
OUTPUT.mkdir(parents=True, exist_ok=True)

HORIZON   = 10
BATCH_SIZE = 512
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── FI-2010 data loading ──────────────────────────────────────────────────────
_LABEL_START = 144; _N_FEAT = 40; _HOR_IDX = {1:0, 2:1, 3:2, 5:3, 10:4}

def _find_fi2010_root(base: Path) -> Path:
    for sub in ["", "BenchmarkDatasets", "BenchmarkDatasets/NoAuction"]:
        p = base / sub
        if any((p).rglob("NoAuction_Zscore_Testing")):
            return p
    return base

def _load_files(paths):
    cache = OUTPUT / "data_cache"; cache.mkdir(exist_ok=True)
    feats, labs = [], []
    for p in sorted(paths):
        npy = cache / (p.name.replace(".txt", ".npy"))
        if npy.exists():
            data = np.load(str(npy))
        else:
            print(f"  Loading {p.name} ...")
            data = np.loadtxt(str(p), dtype=np.float32)
            np.save(str(npy), data); print(f"  Cached → {npy.name}")
        feats.append(data[:_N_FEAT].T)
        labs.append(data[_LABEL_START:_LABEL_START+5].T)
    return np.concatenate(feats), np.concatenate(labs)

class FI2010Dataset(Dataset):
    def __init__(self, feats, labs, horizon=10, seq_len=100):
        hi = _HOR_IDX[horizon]
        self.labels = (labs[:, hi] - 1).astype(np.int64)   # 1-3 → 0-2
        # sliding windows
        X, Y = [], []
        for i in range(seq_len, len(feats)):
            X.append(feats[i-seq_len:i])
            Y.append(self.labels[i])
        self.X = np.stack(X, dtype=np.float32)
        self.Y = np.array(Y, dtype=np.int64)
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), self.Y[i]

fi_root = _find_fi2010_root(DATA_ROOT)
test_folder = None
for candidate in fi_root.rglob("NoAuction_Zscore_Testing"):
    test_folder = candidate; break
if test_folder is None:
    # Try alternative layout
    for candidate in DATA_ROOT.rglob("*Testing*"):
        if candidate.is_dir(): test_folder = candidate; break
print(f"Test folder: {test_folder}")

test_files = sorted(test_folder.glob("Test_Dst_NoAuction_ZScore_CF_*.txt"))
test_files = [f for f in test_files if not f.name.startswith("._")]
print(f"Found {len(test_files)} test files")

test_feat, test_labs = _load_files(test_files)
test_ds = FI2010Dataset(test_feat, test_labs, horizon=HORIZON)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Test samples: {len(test_ds)}")

# ── Model definitions (inline) ────────────────────────────────────────────────

class _InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        bc = out_ch // 3
        self.b1 = nn.Sequential(nn.Conv1d(in_ch, bc, 1), nn.LeakyReLU())
        self.b2 = nn.Sequential(nn.Conv1d(in_ch, bc, 3, padding=1), nn.LeakyReLU())
        self.b3 = nn.Sequential(nn.Conv1d(in_ch, bc, 5, padding=2), nn.LeakyReLU())
    def forward(self, x): return torch.cat([self.b1(x), self.b2(x), self.b3(x)], 1)

class DeepLOB(nn.Module):
    def __init__(self, input_channels=40, cnn_out_channels=96, lstm_hidden=64, n_classes=3, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            _InceptionBlock(input_channels, cnn_out_channels), nn.MaxPool1d(2,2),
            _InceptionBlock(cnn_out_channels, cnn_out_channels), nn.MaxPool1d(2,2),
            _InceptionBlock(cnn_out_channels, cnn_out_channels))
        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout); self.fc = nn.Linear(lstm_hidden, n_classes)
    def forward(self, x):
        x = self.cnn(x.permute(0,2,1)).permute(0,2,1)
        _, (h, _) = self.lstm(x)
        return self.fc(self.dropout(h.squeeze(0)))

class CausalConv1d(nn.Module):
    def __init__(self, ic, oc, k, d):
        super().__init__()
        self.pad = (k-1)*d
        self.conv = nn.Conv1d(ic, oc, k, padding=self.pad, dilation=d)
    def forward(self, x):
        o = self.conv(x); return o[:,:,:-self.pad] if self.pad else o

class TCNBlock(nn.Module):
    def __init__(self, ic, oc, k, d, dr):
        super().__init__()
        self.block = nn.Sequential(CausalConv1d(ic,oc,k,d), nn.BatchNorm1d(oc), nn.ReLU(), nn.Dropout(dr),
                                   CausalConv1d(oc,oc,k,d), nn.BatchNorm1d(oc), nn.ReLU(), nn.Dropout(dr))
        self.res = nn.Conv1d(ic,oc,1) if ic!=oc else nn.Identity(); self.act = nn.ReLU()
    def forward(self, x): return self.act(self.block(x) + self.res(x))

class TCN(nn.Module):
    def __init__(self, input_channels=40, num_channels=None, kernel_size=3, dropout=0.2, n_classes=3):
        super().__init__()
        if num_channels is None: num_channels = [64]*6
        layers = []; ic = input_channels
        for i,(oc) in enumerate(num_channels):
            layers.append(TCNBlock(ic, oc, kernel_size, 2**i, dropout)); ic=oc
        self.net = nn.Sequential(*layers); self.fc = nn.Linear(num_channels[-1], n_classes)
    def forward(self, x):
        x = self.net(x.permute(0,2,1)); return self.fc(x[:,:,-1])

class PositionalEncoding(nn.Module):
    def __init__(self, d, maxlen=200, dr=0.1):
        super().__init__()
        import math; self.drop = nn.Dropout(dr)
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(0,maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d,2).float() * (-math.log(10000.)/d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:,:x.size(1)])

class LOBTransformer(nn.Module):
    def __init__(self, input_dim=40, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.1, n_classes=3):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.cls  = nn.Parameter(torch.zeros(1,1,d_model))
        self.pe   = PositionalEncoding(d_model, dropout=dropout)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc, num_encoder_layers)
        self.head = nn.Linear(d_model, n_classes); nn.init.normal_(self.cls, std=0.02)
    def forward(self, x):
        B = x.size(0); s = torch.cat([self.cls.expand(B,-1,-1), self.proj(x)], 1)
        return self.head(self.enc(self.pe(s))[:,0])

class BahdanauAttention(nn.Module):
    def __init__(self, h):
        super().__init__(); self.W = nn.Linear(h,h,bias=False); self.v = nn.Linear(h,1,bias=False)
    def forward(self, o):
        return (F.softmax(self.v(torch.tanh(self.W(o))), 1) * o).sum(1)

class CNNLSTMAttention(nn.Module):
    def __init__(self, input_channels=40, cnn_filters=None, cnn_kernel=3, lstm_hidden=64, lstm_layers=2, dropout=0.2, n_classes=3, use_attention=True):
        super().__init__()
        if cnn_filters is None: cnn_filters = [32,64,64]
        layers = []; ic = input_channels
        for oc in cnn_filters:
            layers += [nn.Conv1d(ic,oc,cnn_kernel,padding=cnn_kernel//2), nn.BatchNorm1d(oc), nn.ReLU()]; ic=oc
        self.cnn  = nn.Sequential(*layers)
        self.lstm = nn.LSTM(cnn_filters[-1], lstm_hidden, lstm_layers, batch_first=True, dropout=dropout if lstm_layers>1 else 0.)
        self.attn = BahdanauAttention(lstm_hidden) if use_attention else None
        self.drop = nn.Dropout(dropout); self.fc = nn.Linear(lstm_hidden, n_classes)
    def forward(self, x):
        x = self.cnn(x.permute(0,2,1)).permute(0,2,1); o,(h,_) = self.lstm(x)
        ctx = self.attn(o) if self.attn else h[-1]
        return self.fc(self.drop(ctx))

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "deeplob":            (DeepLOB,           {"input_channels":40,"cnn_out_channels":96,"lstm_hidden":64}),
    "tcn":                (TCN,               {"input_channels":40,"num_channels":[64]*6,"kernel_size":3}),
    "lob_transformer":    (LOBTransformer,    {"input_dim":40,"d_model":64,"nhead":4,"num_encoder_layers":2,"dim_feedforward":128}),
    "cnn_lstm_attention": (CNNLSTMAttention,  {"input_channels":40,"cnn_filters":[32,64,64],"lstm_hidden":64}),
}

# ── Inference ─────────────────────────────────────────────────────────────────
all_results = {}

for model_name, (ModelClass, kwargs) in MODELS.items():
    ckpt_path = CKPT_DIR / f"{model_name}_k{HORIZON}.pt"
    if not ckpt_path.exists():
        print(f"\n[SKIP] {ckpt_path} not found"); continue

    print(f"\n── {model_name} ──")
    model = ModelClass(**kwargs).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path.name}")

    all_logits, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y)
            if i % 50 == 0:
                print(f"  batch {i}/{len(test_loader)}", flush=True)

    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels)
    elapsed  = time.time() - t0

    preds    = logits_t.argmax(1).numpy()
    labels_np = labels_t.numpy()
    acc  = float((preds == labels_np).mean())
    f1m  = float(f1_score(labels_np, preds, average="macro"))
    f1w  = float(f1_score(labels_np, preds, average="weighted"))
    kap  = float(cohen_kappa_score(labels_np, preds))
    print(f"  acc={acc:.4f}  f1_mac={f1m:.4f}  kappa={kap:.4f}  ({elapsed:.1f}s)")

    out_path = OUTPUT / f"{model_name}_k{HORIZON}_test_logits.pt"
    torch.save({"logits": logits_t, "labels": labels_t}, out_path)
    print(f"  Saved → {out_path.name}")

    all_results[f"{model_name}_k{HORIZON}"] = {
        "model": model_name, "horizon": HORIZON,
        "test_acc": acc, "test_f1_macro": f1m, "test_f1_weighted": f1w, "test_kappa": kap,
        "n_test": len(labels_np),
    }

out_json = OUTPUT / "baseline_test_results.json"
with open(out_json, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved summary: {out_json}")
print(json.dumps(all_results, indent=2))
