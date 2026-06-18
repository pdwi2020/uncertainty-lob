"""
Compute test-partition bootstrap CIs for all 4 baseline models.
Uses lazy loading (raw arrays kept; windows sliced in __getitem__) to stay within RAM.

Outputs:
  results/baseline_test_ci.json         — acc/F1/kappa CIs (mean ± CI, 95%)
  results/{model}_k10_test_logits.pt    — {logits, labels} tensors
"""
import json, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score

ROOT    = Path("/Volumes/Crucial X9/Research Projects/DL_Paper")
CKPT    = ROOT
TEST_DIR = ROOT / "datasets/fi_2010/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing"
OUT     = ROOT / "results"
OUT.mkdir(exist_ok=True)

HORIZON   = 10
SEQ_LEN   = 100
BATCH     = 512
N_BOOT    = 2000
SEED      = 42
_N_FEAT   = 40
_LABEL_START = 144
_HOR_IDX  = {1:0, 2:1, 3:2, 5:3, 10:4}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}", flush=True)

torch.manual_seed(SEED); np.random.seed(SEED)

# ── Lazy dataset (no window pre-allocation) ───────────────────────────────────
class LazyLOBDataset(Dataset):
    def __init__(self, feats: np.ndarray, labs: np.ndarray, horizon=10, seq_len=100):
        hi = _HOR_IDX[horizon]
        self.feats = feats.astype(np.float32)
        self.labs  = (labs[:, hi] - 1).astype(np.int64)
        self.seq_len = seq_len
        self.n = len(feats) - seq_len

    def __len__(self): return self.n

    def __getitem__(self, i):
        x = torch.from_numpy(self.feats[i: i + self.seq_len])
        y = int(self.labs[i + self.seq_len])
        return x, y


def load_test_data(test_dir: Path) -> tuple:
    files = sorted(test_dir.glob("Test_Dst_NoAuction_ZScore_CF_*.txt"))
    files = [f for f in files if not f.name.startswith("._")]
    print(f"Loading {len(files)} test files from {test_dir} ...", flush=True)
    feats_list, labs_list = [], []
    for f in files:
        npy = OUT / f"_cache_{f.stem}.npy"
        if npy.exists():
            data = np.load(str(npy))
        else:
            print(f"  Parsing {f.name} ...", flush=True)
            data = np.loadtxt(str(f), dtype=np.float32)
            np.save(str(npy), data)
        feats_list.append(data[:_N_FEAT].T)
        labs_list.append(data[_LABEL_START: _LABEL_START + 5].T)
    feats = np.concatenate(feats_list, axis=0)
    labs  = np.concatenate(labs_list, axis=0)
    print(f"  Total test samples (raw): {len(feats)}", flush=True)
    return feats, labs


# ── Model definitions (minimal inline copies) ─────────────────────────────────
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
        self.block = nn.Sequential(
            CausalConv1d(ic,oc,k,d), nn.BatchNorm1d(oc), nn.ReLU(), nn.Dropout(dr),
            CausalConv1d(oc,oc,k,d), nn.BatchNorm1d(oc), nn.ReLU(), nn.Dropout(dr))
        self.res = nn.Conv1d(ic,oc,1) if ic!=oc else nn.Identity()
        self.act = nn.ReLU()
    def forward(self, x): return self.act(self.block(x) + self.res(x))

class TCN(nn.Module):
    def __init__(self, input_channels=40, num_channels=None, kernel_size=3, dropout=0.2, n_classes=3):
        super().__init__()
        if num_channels is None: num_channels = [64]*6
        layers = []; ic = input_channels
        for i, oc in enumerate(num_channels):
            layers.append(TCNBlock(ic, oc, kernel_size, 2**i, dropout)); ic = oc
        self.net = nn.Sequential(*layers); self.fc = nn.Linear(num_channels[-1], n_classes)
    def forward(self, x): return self.fc(self.net(x.permute(0,2,1))[:,:,-1])

class PositionalEncoding(nn.Module):
    def __init__(self, d, maxlen=200, dr=0.1):
        super().__init__()
        import math; self.drop = nn.Dropout(dr)
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(0, maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.) / d))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:,:x.size(1)])

class LOBTransformer(nn.Module):
    def __init__(self, input_dim=40, d_model=64, nhead=4, num_encoder_layers=2,
                 dim_feedforward=128, dropout=0.1, n_classes=3):
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
    def __init__(self, input_channels=40, cnn_filters=None, cnn_kernel=3,
                 lstm_hidden=64, lstm_layers=2, dropout=0.2, n_classes=3, use_attention=True):
        super().__init__()
        if cnn_filters is None: cnn_filters = [32,64,64]
        layers = []; ic = input_channels
        for oc in cnn_filters:
            layers += [nn.Conv1d(ic,oc,cnn_kernel,padding=cnn_kernel//2), nn.BatchNorm1d(oc), nn.ReLU()]
            ic = oc
        self.cnn  = nn.Sequential(*layers)
        self.lstm = nn.LSTM(cnn_filters[-1], lstm_hidden, lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers>1 else 0.)
        self.attn = BahdanauAttention(lstm_hidden) if use_attention else None
        self.drop = nn.Dropout(dropout); self.fc = nn.Linear(lstm_hidden, n_classes)
    def forward(self, x):
        x = self.cnn(x.permute(0,2,1)).permute(0,2,1)
        o, (h, _) = self.lstm(x)
        ctx = self.attn(o) if self.attn else h[-1]
        return self.fc(self.drop(ctx))

MODELS = {
    "deeplob":            (DeepLOB,           {"input_channels":40,"cnn_out_channels":96,"lstm_hidden":64}),
    "tcn":                (TCN,               {"input_channels":40,"num_channels":[64]*6,"kernel_size":3}),
    "lob_transformer":    (LOBTransformer,    {"input_dim":40,"d_model":64,"nhead":4,"num_encoder_layers":2,"dim_feedforward":128}),
    "cnn_lstm_attention": (CNNLSTMAttention,  {"input_channels":40,"cnn_filters":[32,64,64],"lstm_hidden":64}),
}

# ── Bootstrap ─────────────────────────────────────────────────────────────────
def bootstrap_ci(logits_np, labels_np, n_boot=N_BOOT, alpha=0.05, rng=None):
    if rng is None: rng = np.random.default_rng(SEED)
    n = len(labels_np)
    accs, f1s, kaps = [], [], []
    print(f"  Bootstrap {n_boot} iters over {n} samples ...", end="", flush=True)
    t0 = time.time()
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        preds_b = logits_np[idx].argmax(1)
        labs_b  = labels_np[idx]
        accs.append((preds_b == labs_b).mean())
        f1s.append(f1_score(labs_b, preds_b, average="macro", zero_division=0))
        kaps.append(cohen_kappa_score(labs_b, preds_b))
    print(f" {time.time()-t0:.1f}s", flush=True)
    lo, hi = (1-alpha)/2, (1+alpha)/2
    return {
        "acc":   (float(np.percentile(accs, lo*100)),  float(np.percentile(accs, hi*100))),
        "f1":    (float(np.percentile(f1s,  lo*100)),  float(np.percentile(f1s,  hi*100))),
        "kappa": (float(np.percentile(kaps, lo*100)),  float(np.percentile(kaps, hi*100))),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
feats, labs = load_test_data(TEST_DIR)
ds = LazyLOBDataset(feats, labs, horizon=HORIZON, seq_len=SEQ_LEN)
loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)
print(f"Dataset size: {len(ds)} windows", flush=True)

rng = np.random.default_rng(SEED)
all_ci = {"split": "test", "n_boot": N_BOOT, "seed": SEED, "models": {}}

for name, (Cls, kwargs) in MODELS.items():
    ckpt = CKPT / f"{name}_k{HORIZON}.pt"
    if not ckpt.exists():
        print(f"\n[SKIP] {ckpt.name} not found"); continue

    print(f"\n── {name} ──", flush=True)
    model = Cls(**kwargs).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"  Checkpoint loaded", flush=True)

    all_logits, all_labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            logits = model(x.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(y)
            if i % 100 == 0:
                print(f"  batch {i}/{len(loader)}", flush=True)

    logits_t  = torch.cat(all_logits)
    labels_t  = torch.cat(all_labels)
    elapsed   = time.time() - t0

    logits_np = logits_t.numpy()
    labels_np = labels_t.numpy()
    preds     = logits_np.argmax(1)

    acc  = float((preds == labels_np).mean())
    f1m  = float(f1_score(labels_np, preds, average="macro", zero_division=0))
    kap  = float(cohen_kappa_score(labels_np, preds))
    print(f"  acc={acc:.4f}  f1={f1m:.4f}  kappa={kap:.4f}  ({elapsed:.1f}s)", flush=True)

    out_pt = OUT / f"{name}_k{HORIZON}_test_logits.pt"
    torch.save({"logits": logits_t, "labels": labels_t}, str(out_pt))
    print(f"  Saved logits → {out_pt.name}", flush=True)

    ci = bootstrap_ci(logits_np, labels_np, rng=rng)

    all_ci["models"][name] = {
        "n": int(len(labels_np)),
        "acc":   {"point": acc,  "ci_low": ci["acc"][0],   "ci_high": ci["acc"][1]},
        "f1":    {"point": f1m,  "ci_low": ci["f1"][0],    "ci_high": ci["f1"][1]},
        "kappa": {"point": kap,  "ci_low": ci["kappa"][0], "ci_high": ci["kappa"][1]},
    }
    print(f"  acc={acc:.4f} [{ci['acc'][0]:.4f}, {ci['acc'][1]:.4f}]", flush=True)
    print(f"  f1 ={f1m:.4f} [{ci['f1'][0]:.4f},  {ci['f1'][1]:.4f}]", flush=True)
    print(f"  kap={kap:.4f} [{ci['kappa'][0]:.4f}, {ci['kappa'][1]:.4f}]", flush=True)

out_json = OUT / "baseline_test_ci.json"
with open(str(out_json), "w") as f:
    json.dump(all_ci, f, indent=2)
print(f"\nSaved: {out_json}")
print(json.dumps(all_ci, indent=2))
