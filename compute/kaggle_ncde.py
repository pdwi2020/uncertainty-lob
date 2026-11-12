# KAGGLE SETUP: In the first cell, run:
# !pip install torchcde torchdiffeq -q

"""
Self-contained NeuralCDE training script for Kaggle.

Embeds all source code. No project repo required.
Expects:
  - Add ulfricirons/fi-2010 as Kaggle dataset input
  - torchcde, torchdiffeq, scikit-learn installed
Saves:
  - /kaggle/working/neural_cde_results.json
  - /kaggle/working/neural_cde_k10_logits.pt
  - /kaggle/working/neural_cde_k10_h1_norms.pt
"""

import os, sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report

import torchcde

# ─────────────────────────────── Config ──────────────────────────────────────
HORIZON       = 10
BATCH_SIZE    = 128          # Kaggle T4/P100 16GB — larger batch = faster epochs
HIDDEN        = 128
LR            = 2e-4         # was 1e-4 — scaled up for the larger batch (grad-clip=1.0 guards stability)
EPOCHS        = 100
PATIENCE      = 20
SEED          = 42
DATA_ROOT     = Path("/kaggle/input/datasets/ulfricirons/fi-2010")
OUTPUT_DIR    = Path("/kaggle/working")
ADJOINT       = False

# ─────────────────────────────── Data ────────────────────────────────────────
_HORIZON_IDX  = {1: 0, 2: 1, 3: 2, 5: 3, 10: 4}
_LABEL_START  = 144
_N_FEAT       = 40

def _load_files(paths):
    feats, labs = [], []
    cache_dir = OUTPUT_DIR / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(paths):
        npy = cache_dir / (p.name.replace(".txt", ".npy"))  # write to /kaggle/working (writable)
        if npy.exists():
            data = np.load(str(npy))
        else:
            print(f"  Loading {p.name} via np.loadtxt ...")
            data = np.loadtxt(str(p), dtype=np.float32)
            np.save(str(npy), data)
            print(f"  Cached → {npy.name}")
        feats.append(data[:_N_FEAT].T)
        labs.append(data[_LABEL_START:_LABEL_START+5].T)
    return np.concatenate(feats), np.concatenate(labs)

def _find_files(zscore_root, split):
    sub    = "NoAuction_Zscore_Training" if split=="train" else "NoAuction_Zscore_Testing"
    prefix = "Train_Dst" if split=="train" else "Test_Dst"
    files  = sorted((zscore_root/sub).glob(f"{prefix}_NoAuction_ZScore_CF_*.txt"))
    files  = [f for f in files if not f.name.startswith("._")]
    if not files: raise FileNotFoundError(f"No FI-2010 files in {zscore_root/sub}")
    return files

class LOBDataset(Dataset):
    def __init__(self, feat, lab, horizon=10, T=100):
        self.feat = feat.astype(np.float32)
        self.lab  = (lab[:, _HORIZON_IDX[horizon]] - 1).astype(np.int64)
        self.T    = T
    def __len__(self):  return len(self.feat) - self.T
    def __getitem__(self, i):
        return torch.from_numpy(self.feat[i:i+self.T]), torch.tensor(self.lab[i+self.T-1])

def _find_data_root():
    """Auto-detect the FI-2010 NoAuction/Zscore root under any Kaggle mount layout."""
    import subprocess
    candidates = [
        DATA_ROOT / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore",
        DATA_ROOT / "NoAuction" / "1.NoAuction_Zscore",
        DATA_ROOT / "1.NoAuction_Zscore",
    ]
    # Also search recursively (Kaggle sometimes nests differently)
    try:
        result = subprocess.run(
            ["find", str(DATA_ROOT), "-type", "d", "-name", "1.NoAuction_Zscore"],
            capture_output=True, text=True, timeout=30
        )
        for p in result.stdout.strip().split("\n"):
            if p: candidates.insert(0, Path(p))
    except Exception:
        pass
    for c in candidates:
        if c.exists() and any(c.glob("**/*.txt")):
            print(f"Found data at: {c}")
            return c
    # Last resort: print tree to debug
    try:
        result = subprocess.run(["find", str(DATA_ROOT), "-name", "*.txt", "-maxdepth", "6"],
                                capture_output=True, text=True, timeout=30)
        print("Available .txt files:", result.stdout[:2000])
    except Exception:
        pass
    raise FileNotFoundError(f"FI-2010 NoAuction/Zscore not found under {DATA_ROOT}")

def make_loaders():
    zscore = _find_data_root()
    tr_f, tr_l = _load_files(_find_files(zscore, "train"))
    te_f, te_l = _load_files(_find_files(zscore, "test"))
    split = int(len(tr_f) * 0.80)
    kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    train_dl = DataLoader(LOBDataset(tr_f[:split], tr_l[:split], HORIZON), shuffle=True,  **kw)
    val_dl   = DataLoader(LOBDataset(tr_f[split:], tr_l[split:], HORIZON), shuffle=False, **kw)
    test_dl  = DataLoader(LOBDataset(te_f,         te_l,         HORIZON), shuffle=False, **kw)
    print(f"Train: {len(train_dl.dataset):,}  Val: {len(val_dl.dataset):,}  Test: {len(test_dl.dataset):,}")
    return train_dl, val_dl, test_dl

# ───────────────────────── Spline preprocessing (shared) ─────────────────────
CLAMP = 7.0  # preserve LOB volume magnitude. Was clamp(-3,3)/3 → [-1,1], which
             # destroyed the volume-spike information baselines exploit (acc cap ~54%).

def _build_spline(x, T, device):
    """Clamp (magnitude-preserving), prepend normalised time, build a cubic spline.

    Shared by NeuralCDE.forward AND compute_h1_norms so the H¹ trajectory norm is
    computed on exactly the path the model was trained on (previously they diverged).
    Prefers Hermite backward-difference coefficients — local, non-ringing derivatives
    that prevent the spline-coefficient blow-up (±2.5M) seen with natural cubic splines,
    letting us relax the clamp without re-triggering the loss explosion.
    """
    x = torch.clamp(x, -CLAMP, CLAMP)
    t = torch.arange(T, dtype=torch.float32, device=device)  # dt=1 → dX/dt sane (was linspace(0,1) → dt=0.01 → coeff magnitudes ~81M)
    t_ch = t[None, :, None].expand(x.shape[0], T, 1)
    x_aug = torch.cat([t_ch, x], dim=-1)                       # (B, T, 41)
    # Use natural cubic splines — Hermite uses torch.diff which fails on P100 (sm_60) in CUDA 12.x
    coeffs = torchcde.natural_cubic_spline_coeffs(x_aug, t)
    return torchcde.CubicSpline(coeffs), x

# ─────────────────────────────── Model ───────────────────────────────────────
class CDEFunc(nn.Module):
    def __init__(self, input_ch, hidden_ch):
        super().__init__()
        self.hidden_ch  = hidden_ch
        self.input_ch   = input_ch
        self.net = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch * 2), nn.Tanh(),
            nn.Linear(hidden_ch * 2, hidden_ch * input_ch),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, t, z):
        return torch.tanh(self.net(z)).view(z.shape[0], self.hidden_ch, self.input_ch)

class NeuralCDE(nn.Module):
    def __init__(self, input_ch=40, hidden_ch=64, out_ch=3, adjoint=True, dropout=0.1):
        super().__init__()
        self.hidden_ch  = hidden_ch
        self.adjoint    = adjoint
        self.cde_func   = CDEFunc(input_ch+1, hidden_ch)   # +1 for time channel
        self.initial    = nn.Linear(input_ch, hidden_ch)
        self.readout    = nn.Linear(hidden_ch, out_ch)
        self.drop       = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_ch)
    def forward(self, x):
        B, T, _ = x.shape
        X, xc = _build_spline(x, T, x.device)              # shared preprocessing
        z0 = torch.tanh(self.initial(xc[:, 0, :]))
        z_path = torchcde.cdeint(X=X, func=self.cde_func, z0=z0,
                                  t=X.interval, adjoint=self.adjoint,
                                  method="rk4", rtol=1e-3, atol=1e-4)
        logits = self.readout(self.drop(self.final_norm(z_path[:, -1])))
        return logits

# ─────────────────────────────── Training ────────────────────────────────────
def train(model, train_dl, val_dl, device, ckpt):
    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit  = nn.CrossEntropyLoss()
    best_val, patience_count, best_meta = float("inf"), 0, {}

    for ep in range(1, EPOCHS+1):
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
        acc = float((ps==ls).mean())
        f1  = float(f1_score(ls, ps, average="macro", zero_division=0))
        sched.step(vl)
        print(f"[Ep {ep:3d}] train={tr_loss:.4f} val={vl:.4f} acc={acc:.4f} "
              f"f1={f1:.4f} lr={opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s")
        sys.stdout.flush()

        if vl < best_val:
            best_val, patience_count = vl, 0
            torch.save(model.state_dict(), ckpt)
            best_meta = {"epoch": ep, "val_loss": vl, "val_acc": acc, "val_f1": f1}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
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
            ps.append(logits.argmax(-1).cpu()); ls.append(y.cpu())
            logits_all.append(logits.cpu())
    loss /= len(loader.dataset)
    ps, ls = np.concatenate([p.numpy() for p in ps]), np.concatenate([l.numpy() for l in ls])
    return loss, ps, ls, torch.cat(logits_all)

def compute_h1_norms(model, loader, device):
    model.eval()
    norms = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device); B, T, _ = x.shape
            X, xc = _build_spline(x, T, device)            # SAME preprocessing as forward
            z0 = torch.tanh(model.initial(xc[:, 0, :]))    # SAME tanh init as forward
            t_eval = torch.linspace(0, float(T - 1), 10, device=device)  # match arange domain [0, T-1]
            zp = torchcde.cdeint(X=X, func=model.cde_func, z0=z0, t=t_eval,
                                  adjoint=False, method="rk4", rtol=1e-3, atol=1e-4)
            dz = zp[:, 1:] - zp[:, :-1]
            norms.append(((dz**2).sum(-1).sum(-1)).cpu().numpy())
    return np.concatenate(norms)

def compute_mc_dropout(model, loader, device, S=30):
    """Monte-Carlo Dropout on the test set: S stochastic passes (dropout left ON),
    mean softmax → predictive entropy. Mirrors src/calibration/mc_dropout.py so the
    NeuralCDE row in Table 3 is computed consistently with the other models — but
    self-contained here (the pod is ephemeral; avoids re-instantiating via the pipeline)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()                                  # keep dropout stochastic at eval
    probs_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = torch.zeros(x.shape[0], 3, device=device)
            for _ in range(S):
                p += torch.softmax(model(x), dim=-1)
            probs_all.append((p / S).cpu()); labels_all.append(y)
    P = torch.cat(probs_all); L = torch.cat(labels_all)
    mc_acc  = float((P.argmax(-1) == L).float().mean())
    entropy = -(P * (P + 1e-12).log()).sum(-1)
    return mc_acc, float(entropy.mean()), entropy.numpy()

# ─────────────────────────────── Main ────────────────────────────────────────
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  GPU: {gpu_name}")
    print(f"Config: horizon={HORIZON} batch={BATCH_SIZE} hidden={HIDDEN} "
          f"lr={LR} adjoint={ADJOINT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = OUTPUT_DIR / "neural_cde_k10.pt"

    print("\n=== Loading data ===")
    train_dl, val_dl, test_dl = make_loaders()

    print("\n=== Building model ===")
    model = NeuralCDE(input_ch=40, hidden_ch=HIDDEN, adjoint=ADJOINT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NeuralCDE params: {n_params:,}")

    print("\n=== Training ===")
    best_meta = train(model, train_dl, val_dl, device, ckpt)

    print("\n=== Test evaluation ===")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model = model.to(device)
    test_loss, preds, labels, test_logits = evaluate(model, test_dl, device)
    acc    = float((preds==labels).mean())
    f1_mac = float(f1_score(labels, preds, average="macro",    zero_division=0))
    f1_wt  = float(f1_score(labels, preds, average="weighted", zero_division=0))
    kappa  = float(cohen_kappa_score(labels, preds))
    print(classification_report(labels, preds, target_names=["up","stationary","down"]))
    print(f"\n[NeuralCDE k=10]  acc={acc:.4f}  f1_mac={f1_mac:.4f}  kappa={kappa:.4f}")

    # Val logits + ALIGNED labels (calibration pipeline reads these for temperature/ECE).
    # Use the labels evaluate() actually iterated — NOT the off-by-one lab[T-1:] re-derivation.
    _, _, val_labels_arr, val_logits_t = evaluate(model, val_dl, device)
    torch.save({"logits": val_logits_t, "labels": torch.tensor(val_labels_arr)},
               OUTPUT_DIR / "neural_cde_k10_logits.pt")
    # Test logits too (selective prediction / completeness)
    torch.save({"logits": test_logits, "labels": torch.tensor(labels)},
               OUTPUT_DIR / "neural_cde_k10_test_logits.pt")

    # MC-Dropout on test (S=30) → fills Table 3 MC columns self-contained
    print("\nComputing MC-Dropout (S=30)...")
    mc_acc, mc_entropy_mean, mc_entropy = compute_mc_dropout(model, test_dl, device, S=30)
    torch.save(torch.from_numpy(mc_entropy), OUTPUT_DIR / "neural_cde_k10_mc_entropy.pt")
    print(f"MC-Dropout: acc={mc_acc:.4f}  mean_entropy={mc_entropy_mean:.4f}")

    # H¹ norms (uncertainty proxy; SAME preprocessing as training — see _build_spline)
    print("\nComputing H¹ norms...")
    h1 = compute_h1_norms(model, test_dl, device)
    torch.save(torch.from_numpy(h1), OUTPUT_DIR / "neural_cde_k10_h1_norms.pt")
    print(f"H¹ norm: mean={h1.mean():.4f}  std={h1.std():.4f}")

    result = {
        "model": "neural_cde", "horizon": HORIZON,
        "test_loss": float(test_loss), "test_acc": acc,
        "test_f1_macro": f1_mac, "test_f1_weighted": float(f1_wt),
        "test_kappa": kappa, "best_val_metrics": best_meta,
        "mc_acc": mc_acc, "mc_entropy_mean": mc_entropy_mean,
        "h1_norm_mean": float(h1.mean()), "h1_norm_std": float(h1.std()),
        "config": {"hidden_channels": HIDDEN, "lr": LR, "clamp": CLAMP,
                   "batch_size": BATCH_SIZE, "adjoint": ADJOINT,
                   "spline": ("hermite_backward_diff"
                              if hasattr(torchcde, "hermite_cubic_coefficients_with_backward_differences")
                              else "natural_cubic")},
        "gpu": gpu_name,
    }
    out_path = OUTPUT_DIR / "neural_cde_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults → {out_path}")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
