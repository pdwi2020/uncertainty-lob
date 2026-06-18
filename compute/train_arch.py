"""Unified training script for Track-A architectures: EvLOB, LOB-SDE, RAC-MoE.

Usage:
    python train_arch.py --arch evlob --seed 0 --horizon 10 --epochs 50
    python train_arch.py --arch lob_sde --seed 0 --horizon 10 --epochs 50 --verify-only
    python train_arch.py --arch rac_moe --seed 0 --horizon 10 --epochs 50
    python train_arch.py --arch evlob --seed 0 --horizon 10 --epochs 50 --no-gate  # ablation

Outputs (to OUTPUT_DIR):
    {arch}_{gate_tag}_k{horizon}_seed{seed}.pt          checkpoint
    {arch}_{gate_tag}_k{horizon}_seed{seed}_test_logits.pt   {logits, labels, alpha, vacuity, ...}
    {arch}_{gate_tag}_k{horizon}_seed{seed}_val_logits.pt
    {arch}_{gate_tag}_k{horizon}_seed{seed}_train_log.json

LOB-SDE only:
    If --verify-only: runs 3-epoch gate; exits 0 if stable, exits 1 if unstable.
    Full run: only launched after gate passes (enforced by caller / train script).

Microstructure features (n_micro=7):
    Not available in base FI-2010 → computed on-the-fly from the raw LOB snapshots
    as approximations: [spread, depth_imb, vol_ratio_l1, ofi_l1, ofi_l2, ofi_l3, ofi_l4]
    If unable to compute (no raw data), n_micro is set to 0 (gate disabled, plain EDL/MoE).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, cohen_kappa_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_ROOT  = Path(os.environ.get("FI2010_ROOT", "/kaggle/input/fi-2010"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "results"))
CKPT_DIR   = Path(os.environ.get("CKPT_DIR", OUTPUT_DIR))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.models.evlob import EvLOB, edl_bayes_risk_loss, edl_uce_loss, edl_ce_warmup_loss, edl_ce_warmup_bayes_loss
from src.models.lob_sde import LOB_SDE, lob_sde_loss, LOBSDEVerifyGate
from src.models.rac_moe import RAC_MoE

# ── Constants ─────────────────────────────────────────────────────────────────
_LABEL_START = 144
_N_FEAT      = 40
_HOR_IDX     = {1: 0, 2: 1, 3: 2, 5: 3, 10: 4}
SEQ_LEN      = 100

# ── Data loading ──────────────────────────────────────────────────────────────

def _find_fi2010_root(base: Path) -> Path:
    for sub in ["", "BenchmarkDatasets", "BenchmarkDatasets/NoAuction"]:
        p = base / sub
        if any(p.rglob("NoAuction_Zscore_Training")):
            return p
    return base


def _load_files(paths, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    feats, labs = [], []
    for p in sorted(paths):
        npy = cache_dir / p.name.replace(".txt", ".npy")
        if npy.exists():
            data = np.load(str(npy))
        else:
            print(f"  Loading {p.name} ...")
            data = np.loadtxt(str(p), dtype=np.float32)
            np.save(str(npy), data)
        feats.append(data[:_N_FEAT].T)
        labs.append(data[_LABEL_START: _LABEL_START + 5].T)
    return np.concatenate(feats, axis=0), np.concatenate(labs, axis=0)


def _compute_micro(feats_window: np.ndarray) -> np.ndarray:
    """Approximate microstructure vector from last snapshot in window.

    FI-2010 column layout (0-indexed):
      bid_p1=0, bid_v1=1, ask_p1=2, ask_v1=3
      bid_p2=4, bid_v2=5, ask_p2=6, ask_v2=7  ...  (10 levels, 4 cols each)

    Returns: [spread_norm, depth_imb_l1, vol_ratio, ofi_l1, ofi_l2, ofi_l3, ofi_l4]
    """
    snap = feats_window[-1]   # last row of window, shape (40,)

    def bid_p(l): return snap[4 * l]
    def bid_v(l): return snap[4 * l + 1]
    def ask_p(l): return snap[4 * l + 2]
    def ask_v(l): return snap[4 * l + 3]

    spread = (ask_p(0) - bid_p(0)) / (ask_p(0) + 1e-8)
    depth_imb = (bid_v(0) - ask_v(0)) / (bid_v(0) + ask_v(0) + 1e-8)
    total_bid = sum(bid_v(l) for l in range(4))
    total_ask = sum(ask_v(l) for l in range(4))
    vol_ratio = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)

    def ofi(l):
        return bid_v(l) * (1 if bid_p(l) >= 0 else -1) - ask_v(l) * (1 if ask_p(l) <= 0 else -1)

    ofis = [np.tanh(ofi(l)) for l in range(4)]
    return np.array([spread, depth_imb, vol_ratio] + ofis, dtype=np.float32)


class LOBDataset(Dataset):
    def __init__(
        self,
        feats: np.ndarray,
        labs: np.ndarray,
        horizon: int = 10,
        seq_len: int = SEQ_LEN,
        compute_micro: bool = True,
    ):
        hi = _HOR_IDX[horizon]
        labels_raw = (labs[:, hi] - 1).astype(np.int64)

        X, Y, M = [], [], []
        for i in range(seq_len, len(feats)):
            window = feats[i - seq_len: i]
            X.append(window)
            Y.append(labels_raw[i])
            if compute_micro:
                M.append(_compute_micro(window))

        self.X = np.stack(X).astype(np.float32)
        self.Y = np.array(Y, dtype=np.int64)
        self.M = np.stack(M).astype(np.float32) if M else None

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])
        y = int(self.Y[i])
        if self.M is not None:
            m = torch.from_numpy(self.M[i])
            return x, y, m
        return x, y, torch.zeros(0)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds, labels):
    acc = float((preds == labels).mean())
    f1m = float(f1_score(labels, preds, average="macro", zero_division=0))
    f1w = float(f1_score(labels, preds, average="weighted", zero_division=0))
    kap = float(cohen_kappa_score(labels, preds))
    return {"acc": acc, "f1_macro": f1m, "f1_weighted": f1w, "kappa": kap}


# ── Training loops ────────────────────────────────────────────────────────────

def train_evlob(model, loader, optimizer, epoch, n_epochs, lambda_kl=1.0, device="cpu", loss_fn="uce", warm_epochs=20):
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    for x, y, m in loader:
        x, y = x.to(device), y.to(device)
        m = m.to(device) if m.size(-1) > 0 else None
        optimizer.zero_grad()
        alpha, S, p_hat, _ = model(x, m)
        if loss_fn == "ce_warmup_bayes":
            loss = edl_ce_warmup_bayes_loss(alpha, y, epoch, n_epochs, lambda_kl, warm_epochs)
        elif loss_fn == "uce":
            loss = edl_uce_loss(alpha, y, epoch, n_epochs, lambda_kl)
        elif loss_fn == "ce_warmup":
            loss = edl_ce_warmup_loss(alpha, y, epoch, n_epochs, lambda_kl, warm_epochs)
        else:
            loss = edl_bayes_risk_loss(alpha, y, epoch, n_epochs, lambda_kl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_correct  += (p_hat.argmax(1) == y).sum().item()
        n_total    += x.size(0)
    return total_loss / max(n_total, 1), n_correct / max(n_total, 1)


def train_lob_sde(model, loader, optimizer, device="cpu"):
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    for x, y, m in loader:
        x, y = x.to(device), y.to(device)
        m = m.to(device) if m.size(-1) > 0 else None
        optimizer.zero_grad()
        logits, pe = model(x, m)
        loss = lob_sde_loss(logits, pe, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_correct  += (logits.argmax(1) == y).sum().item()
        n_total    += x.size(0)
    return total_loss / max(n_total, 1), n_correct / max(n_total, 1)


def train_rac_moe(model, loader, optimizer, device="cpu"):
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    for x, y, m in loader:
        x, y = x.to(device), y.to(device)
        m = m.to(device) if m.size(-1) > 0 else None
        optimizer.zero_grad()
        logits, _ = model(x, m)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_correct  += (logits.argmax(1) == y).sum().item()
        n_total    += x.size(0)
    return total_loss / max(n_total, 1), n_correct / max(n_total, 1)


@torch.no_grad()
def evaluate(model, loader, arch, device="cpu", n_epochs=50, epoch=0, is_final=False):
    model.eval()
    all_logits, all_labels = [], []
    extra = {}

    for x, y, m in loader:
        x = x.to(device)
        m_dev = m.to(device) if m.size(-1) > 0 else None

        if arch == "evlob":
            alpha, S, p_hat, logits = model(x, m_dev)
            all_logits.append(logits.cpu())
            # Collect uncertainties on last eval
            if "alpha" not in extra:
                extra["alpha"] = []
                extra["S"]     = []
            extra["alpha"].append(alpha.cpu())
            extra["S"].append(S.cpu())
        elif arch == "lob_sde":
            if is_final:
                # MC inference: use n_rollouts for proper UQ estimates
                pred_dict = model.predict(x, m_dev)
                logits = pred_dict["logits"]
                all_logits.append(logits.cpu())
                for k in ("aleatoric", "epistemic", "h1_energy"):
                    if k not in extra:
                        extra[k] = []
                    extra[k].append(pred_dict[k].cpu())
            else:
                logits, _ = model(x, m_dev)
                all_logits.append(logits.cpu())
        elif arch == "rac_moe":
            logits, gw = model(x, m_dev)
            all_logits.append(logits.cpu())
        else:
            raise ValueError(f"Unknown arch: {arch}")

        all_labels.append(y)

    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels)
    preds    = logits_t.argmax(1).numpy()
    labels_np = labels_t.numpy()
    metrics = compute_metrics(preds, labels_np)

    out = {"logits": logits_t, "labels": labels_t}
    for k, v in extra.items():
        out[k] = torch.cat(v) if isinstance(v, list) else v
    return metrics, out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Track-A architecture")
    parser.add_argument("--arch",        choices=["evlob", "lob_sde", "rac_moe"], required=True)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--horizon",     type=int, default=10)
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--batch-size",  type=int, default=256)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--lambda-kl",   type=float, default=1.0, help="EDL KL weight (EvLOB)")
    parser.add_argument("--loss-fn",     type=str,   default="bayes_risk", choices=["bayes_risk","ce_warmup_bayes","uce","ce_warmup"], help="EvLOB loss function")
    parser.add_argument("--warm-epochs", type=int,   default=20, help="CE warm-up epochs for ce_warmup loss")
    parser.add_argument("--lr-backbone",  type=float, default=None, help="Separate LR for backbone (None=same as --lr)")
    parser.add_argument("--no-gate",     action="store_true", help="Disable microstructure gate (ablation)")
    parser.add_argument("--tag",         type=str,   default="", help="Extra tag appended to output filenames (e.g. for parallel runs)")
    parser.add_argument("--verify-only", action="store_true", help="LOB-SDE: run 3-epoch gate only")
    parser.add_argument("--data-root",   type=str, default=str(DATA_ROOT))
    parser.add_argument("--output-dir",  type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Arch: {args.arch}  Seed: {args.seed}  Gate: {not args.no_gate}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    # Find FI-2010 layout
    fi_root  = _find_fi2010_root(data_root)
    cache    = output_dir / "data_cache"

    # Training files (Days 1–7 = Declarations)
    train_folders = []
    for cand in fi_root.rglob("NoAuction_Zscore_Training"):
        train_folders.append(cand)
        break
    if not train_folders:
        # Try flat layout
        for cand in data_root.rglob("*Training*"):
            if cand.is_dir():
                train_folders.append(cand)
                break
    if not train_folders:
        print("ERROR: Could not find FI-2010 Training folder. Set FI2010_ROOT env var.")
        sys.exit(1)

    train_folder = train_folders[0]
    test_folder  = None
    for cand in fi_root.rglob("NoAuction_Zscore_Testing"):
        test_folder = cand
        break
    if test_folder is None:
        for cand in data_root.rglob("*Testing*"):
            if cand.is_dir():
                test_folder = cand
                break

    all_train_files = sorted(f for f in train_folder.glob("Train_Dst_NoAuction_ZScore_CF_*.txt")
                              if not f.name.startswith("._"))
    test_files      = sorted(f for f in test_folder.glob("Test_Dst_NoAuction_ZScore_CF_*.txt")
                              if not f.name.startswith("._")) if test_folder else []

    print(f"Train files: {len(all_train_files)}  Test files: {len(test_files)}")

    # Val = last 20% of training data (Day 8 proxy)
    split_idx = int(len(all_train_files) * 0.8)
    train_files = all_train_files[:split_idx]
    val_files   = all_train_files[split_idx:]

    print("Loading train data...")
    tr_feats, tr_labs = _load_files(train_files, cache)
    print("Loading val data...")
    va_feats, va_labs = _load_files(val_files, cache)
    print("Loading test data...")
    if test_files:
        te_feats, te_labs = _load_files(test_files, cache)

    n_micro = 0 if args.no_gate else 7
    gate_tag = "nogate" if args.no_gate else "gate"

    train_ds = LOBDataset(tr_feats, tr_labs, args.horizon, compute_micro=(n_micro > 0))
    val_ds   = LOBDataset(va_feats, va_labs, args.horizon, compute_micro=(n_micro > 0))
    test_ds  = LOBDataset(te_feats, te_labs, args.horizon, compute_micro=(n_micro > 0)) if test_files else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=512,              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512,              shuffle=False, num_workers=4, pin_memory=True) if test_ds else None

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds) if test_ds else 0}")

    # ── Build model ───────────────────────────────────────────────────────────
    if args.arch == "evlob":
        model = EvLOB(
            input_channels=40, lstm_hidden=64, n_classes=3,
            n_micro=n_micro, gate_hidden=32, dropout=0.2,
        ).to(device)
    elif args.arch == "lob_sde":
        model = LOB_SDE(
            input_channels=40, lstm_hidden=64, n_classes=3,
            n_micro=n_micro, latent_dim=64, sde_hidden=64,
            dropout=0.2, n_rollouts=20, dt=0.05,
        ).to(device)
        # Verify gate for LOB-SDE
        print("\nRunning LOB-SDE 3-epoch verify gate ...")
        gate_result = LOBSDEVerifyGate.run(model, train_loader, device, n_epochs=3, lr=args.lr)
        print(f"  Stable: {gate_result['stable']}  Reason: {gate_result['reason']}")
        print(f"  Losses: {gate_result['losses']}")
        gate_path = output_dir / f"lob_sde_{gate_tag}_k{args.horizon}_seed{args.seed}_verify.json"
        with open(gate_path, "w") as f:
            json.dump(gate_result, f, indent=2)
        if args.verify_only or not gate_result["stable"]:
            print("Verify-only or gate failed — exiting.")
            sys.exit(0 if gate_result["stable"] else 1)
        # Re-initialize model for clean full training
        torch.manual_seed(args.seed)
        model = LOB_SDE(
            input_channels=40, lstm_hidden=64, n_classes=3,
            n_micro=n_micro, latent_dim=64, sde_hidden=64,
            dropout=0.2, n_rollouts=20, dt=0.05,
        ).to(device)
    elif args.arch == "rac_moe":
        model = RAC_MoE(
            input_channels=40, lstm_hidden=64, n_classes=3,
            n_micro=n_micro, n_experts=2, gate_hidden=32, dropout=0.2,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    if hasattr(args, 'lr_backbone') and args.lr_backbone is not None and args.arch == "evlob":
        optimizer = torch.optim.Adam([
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": list(model.evid_head.parameters()) + (list(model.gate.parameters()) if model.gate else []), "lr": args.lr},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    run_tag = f"_{args.tag}" if args.tag else ""
    best_ckpt    = output_dir / f"{args.arch}_{gate_tag}_k{args.horizon}_seed{args.seed}{run_tag}.pt"
    train_log    = []

    for epoch in range(args.epochs):
        t0 = time.time()

        if args.arch == "evlob":
            tr_loss, tr_acc = train_evlob(model, train_loader, optimizer, epoch, args.epochs, args.lambda_kl, device, loss_fn=args.loss_fn, warm_epochs=args.warm_epochs)
        elif args.arch == "lob_sde":
            tr_loss, tr_acc = train_lob_sde(model, train_loader, optimizer, device)
        elif args.arch == "rac_moe":
            tr_loss, tr_acc = train_rac_moe(model, train_loader, optimizer, device)

        val_metrics, _ = evaluate(model, val_loader, args.arch, device, args.epochs, epoch)
        scheduler.step(val_metrics["acc"])  # ReduceLROnPlateau
        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "train_acc":  round(tr_acc, 4),
            "val_acc":    round(val_metrics["acc"], 4),
            "val_f1_mac": round(val_metrics["f1_macro"], 4),
            "val_kappa":  round(val_metrics["kappa"], 4),
            "elapsed_s":  round(elapsed, 1),
        }
        train_log.append(log_entry)
        print(f"  Epoch {epoch+1:3d}/{args.epochs}  loss={tr_loss:.4f}  "
              f"tr_acc={tr_acc:.4f}  val_acc={val_metrics['acc']:.4f}  "
              f"kappa={val_metrics['kappa']:.4f}  ({elapsed:.1f}s)")

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": best_val_acc,
                "args": vars(args),
            }, best_ckpt)
            print(f"    ↑ New best val_acc={best_val_acc:.4f} → {best_ckpt.name}")

    print(f"\nBest val acc: {best_val_acc:.4f}")

    # ── Save train log ────────────────────────────────────────────────────────
    log_path = output_dir / f"{args.arch}_{gate_tag}_k{args.horizon}_seed{args.seed}{run_tag}_train_log.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"Train log → {log_path.name}")

    # ── Generate logits from best checkpoint ─────────────────────────────────
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    for split, loader, tag in [
        ("val", val_loader, "val"),
        ("test", test_loader, "test"),
    ]:
        if loader is None:
            continue
        metrics, artifact = evaluate(model, loader, args.arch, device, args.epochs, args.epochs - 1, is_final=True)
        out_path = output_dir / f"{args.arch}_{gate_tag}_k{args.horizon}_seed{args.seed}{run_tag}_{tag}_logits.pt"
        torch.save(artifact, out_path)
        print(f"{split}: acc={metrics['acc']:.4f}  f1_mac={metrics['f1_macro']:.4f}  kappa={metrics['kappa']:.4f}")
        print(f"Saved → {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
