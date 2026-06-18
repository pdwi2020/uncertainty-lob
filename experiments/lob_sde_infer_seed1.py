"""Regenerate LOB-SDE *no-gate* seed-1 TEST logits by inference from the saved
checkpoint (the original run's test logits were never synced off the pod).

This upgrades the no-gate ablation row from "seed-0 only" for ECE/AUROC to a
clean 2-seed result for all metrics (seed-0 logits already on disk). It is a
faithful re-inference: the trained weights are loaded from
`lob_sde_nogate_k10_seed1.pt`; only the 20-rollout MC noise differs from the
original CUDA run, which the rollout average absorbs.

Reuses the EXACT data pipeline from compute/train_arch.py so the 354,725-window
test set aligns with seed-0. Runs on MPS (Apple GPU) locally; no torchsde, no
external compute.

Output: results/lob_sde_nogate_k10_seed1_test_logits.pt
        {logits, labels, aleatoric, epistemic, h1_energy}  (matches seed-0 schema)
"""

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path("/Volumes/Crucial X9/Research Projects/DL_Paper")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "compute"))

import train_arch as ta                       # noqa: E402  (data pipeline + evaluate)
from src.models.lob_sde import LOB_SDE        # noqa: E402

SEED = 1
HORIZON = 10
DATA_ROOT = ROOT / "datasets/fi_2010/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore"
CKPT = ROOT / "results/lob_sde_nogate_k10_seed1.pt"
OUT = ROOT / "results/lob_sde_nogate_k10_seed1_test_logits.pt"
CACHE = ROOT / "results/data_cache_infer"     # fresh cache (real .npy cache was evicted)


def main():
    torch.manual_seed(SEED)
    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={dev}", flush=True)

    fi_root = ta._find_fi2010_root(DATA_ROOT)
    test_folder = next(fi_root.rglob("NoAuction_Zscore_Testing"))
    test_files = sorted(f for f in test_folder.glob("Test_Dst_NoAuction_ZScore_CF_*.txt")
                        if not f.name.startswith("._"))
    print(f"test files: {len(test_files)}", flush=True)

    t0 = time.time()
    te_feats, te_labs = ta._load_files(test_files, CACHE)
    test_ds = ta.LOBDataset(te_feats, te_labs, HORIZON, compute_micro=False)  # no-gate
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    print(f"test windows: {len(test_ds)}  (load {time.time()-t0:.1f}s)", flush=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"ckpt epoch={ck.get('epoch')} val_acc={ck.get('val_acc')} "
          f"args={ck.get('args')}", flush=True)

    model = LOB_SDE(input_channels=40, lstm_hidden=64, n_classes=3, n_micro=0,
                    latent_dim=64, sde_hidden=64, dropout=0.2,
                    n_rollouts=20, dt=0.05).to(dev)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()

    t1 = time.time()
    metrics, out = ta.evaluate(model, test_loader, "lob_sde", device=dev, is_final=True)
    print(f"metrics={metrics}  infer={time.time()-t1:.1f}s", flush=True)

    torch.save({"logits": out["logits"], "labels": out["labels"],
                "aleatoric": out["aleatoric"], "epistemic": out["epistemic"],
                "h1_energy": out["h1_energy"]}, OUT)
    print(f"SAVED {OUT}", flush=True)
    print("LOB_SDE_SEED1_INFER_DONE", flush=True)


if __name__ == "__main__":
    main()
