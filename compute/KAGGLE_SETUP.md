# Kaggle Setup — DL Paper

## Overview

The DL paper uses Kaggle T4 GPU notebooks (free tier, 30h/week).
Primary platform for DeepLOB, TCN, LOB-Transformer, and CNN-LSTM+Attention.
Neural CDE fallback: RunPod RTX 3090 (see `RUNPOD_SETUP.md`).

## Prerequisites

1. Kaggle account + API key at `~/.kaggle/kaggle.json`
2. FI-2010 dataset: available on Kaggle as public dataset

## FI-2010 Dataset Access

```bash
# Option A: Download public FI-2010 dataset from Kaggle
kaggle datasets download -d username/fi-2010-lob --path data/fi_2010/

# Option B: Upload your X9 copy as private dataset
kaggle datasets create -p /tmp/fi2010_upload/
```

## Upload X9 Data to Kaggle (One-Time)

```bash
mkdir -p /tmp/fi2010_upload

# FI-2010 is ~30 GB; upload NoAuction/Zscore subset only (~2 GB)
cp -r "/Volumes/Crucial X9/kaggle/fi-2010/NoAuction_Zscore" /tmp/fi2010_upload/

cat > /tmp/fi2010_upload/dataset-metadata.json <<'EOF'
{
  "title": "fi2010-lob-noauction-zscore",
  "id": "paritoshdwi2019/fi2010-lob-noauction-zscore",
  "licenses": [{"name": "other"}]
}
EOF

kaggle datasets create -p /tmp/fi2010_upload/
```

## Kaggle Notebook Configuration

1. New Notebook → Add Dataset → search "fi2010-lob-noauction-zscore"
2. Settings → **Accelerator → GPU T4 x1** (essential for neural network training)
3. Settings → Internet → On (for pip installs)
4. Settings → Persistence → Files (keep outputs between runs)

Install PyTorch extensions:
```python
!pip install torchcde torchdiffeq hmmlearn mapie -q
```

## GPU Memory Management

T4 has 16 GB VRAM. Keep batch sizes within budget:

| Model | Batch Size | VRAM Usage |
|-------|-----------|------------|
| DeepLOB | 64 | ~4 GB |
| TCN | 64 | ~5 GB |
| LOB-Transformer | 64 | ~6 GB |
| Neural CDE | 32 | ~12 GB (adjoint) |
| CNN-LSTM+Att | 64 | ~5 GB |

```python
# Check GPU at start of each notebook
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Training Checkpoint Strategy

```python
# Save every 10 epochs to /kaggle/working/
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_f1": val_f1,
}
torch.save(checkpoint, f"/kaggle/working/deeplob_epoch{epoch}.pt")
```

## Quota Management

- GPU quota: 30h/week; session limit: 9h
- DeepLOB (100 epochs): ~2–3h → 1 session
- TCN (100 epochs): ~2h → 1 session
- Neural CDE (100 epochs): ~6–8h → borderline; use RunPod if needed
- Strategy: run DeepLOB + TCN on Day 1, Transformer + CNN-LSTM+Att on Day 2

## Bybit Data Access

Bybit BTCUSDT L2 data (24 GB) is too large to upload fully. Use a 1-week sample:

```bash
# Extract 1-week sample locally, upload to Kaggle private dataset
python - <<'EOF'
import pandas as pd, os
bybit_path = "/Volumes/Crucial X9/exchange=bybit/spot/BTCUSDT/"
# Sample 7 days
sample = pd.read_parquet(bybit_path, filters=[("date", ">=", "2025-05-01"), ("date", "<=", "2025-05-07")])
sample.to_parquet("/tmp/bybit_sample.parquet")
print(f"Sample shape: {sample.shape}")
EOF
```
