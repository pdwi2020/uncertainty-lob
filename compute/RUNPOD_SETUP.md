# RunPod Setup — DL Paper (NeuralCDE + LOBTransformer v2)

## Scripts

| Script | Model | GPU | Est. time | Est. cost |
|--------|-------|-----|-----------|-----------|
| `runpod_ncde.py` | NeuralCDE (hidden=128, batch=32, patience=20, adjoint) | RTX 4070 Ti | 3–5 hrs | ~$0.60–$0.95 |
| `runpod_lob_transformer_v2.py` | LOBTransformer v2 (d_model=128, warmup+cosine) | RTX 4070 Ti | 1–2 hrs | ~$0.19–$0.38 |

Run them sequentially via `run_all.sh` or individually.

## GPU Selection (community cloud)

| GPU | VRAM | $/hr | Notes |
|-----|------|------|-------|
| **RTX 4070 Ti** | 24 GB | $0.190 | Recommended — lowest total cost |
| RTX 4090 | 41 GB | $0.340 | ~2× faster, higher rate |

Check availability: `runpodctl get cloud --communityCloud`

## Step-by-Step

### 1. Create pod

```bash
runpodctl create pod \
  --name dl-paper \
  --imageName runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel \
  --gpuType "NVIDIA GeForce RTX 4070 Ti" \
  --containerDiskSize 30 \
  --ports "22/tcp" \
  --startSSH
```

### 2. Upload scripts

```bash
# Get SSH details
runpodctl get pod <POD_ID>

# Upload compute/ directory
scp -P <PORT> -r \
  "/Volumes/Crucial X9/Research Projects/DL_Paper/compute/" \
  root@<HOST>:/workspace/compute/
```

### 3. Run setup (installs packages + downloads FI-2010)

```bash
ssh -p <PORT> root@<HOST>
bash /workspace/compute/setup.sh <YOUR_KAGGLE_API_KEY>
```

Kaggle API key: `~/.kaggle/kaggle.json` → `"key"` field on your Mac.

### 4. Train

**Option A — sequential (recommended, unattended):**
```bash
nohup bash /workspace/compute/run_all.sh > /workspace/run_all.log 2>&1 &
tail -f /workspace/run_all.log
```

**Option B — individual:**
```bash
# NeuralCDE
nohup python /workspace/compute/runpod_ncde.py > /workspace/ncde.log 2>&1 &
tail -f /workspace/ncde.log

# LOBTransformer v2 (after NeuralCDE finishes, or in a second SSH window)
nohup python /workspace/compute/runpod_lob_transformer_v2.py > /workspace/lob.log 2>&1 &
tail -f /workspace/lob.log
```

### 5. Download results to X9

```bash
# From your Mac (after training completes)
scp -P <PORT> root@<HOST>:/workspace/neural_cde_results.json \
    "/Volumes/Crucial X9/Research Projects/DL_Paper/results/"

scp -P <PORT> root@<HOST>:/workspace/neural_cde_k10_logits.pt \
    "/Volumes/Crucial X9/Research Projects/DL_Paper/results/"

scp -P <PORT> root@<HOST>:/workspace/neural_cde_k10_h1_norms.pt \
    "/Volumes/Crucial X9/Research Projects/DL_Paper/results/"

scp -P <PORT> root@<HOST>:/workspace/lob_transformer_v2_results.json \
    "/Volumes/Crucial X9/Research Projects/DL_Paper/results/"

scp -P <PORT> root@<HOST>:/workspace/lob_transformer_v2_k10_logits.pt \
    "/Volumes/Crucial X9/Research Projects/DL_Paper/results/"

# Model checkpoints
scp -P <PORT> root@<HOST>:/workspace/neural_cde_k10.pt \
    "/Volumes/Crucial X9/Research Projects/DL_Paper/checkpoints/"

scp -P <PORT> root@<HOST>:/workspace/lob_transformer_v2_k10.pt \
    "/Volumes/Crucial X9/Research Projects/DL_Paper/checkpoints/"
```

### 6. Stop and remove pod

```bash
runpodctl remove pod <POD_ID>
```

## Expected Outputs

After training completes, `/workspace/` contains:

```
neural_cde_results.json          ← test acc, F1, kappa + config
neural_cde_k10.pt                ← model checkpoint
neural_cde_k10_logits.pt         ← val logits for calibration
neural_cde_k10_h1_norms.pt       ← H¹ trajectory norms (UQ)

lob_transformer_v2_results.json  ← test acc, F1, kappa + config
lob_transformer_v2_k10.pt        ← model checkpoint
lob_transformer_v2_k10_logits.pt ← val logits for calibration
```
