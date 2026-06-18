#!/bin/bash
# RunPod setup for DL Paper — NeuralCDE + LOBTransformer v2
# Run once on a fresh pod before training.
# Usage: bash setup.sh <KAGGLE_KEY>
#   KAGGLE_KEY: your Kaggle API key (from ~/.kaggle/kaggle.json → "key" field)

set -e

KAGGLE_USER="paritoshdwivedi"
KAGGLE_KEY="${1:-}"

if [ -z "$KAGGLE_KEY" ]; then
    echo "Usage: bash setup.sh <KAGGLE_API_KEY>"
    echo "Find your key at: https://www.kaggle.com/settings/account → API → Create Token"
    exit 1
fi

echo "========================================"
echo " DL Paper — RunPod Setup"
echo "========================================"

# 1. Install packages
echo "[1/4] Installing packages..."
pip install torchcde torchdiffeq scikit-learn kaggle -q
command -v unzip >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y unzip -qq; }
echo "  torchcde, torchdiffeq, scikit-learn, kaggle, unzip installed"

# 2. Kaggle credentials
echo "[2/4] Setting up Kaggle credentials..."
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<EOF
{"username":"${KAGGLE_USER}","key":"${KAGGLE_KEY}"}
EOF
chmod 600 ~/.kaggle/kaggle.json
echo "  Credentials written to ~/.kaggle/kaggle.json"

# 3. Download FI-2010 dataset
echo "[3/4] Downloading FI-2010 dataset..."
mkdir -p /workspace/data
cd /workspace/data

if [ -d "fi-2010/BenchmarkDatasets" ]; then
    echo "  FI-2010 already present, skipping download"
else
    # Correct public slug — the old paritoshdwivedi/fi2010-... slug 403s.
    kaggle datasets download -d "ulfricirons/fi-2010" -p /workspace/data/ --quiet
    echo "  Unzipping ONLY NoAuction/Zscore (full ~30G extract overflows the 30G disk)..."
    unzip -q /workspace/data/fi-2010.zip 'BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/*' -d /workspace/data/fi-2010/
    rm /workspace/data/fi-2010.zip
    echo "  FI-2010 (NoAuction/Zscore) ready at /workspace/data/fi-2010/"
fi

# 4. Verify
echo "[4/4] Verifying setup..."
python - <<'PYEOF'
import torch, torchcde, sklearn
print(f"  PyTorch:  {torch.__version__}")
print(f"  torchcde: {torchcde.__version__}")
print(f"  CUDA:     {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
from pathlib import Path
zscore = Path("/workspace/data/fi-2010/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore")
train_files = list((zscore / "NoAuction_Zscore_Training").glob("Train_Dst*.txt"))
test_files  = list((zscore / "NoAuction_Zscore_Testing").glob("Test_Dst*.txt"))
print(f"  FI-2010 train files: {len(train_files)}")
print(f"  FI-2010 test files:  {len(test_files)}")
assert len(train_files) > 0 and len(test_files) > 0, "Data files missing!"
print("  Setup OK")
PYEOF

echo ""
echo "========================================"
echo " Setup complete. Ready to train."
echo ""
echo " Run NeuralCDE:"
echo "   nohup python runpod_ncde.py > ncde.log 2>&1 &"
echo "   tail -f ncde.log"
echo ""
echo " Run LOBTransformer v2:"
echo "   nohup python runpod_lob_transformer_v2.py > lob.log 2>&1 &"
echo "   tail -f lob.log"
echo ""
echo " Run both sequentially:"
echo "   bash run_all.sh"
echo "========================================"
