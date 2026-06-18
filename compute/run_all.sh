#!/bin/bash
# Run NeuralCDE then LOBTransformer v2 sequentially on RunPod.
# Both results land in /workspace/.
# Usage: bash run_all.sh   (after setup.sh has been run)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo " DL Paper — Full Training Run"
echo " $(date)"
echo "========================================"

# --- NeuralCDE ---
echo ""
echo "[1/2] NeuralCDE (hidden=128, batch=32, patience=20, adjoint=True)"
echo "      Log → /workspace/ncde.log"
python "${SCRIPT_DIR}/runpod_ncde.py" 2>&1 | tee /workspace/ncde.log
echo ""
echo "  NeuralCDE done. Results: /workspace/neural_cde_results.json"

# --- LOBTransformer v2 ---
echo ""
echo "[2/2] LOBTransformer v2 (d_model=128, n_heads=8, warmup+cosine LR)"
echo "      Log → /workspace/lob.log"
python "${SCRIPT_DIR}/runpod_lob_transformer_v2.py" 2>&1 | tee /workspace/lob.log
echo ""
echo "  LOBTransformer v2 done. Results: /workspace/lob_transformer_v2_results.json"

echo ""
echo "========================================"
echo " All done. Files in /workspace/:"
ls /workspace/*.json /workspace/*.pt 2>/dev/null
echo "========================================"
echo " Download to Mac:"
echo "   scp -P <PORT> root@<HOST>:/workspace/*.json ."
echo "   scp -P <PORT> root@<HOST>:/workspace/*.pt ."
echo "========================================"
