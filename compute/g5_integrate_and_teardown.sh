#!/bin/bash
# G5: Post-training integration — download artifacts, fill placeholders, compile PDF.
# Run AFTER the chain_g3_g4 tmux session prints "All done."
set -euo pipefail

POD_HOST="64.119.209.250"
POD_PORT="15123"
POD_RESULTS="/workspace/results"
X9_RESULTS="/Volumes/Crucial X9/Research Projects/DL_Paper/results"
X9_PAPER="/Volumes/Crucial X9/Research Projects/DL_Paper/paper"
X9_COMPUTE="/Volumes/Crucial X9/Research Projects/DL_Paper/compute"
BUILD_DIR="/tmp/dl_paper_build"
SSH="ssh -o StrictHostKeyChecking=no -p $POD_PORT root@$POD_HOST"
SCP="scp -P $POD_PORT -o StrictHostKeyChecking=no"

echo "=========================================="
echo "G5: Download → Fill → Compile → Teardown"
echo "=========================================="

# 1 — Download artifacts from pod
echo ""
echo "[1/4] Downloading results from pod..."
mkdir -p "$X9_RESULTS/track_a_logs"

# JSON analysis files
$SCP "root@$POD_HOST:$POD_RESULTS/*.json" "$X9_RESULTS/"

# Checkpoint and logit .pt files (Track A)
$SCP "root@$POD_HOST:$POD_RESULTS/evlob_*.pt"   "$X9_RESULTS/" 2>/dev/null || true
$SCP "root@$POD_HOST:$POD_RESULTS/lob_sde_*.pt" "$X9_RESULTS/" 2>/dev/null || true
$SCP "root@$POD_HOST:$POD_RESULTS/rac_moe_*.pt" "$X9_RESULTS/" 2>/dev/null || true

# Training logs
$SCP "root@$POD_HOST:$POD_RESULTS/evlob_full_*_ce_warmup_bayes.log" \
    "$X9_RESULTS/track_a_logs/" 2>/dev/null || true
$SCP "root@$POD_HOST:$POD_RESULTS/lob_sde_gate_k10_seed*_full.log" \
    "$X9_RESULTS/track_a_logs/" 2>/dev/null || true
$SCP "root@$POD_HOST:$POD_RESULTS/rac_moe_gate_k10_seed*_full.log" \
    "$X9_RESULTS/track_a_logs/" 2>/dev/null || true
$SCP "root@$POD_HOST:$POD_RESULTS/track_a_latex_rows.txt" \
    "$X9_RESULTS/" 2>/dev/null || true

echo "Download complete."
ls -lh "$X9_RESULTS/track_a_full_analysis.json" 2>/dev/null || echo "WARNING: analysis JSON missing!"

# 2 — Fill ?? placeholders in main.tex
echo ""
echo "[2/4] Filling Track-A placeholders in main.tex..."
python3 "$X9_COMPUTE/fill_track_a_placeholders.py" \
    --results-dir "$X9_RESULTS" \
    --tex-in      "$X9_PAPER/main.tex" \
    --tex-out     "$X9_PAPER/main.tex"
echo "Placeholder fill complete."

# 3 — Final PDF compile
echo ""
echo "[3/4] Compiling final PDF..."
cp "$X9_PAPER/main.tex" "$BUILD_DIR/main.tex"
cp "$X9_PAPER/refs.bib" "$BUILD_DIR/refs.bib" 2>/dev/null || true

cd "$BUILD_DIR"
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "Output written|Overfull|^!"

cp "$BUILD_DIR/main.pdf" "$X9_PAPER/main.pdf"
echo "PDF written to $X9_PAPER/main.pdf"

echo ""
echo "=========================================="
echo "G5 integration complete."
echo "Review results, then stop the RunPod pod."
echo "=========================================="
echo ""
echo "Verify ?? still remaining:"
grep -cF "??" "$X9_PAPER/main.tex" || echo "  (none — all filled)"
echo "Page count:"
ls -lh "$X9_PAPER/main.pdf"
