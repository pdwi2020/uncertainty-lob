#!/usr/bin/env bash
# G5 Local Integration: run this AFTER the pod chain finishes.
# Usage: bash g5_local_integrate.sh [pod_ip] [pod_port]
# Example: bash g5_local_integrate.sh 64.119.209.250 15123
set -e

POD_IP="${1:-64.119.209.250}"
POD_PORT="${2:-15123}"
X9="/Volumes/Crucial X9/Research Projects/DL_Paper"
RESULTS="$X9/results"
TEX="$X9/paper/main.tex"
FILL="$X9/compute/fill_track_a_placeholders.py"

echo "=== G5 Integration ==="
echo "Pod: $POD_IP:$POD_PORT"
echo "X9:  $X9"

# ── 1. Download artifacts from pod ──────────────────────────────────────────
echo ""
echo "Step 1: rsync results from pod..."
rsync -avz -e "ssh -o ConnectTimeout=30 -p $POD_PORT" \
    root@$POD_IP:/workspace/results/ "$RESULTS/" \
    --exclude "_cache_*.npy" \
    --include "*.pt" --include "*.json" --include "*.npy" --include "*.log" \
    --include "*/" --exclude "*"
echo "rsync complete."

# ── 2. Verify critical files ─────────────────────────────────────────────────
echo ""
echo "Step 2: Verify key artifacts..."
MISSING=0
for arch in lob_sde rac_moe; do
    for seed in 0 1 2; do
        f="$RESULTS/${arch}_gate_k10_seed${seed}_test_logits.pt"
        if [ ! -f "$f" ]; then
            echo "  MISSING: $f"
            MISSING=$((MISSING+1))
        else
            echo "  OK: ${arch} seed ${seed}"
        fi
    done
done
for seed in 0 1 2; do
    for tag in gate nogate; do
        f="$RESULTS/evlob_${tag}_k10_seed${seed}_test_logits.pt"
        if [ ! -f "$f" ]; then
            echo "  MISSING: $f"
            MISSING=$((MISSING+1))
        else
            echo "  OK: evlob_${tag} seed ${seed}"
        fi
    done
done
if [ $MISSING -gt 0 ]; then
    echo "ERROR: $MISSING artifact(s) missing. Aborting."
    exit 1
fi
echo "All artifacts present."

# ── 3. Run Track-A analysis (on X9 Python) ──────────────────────────────────
echo ""
echo "Step 3: Running track_a_analysis.py locally..."
python3 "$X9/compute/track_a_analysis.py" --results-dir "$RESULTS"
echo "Analysis complete."

# ── 4. Fill ?? placeholders in main.tex ──────────────────────────────────────
echo ""
echo "Step 4: Filling LaTeX placeholders..."
cp "$TEX" "${TEX}.bak_g5"
python3 "$FILL" \
    --results-dir "$RESULTS" \
    --tex-in  "$TEX" \
    --tex-out "$TEX"
echo "Placeholders filled."

# ── 5. Rebuild PDF ───────────────────────────────────────────────────────────
echo ""
echo "Step 5: Rebuilding PDF..."
BUILD_DIR=$(mktemp -d /tmp/dl_paper_g5_XXXX)
cp -r "$X9/paper" "$BUILD_DIR/paper"
cp "$X9"/fig*.pdf "$BUILD_DIR/" 2>/dev/null || true
cd "$BUILD_DIR/paper"
pdflatex -interaction=nonstopmode main.tex > /tmp/g5_build.log 2>&1
bibtex main >> /tmp/g5_build.log 2>&1
pdflatex -interaction=nonstopmode main.tex >> /tmp/g5_build.log 2>&1
pdflatex -interaction=nonstopmode main.tex >> /tmp/g5_build.log 2>&1
if [ -f main.pdf ]; then
    cp main.pdf "$X9/paper/main.pdf"
    echo "PDF rebuilt: $(ls -lh main.pdf | awk '{print $5, $6, $7, $8}')"
else
    echo "ERROR: PDF build failed. Check /tmp/g5_build.log"
    exit 1
fi

echo ""
echo "=== G5 Integration Complete ==="
echo "Remaining ??: $(grep -c '[?][?]' "$TEX" || echo 0)"
echo "PDF: $X9/paper/main.pdf"
echo ""
echo "Next: stop the RunPod (ID: ggf9t69aesjcjb)"
echo "  Use mcp__runpod__stop-pod with podId=ggf9t69aesjcjb"
