#!/usr/bin/env zsh
# integrate_batches.sh — Fetch completed Kaggle batch runs and rsync to X9.
# Handles batch-a (tcn/lob_transformer/lob_transformer_v2) and batch-b (rac_moe)
# Usage: zsh integrate_batches.sh [--dry-run] [--batch a|b|both]

set -e

BATCH="${2:-both}"  # a, b, or both
DRY_RUN=0
[[ "${1}" == "--dry-run" ]] && DRY_RUN=1

FETCH_DIR="/tmp/campaign_batches"
EXTRACT_DIR="${FETCH_DIR}/extracted"
X9_DEST="/Volumes/Crucial X9/Research Projects/DL_Paper/experiments/new/DL_Paper_v2_complete/runs/protocol_v2_long"

echo "=== Integrate anchored FI-2010 batches ==="
echo "Batch: ${BATCH}"
echo "X9 destination: ${X9_DEST}"
[[ $DRY_RUN -eq 1 ]] && echo "(DRY RUN)"

mkdir -p "${FETCH_DIR}" "${EXTRACT_DIR}"

# Function to fetch and extract a batch
fetch_batch() {
    local kernel_slug="$1"
    local zip_name="$2"
    local out_dir="${FETCH_DIR}"
    
    echo ""
    echo "--- Fetching ${kernel_slug} ---"
    kaggle kernels status "${kernel_slug}" 2>&1 || true
    
    local prev_status=$(kaggle kernels status "${kernel_slug}" 2>&1)
    if ! echo "$prev_status" | grep -q "COMPLETE"; then
        echo "WARNING: ${kernel_slug} may not be complete yet"
    fi
    
    kaggle kernels output "${kernel_slug}" -p "${out_dir}" 2>&1
    
    local zip="${out_dir}/${zip_name}"
    if [[ -f "${zip}" ]]; then
        echo "ZIP found: ${zip} ($(du -sh "${zip}" | cut -f1))"
        echo "Extracting..."
        unzip -q "${zip}" -d "${EXTRACT_DIR}"
        echo "Done extracting ${zip_name}"
    else
        echo "No ${zip_name} found in output — kernel may not have completed successfully"
        ls "${out_dir}/" 2>/dev/null || true
    fi
}

# Fetch requested batches
if [[ "${BATCH}" == "a" || "${BATCH}" == "both" ]]; then
    fetch_batch "paritoshdwivedi/anchored-lob-batch-a" "anchored_runs_batch_a.zip"
fi

if [[ "${BATCH}" == "b" || "${BATCH}" == "both" ]]; then
    fetch_batch "paritoshdwivedi/anchored-lob-batch-b" "anchored_runs_batch_b.zip"
fi

# Count extracted runs
N_EXTRACTED=$(ls "${EXTRACT_DIR}" 2>/dev/null | grep "anchored_references__" | wc -l | tr -d ' ')
echo ""
echo "--- ${N_EXTRACTED} run directories extracted ---"

# Verify X9 is mounted
if [[ ! -d "${X9_DEST}" ]]; then
    echo "ERROR: X9 not mounted at expected path"
    echo "Mount with: diskutil unmount '/Volumes/Crucial X9' && diskutil mount /dev/disk4s2"
    exit 1
fi

# Rsync to X9 (no -a for ExFAT)
echo ""
echo "--- Rsyncing to X9 ---"
RSYNC_FLAGS="-rt --modify-window=2 --no-perms --no-owner --no-group"
[[ $DRY_RUN -eq 1 ]] && RSYNC_FLAGS="${RSYNC_FLAGS} --dry-run"

rsync ${RSYNC_FLAGS} "${EXTRACT_DIR}/" "${X9_DEST}/"
rc=$?

if [[ $rc -ne 0 ]]; then
    echo "ERROR: rsync failed with exit code ${rc}"
    exit 1
fi

# Count completed runs on X9 by model type
echo ""
echo "--- X9 Run Counts ---"
for model in tcn lob_transformer lob_transformer_v2 rac_moe neural_cde; do
    count=$(ls "${X9_DEST}" | grep "anchored_references__${model}__" | wc -l | tr -d ' ')
    expected=27
    echo "  ${model}: ${count}/27"
done

# Count total completed
COMPLETED=0
MISSING=0
for d in "${X9_DEST}"/anchored_references__tcn__* \
          "${X9_DEST}"/anchored_references__lob_transformer__* \
          "${X9_DEST}"/anchored_references__lob_transformer_v2__* \
          "${X9_DEST}"/anchored_references__neural_cde__* \
          "${X9_DEST}"/anchored_references__rac_moe__*; do
    [[ -d "$d" ]] || continue
    s="${d}/summary.json"
    if [[ -f "$s" ]] && python3 -c "import json,sys; d=json.load(open('${s}')); sys.exit(0 if d.get('state')=='completed' else 1)" 2>/dev/null; then
        ((COMPLETED++))
    else
        ((MISSING++))
        echo "  MISSING/INCOMPLETE: $(basename $d)"
    fi
done

echo ""
echo "Completed runs on X9: ${COMPLETED} / 135"
echo "Missing/incomplete:   ${MISSING}"
[[ $COMPLETED -eq 135 ]] && echo "ALL 135 RUNS COMPLETE" || echo "Resume with another kernel push when quota resets"
