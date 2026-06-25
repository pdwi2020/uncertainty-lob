#!/usr/bin/env zsh
# fetch_and_integrate.sh — Fetch completed Kaggle runs and rsync to X9.
#
# Usage:
#   zsh fetch_and_integrate.sh
#   zsh fetch_and_integrate.sh --dry-run
#
# Fetches the anchored_runs.zip from the Kaggle kernel output,
# extracts it, and rsyncs the run dirs to the X9 campaign folder.

set -e

KERNEL_SLUG="paritoshdwivedi/anchored-lob-campaign-v2"
FETCH_DIR="/tmp/campaign_out"
X9_DEST="/Volumes/Crucial X9/Research Projects/DL_Paper/experiments/new/DL_Paper_v2_complete/runs/protocol_v2_long"
EXTRACT_DIR="${FETCH_DIR}/extracted"

DRY_RUN=0
[[ "${1}" == "--dry-run" ]] && DRY_RUN=1

echo "=== Fetch and integrate anchored FI-2010 campaign runs ==="
echo "Kernel: ${KERNEL_SLUG}"
echo "X9 destination: ${X9_DEST}"
[[ $DRY_RUN -eq 1 ]] && echo "(DRY RUN)"

# Step 1: Check kernel status
echo ""
echo "--- Kernel status ---"
kaggle kernels status "${KERNEL_SLUG}" 2>&1 || true

# Step 2: Fetch output
echo ""
echo "--- Fetching kernel output ---"
mkdir -p "${FETCH_DIR}"
kaggle kernels output "${KERNEL_SLUG}" -p "${FETCH_DIR}" 2>&1

ZIP="${FETCH_DIR}/anchored_runs.zip"
if [[ ! -f "${ZIP}" ]]; then
    echo "ERROR: ${ZIP} not found. Has the kernel completed?"
    exit 1
fi
echo "ZIP size: $(du -sh "${ZIP}" | cut -f1)"

# Step 3: Extract
echo ""
echo "--- Extracting ---"
rm -rf "${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}"
unzip -q "${ZIP}" -d "${EXTRACT_DIR}"
echo "Extracted $(ls "${EXTRACT_DIR}" | wc -l | tr -d ' ') run directories"

# Step 4: Verify X9 is mounted
if [[ ! -d "${X9_DEST}" ]]; then
    echo "ERROR: X9 not mounted at expected path"
    echo "Mount with: diskutil mount /dev/disk4s2"
    exit 1
fi

# Step 5: Rsync (NO -a to avoid ExFAT permission errors)
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

echo ""
echo "--- Done ---"
echo "Run dirs in X9: $(ls "${X9_DEST}" | grep "anchored_references" | wc -l | tr -d ' ')"

# Step 6: Verify summary.json files
echo "Checking summary.json completeness..."
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
