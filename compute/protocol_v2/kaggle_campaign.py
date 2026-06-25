"""
kaggle_campaign.py — Kaggle GPU kernel for the anchored FI-2010 campaign.

Trains 5 architectures × 9 folds × 3 seeds = 135 runs.
Fully resumable: each run is skipped if its summary.json already exists.

To use on Kaggle:
  1. Upload this file + the models/ directory as a Kaggle dataset or utility script.
  2. Add the fi-2010 dataset (ulfricirons/fi-2010) as an input.
  3. Run this as a script kernel with GPU accelerator enabled.

The script writes results to /kaggle/working/runs/ and also prints a ZIP path
at the end for download.
"""

import os
import sys
import subprocess
import json
import shutil
import zipfile
from pathlib import Path
import time

# ─── Install required packages ───────────────────────────────────────────────
print("Installing packages …")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torchcde", "torchdiffeq", "scikit-learn", "scipy",
])
print("Packages installed.")

# ─── Path setup ──────────────────────────────────────────────────────────────
WORKING    = Path("/kaggle/working")
RUNS_DIR   = WORKING / "runs"
CODE_DIR   = WORKING / "code"
MODELS_DIR = CODE_DIR / "src" / "models"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Add code dir to path so 'from src.models.xxx import' works
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# ─── Embed model source files ─────────────────────────────────────────────────
# Each model source is embedded as a string and written to the code tree.
# This avoids needing a separate dataset upload for the model files.

_INIT_PY = """\
from .deeplob import DeepLOB
from .tcn import TCN
from .lob_transformer import LOBTransformer
from .cnn_lstm_attention import CNNLSTMAttention
try:
    from .neural_cde import NeuralCDE
except ImportError:
    NeuralCDE = None
"""

# We embed these as base64 to avoid any quoting issues.
# Instead, we copy them from /kaggle/input if they were uploaded, or write inline.
# For simplicity, write them from embedded strings.

import importlib

def _write_from_repo(src_dir: Path, dst_dir: Path, filename: str) -> bool:
    """Copy a model file from src_dir to dst_dir. Returns True if found."""
    src = src_dir / filename
    if src.exists():
        shutil.copy(src, dst_dir / filename)
        return True
    return False


# Try to find models from a utility dataset upload or the working directory
_REPO_CANDIDATES = [
    Path("/kaggle/input/uncertainty-lob-models/src/models"),
    Path("/kaggle/input/uncertainty-lob/src/models"),
    Path("/kaggle/working/uncertainty-lob/src/models"),
]

_NEEDED_FILES = [
    "__init__.py",
    "tcn.py",
    "lob_transformer.py",
    "cnn_lstm_attention.py",
    "neural_cde.py",
    "rac_moe.py",
]

_copied = False
for _candidate in _REPO_CANDIDATES:
    if _candidate.is_dir():
        for fname in _NEEDED_FILES:
            _write_from_repo(_candidate, MODELS_DIR, fname)
        print(f"Copied model files from {_candidate}")
        _copied = True
        break

if not _copied:
    # Write __init__.py at minimum; the individual model files must be present
    (MODELS_DIR / "__init__.py").write_text(_INIT_PY)
    print("WARNING: Could not find model source files. "
          "Please upload uncertainty-lob-models dataset or copy files manually.")

# Also copy anchored_runner.py if available
_RUNNER_CANDIDATES = [
    Path("/kaggle/input/uncertainty-lob-models/compute/protocol_v2/anchored_runner.py"),
    Path("/kaggle/input/uncertainty-lob/compute/protocol_v2/anchored_runner.py"),
    Path("/kaggle/working/uncertainty-lob/compute/protocol_v2/anchored_runner.py"),
]
_RUNNER_DST = WORKING / "anchored_runner.py"
for _cand in _RUNNER_CANDIDATES:
    if _cand.exists():
        shutil.copy(_cand, _RUNNER_DST)
        print(f"Copied anchored_runner.py from {_cand}")
        break

# ─── FI-2010 data path ───────────────────────────────────────────────────────
_FI2010_ROOT = Path("/kaggle/input/fi-2010")
_ZSCORE = (
    _FI2010_ROOT
    / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore"
)
assert _ZSCORE.is_dir(), f"FI-2010 data not found at {_ZSCORE}"
print(f"FI-2010 root: {_FI2010_ROOT}")

# ─── Campaign definition ─────────────────────────────────────────────────────
MODELS = ["tcn", "lob_transformer", "lob_transformer_v2", "neural_cde", "rac_moe"]
FOLDS  = list(range(1, 10))
SEEDS  = [0, 1, 2]
EPOCHS = 80

# Full campaign: 5 × 9 × 3 = 135 jobs
jobs = [
    (model, fold, seed)
    for model in MODELS
    for fold in FOLDS
    for seed in SEEDS
]

print(f"\nCampaign: {len(jobs)} jobs ({len(MODELS)} models × {len(FOLDS)} folds × {len(SEEDS)} seeds)")


# ─── Run one job ─────────────────────────────────────────────────────────────

def run_job(model: str, fold: int, seed: int, smoke: bool = False) -> dict:
    """Run anchored_runner.py for one (model, fold, seed) job.

    Returns a dict with run_id and status.
    """
    run_id = f"anchored_references__{model}__f{fold}__k10__s{seed}__none"
    run_dir = RUNS_DIR / run_id
    summary_path = run_dir / "summary.json"

    # Skip if already completed
    if summary_path.exists():
        try:
            s = json.loads(summary_path.read_text())
            if s.get("state") == "completed":
                print(f"SKIP {run_id}")
                return {"run_id": run_id, "status": "skipped"}
        except Exception:
            pass

    print(f"\n{'─'*60}")
    print(f"START: {run_id}  (smoke={smoke})")
    print(f"{'─'*60}")

    cmd = [
        sys.executable,
        str(_RUNNER_DST),
        "--model", model,
        "--fold",  str(fold),
        "--seed",  str(seed),
        "--data-root", str(_FI2010_ROOT),
        "--out-dir",   str(RUNS_DIR),
        "--epochs",    "3" if smoke else str(EPOCHS),
    ]
    if smoke:
        cmd.append("--smoke")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"FAILED: {run_id}  (exit {result.returncode})")
        return {"run_id": run_id, "status": "failed", "elapsed": elapsed}

    if not summary_path.exists():
        print(f"WARNING: {run_id} exited 0 but no summary.json found")
        return {"run_id": run_id, "status": "missing_summary", "elapsed": elapsed}

    print(f"DONE: {run_id}  ({elapsed:.0f}s)")
    return {"run_id": run_id, "status": "completed", "elapsed": elapsed}


# ─── Smoke test ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Smoke test — tcn, fold 9, seed 0, epochs=3")
print("="*60)

smoke_result = run_job("tcn", fold=9, seed=0, smoke=True)
print(f"\nSmoke result: {smoke_result}")

if smoke_result["status"] in ("failed", "missing_summary"):
    print("ERROR: Smoke test failed. Aborting campaign.")
    sys.exit(1)

# Verify smoke run outputs
_smoke_dir = RUNS_DIR / "anchored_references__tcn__f9__k10__s0__none"
assert (_smoke_dir / "summary.json").exists(),       "Missing summary.json"
assert (_smoke_dir / "test_logits.pt").exists(),     "Missing test_logits.pt"
assert (_smoke_dir / "validation_logits.pt").exists(), "Missing validation_logits.pt"

smoke_summary = json.loads((_smoke_dir / "summary.json").read_text())
print(f"\nSmoke test metrics (test):")
for k, v in sorted(smoke_summary["test"].items()):
    print(f"  {k}: {v:.4f}")

print("\nSmoke test PASSED. Proceeding to full campaign.")


# ─── Full campaign ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Full campaign — 135 runs")
print("="*60)

results = []
for i, (model, fold, seed) in enumerate(jobs, 1):
    print(f"\n[{i}/{len(jobs)}] {model} fold={fold} seed={seed}")
    r = run_job(model, fold, seed, smoke=False)
    results.append(r)

# Summary
completed  = [r for r in results if r["status"] == "completed"]
skipped    = [r for r in results if r["status"] == "skipped"]
failed     = [r for r in results if r["status"] == "failed"]

print("\n" + "="*60)
print("CAMPAIGN SUMMARY")
print("="*60)
print(f"  Total jobs:  {len(jobs)}")
print(f"  Completed:   {len(completed)}")
print(f"  Skipped:     {len(skipped)}")
print(f"  Failed:      {len(failed)}")

if failed:
    print("\nFailed jobs:")
    for r in failed:
        print(f"  {r['run_id']}")

# Save campaign results
results_json = WORKING / "campaign_results.json"
with open(results_json, "w") as f:
    json.dump({
        "total": len(jobs),
        "completed": len(completed),
        "skipped": len(skipped),
        "failed": len(failed),
        "jobs": results,
    }, f, indent=2)
print(f"\nCampaign log → {results_json}")


# ─── Package outputs ─────────────────────────────────────────────────────────
print("\nPackaging outputs …")
zip_path = WORKING / "anchored_runs.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        for f in ["summary.json", "test_logits.pt", "validation_logits.pt"]:
            fp = run_dir / f
            if fp.exists():
                arcname = f"{run_dir.name}/{f}"
                zf.write(fp, arcname)

zip_size_mb = zip_path.stat().st_size / 1e6
print(f"Output ZIP: {zip_path}  ({zip_size_mb:.1f} MB)")
print("\nDownload with:")
print(f"  kaggle kernels output <kernel-slug> -p /tmp/campaign_out/")
