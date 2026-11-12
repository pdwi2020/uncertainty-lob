"""Retrain the four fold-9 seed-0 smoke-test runs on Kaggle through the kaggle-exec MCP engine.

Usage: python launch.py <kernel-name> <model> [<model> ...]
Writes <kernel-name>.result.json and the per-run zips into ./out_<kernel-name>/.
"""
import base64
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path.home() / "projects/mcp-server-kaggle-exec/src"))
from mcp_server_kaggle_exec.kaggle_runtime import run_on_kaggle_with_artifacts  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def bundle() -> str:
    """Base64 tar.gz of src/ and the anchored runner, laid out as in the repo."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted((REPO / "src").rglob("*.py")):
            if "__pycache__" not in path.parts and not path.name.startswith("._"):
                tar.add(path, arcname=str(path.relative_to(REPO)))
        runner = REPO / "compute/protocol_v2/anchored_runner.py"
        tar.add(runner, arcname=str(runner.relative_to(REPO)))
    return base64.b64encode(buf.getvalue()).decode()


KERNEL = r'''
import base64, io, json, os, shutil, subprocess, sys, tarfile, time
CODE, RUNS, WORK = "/tmp/code", "/tmp/runs", "/kaggle/working"
os.makedirs(CODE, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(CODE)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
import glob
# Kaggle mounts datasets via symlinks under /kaggle/input/datasets/<owner>/<slug>; glob follows them
found = sorted(glob.glob("/kaggle/input/**/NoAuction_Zscore_Training", recursive=True))
print("FI-2010 candidates:", found, flush=True)
assert found, "FI-2010 NoAuction_Zscore_Training not found under /kaggle/input"
DATA = os.path.dirname(found[0])
for model in MODELS:
    run_id = f"anchored_references__{model}__f9__k10__s0__none"
    env = dict(os.environ)
    if model == "tcn":
        env["CUDNN_BENCHMARK"] = "1"   # deterministic cuDNN makes dilated Conv1d 50-300x slower
    cmd = [sys.executable, f"{CODE}/compute/protocol_v2/anchored_runner.py", "--model", model,
           "--fold", "9", "--seed", "0", "--data-root", DATA, "--out-dir", RUNS,
           "--epochs", "80"]
    t0 = time.time()
    print(f"=== START {run_id}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    print(f"=== END {run_id} rc={rc} {time.time() - t0:.0f}s", flush=True)
    summary = os.path.join(RUNS, run_id, "summary.json")
    if os.path.isfile(summary):
        s = json.load(open(summary))
        print("RESULT", model, json.dumps({"state": s.get("state"), "epochs": s.get("config", {}).get("epochs"),
              "best_epoch": s.get("best_epoch"), "test_acc": s.get("test", {}).get("accuracy")}), flush=True)
    if os.path.isdir(os.path.join(RUNS, run_id)):
        shutil.make_archive(os.path.join(WORK, f"retrain_{model}_f9_s0"), "zip", RUNS, run_id)
print("ALL DONE", flush=True)
'''


def main() -> None:
    name, models = sys.argv[1], sys.argv[2:]
    code = f"BUNDLE = {bundle()!r}\nMODELS = {models!r}\n" + KERNEL
    out = HERE / f"out_{name}"
    result = run_on_kaggle_with_artifacts(
        code=code,
        local_output_dir=str(out),
        enable_gpu=True,
        enable_internet=False,
        timeout=4 * 3600,
        cleanup=False,                       # a timeout must never delete a live kernel
        dataset_sources=["wzy841127938/fi2010"],
        machine_shape="NvidiaTeslaP100",     # same GPU as the other 158 runs
    )
    (HERE / f"{name}.result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("status", "execution_time", "kernel_id", "output_files")}, indent=2))


if __name__ == "__main__":
    main()
