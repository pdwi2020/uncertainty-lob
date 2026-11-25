"""Sample GPU and host telemetry for the monitoring dashboard."""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil

from protocol_v2.io_utils import append_jsonl, atomic_write_json, utc_now


GPU_FIELDS = [
    "gpu_index",
    "gpu_name",
    "gpu_utilization",
    "gpu_memory_utilization",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "gpu_temperature_c",
    "gpu_power_w",
]


def gpu_rows() -> List[Dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    query = (
        "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "temperature.gpu,power.draw"
    )
    result = subprocess.run(
        [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        return []
    rows = []
    for values in csv.reader(io.StringIO(result.stdout)):
        if len(values) != len(GPU_FIELDS):
            continue
        cleaned = [value.strip() for value in values]
        row: Dict[str, Any] = dict(zip(GPU_FIELDS, cleaned))
        for key in GPU_FIELDS:
            if key in {"gpu_name"}:
                continue
            try:
                row[key] = float(row[key])
            except ValueError:
                row[key] = None
        rows.append(row)
    return rows


def sample(runs_root: Path) -> Dict[str, Any]:
    memory = psutil.virtual_memory()
    disk_target = runs_root.anchor or str(runs_root)
    disk = psutil.disk_usage(disk_target)
    return {
        "timestamp": utc_now(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": memory.percent,
        "memory_used_gb": memory.used / (1024 ** 3),
        "memory_total_gb": memory.total / (1024 ** 3),
        "disk_percent": disk.percent,
        "disk_free_gb": disk.free / (1024 ** 3),
        "gpus": gpu_rows(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    runs_root = Path(args.runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    telemetry_path = runs_root / "telemetry.jsonl"
    heartbeat_path = runs_root / "telemetry_status.json"

    while True:
        try:
            payload = sample(runs_root)
            append_jsonl(telemetry_path, payload)
            atomic_write_json(heartbeat_path, {"state": "running", **payload})
        except Exception as exc:
            atomic_write_json(heartbeat_path, {
                "state": "degraded",
                "timestamp": utc_now(),
                "error": str(exc),
            })
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
