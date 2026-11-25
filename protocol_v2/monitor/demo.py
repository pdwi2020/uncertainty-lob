"""Generate deterministic dashboard fixtures in a temporary runs directory."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from protocol_v2.io_utils import append_jsonl, atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    args = parser.parse_args()
    root = Path(args.runs_root)
    campaign = root / "protocol_v2_demo"
    run = campaign / "gate_integrity__evlob__f9__k10__s0__corrected"
    run.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    atomic_write_json(campaign / "campaign_status.json", {
        "state": "running",
        "campaign": "protocol_v2_demo",
        "current_index": 7,
        "total_runs": 201,
        "completed": 6,
        "failed": 0,
        "current_run": run.name,
        "updated_at": now.isoformat(),
    })
    for epoch in range(1, 19):
        append_jsonl(run / "metrics.jsonl", {
            "type": "epoch",
            "timestamp": (now - timedelta(minutes=19 - epoch)).isoformat(),
            "epoch": epoch,
            "epochs": 80,
            "train_loss": 1.1 * math.exp(-epoch / 20),
            "val_nll": 1.05 * math.exp(-epoch / 24) + 0.25,
            "train_accuracy": 0.42 + epoch * 0.022,
            "val_accuracy": 0.40 + epoch * 0.018,
            "val_f1_macro": 0.38 + epoch * 0.017,
            "val_kappa": 0.20 + epoch * 0.02,
            "val_ece": 0.16 * math.exp(-epoch / 12),
            "val_brier": 0.52 * math.exp(-epoch / 30),
            "learning_rate": 0.0003,
            "elapsed_seconds": 34.2,
            "eta_seconds": (80 - epoch) * 34.2,
        })
    status = {
        "state": "running",
        "run_id": run.name,
        "updated_at": now.isoformat(),
        "progress": {
            "epoch": 18,
            "epochs": 80,
            "eta_seconds": 2120,
            "val_accuracy": 0.724,
            "val_ece": 0.035,
        },
        "config": {
            "model": "evlob",
            "fold": 9,
            "horizon": 10,
            "seed": 0,
            "micro_mode": "corrected",
        },
    }
    atomic_write_json(run / "status.json", status)
    (run / "console.log").write_text(
        "epoch=17/80 train_loss=0.47011 val_nll=0.76110 val_acc=0.7060\n"
        "epoch=18/80 train_loss=0.44718 val_nll=0.74503 val_acc=0.7240\n",
        encoding="utf-8",
    )
    for index in range(120):
        timestamp = now - timedelta(seconds=(119 - index) * 5)
        append_jsonl(root / "telemetry.jsonl", {
            "timestamp": timestamp.isoformat(),
            "cpu_percent": 18 + 4 * math.sin(index / 10),
            "memory_percent": 42 + 2 * math.sin(index / 16),
            "disk_percent": 11.5,
            "gpus": [{
                "gpu_index": 0,
                "gpu_name": "NVIDIA Quadro RTX 8000",
                "gpu_utilization": 91 + 5 * math.sin(index / 8),
                "gpu_memory_utilization": 61 + 3 * math.sin(index / 9),
                "gpu_memory_used_mb": 21840,
                "gpu_memory_total_mb": 49152,
                "gpu_temperature_c": 67 + 2 * math.sin(index / 13),
                "gpu_power_w": 212,
            }],
        })


if __name__ == "__main__":
    main()
