"""Expand and execute a resumable sequence of isolated training runs."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from .io_utils import append_jsonl, atomic_write_json, runtime_metadata, utc_now


def expand_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: expand_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_value(item) for item in value]
    return value


def expand_phase(defaults: Dict[str, Any], phase: Dict[str, Any]) -> List[Dict[str, Any]]:
    fixed = {**defaults, **phase.get("fixed", {})}
    matrix = phase.get("matrix", {})
    keys = list(matrix)
    values = [value if isinstance(value, list) else [value] for value in matrix.values()]
    combinations = itertools.product(*values) if values else [()]
    runs = []
    for combination in combinations:
        run = {**fixed, **dict(zip(keys, combination))}
        run["phase"] = phase["name"]
        runs.append(run)
    return runs


def safe_token(value: Any) -> str:
    text = str(value).lower().replace(".", "p")
    return re.sub(r"[^a-z0-9_-]+", "-", text).strip("-")


def run_id(config: Dict[str, Any]) -> str:
    fields = [
        config.get("phase"),
        config.get("model"),
        f"f{config.get('fold')}",
        f"k{config.get('horizon')}",
        f"s{config.get('seed')}",
        config.get("micro_mode", "none"),
    ]
    if config.get("model") == "evlob":
        fields.extend([config.get("loss_fn", "bayes_risk"), f"kl{config.get('lambda_kl', 1.0)}"])
    if config.get("model") == "lob_sde":
        fields.extend([f"dt{config.get('dt', 0.05)}", f"r{config.get('n_rollouts', 20)}"])
    return "__".join(safe_token(field) for field in fields if field is not None)


def load_campaign(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return expand_value(raw)


def build_runs(campaign: Dict[str, Any]) -> List[Dict[str, Any]]:
    defaults = campaign.get("defaults", {})
    runs: List[Dict[str, Any]] = []
    for phase in campaign.get("phases", []):
        if phase.get("enabled", True):
            runs.extend(expand_phase(defaults, phase))
    return runs


def status_state(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["state"]
    except (json.JSONDecodeError, KeyError):
        return "unknown"


def attempt_succeeded(returncode: int, status_path: Path) -> bool:
    return returncode == 0 or status_state(status_path) == "completed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    campaign = load_campaign(config_path)
    name = str(campaign["name"])
    runs_root = Path(campaign["defaults"]["runs_root"]).resolve() / name
    runs_root.mkdir(parents=True, exist_ok=True)
    state_path = runs_root / "campaign_status.json"
    runs = build_runs(campaign)
    prepared = []
    for index, run in enumerate(runs, start=1):
        identifier = run_id(run)
        directory = runs_root / identifier
        run.update({
            "run_id": identifier,
            "run_dir": str(directory),
            "campaign": name,
            "campaign_index": index,
            "campaign_total": len(runs),
        })
        prepared.append(run)

    manifest = {
        "name": name,
        "created_at": utc_now(),
        "config_path": str(config_path),
        "run_count": len(prepared),
        "runs": prepared,
        "runtime": runtime_metadata(),
    }
    atomic_write_json(runs_root / "campaign_manifest.json", manifest)
    print(f"Campaign {name}: {len(prepared)} runs")
    if args.dry_run:
        for run in prepared:
            print(run["run_id"])
        return

    completed = 0
    failed = 0
    skipped = 0
    max_run_attempts = max(1, int(campaign.get("max_run_attempts", 3)))
    retry_backoff_seconds = max(0.0, float(campaign.get("retry_backoff_seconds", 15)))
    campaign_started = time.time()
    for run in prepared:
        directory = Path(run["run_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        run_config_path = directory / "run_config.json"
        atomic_write_json(run_config_path, run)
        state = status_state(directory / "status.json")
        if state == "completed":
            skipped += 1
            continue

        command = [sys.executable, "-m", "protocol_v2.train", "--config", str(run_config_path)]
        log_path = directory / "console.log"
        attempts_path = directory / "attempts.jsonl"
        succeeded = False
        for attempt in range(1, max_run_attempts + 1):
            atomic_write_json(state_path, {
                "state": "running",
                "updated_at": utc_now(),
                "campaign": name,
                "current_run": run["run_id"],
                "current_index": run["campaign_index"],
                "current_attempt": attempt,
                "max_run_attempts": max_run_attempts,
                "total_runs": len(prepared),
                "completed": completed,
                "failed": failed,
                "skipped": skipped,
            })
            print(
                f"[{run['campaign_index']}/{len(prepared)}] {run['run_id']} "
                f"attempt={attempt}/{max_run_attempts}",
                flush=True,
            )
            append_jsonl(attempts_path, {
                "type": "attempt_started",
                "timestamp": utc_now(),
                "attempt": attempt,
                "max_attempts": max_run_attempts,
            })
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            run_state = status_state(directory / "status.json")
            succeeded = attempt_succeeded(process.returncode, directory / "status.json")
            append_jsonl(attempts_path, {
                "type": "attempt_finished",
                "timestamp": utc_now(),
                "attempt": attempt,
                "returncode": process.returncode,
                "run_state": run_state,
                "succeeded": succeeded,
            })
            if succeeded:
                break
            if attempt < max_run_attempts:
                time.sleep(retry_backoff_seconds)

        if succeeded:
            completed += 1
        else:
            failed += 1
            if args.stop_on_failure:
                break

    final_state = "completed" if failed == 0 else "completed_with_failures"
    atomic_write_json(state_path, {
        "state": final_state,
        "updated_at": utc_now(),
        "campaign": name,
        "total_runs": len(prepared),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "duration_seconds": time.time() - campaign_started,
    })


if __name__ == "__main__":
    main()
