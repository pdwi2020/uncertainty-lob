"""Audit FI-2010 fold structure before training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .fi2010 import audit_fold_sizes, discover_fold_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root")
    parser.add_argument("--output", default="protocol_v2_audit.json")
    args = parser.parse_args()

    cache_root = Path(args.cache_root) if args.cache_root else None
    pairs = discover_fold_pairs(args.data_root, "ZScore")
    rows = audit_fold_sizes(pairs, cache_root)
    all_anchored = all(
        row["increment_matches_previous_test"] is not False for row in rows
    )
    final_train = int(rows[-1]["train_samples"])
    concatenated = sum(int(row["train_samples"]) for row in rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(Path(args.data_root).resolve()),
        "fold_count": len(rows),
        "anchored_growth_verified": all_anchored,
        "final_fold_train_samples": final_train,
        "concatenated_train_samples": concatenated,
        "concatenation_inflation": concatenated / final_train,
        "folds": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all_anchored:
        raise SystemExit("Fold growth does not match the anchored FI-2010 protocol")


if __name__ == "__main__":
    main()
