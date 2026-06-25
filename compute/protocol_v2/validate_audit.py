#!/usr/bin/env python3
"""CPU-only validation gate for the anchored FI-2010 protocol."""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


T = 100
FOLDS = tuple(range(1, 10))
METRIC_TOL = 1e-4

RUNS_ROOT = Path(
    "/Volumes/Crucial X9/Research Projects/DL_Paper/experiments/new/"
    "DL_Paper_v2_complete/runs"
)
AUDIT_PATH = RUNS_ROOT / "protocol_audit.json"
REFERENCE_ROOT = RUNS_ROOT / "protocol_v2_long"

DATA_ROOT_SUFFIX = Path("BenchmarkDatasets/NoAuction/1.NoAuction_Zscore")
FIXED_DATA_ROOTS = (
    Path("/kaggle/input/fi-2010") / DATA_ROOT_SUFFIX,
    Path("/tmp/fi-2010") / DATA_ROOT_SUFFIX,
)

METRIC_NAMES = (
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "kappa",
    "nll",
    "brier",
    "ece",
    "error_auroc",
)


@dataclass(frozen=True)
class FoldCounts:
    fold: int
    train_rows: int
    test_rows: int
    split_row: int
    train_win: int
    val_win: int
    test_win: int
    summary_match: bool


@dataclass(frozen=True)
class SignatureRow:
    fold: int
    raw_increment: int
    previous_test_rows: int
    raw_ok: bool
    split_train_win_increment: int
    previous_test_win: int
    split_window_matches_previous_test_win: bool


def add_common_site_packages() -> None:
    """Add common user/Conda site-package paths before optional imports."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = (
        Path.home() / ".local" / "lib" / f"python{version}" / "site-packages",
        Path("/opt/conda/lib") / f"python{version}" / "site-packages",
        Path("/usr/local/lib") / f"python{version}" / "site-packages",
        Path(sys.prefix) / "lib" / f"python{version}" / "site-packages",
        Path(sys.exec_prefix) / "lib" / f"python{version}" / "site-packages",
    )
    for candidate in candidates:
        text = str(candidate)
        if candidate.exists() and text not in sys.path:
            sys.path.insert(0, text)


def import_required(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        add_common_site_packages()
        return importlib.import_module(module_name)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fail_missing(path: Path, failures: list[str], label: str) -> bool:
    if path.exists():
        return False
    failures.append(f"{label} not found: {path}")
    return True


def reference_dir(fold: int) -> Path:
    return (
        REFERENCE_ROOT
        / f"anchored_references__deeplob__f{fold}__k10__s0__none"
    )


def compute_windows(train_rows: int, test_rows: int) -> tuple[int, int, int, int]:
    split_row = int(train_rows * 0.9)
    train_win = split_row - T + 1
    val_win = (train_rows - split_row) - T + 1
    test_win = test_rows - T + 1
    return split_row, train_win, val_win, test_win


def load_audit_rows(failures: list[str]) -> dict[int, tuple[int, int]]:
    if fail_missing(AUDIT_PATH, failures, "Protocol audit"):
        return {}

    audit = load_json(AUDIT_PATH)
    raw_folds = audit.get("folds")
    if not isinstance(raw_folds, list):
        failures.append(f"Protocol audit has no folds[] array: {AUDIT_PATH}")
        return {}

    rows: dict[int, tuple[int, int]] = {}
    for item in raw_folds:
        try:
            fold = int(item["fold"])
            train_rows = int(item["train_samples"])
            test_rows = int(item["test_samples"])
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"Malformed fold entry in protocol audit: {item!r} ({exc})")
            continue
        rows[fold] = (train_rows, test_rows)

    missing = [fold for fold in FOLDS if fold not in rows]
    if missing:
        failures.append(f"Protocol audit missing folds: {missing}")
    return {fold: rows[fold] for fold in FOLDS if fold in rows}


def load_summary_counts(fold: int, failures: list[str]) -> tuple[int, int, int] | None:
    summary_path = reference_dir(fold) / "summary.json"
    if fail_missing(summary_path, failures, f"Fold {fold} deeplob summary"):
        return None

    summary = load_json(summary_path)
    data = summary.get("data")
    if not isinstance(data, dict):
        failures.append(f"Fold {fold} summary has no data object: {summary_path}")
        return None

    try:
        return (
            int(data["train_samples"]),
            int(data["validation_samples"]),
            int(data["test_samples"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"Fold {fold} summary data counts are malformed ({exc})")
        return None


def build_fold_counts(
    audit_rows: dict[int, tuple[int, int]], failures: list[str]
) -> list[FoldCounts]:
    counts: list[FoldCounts] = []

    for fold in FOLDS:
        if fold not in audit_rows:
            continue
        train_rows, test_rows = audit_rows[fold]
        split_row, train_win, val_win, test_win = compute_windows(
            train_rows, test_rows
        )

        if min(train_win, val_win, test_win) <= 0:
            failures.append(
                f"Fold {fold} produced non-positive windows: "
                f"train={train_win}, val={val_win}, test={test_win}"
            )

        summary_counts = load_summary_counts(fold, failures)
        summary_match = summary_counts == (train_win, val_win, test_win)
        if summary_counts is not None and not summary_match:
            failures.append(
                f"Fold {fold} deeplob count mismatch: computed "
                f"train/val/test={train_win}/{val_win}/{test_win}, stored "
                f"train/val/test={summary_counts[0]}/{summary_counts[1]}/"
                f"{summary_counts[2]}"
            )

        counts.append(
            FoldCounts(
                fold=fold,
                train_rows=train_rows,
                test_rows=test_rows,
                split_row=split_row,
                train_win=train_win,
                val_win=val_win,
                test_win=test_win,
                summary_match=summary_match,
            )
        )

    return counts


def check_anchored_signature(
    counts: list[FoldCounts], failures: list[str]
) -> list[SignatureRow]:
    by_fold = {item.fold: item for item in counts}
    rows: list[SignatureRow] = []

    for fold in range(2, 10):
        if fold not in by_fold or fold - 1 not in by_fold:
            continue
        current = by_fold[fold]
        previous = by_fold[fold - 1]

        raw_increment = current.train_rows - previous.train_rows
        raw_ok = raw_increment == previous.test_rows
        if not raw_ok:
            failures.append(
                f"Fold {fold} raw anchored increment mismatch: "
                f"{raw_increment} != previous test rows {previous.test_rows}"
            )

        split_train_win_increment = current.train_win - previous.train_win
        split_window_ok = split_train_win_increment == previous.test_win

        rows.append(
            SignatureRow(
                fold=fold,
                raw_increment=raw_increment,
                previous_test_rows=previous.test_rows,
                raw_ok=raw_ok,
                split_train_win_increment=split_train_win_increment,
                previous_test_win=previous.test_win,
                split_window_matches_previous_test_win=split_window_ok,
            )
        )

    return rows


def candidate_data_roots() -> list[Path]:
    roots = list(FIXED_DATA_ROOTS)
    env_root = os.environ.get("DATA_ROOT")
    if env_root:
        env_path = Path(env_root)
        roots.append(env_path / DATA_ROOT_SUFFIX)
        roots.append(env_path)

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def find_data_root() -> Path | None:
    for root in candidate_data_roots():
        if root.exists() and root.is_dir():
            return root
    return None


def find_fold_file(root: Path, fold: int, kind: str) -> Path | None:
    stem = f"{kind}_Dst_NoAuction_ZScore_CF_{fold}"
    direct_candidates = (
        root / f"NoAuction_Zscore_{'Training' if kind == 'Train' else 'Testing'}" / f"{stem}.txt",
        root / f"NoAuction_Zscore_{'Training' if kind == 'Train' else 'Testing'}" / f"{stem}.npy",
        root / f"{stem}.txt",
        root / f"{stem}.npy",
    )
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    matches = sorted(root.glob(f"**/{stem}.txt")) + sorted(root.glob(f"**/{stem}.npy"))
    return matches[0] if matches else None


def load_raw_row_count(path: Path) -> int:
    np = import_required("numpy")
    if path.suffix.lower() == ".npy":
        raw = np.load(path)
    else:
        raw = np.loadtxt(path, dtype=np.float32)

    if raw.ndim != 2 or raw.shape[0] != 149:
        raise ValueError(f"expected raw shape (149, num_rows), got {raw.shape}")

    features = raw[:40].T
    labels = (raw[144:149].T)[:, 4] - 1
    if len(features) != len(labels):
        raise ValueError(
            f"feature/label length mismatch: {len(features)} != {len(labels)}"
        )
    return int(len(features))


def validate_actual_files(
    data_root: Path | None,
    audit_rows: dict[int, tuple[int, int]],
    counts: list[FoldCounts],
    failures: list[str],
) -> list[tuple[int, str, str]]:
    if data_root is None:
        return []

    expected_counts = {item.fold: item for item in counts}
    rows: list[tuple[int, str, str]] = []

    for fold in FOLDS:
        train_path = find_fold_file(data_root, fold, "Train")
        test_path = find_fold_file(data_root, fold, "Test")
        if train_path is None or test_path is None:
            failures.append(
                f"Fold {fold} data files missing under {data_root}: "
                f"train={train_path}, test={test_path}"
            )
            rows.append((fold, "MISSING", "FAIL"))
            continue

        try:
            train_rows = load_raw_row_count(train_path)
            test_rows = load_raw_row_count(test_path)
        except Exception as exc:
            failures.append(f"Fold {fold} failed loading data files: {exc}")
            rows.append((fold, "LOAD_ERROR", "FAIL"))
            continue

        expected_train_rows, expected_test_rows = audit_rows[fold]
        row_ok = (
            train_rows == expected_train_rows and test_rows == expected_test_rows
        )

        split_row, train_win, val_win, test_win = compute_windows(
            train_rows, test_rows
        )
        expected = expected_counts[fold]
        window_ok = (
            split_row == expected.split_row
            and train_win == expected.train_win
            and val_win == expected.val_win
            and test_win == expected.test_win
        )

        if not row_ok:
            failures.append(
                f"Fold {fold} data row mismatch: actual "
                f"train/test={train_rows}/{test_rows}, audit "
                f"train/test={expected_train_rows}/{expected_test_rows}"
            )
        if not window_ok:
            failures.append(
                f"Fold {fold} data window mismatch after file load: actual "
                f"split/train/val/test={split_row}/{train_win}/{val_win}/{test_win}, "
                f"audit-derived {expected.split_row}/{expected.train_win}/"
                f"{expected.val_win}/{expected.test_win}"
            )

        rows.append(
            (
                fold,
                f"{train_rows}/{test_rows}",
                "PASS" if row_ok and window_ok else "FAIL",
            )
        )

    return rows


def to_numpy(value: Any, np: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def expected_calibration_error(
    np: Any, predictions: Any, labels: Any, confidence: Any, n_bins: int = 15
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    ece = 0.0
    for index in range(n_bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == 0:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence > lower) & (confidence <= upper)
        bin_count = int(in_bin.sum())
        if bin_count == 0:
            continue
        bin_acc = (predictions[in_bin] == labels[in_bin]).mean()
        bin_conf = confidence[in_bin].mean()
        ece += (bin_count / n) * abs(float(bin_acc) - float(bin_conf))
    return float(ece)


def compute_metrics_lock(failures: list[str]) -> list[tuple[str, str, str, str, str]]:
    fold = 9
    run_dir = reference_dir(fold)
    summary_path = run_dir / "summary.json"
    logits_path = run_dir / "test_logits.pt"

    if fail_missing(summary_path, failures, "Fold 9 summary for metrics lock"):
        return []
    if fail_missing(logits_path, failures, "Fold 9 test_logits.pt"):
        return []

    try:
        np = import_required("numpy")
        torch = import_required("torch")
        scipy_special = import_required("scipy.special")
        sklearn_metrics = import_required("sklearn.metrics")
    except ImportError as exc:
        failures.append(f"Metrics lock dependency missing: {exc}")
        return []

    summary = load_json(summary_path)
    stored = summary.get("test")
    if not isinstance(stored, dict):
        failures.append(f"Fold 9 summary has no test metrics object: {summary_path}")
        return []

    artifact = torch.load(logits_path, map_location="cpu")
    if not isinstance(artifact, dict) or "logits" not in artifact or "labels" not in artifact:
        failures.append(f"Fold 9 logits artifact must contain logits and labels: {logits_path}")
        return []

    logits = to_numpy(artifact["logits"], np)
    labels = to_numpy(artifact["labels"], np).astype(np.int64)

    if logits.ndim != 2:
        failures.append(f"Fold 9 logits must be rank 2, got shape {logits.shape}")
        return []
    if labels.ndim != 1 or len(labels) != logits.shape[0]:
        failures.append(
            f"Fold 9 labels shape mismatch: labels={labels.shape}, logits={logits.shape}"
        )
        return []
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        failures.append(
            f"Fold 9 labels outside class range 0..{logits.shape[1] - 1}"
        )
        return []

    probs = scipy_special.softmax(logits, axis=1)
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    one_hot = np.eye(probs.shape[1], dtype=probs.dtype)[labels]
    row_index = np.arange(len(labels))
    errors = predictions != labels

    computed = {
        "accuracy": float((predictions == labels).mean()),
        "f1_macro": float(
            sklearn_metrics.f1_score(labels, predictions, average="macro")
        ),
        "f1_weighted": float(
            sklearn_metrics.f1_score(labels, predictions, average="weighted")
        ),
        "kappa": float(sklearn_metrics.cohen_kappa_score(labels, predictions)),
        "nll": float(np.mean(-np.log(probs[row_index, labels]))),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece": expected_calibration_error(
            np, predictions, labels, confidence, n_bins=15
        ),
        "error_auroc": float(
            sklearn_metrics.roc_auc_score(errors.astype(np.int64), 1.0 - confidence)
        ),
    }

    rows: list[tuple[str, str, str, str, str]] = []
    for metric in METRIC_NAMES:
        if metric not in stored:
            failures.append(f"Fold 9 stored test metric missing: {metric}")
            continue
        stored_value = float(stored[metric])
        computed_value = computed[metric]
        diff = abs(computed_value - stored_value)
        passed = diff < METRIC_TOL
        if not passed:
            failures.append(
                f"Fold 9 metric mismatch for {metric}: computed={computed_value}, "
                f"stored={stored_value}, diff={diff}"
            )
        rows.append(
            (
                metric,
                f"{computed_value:.10f}",
                f"{stored_value:.10f}",
                f"{diff:.3e}",
                "PASS" if passed else "FAIL",
            )
        )

    return rows


def print_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    string_rows = [tuple(str(value) for value in row) for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in string_rows))
        if string_rows
        else len(headers[index])
        for index in range(len(headers))
    ]
    print(" | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in string_rows:
        print(" | ".join(row[index].ljust(widths[index]) for index in range(len(headers))))


def print_report(
    counts: list[FoldCounts],
    signature_rows: list[SignatureRow],
    data_root: Path | None,
    file_rows: list[tuple[int, str, str]],
    metric_rows: list[tuple[str, str, str, str, str]],
    failures: list[str],
) -> None:
    print("Per-fold window counts")
    print_table(
        (
            "fold",
            "train_rows",
            "split_row",
            "train_win",
            "val_win",
            "test_win",
            "match_existing?",
        ),
        [
            (
                item.fold,
                item.train_rows,
                item.split_row,
                item.train_win,
                item.val_win,
                item.test_win,
                "PASS" if item.summary_match else "FAIL",
            )
            for item in counts
        ],
    )

    print("\nAnchored signature check")
    print_table(
        (
            "fold",
            "row_inc",
            "prev_test_rows",
            "rows",
            "split_train_win_inc",
            "prev_test_win",
            "split_win_eq_prev_test_win",
        ),
        [
            (
                item.fold,
                item.raw_increment,
                item.previous_test_rows,
                "PASS" if item.raw_ok else "FAIL",
                item.split_train_win_increment,
                item.previous_test_win,
                "PASS" if item.split_window_matches_previous_test_win else "INFO",
            )
            for item in signature_rows
        ],
    )
    print(
        "Note: split_train_win_inc is informational because the protocol splits "
        "rows 90/10 before windowing; the hard anchored assertion is row_inc == "
        "prev_test_rows."
    )

    print("\nActual FI-2010 file row-count check")
    if data_root is None:
        considered = ", ".join(str(path) for path in candidate_data_roots())
        print(f"SKIP: no FI-2010 data root found. Considered: {considered}")
        print("Formula-only validation is using protocol_audit.json raw row counts.")
    else:
        print(f"Data root: {data_root}")
        print_table(("fold", "actual_train/test_rows", "status"), file_rows)

    print("\nMetrics lock")
    if metric_rows:
        print_table(("metric", "computed", "stored", "diff", "status"), metric_rows)
    else:
        print("FAIL: metrics lock could not be computed.")

    if failures:
        print("\nAUDIT GATE: FAIL")
        print("Reasons:")
        for failure in failures:
            print(f"- {failure}")
        print("STOP: Do not train. Fix the protocol first.")
    else:
        print("\nAUDIT GATE: PASS")
        print("ALL GATES PASSED. Ready for GPU training.")


def main() -> int:
    failures: list[str] = []
    audit_rows = load_audit_rows(failures)
    counts = build_fold_counts(audit_rows, failures) if audit_rows else []
    signature_rows = check_anchored_signature(counts, failures) if counts else []

    data_root = find_data_root()
    file_rows = validate_actual_files(data_root, audit_rows, counts, failures)

    metric_rows = compute_metrics_lock(failures)
    print_report(counts, signature_rows, data_root, file_rows, metric_rows, failures)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
