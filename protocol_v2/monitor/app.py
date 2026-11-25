"""Live Streamlit dashboard for Protocol-V2 training campaigns."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from protocol_v2.io_utils import read_jsonl


st.set_page_config(page_title="Arti Training Monitor", page_icon="A", layout="wide")

COLORS = {
    "teal": "#2dd4bf",
    "amber": "#f59e0b",
    "coral": "#fb7185",
    "sky": "#38bdf8",
    "violet": "#a78bfa",
    "green": "#4ade80",
    "muted": "#87928d",
}

st.markdown(
    """
    <style>
    .stApp { background: #101312; color: #edf3ef; }
    [data-testid="stHeader"] { background: rgba(16,19,18,.96); }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1600px; }
    [data-testid="stMetric"] {
        background: #181d1a; border: 1px solid #2d3732; border-radius: 6px;
        padding: 12px 14px; min-height: 106px;
    }
    [data-testid="stMetricLabel"] { color: #9da9a3 !important; }
    [data-testid="stMetricValue"] { color: #f5faf7 !important; }
    [data-testid="stMetricDelta"] { color: #2dd4bf !important; }
    button[data-baseweb="tab"] { padding-top: .75rem; padding-bottom: .75rem; }
    button[data-baseweb="tab"] p { color: #aab5af !important; font-weight: 600; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #2dd4bf !important; }
    h1, h2, h3 { letter-spacing: 0; color: #f2f7f4; }
    h1 { font-size: 2rem !important; margin-bottom: .2rem !important; }
    h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1rem !important; }
    .monitor-kicker { color: #2dd4bf; font-size: .72rem; font-weight: 700; text-transform: uppercase; }
    .monitor-subtitle { color: #93a099; margin-bottom: 1rem; }
    .status-row { display: flex; align-items: center; gap: .55rem; flex-wrap: wrap; margin: .35rem 0 1rem; }
    .status-badge {
        display: inline-flex; align-items: center; gap: .4rem; border: 1px solid #34413a;
        border-radius: 999px; padding: .26rem .62rem; color: #cbd5d0;
        background: #181d1a; font-size: .78rem;
    }
    .status-dot { width: .48rem; height: .48rem; border-radius: 50%; display: inline-block; }
    .section-note { color: #8d9993; font-size: .84rem; margin-top: -.4rem; margin-bottom: .75rem; }
    [data-testid="stDataFrame"] { border: 1px solid #29332e; border-radius: 6px; overflow: hidden; }
    div[data-baseweb="select"] > div { border-radius: 6px; }
    code { font-size: .78rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def discover_runs(runs_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in runs_root.glob("*/*/status.json"):
        payload = load_json(path)
        if not payload:
            continue
        payload["_run_dir"] = str(path.parent)
        rows.append(payload)
    return sorted(rows, key=lambda row: row.get("updated_at", row.get("completed_at", "")), reverse=True)


def campaign_statuses(runs_root: Path) -> List[Dict[str, Any]]:
    statuses = []
    for path in runs_root.glob("*/campaign_status.json"):
        payload = load_json(path)
        if payload:
            payload["_campaign_dir"] = str(path.parent)
            statuses.append(payload)
    return sorted(statuses, key=lambda row: row.get("updated_at", ""), reverse=True)


def parse_time(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(parsed) else parsed


def age_seconds(value: Any) -> float | None:
    parsed = parse_time(value)
    if parsed is None:
        return None
    return max((pd.Timestamp.now(tz="UTC") - parsed).total_seconds(), 0.0)


def latest_gpu(telemetry: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not telemetry:
        return {}
    gpus = telemetry[-1].get("gpus", [])
    return gpus[0] if gpus else {}


def metric_frame(path: Path) -> pd.DataFrame:
    events = [event for event in read_jsonl(path) if event.get("type") == "epoch"]
    frame = pd.DataFrame(events)
    if not frame.empty and "epoch" in frame:
        frame = frame.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
    return frame


def format_duration(seconds: Any) -> str:
    try:
        seconds = max(int(float(seconds)), 0)
    except (TypeError, ValueError):
        return "-"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def current_run_for_campaign(
    runs: Sequence[Dict[str, Any]], active_campaign: Dict[str, Any]
) -> Dict[str, Any]:
    current_id = active_campaign.get("current_run")
    if current_id:
        exact = next((run for run in runs if run.get("run_id") == current_id), None)
        if exact:
            return exact
    return next((run for run in runs if run.get("state") == "running"), runs[0] if runs else {})


def run_health(current: Dict[str, Any], active_campaign: Dict[str, Any]) -> Dict[str, Any]:
    state = str(current.get("state", "idle"))
    progress = current.get("progress", {})
    age = age_seconds(current.get("updated_at", current.get("completed_at")))
    epoch_seconds = float(progress.get("elapsed_seconds") or 0)
    fresh_limit = max(180.0, epoch_seconds * 3.0)
    campaign_age = age_seconds(active_campaign.get("updated_at"))

    if state == "failed":
        return {"label": "Failed", "color": COLORS["coral"], "score": 0, "age": age}
    if active_campaign.get("state") != "running":
        return {"label": "Idle", "color": COLORS["muted"], "score": 0, "age": age}
    if state == "initializing":
        return {"label": "Starting", "color": COLORS["amber"], "score": 45, "age": age}
    if age is None:
        return {"label": "No heartbeat", "color": COLORS["coral"], "score": 10, "age": age}
    if age <= fresh_limit:
        return {"label": "Healthy", "color": COLORS["green"], "score": 100, "age": age}
    if campaign_age is not None and campaign_age <= 90 and age <= fresh_limit * 4:
        return {"label": "Resuming", "color": COLORS["amber"], "score": 65, "age": age}
    if age <= fresh_limit * 4:
        return {"label": "Delayed", "color": COLORS["amber"], "score": 35, "age": age}
    return {"label": "Stale", "color": COLORS["coral"], "score": 10, "age": age}


def gpu_frame(telemetry: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in telemetry:
        if row.get("gpus"):
            rows.append({"timestamp": row.get("timestamp"), **row["gpus"][0]})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        frame = frame.dropna(subset=["timestamp"])
    return frame


def chart_lines(
    frame: pd.DataFrame,
    x: str,
    columns: Sequence[str],
    labels: Dict[str, str],
    colors: Sequence[str],
    *,
    height: int = 280,
    y_title: str | None = None,
) -> None:
    selected = [column for column in columns if column in frame.columns]
    if frame.empty or not selected or x not in frame:
        st.info("Waiting for data")
        return
    plot = frame[[x, *selected]].copy().melt(id_vars=[x], var_name="metric", value_name="value")
    plot["metric"] = plot["metric"].map(labels).fillna(plot["metric"])
    plot = plot.dropna(subset=["value"])
    domain = [labels.get(column, column) for column in selected]
    x_encoding = f"{x}:Q" if pd.api.types.is_numeric_dtype(plot[x]) else f"{x}:T"
    chart = (
        alt.Chart(plot)
        .mark_line(strokeWidth=2, interpolate="monotone")
        .encode(
            x=alt.X(x_encoding, title=None),
            y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "metric:N",
                title=None,
                scale=alt.Scale(domain=domain, range=list(colors)[: len(domain)]),
                legend=alt.Legend(orient="top", direction="horizontal"),
            ),
            tooltip=[alt.Tooltip(x_encoding), alt.Tooltip("metric:N"), alt.Tooltip("value:Q", format=".5f")],
        )
        .properties(height=height)
        .interactive(bind_y=False)
    )
    st.altair_chart(chart, use_container_width=True)


def completed_frame(runs: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        if run.get("state") != "completed":
            continue
        config = run.get("config", {})
        validation = run.get("validation", {})
        test = run.get("test", {})
        rows.append(
            {
                "run": run.get("run_id"),
                "model": config.get("model"),
                "fold": config.get("fold"),
                "seed": config.get("seed"),
                "micro": config.get("micro_mode"),
                "best_epoch": run.get("best_epoch"),
                "duration_min": (run.get("duration_seconds") or 0) / 60,
                "val_f1": validation.get("f1_macro"),
                "test_f1": test.get("f1_macro"),
                "test_accuracy": test.get("accuracy"),
                "test_kappa": test.get("kappa"),
                "test_ece": test.get("ece"),
                "test_nll": test.get("nll"),
            }
        )
    return pd.DataFrame(rows)


def attempt_summary(runs: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    attempts = 0
    retried_runs = 0
    exhausted = 0
    for run in runs:
        events = read_jsonl(Path(run["_run_dir"]) / "attempts.jsonl")
        started = [event for event in events if event.get("type") == "attempt_started"]
        attempts += len(started)
        retried_runs += int(len(started) > 1)
        exhausted += int(any(event.get("succeeded") is False for event in events) and run.get("state") == "failed")
    return {"attempts": attempts, "retried_runs": retried_runs, "exhausted": exhausted}


def campaign_eta_seconds(runs: Sequence[Dict[str, Any]], remaining_after_current: int) -> float | None:
    durations = [float(run["duration_seconds"]) for run in runs if run.get("state") == "completed" and run.get("duration_seconds")]
    if not durations:
        return None
    return float(pd.Series(durations).median()) * max(remaining_after_current, 0)


def effective_state(run: Dict[str, Any], current_id: str | None) -> str:
    state = str(run.get("state", "unknown"))
    if state == "running" and run.get("run_id") != current_id:
        return "interrupted"
    return state


@st.fragment(run_every="3s")
def live_dashboard() -> None:
    default_root = Path.cwd() / "runs"
    runs_root = Path(os.environ.get("DL_PAPER_RUNS_ROOT", default_root)).resolve()
    runs = discover_runs(runs_root)
    campaigns = campaign_statuses(runs_root)
    telemetry = read_jsonl(runs_root / "telemetry.jsonl", limit=1800)
    gpu = latest_gpu(telemetry)
    active_campaign = campaigns[0] if campaigns else {}
    current = current_run_for_campaign(runs, active_campaign)
    progress = current.get("progress", {})
    health = run_health(current, active_campaign)

    total = int(active_campaign.get("total_runs") or 0)
    current_index = int(active_campaign.get("current_index") or 0)
    position = min(current_index + 1, total) if total else 0
    remaining_after_current = max(total - position, 0)
    completed = int(active_campaign.get("completed") or 0)
    failed = int(active_campaign.get("failed") or 0)
    completion_ratio = position / total if total else 0.0

    st.markdown("<div class='monitor-kicker'>Protocol V2 / Live operations</div>", unsafe_allow_html=True)
    st.title("Arti Training Monitor")
    st.markdown(
        "<div class='monitor-subtitle'>Campaign progress, model quality, calibration and hardware telemetry.</div>",
        unsafe_allow_html=True,
    )

    columns = st.columns(7)
    columns[0].metric("Campaign", str(active_campaign.get("state", "idle")).title())
    columns[1].metric("Queue position", f"{position}/{total}" if total else "-")
    columns[2].metric("Pending after run", remaining_after_current)
    columns[3].metric("Epoch", f"{progress.get('epoch', 0)}/{progress.get('epochs', 0)}")
    columns[4].metric("Health", health["label"])
    columns[5].metric("GPU", f"{gpu.get('gpu_utilization', 0):.0f}%" if gpu else "-")
    columns[6].metric("Run ETA", format_duration(progress.get("eta_seconds")))

    current_id = active_campaign.get("current_run", current.get("run_id", "-"))
    heartbeat_badge = (
        f"<span class='status-badge'>{health['age']:.0f}s heartbeat</span>"
        if health["age"] is not None
        else ""
    )
    st.markdown(
        "<div class='status-row'>"
        f"<span class='status-badge'><span class='status-dot' style='background:{health['color']}'></span>{health['label']}</span>"
        f"{heartbeat_badge}"
        f"<span class='status-badge'>Run {current_id}</span>"
        f"<span class='status-badge'>Attempt {active_campaign.get('current_attempt', '-')}/{active_campaign.get('max_run_attempts', '-')}</span>"
        f"<span class='status-badge'>{failed} failed</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(max(completion_ratio, 0.0), 1.0), text=f"Campaign position {position} of {total}")

    overview_tab, training_tab, system_tab, experiments_tab, logs_tab = st.tabs(
        ["Overview", "Training", "System", "Experiments", "Logs"]
    )

    metrics = metric_frame(Path(current.get("_run_dir", ".")) / "metrics.jsonl") if current else pd.DataFrame()
    complete = completed_frame(runs)
    attempts = attempt_summary(runs)

    with overview_tab:
        left, right = st.columns([1.45, 1])
        with left:
            st.subheader("Current run trajectory")
            st.markdown("<div class='section-note'>Optimization and validation loss for the controller-selected run.</div>", unsafe_allow_html=True)
            chart_lines(
                metrics,
                "epoch",
                ["train_loss", "val_nll", "best_val_nll"],
                {"train_loss": "Train loss", "val_nll": "Validation NLL", "best_val_nll": "Best validation NLL"},
                [COLORS["sky"], COLORS["amber"], COLORS["teal"]],
                height=310,
                y_title="Loss",
            )
        with right:
            st.subheader("Campaign reliability")
            state_rows = [effective_state(run, current_id) for run in runs]
            state_counts = pd.Series(state_rows).value_counts().rename_axis("state").reset_index(name="runs")
            if not state_counts.empty:
                state_chart = (
                    alt.Chart(state_counts)
                    .mark_bar(cornerRadiusEnd=3)
                    .encode(
                        y=alt.Y("state:N", sort="-x", title=None),
                        x=alt.X("runs:Q", title="Runs"),
                        color=alt.Color(
                            "state:N",
                            title=None,
                            scale=alt.Scale(
                                domain=["completed", "running", "initializing", "interrupted", "failed"],
                                range=[COLORS["teal"], COLORS["sky"], COLORS["amber"], COLORS["violet"], COLORS["coral"]],
                            ),
                            legend=None,
                        ),
                        tooltip=["state:N", "runs:Q"],
                    )
                    .properties(height=210)
                )
                st.altair_chart(state_chart, use_container_width=True)
            reliability = st.columns(3)
            reliability[0].metric("Completed this session", completed)
            reliability[1].metric("Retried runs", attempts["retried_runs"])
            reliability[2].metric("Exhausted", attempts["exhausted"])

        eta = campaign_eta_seconds(runs, remaining_after_current)
        summary_cols = st.columns(5)
        summary_cols[0].metric("Best validation NLL", format_number(progress.get("best_val_nll"), 4))
        summary_cols[1].metric("Validation F1", format_number(progress.get("val_f1_macro"), 4))
        summary_cols[2].metric("Calibration ECE", format_number(progress.get("val_ece"), 4))
        summary_cols[3].metric("Mean epoch", format_duration(metrics["elapsed_seconds"].tail(10).mean()) if "elapsed_seconds" in metrics else "-")
        summary_cols[4].metric("Campaign ETA", format_duration(eta))

        st.subheader("Run configuration")
        config = current.get("config", {})
        config_rows = {
            "Model": config.get("model"),
            "Fold": config.get("fold"),
            "Horizon": config.get("horizon"),
            "Seed": config.get("seed"),
            "Micro mode": config.get("micro_mode"),
            "Objective": config.get("objective", config.get("loss", "default")),
            "Batch size": config.get("batch_size"),
            "Learning rate": progress.get("learning_rate", config.get("learning_rate")),
        }
        st.dataframe(pd.DataFrame([config_rows]), hide_index=True, use_container_width=True)

    with training_tab:
        st.subheader("Optimization")
        optimization_left, optimization_right = st.columns(2)
        with optimization_left:
            chart_lines(
                metrics, "epoch", ["train_loss", "val_nll"],
                {"train_loss": "Train loss", "val_nll": "Validation NLL"},
                [COLORS["sky"], COLORS["amber"]], y_title="Loss",
            )
        with optimization_right:
            chart_lines(
                metrics, "epoch", ["train_accuracy", "val_accuracy", "val_f1_macro", "val_kappa"],
                {"train_accuracy": "Train accuracy", "val_accuracy": "Validation accuracy", "val_f1_macro": "Macro F1", "val_kappa": "Kappa"},
                [COLORS["sky"], COLORS["teal"], COLORS["amber"], COLORS["violet"]], y_title="Score",
            )

        st.subheader("Calibration and stability")
        calibration_left, calibration_right = st.columns(2)
        with calibration_left:
            chart_lines(
                metrics, "epoch", ["val_ece", "val_brier"],
                {"val_ece": "Expected calibration error", "val_brier": "Brier score"},
                [COLORS["coral"], COLORS["amber"]], y_title="Error",
            )
        with calibration_right:
            chart_lines(
                metrics, "epoch", ["learning_rate", "elapsed_seconds", "skipped_batches"],
                {"learning_rate": "Learning rate", "elapsed_seconds": "Epoch seconds", "skipped_batches": "Skipped batches"},
                [COLORS["teal"], COLORS["sky"], COLORS["coral"]], y_title="Value",
            )

        if not metrics.empty:
            st.subheader("Epoch detail")
            detail_columns = [
                "epoch", "train_loss", "train_accuracy", "val_nll", "val_accuracy",
                "val_f1_macro", "val_kappa", "val_ece", "val_brier", "learning_rate",
                "elapsed_seconds", "skipped_batches",
            ]
            st.dataframe(metrics[[column for column in detail_columns if column in metrics]].tail(25), hide_index=True, use_container_width=True)

    with system_tab:
        host = pd.DataFrame(telemetry)
        if not host.empty:
            host["timestamp"] = pd.to_datetime(host["timestamp"], errors="coerce", utc=True)
            host = host.dropna(subset=["timestamp"])
        gpu_history = gpu_frame(telemetry)

        hardware = st.columns(6)
        hardware[0].metric("GPU utilization", f"{gpu.get('gpu_utilization', 0):.0f}%" if gpu else "-")
        hardware[1].metric("VRAM", f"{gpu.get('gpu_memory_used_mb', 0) / 1024:.1f}/{gpu.get('gpu_memory_total_mb', 0) / 1024:.0f} GB" if gpu else "-")
        hardware[2].metric("Temperature", f"{gpu.get('gpu_temperature_c', 0):.0f} C" if gpu else "-")
        hardware[3].metric("GPU power", f"{gpu.get('gpu_power_w', 0):.0f} W" if gpu else "-")
        hardware[4].metric("Host memory", f"{telemetry[-1].get('memory_percent', 0):.0f}%" if telemetry else "-")
        hardware[5].metric("Disk", f"{telemetry[-1].get('disk_percent', 0):.0f}%" if telemetry else "-")

        st.subheader("GPU telemetry")
        gpu_left, gpu_right = st.columns(2)
        with gpu_left:
            chart_lines(
                gpu_history, "timestamp", ["gpu_utilization", "gpu_memory_utilization"],
                {"gpu_utilization": "Compute utilization", "gpu_memory_utilization": "Memory controller"},
                [COLORS["teal"], COLORS["sky"]], y_title="Percent",
            )
        with gpu_right:
            chart_lines(
                gpu_history, "timestamp", ["gpu_temperature_c", "gpu_power_w"],
                {"gpu_temperature_c": "Temperature C", "gpu_power_w": "Power W"},
                [COLORS["coral"], COLORS["amber"]], y_title="Reading",
            )

        st.subheader("Host telemetry")
        chart_lines(
            host, "timestamp", ["cpu_percent", "memory_percent", "disk_percent"],
            {"cpu_percent": "CPU", "memory_percent": "Memory", "disk_percent": "Disk"},
            [COLORS["sky"], COLORS["amber"], COLORS["violet"]], y_title="Percent", height=300,
        )

    with experiments_tab:
        if not runs:
            st.info("No runs discovered")
        else:
            table_rows = []
            for run in runs:
                config = run.get("config", {})
                table_rows.append(
                    {
                        "state": effective_state(run, current_id),
                        "run": run.get("run_id"),
                        "model": config.get("model"),
                        "fold": config.get("fold"),
                        "horizon": config.get("horizon"),
                        "seed": config.get("seed"),
                        "micro": config.get("micro_mode"),
                        "best_epoch": run.get("best_epoch", run.get("progress", {}).get("best_epoch")),
                        "updated": run.get("updated_at", run.get("completed_at")),
                    }
                )
            queue_frame = pd.DataFrame(table_rows)
            filters = ["all", *sorted(queue_frame["state"].dropna().unique().tolist())]
            selected_state = st.selectbox("Run state", filters, index=0)
            visible_queue = queue_frame if selected_state == "all" else queue_frame[queue_frame["state"] == selected_state]
            st.dataframe(visible_queue, hide_index=True, use_container_width=True, height=360)

        st.subheader("Completed-run leaderboard")
        if complete.empty:
            st.info("Completed summaries will appear here")
        else:
            leaderboard_left, leaderboard_right = st.columns([1.4, 1])
            with leaderboard_left:
                ranked = complete.sort_values("test_f1", ascending=False, na_position="last")
                display_columns = ["model", "fold", "seed", "micro", "test_f1", "test_accuracy", "test_kappa", "test_ece", "test_nll", "duration_min", "run"]
                st.dataframe(ranked[display_columns].head(40), hide_index=True, use_container_width=True, height=390)
            with leaderboard_right:
                model_scores = complete.groupby("model", dropna=False)["test_f1"].agg(["mean", "std", "count"]).reset_index()
                model_chart = (
                    alt.Chart(model_scores.dropna(subset=["mean"]))
                    .mark_bar(cornerRadiusEnd=3, color=COLORS["teal"])
                    .encode(
                        y=alt.Y("model:N", sort="-x", title=None),
                        x=alt.X("mean:Q", title="Mean test macro F1", scale=alt.Scale(zero=False)),
                        tooltip=["model:N", alt.Tooltip("mean:Q", format=".4f"), alt.Tooltip("std:Q", format=".4f"), "count:Q"],
                    )
                    .properties(height=350)
                )
                st.altair_chart(model_chart, use_container_width=True)

    with logs_tab:
        if current:
            log_path = Path(current["_run_dir"]) / "console.log"
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-140:]
                st.code("\n".join(lines), language="text")
            else:
                st.info("Waiting for console output")
        else:
            st.info("No active run")
        with st.expander("Raw current status"):
            st.json({key: value for key, value in current.items() if key != "_run_dir"})


live_dashboard()
