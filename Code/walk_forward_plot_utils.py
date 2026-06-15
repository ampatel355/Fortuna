"""Helpers for plotting current-run multi-asset walk-forward artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from artifact_provenance import (
        artifact_run_id,
        current_run_id,
        load_artifact_metadata,
        versioned_artifact_path,
    )
    from plot_config import data_clean_dir, load_csv_checked
    from strategy_config import AGENT_ORDER
except ModuleNotFoundError:
    from Code.artifact_provenance import (
        artifact_run_id,
        current_run_id,
        load_artifact_metadata,
        versioned_artifact_path,
    )
    from Code.plot_config import data_clean_dir, load_csv_checked
    from Code.strategy_config import AGENT_ORDER


WALK_FORWARD_RUNS_FILENAME = "multi_asset_walk_forward_runs.csv"
WALK_FORWARD_PANEL_FILENAME = "multi_asset_walk_forward_panel_summary.csv"
WALK_FORWARD_SUMMARY_FILENAME = "multi_asset_walk_forward_agent_summary.csv"


def resolve_current_run_artifact(filename: str) -> Path:
    """Return the current run's versioned artifact path for one walk-forward table."""
    canonical_path = data_clean_dir() / filename
    run_id = current_run_id()
    versioned_path = versioned_artifact_path(canonical_path, run_id=run_id)

    if versioned_path.exists():
        return versioned_path

    if canonical_path.exists() and artifact_run_id(canonical_path) == run_id:
        return canonical_path

    raise FileNotFoundError(
        "The current walk-forward run did not produce the expected artifact "
        f"for run_id={run_id}: {canonical_path}"
    )


def load_walk_forward_runs() -> tuple[pd.DataFrame, Path]:
    """Load the current run's per-run walk-forward table."""
    input_path = resolve_current_run_artifact(WALK_FORWARD_RUNS_FILENAME)
    df = load_csv_checked(
        input_path,
        required_columns=[
            "ticker",
            "fold_id",
            "fold_start",
            "fold_end",
            "agent",
            "actual_cumulative_return",
            "actual_percentile",
            "p_value",
            "RCSI_z",
            "outer_run",
        ],
    ).copy()
    df["fold_start"] = pd.to_datetime(df["fold_start"], errors="coerce")
    df["fold_end"] = pd.to_datetime(df["fold_end"], errors="coerce")
    numeric_columns = [
        "actual_cumulative_return",
        "actual_percentile",
        "p_value",
        "RCSI_z",
        "outer_run",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["fold_start", "fold_end", "agent"]).reset_index(drop=True)
    return df, input_path


def load_walk_forward_panel_summary() -> tuple[pd.DataFrame, Path]:
    """Load the current run's per-panel walk-forward summary table."""
    input_path = resolve_current_run_artifact(WALK_FORWARD_PANEL_FILENAME)
    df = load_csv_checked(
        input_path,
        required_columns=[
            "ticker",
            "fold_id",
            "fold_start",
            "fold_end",
            "agent",
            "mean_actual_cumulative_return",
            "mean_p_value",
            "mean_actual_percentile",
            "mean_RCSI_z",
        ],
    ).copy()
    df["fold_start"] = pd.to_datetime(df["fold_start"], errors="coerce")
    df["fold_end"] = pd.to_datetime(df["fold_end"], errors="coerce")
    numeric_columns = [
        "mean_actual_cumulative_return",
        "mean_p_value",
        "mean_actual_percentile",
        "mean_RCSI_z",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["fold_start", "fold_end", "agent"]).reset_index(drop=True)
    return df, input_path


def load_walk_forward_agent_summary() -> tuple[pd.DataFrame, Path]:
    """Load the current run's per-agent walk-forward summary table."""
    input_path = resolve_current_run_artifact(WALK_FORWARD_SUMMARY_FILENAME)
    df = load_csv_checked(
        input_path,
        required_columns=[
            "agent",
            "panel_count",
            "mean_mean_p_value",
            "mean_mean_actual_percentile",
            "mean_mean_RCSI_z",
            "evidence_label",
            "verdict_label",
            "confidence_label",
        ],
    ).copy()
    numeric_columns = [
        "panel_count",
        "mean_mean_p_value",
        "mean_mean_actual_percentile",
        "mean_mean_RCSI_z",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["agent"] = df["agent"].astype(str).str.strip()
    available_agents = [agent for agent in AGENT_ORDER if agent in df["agent"].tolist()]
    if available_agents:
        df = df.set_index("agent").reindex(available_agents).reset_index()
    return df.dropna(subset=["agent"]).reset_index(drop=True), input_path


def walk_forward_metadata(path: Path) -> dict:
    """Return metadata for one walk-forward artifact when available."""
    return load_artifact_metadata(path) or {}


def walk_forward_tickers(panel_df: pd.DataFrame, metadata: dict | None = None) -> list[str]:
    """Return the active ticker list for captions and titles."""
    metadata = metadata or {}
    parameter_tickers = metadata.get("parameters", {}).get("tickers")
    if isinstance(parameter_tickers, list) and parameter_tickers:
        return [str(ticker).strip().upper() for ticker in parameter_tickers if str(ticker).strip()]
    return sorted(
        {
            str(value).strip().upper()
            for value in panel_df.get("ticker", pd.Series(dtype="object")).dropna().tolist()
            if str(value).strip()
        }
    )
