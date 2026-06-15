"""Shared helpers for single-ticker strategy wrapper scripts."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

try:
    from artifact_provenance import write_dataframe_artifact
    from regimes import main as create_regimes
    from timeframe_config import RESEARCH_INTERVAL, RESEARCH_TIMEFRAME_LABEL, infer_months_covered, normalize_timestamp_series
except ModuleNotFoundError:
    from Code.artifact_provenance import write_dataframe_artifact
    from Code.regimes import main as create_regimes
    from Code.timeframe_config import RESEARCH_INTERVAL, RESEARCH_TIMEFRAME_LABEL, infer_months_covered, normalize_timestamp_series


ticker = os.environ.get("TICKER", "SPY")
SAVE_DEBUG_OUTPUTS = os.environ.get("SAVE_DEBUG_OUTPUTS", "0") == "1"


def load_regime_data(input_path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a regime-tagged market table, creating it first when needed."""
    if not input_path.exists():
        create_regimes()

    df = pd.read_csv(input_path)
    interval_is_compatible = (
        df.get("data_interval", pd.Series(dtype="object")).astype(str).str.lower().eq(RESEARCH_INTERVAL).any()
        if "data_interval" in df.columns
        else False
    )
    if df.empty or any(column not in df.columns for column in required_columns) or not interval_is_compatible:
        create_regimes()
        df = pd.read_csv(input_path)

    df["Date"] = normalize_timestamp_series(df["Date"])
    numeric_columns = [column for column in required_columns if column not in {"Date", "regime"}]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    cleaned_df = (
        df.dropna(subset=required_columns)
        .sort_values("Date", ascending=True)
        .reset_index(drop=True)
    )
    return cleaned_df


def save_trade_outputs(
    *,
    current_ticker: str,
    agent_name: str,
    trades_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Save the standard trade file and a compact signal summary.

    Debug CSVs are useful while developing a strategy, but they create a large
    amount of output noise during routine research runs. They are therefore
    opt-in through SAVE_DEBUG_OUTPUTS=1.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trade_output_path = output_dir / f"{current_ticker}_{agent_name}_trades.csv"
    debug_output_path = output_dir / f"{current_ticker}_{agent_name}_trade_debug.csv"
    legacy_debug_output_path = output_dir / f"{current_ticker}_{agent_name}_debug.csv"
    signal_summary_path = output_dir / f"{current_ticker}_{agent_name}_signal_summary.csv"

    trades_to_save = trades_df.copy()
    trades_to_save["timeframe_label"] = RESEARCH_TIMEFRAME_LABEL
    trades_to_save["data_interval"] = RESEARCH_INTERVAL
    write_dataframe_artifact(
        trades_to_save,
        trade_output_path,
        producer="save_trade_outputs",
        current_ticker=current_ticker,
        research_grade=True,
        canonical_policy="always",
        parameters={
            "agent_name": agent_name,
            "artifact_type": "trade_log",
        },
    )

    debug_df = trades_df.copy()
    if not debug_df.empty:
        debug_df["entry_time"] = pd.to_datetime(debug_df["entry_date"], errors="coerce")
        debug_df["exit_time"] = pd.to_datetime(debug_df["exit_date"], errors="coerce")
        if "holding_period_hours" not in debug_df.columns:
            debug_df["holding_period_hours"] = (
                (debug_df["exit_time"] - debug_df["entry_time"]).dt.total_seconds() / 3600.0
            )
        debug_df["holding_period_bars"] = pd.to_numeric(
            debug_df.get("holding_bars"),
            errors="coerce",
        )
    debug_columns = [
        "asset_ticker",
        "strategy_name",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "return",
        "holding_period_bars",
        "holding_period_days",
        "holding_period_hours",
        "entry_reason",
        "exit_reason",
        "regime_at_entry",
        "stop_loss_used",
        "take_profit_used",
        "capital_before",
        "capital_after",
    ]
    existing_debug_columns = [column for column in debug_columns if column in debug_df.columns]
    if SAVE_DEBUG_OUTPUTS:
        debug_df[existing_debug_columns].to_csv(debug_output_path, index=False)
        debug_df[existing_debug_columns].to_csv(legacy_debug_output_path, index=False)

    date_source = trades_df["entry_date"] if "entry_date" in trades_df.columns else pd.Series(dtype="datetime64[ns]")
    months_covered = infer_months_covered(date_source)
    signal_summary_df = pd.DataFrame(
        [
            {
                "ticker": current_ticker,
                "agent": agent_name,
                "timeframe_label": RESEARCH_TIMEFRAME_LABEL,
                "data_interval": RESEARCH_INTERVAL,
                "signal_count": int(trades_df.attrs.get("signal_count", len(trades_df))),
                "executed_trade_count": int(len(trades_df)),
                "rejected_signal_count": int(trades_df.attrs.get("rejected_signal_count", 0)),
                "months_covered": float(months_covered),
            }
        ]
    )
    write_dataframe_artifact(
        signal_summary_df,
        signal_summary_path,
        producer="save_trade_outputs",
        current_ticker=current_ticker,
        research_grade=True,
        canonical_policy="always",
        parameters={
            "agent_name": agent_name,
            "artifact_type": "signal_summary",
        },
    )
    return trade_output_path, debug_output_path, signal_summary_path
