"""Assign forward-safe volatility regimes to one ticker's feature table."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from features import build_features_for_ticker
    from strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        MEAN_REVERSION_STOP_LOW_COLUMN,
        RELATIVE_STRENGTH_RETURN_COLUMN,
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
    )
except ModuleNotFoundError:
    from Code.features import build_features_for_ticker
    from Code.strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        MEAN_REVERSION_STOP_LOW_COLUMN,
        RELATIVE_STRENGTH_RETURN_COLUMN,
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
    )
try:
    from timeframe_config import RESEARCH_INTERVAL, RESEARCH_TIMEFRAME_LABEL, normalize_timestamp_series, scale_daily_bars, timeframe_output_suffix
except ModuleNotFoundError:
    from Code.timeframe_config import RESEARCH_INTERVAL, RESEARCH_TIMEFRAME_LABEL, normalize_timestamp_series, scale_daily_bars, timeframe_output_suffix


ticker = os.environ.get("TICKER", "SPY")
_regime_history_explicit = os.environ.get("MIN_REGIME_HISTORY")
MIN_REGIME_HISTORY = (
    int(_regime_history_explicit)
    if _regime_history_explicit is not None
    else scale_daily_bars(120)
)


def resolve_data_clean_dir(project_root: Path) -> Path:
    """Return the project's clean-data folder, preferring the uppercase path."""
    suffix = timeframe_output_suffix()
    lowercase_dir = project_root / f"data_clean{suffix}"
    uppercase_dir = project_root / f"Data_Clean{suffix}"

    if uppercase_dir.exists():
        return uppercase_dir
    if lowercase_dir.exists():
        return lowercase_dir

    uppercase_dir.mkdir(parents=True, exist_ok=True)
    return uppercase_dir


def load_feature_data(input_path: Path, current_ticker: str | None = None) -> pd.DataFrame:
    """Load one feature table, creating it first when needed."""
    current_ticker = current_ticker or ticker
    if not input_path.exists():
        build_features_for_ticker(current_ticker, save_output=True)

    df = pd.read_csv(input_path)
    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "daily_return",
        "ema_20",
        "sma_20",
        "sma_50",
        "sma_100",
        "sma_200",
        "ma_20",
        "ma_50",
        "ma_100",
        "ma_200",
        "std_20",
        "price_std_20",
        "avg_volume_20",
        "volume_ratio_20",
        "rsi_14",
        "atr_14",
        "atr_percent",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "bollinger_mid",
        "bollinger_upper",
        "bollinger_lower",
        "bollinger_width",
        "macd_line",
        "macd_signal",
        "macd_hist",
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        RELATIVE_STRENGTH_RETURN_COLUMN,
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
        MEAN_REVERSION_STOP_LOW_COLUMN,
        "atr_percent_ratio_60",
        "bollinger_width_ratio_60",
        "distance_from_ma_20",
        "distance_from_ma_50",
        "ma_50_over_ma_200",
        "zscore_20",
    ]
    if df.empty or any(column not in df.columns for column in required_columns):
        df = build_features_for_ticker(current_ticker, save_output=True)

    interval_is_compatible = (
        df.get("data_interval", pd.Series(dtype="object")).astype(str).str.lower().eq(RESEARCH_INTERVAL).any()
        if "data_interval" in df.columns
        else False
    )
    if not interval_is_compatible:
        df = build_features_for_ticker(current_ticker, save_output=True)

    df["Date"] = normalize_timestamp_series(df["Date"])
    numeric_columns = [column for column in required_columns if column != "Date"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return (
        df.dropna(subset=["Date", *numeric_columns])
        .sort_values("Date", ascending=True)
        .reset_index(drop=True)
    )


def add_regime_labels(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add calm / neutral / stressed labels using only past information."""
    df = feature_df.copy()
    df["timeframe_label"] = RESEARCH_TIMEFRAME_LABEL
    df["data_interval"] = RESEARCH_INTERVAL

    # We use expanding quantiles of recent close-to-close bar volatility. Each
    # threshold is shifted by one bar, so the label for the current bar depends
    # only on information that was already known before the current bar closed.
    lower_quantile = (
        df["std_20"]
        .expanding(min_periods=MIN_REGIME_HISTORY)
        .quantile(1 / 3)
        .shift(1)
    )
    upper_quantile = (
        df["std_20"]
        .expanding(min_periods=MIN_REGIME_HISTORY)
        .quantile(2 / 3)
        .shift(1)
    )

    df["regime"] = pd.Series(np.nan, index=df.index, dtype="object")
    valid_rows = ~(lower_quantile.isna() | upper_quantile.isna())
    df.loc[valid_rows, "regime"] = "neutral"
    df.loc[valid_rows & (df["std_20"] <= lower_quantile), "regime"] = "calm"
    df.loc[valid_rows & (df["std_20"] >= upper_quantile), "regime"] = "stressed"
    return df.dropna(subset=["regime"]).reset_index(drop=True)


def build_regime_dataframe_for_ticker(
    current_ticker: str,
    *,
    project_root: Path | None = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """Build the regime table for one ticker and optionally save it."""
    project_root = project_root or Path(__file__).resolve().parents[1]
    output_dir = resolve_data_clean_dir(project_root)
    input_path = output_dir / f"{current_ticker}_features.csv"
    output_path = output_dir / f"{current_ticker}_regimes.csv"

    feature_df = load_feature_data(input_path, current_ticker=current_ticker)
    regime_df = add_regime_labels(feature_df)

    if save_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        regime_df.to_csv(output_path, index=False)

    return regime_df


def main() -> None:
    """Create the regime-tagged dataset for the active ticker."""
    regime_df = build_regime_dataframe_for_ticker(ticker)
    print(regime_df[["Date", "Close", "std_20", "regime"]].head(10))
    print("\nRows in each regime:")
    print(regime_df["regime"].value_counts().reindex(["calm", "neutral", "stressed"]))


if __name__ == "__main__":
    main()
