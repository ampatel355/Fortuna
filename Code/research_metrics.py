"""Shared performance metrics and equity-curve builders for the research pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from artifact_provenance import write_dataframe_artifact
    from execution_model import STARTING_CAPITAL
    from timeframe_config import (
        RESEARCH_TIMEFRAME_LABEL,
        infer_bars_per_year,
        normalize_timestamp_series,
    )
except ModuleNotFoundError:
    from Code.artifact_provenance import write_dataframe_artifact
    from Code.execution_model import STARTING_CAPITAL
    from Code.timeframe_config import (
        RESEARCH_TIMEFRAME_LABEL,
        infer_bars_per_year,
        normalize_timestamp_series,
    )


TRADING_DAYS_PER_YEAR = 252
MIN_P_VALUE = 1e-12


def calculate_trade_level_return_ratio(returns: np.ndarray) -> float:
    """Calculate a dispersion-adjusted trade-level return ratio."""
    if len(returns) <= 1:
        return 0.0

    std_return = float(np.std(returns, ddof=0))
    if std_return == 0:
        return 0.0

    return float(np.mean(returns) / std_return)


def calculate_annualized_sharpe_from_daily_returns(
    daily_returns: pd.Series,
    dates: pd.Series | None = None,
) -> float:
    """Calculate an annualized Sharpe ratio from one bar-return series.

    The function keeps its legacy name for compatibility, but the math now uses
    the observed bar density from the supplied timestamps whenever possible.
    """
    clean_returns = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(clean_returns) <= 1:
        return 0.0

    std_daily_return = float(clean_returns.std(ddof=0))
    if std_daily_return == 0:
        return 0.0

    mean_daily_return = float(clean_returns.mean())
    periods_per_year = infer_bars_per_year(dates) if dates is not None else float(TRADING_DAYS_PER_YEAR)
    return float((mean_daily_return / std_daily_return) * math.sqrt(periods_per_year))


def calculate_p_value_prominence(p_value: float) -> float:
    """Convert a p-value into a more visually interpretable prominence score."""
    clipped_p_value = min(max(float(p_value), MIN_P_VALUE), 1.0)
    return float(-math.log10(clipped_p_value))


def _load_required_trade_columns(trade_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort the trade log columns needed for daily equity reconstruction."""
    required_columns = [
        "entry_date",
        "exit_date",
        "shares",
        "entry_price",
        "capital_before",
        "capital_after",
        "capital_deployed",
    ]
    missing_columns = [column for column in required_columns if column not in trade_df.columns]
    if missing_columns:
        raise KeyError(
            "Trade data is missing the columns required for daily equity reconstruction: "
            + ", ".join(missing_columns)
        )

    validated_df = trade_df.copy()
    validated_df["entry_date"] = normalize_timestamp_series(validated_df["entry_date"])
    validated_df["exit_date"] = normalize_timestamp_series(validated_df["exit_date"])

    numeric_columns = [
        "shares",
        "entry_price",
        "capital_before",
        "capital_after",
        "capital_deployed",
    ]
    for column in numeric_columns:
        validated_df[column] = pd.to_numeric(validated_df[column], errors="coerce")

    validated_df = validated_df.dropna(subset=required_columns).sort_values("entry_date").reset_index(
        drop=True
    )
    return validated_df


def _prepare_market_curve_df(market_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort the market data needed for daily curve construction."""
    required_columns = ["Date", "Close"]
    missing_columns = [column for column in required_columns if column not in market_df.columns]
    if missing_columns:
        raise KeyError(
            "Market data is missing the columns required for daily curve construction: "
            + ", ".join(missing_columns)
        )

    curve_df = market_df[required_columns].copy()
    curve_df["Date"] = normalize_timestamp_series(curve_df["Date"])
    curve_df["Close"] = pd.to_numeric(curve_df["Close"], errors="coerce")
    curve_df = curve_df.dropna(subset=required_columns).sort_values("Date").reset_index(drop=True)

    if curve_df.empty:
        raise ValueError("No usable market data was available for daily curve construction.")

    if (curve_df["Close"] <= 0).any():
        raise ValueError("Market close prices must remain positive for daily curve construction.")

    return curve_df


def build_daily_strategy_curve(
    trade_df: pd.DataFrame,
    market_df: pd.DataFrame,
    starting_capital: float = STARTING_CAPITAL,
) -> pd.DataFrame:
    """Reconstruct a bar-based marked-to-market equity curve from completed trades."""
    curve_df = _prepare_market_curve_df(market_df)
    validated_trades = _load_required_trade_columns(trade_df)
    date_to_index = pd.Series(curve_df.index.to_numpy(), index=curve_df["Date"])
    close_prices = curve_df["Close"].to_numpy(dtype=float)

    if validated_trades.empty:
        equity = np.full(len(curve_df), float(starting_capital), dtype=float)
        curve_df["equity"] = equity
        curve_df["wealth_index"] = curve_df["equity"] / float(starting_capital)
        curve_df["bar_return"] = curve_df["wealth_index"].pct_change().fillna(0.0)
        curve_df["daily_return"] = curve_df["bar_return"]
        curve_df["cumulative_return"] = curve_df["wealth_index"] - 1.0
        curve_df["rolling_peak"] = curve_df["wealth_index"].cummax()
        curve_df["drawdown"] = (curve_df["wealth_index"] / curve_df["rolling_peak"]) - 1.0
        curve_df["timeframe_label"] = RESEARCH_TIMEFRAME_LABEL
        return curve_df

    initial_capital = float(validated_trades["capital_before"].iloc[0])
    equity = np.empty(len(curve_df), dtype=float)
    fill_start_index = 0
    current_capital = initial_capital

    for _, trade_row in validated_trades.iterrows():
        entry_index = date_to_index.get(pd.Timestamp(trade_row["entry_date"]))
        exit_index = date_to_index.get(pd.Timestamp(trade_row["exit_date"]))

        if pd.isna(entry_index) or pd.isna(exit_index):
            raise ValueError("Trade dates do not align with the market dates used for the curve.")

        entry_index = int(entry_index)
        exit_index = int(exit_index)
        if exit_index <= entry_index:
            raise ValueError("Trade exits must occur after entries for curve construction.")
        if entry_index < fill_start_index:
            raise ValueError("Trade data contains overlapping or out-of-order positions.")

        equity[fill_start_index:entry_index] = current_capital

        cash_after_entry = float(trade_row["capital_before"]) - float(trade_row["capital_deployed"])
        shares = float(trade_row["shares"])
        if shares <= 0:
            raise ValueError("Trade data contains a non-positive position quantity.")

        equity[entry_index:exit_index] = cash_after_entry + (shares * close_prices[entry_index:exit_index])

        current_capital = float(trade_row["capital_after"])
        # Keep the exit-date slot available for cash or a same-day rotation into
        # the next trade. This matters for strategies like relative-strength
        # rotation where one position can exit at the next open and the next one
        # can enter on that same open. The later trade should own the close of
        # that date rather than being treated as an overlap.
        fill_start_index = exit_index

    equity[fill_start_index:] = current_capital

    curve_df["equity"] = equity
    curve_df["wealth_index"] = curve_df["equity"] / initial_capital
    curve_df["bar_return"] = curve_df["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    curve_df["daily_return"] = curve_df["bar_return"]
    curve_df["cumulative_return"] = curve_df["wealth_index"] - 1.0
    curve_df["rolling_peak"] = curve_df["wealth_index"].cummax()
    curve_df["drawdown"] = (curve_df["wealth_index"] / curve_df["rolling_peak"]) - 1.0
    curve_df["timeframe_label"] = RESEARCH_TIMEFRAME_LABEL
    return curve_df


def build_buy_and_hold_curve(
    market_df: pd.DataFrame,
    transaction_cost: float = 0.0,
) -> pd.DataFrame:
    """Build a bar-based buy-and-hold benchmark curve from market closes."""
    curve_df = _prepare_market_curve_df(market_df)
    first_close = float(curve_df["Close"].iloc[0])
    if first_close <= 0:
        raise ValueError("The first close price must be positive for the benchmark curve.")

    curve_df["wealth_index"] = curve_df["Close"] / first_close
    if transaction_cost > 0:
        curve_df["wealth_index"] = curve_df["wealth_index"] * (1.0 - float(transaction_cost))

    curve_df["equity"] = curve_df["wealth_index"] * float(STARTING_CAPITAL)
    curve_df["bar_return"] = curve_df["wealth_index"].pct_change().fillna(0.0)
    curve_df["daily_return"] = curve_df["bar_return"]
    curve_df["cumulative_return"] = curve_df["wealth_index"] - 1.0
    curve_df["rolling_peak"] = curve_df["wealth_index"].cummax()
    curve_df["drawdown"] = (curve_df["wealth_index"] / curve_df["rolling_peak"]) - 1.0
    curve_df["timeframe_label"] = RESEARCH_TIMEFRAME_LABEL
    return curve_df


def summarize_daily_curve(curve_df: pd.DataFrame) -> dict[str, float | int]:
    """Summarize a bar-based equity curve into core performance metrics."""
    return {
        "cumulative_return": float(curve_df["cumulative_return"].iloc[-1]),
        "annualized_sharpe": calculate_annualized_sharpe_from_daily_returns(
            curve_df["daily_return"],
            dates=curve_df.get("Date"),
        ),
        "max_drawdown": float(curve_df["drawdown"].min()),
        "number_of_periods": int(len(curve_df)),
    }


def build_excess_curve(
    strategy_curve_df: pd.DataFrame,
    benchmark_curve_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a relative-performance curve against buy-and-hold.

    We model excess performance as the ratio of the strategy wealth index to the
    benchmark wealth index. If the ratio rises above 1.0, the strategy is
    outperforming buy-and-hold. This keeps the comparison intuitive and avoids
    mixing raw dollar gains from two different exposure paths.
    """
    required_columns = [
        "Date",
        "wealth_index",
        "equity",
        "cumulative_return",
        "daily_return",
        "drawdown",
    ]
    strategy_missing = [
        column for column in required_columns if column not in strategy_curve_df.columns
    ]
    benchmark_missing = [
        column for column in required_columns if column not in benchmark_curve_df.columns
    ]
    if strategy_missing:
        raise KeyError(
            "Strategy curve is missing required columns for excess analysis: "
            + ", ".join(strategy_missing)
        )
    if benchmark_missing:
        raise KeyError(
            "Benchmark curve is missing required columns for excess analysis: "
            + ", ".join(benchmark_missing)
        )

    working_strategy = strategy_curve_df[["Date", "wealth_index", "equity"]].copy()
    working_benchmark = benchmark_curve_df[["Date", "wealth_index", "equity"]].copy()

    working_strategy["Date"] = pd.to_datetime(working_strategy["Date"], errors="coerce")
    working_benchmark["Date"] = pd.to_datetime(working_benchmark["Date"], errors="coerce")
    working_strategy["wealth_index"] = pd.to_numeric(
        working_strategy["wealth_index"],
        errors="coerce",
    )
    working_benchmark["wealth_index"] = pd.to_numeric(
        working_benchmark["wealth_index"],
        errors="coerce",
    )
    working_strategy["equity"] = pd.to_numeric(working_strategy["equity"], errors="coerce")
    working_benchmark["equity"] = pd.to_numeric(working_benchmark["equity"], errors="coerce")

    aligned_df = working_strategy.rename(
        columns={
            "wealth_index": "strategy_wealth_index",
            "equity": "strategy_equity",
        }
    ).merge(
        working_benchmark.rename(
            columns={
                "wealth_index": "benchmark_wealth_index",
                "equity": "benchmark_equity",
            }
        ),
        on="Date",
        how="inner",
    )

    aligned_df = aligned_df.dropna().sort_values("Date").reset_index(drop=True)
    if aligned_df.empty:
        raise ValueError("No shared dates were available to build the excess curve.")

    if (aligned_df["benchmark_wealth_index"] <= 0).any():
        raise ValueError(
            "Benchmark wealth index must stay positive for excess-curve construction."
        )
    if (aligned_df["strategy_wealth_index"] <= 0).any():
        raise ValueError(
            "Strategy wealth index must stay positive for excess-curve construction."
        )

    aligned_df["wealth_index"] = (
        aligned_df["strategy_wealth_index"] / aligned_df["benchmark_wealth_index"]
    )
    aligned_df["equity"] = aligned_df["wealth_index"] * float(STARTING_CAPITAL)
    aligned_df["bar_return"] = aligned_df["wealth_index"].pct_change().fillna(0.0)
    aligned_df["daily_return"] = aligned_df["bar_return"]
    aligned_df["cumulative_return"] = aligned_df["wealth_index"] - 1.0
    aligned_df["rolling_peak"] = aligned_df["wealth_index"].cummax()
    aligned_df["drawdown"] = (
        aligned_df["wealth_index"] / aligned_df["rolling_peak"]
    ) - 1.0
    aligned_df["timeframe_label"] = RESEARCH_TIMEFRAME_LABEL
    return aligned_df


def save_curve_csv(curve_df: pd.DataFrame, output_path: Path) -> None:
    """Save a bar-based curve to CSV with consistent column ordering."""
    ordered_columns = [
        "Date",
        "Close",
        "equity",
        "wealth_index",
        "bar_return",
        "daily_return",
        "cumulative_return",
        "rolling_peak",
        "drawdown",
        "timeframe_label",
    ]
    existing_columns = [column for column in ordered_columns if column in curve_df.columns]
    inferred_ticker = output_path.name.split("_", 1)[0].strip().upper()
    write_dataframe_artifact(
        curve_df[existing_columns],
        output_path,
        producer="save_curve_csv",
        current_ticker=inferred_ticker,
        research_grade=True,
        canonical_policy="always",
        parameters={
            "artifact_type": "strategy_curve",
        },
    )
