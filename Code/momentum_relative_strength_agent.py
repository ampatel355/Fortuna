"""Run the Momentum + Relative Strength strategy on a configurable asset universe."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from artifact_provenance import write_dataframe_artifact
    from asset_class_universe import resolve_relative_strength_setup
    from execution_model import (
        STARTING_CAPITAL,
        OpenPosition,
        build_execution_rng,
        close_position_from_signal,
        open_position_from_signal,
    )
    from research_metrics import calculate_annualized_sharpe_from_daily_returns, save_curve_csv
    from regimes import build_regime_dataframe_for_ticker
    from single_ticker_agent_common import load_regime_data, save_trade_outputs
    from strategy_config import (
        RELATIVE_STRENGTH_ABSOLUTE_MOMENTUM_THRESHOLD,
        RELATIVE_STRENGTH_LOOKBACK_DAYS,
        RELATIVE_STRENGTH_REBALANCE_FREQUENCY,
        RELATIVE_STRENGTH_RETURN_COLUMN,
        RELATIVE_STRENGTH_TOP_N,
        RELATIVE_STRENGTH_TRAILING_STOP_ATR_MULTIPLIER,
    )
    from strategy_simulator import (
        _entry_is_still_valid,
        _finalize_trades,
        _first_bar_exit_reason,
        _is_rebalance_bar,
    )
    from timeframe_config import RESEARCH_INTERVAL, RESEARCH_TIMEFRAME_LABEL, timeframe_output_suffix
except ModuleNotFoundError:
    from Code.artifact_provenance import write_dataframe_artifact
    from Code.asset_class_universe import resolve_relative_strength_setup
    from Code.execution_model import (
        STARTING_CAPITAL,
        OpenPosition,
        build_execution_rng,
        close_position_from_signal,
        open_position_from_signal,
    )
    from Code.research_metrics import calculate_annualized_sharpe_from_daily_returns, save_curve_csv
    from Code.regimes import build_regime_dataframe_for_ticker
    from Code.single_ticker_agent_common import load_regime_data, save_trade_outputs
    from Code.strategy_config import (
        RELATIVE_STRENGTH_ABSOLUTE_MOMENTUM_THRESHOLD,
        RELATIVE_STRENGTH_LOOKBACK_DAYS,
        RELATIVE_STRENGTH_REBALANCE_FREQUENCY,
        RELATIVE_STRENGTH_RETURN_COLUMN,
        RELATIVE_STRENGTH_TOP_N,
        RELATIVE_STRENGTH_TRAILING_STOP_ATR_MULTIPLIER,
    )
    from Code.strategy_simulator import (
        _entry_is_still_valid,
        _finalize_trades,
        _first_bar_exit_reason,
        _is_rebalance_bar,
    )
    from Code.timeframe_config import RESEARCH_INTERVAL, RESEARCH_TIMEFRAME_LABEL, timeframe_output_suffix


ticker = os.environ.get("TICKER", "SPY").strip().upper()


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


def resolve_universe(anchor_ticker: str) -> list[str]:
    """Build the peer universe used for cross-asset ranking."""
    setup = resolve_relative_strength_setup(anchor_ticker)
    universe = [str(symbol).strip().upper() for symbol in setup["universe"]]

    # The current luck-vs-skill pipeline assumes one active position at a time
    # and a trade-level Monte Carlo null. Because of that, we intentionally keep
    # this strategy to the top single asset rather than a concurrent basket.
    if RELATIVE_STRENGTH_TOP_N != 1:
        print(
            "RELATIVE_STRENGTH_TOP_N is currently constrained to 1 so the existing "
            "trade-level Monte Carlo and non-overlapping trade framework remain valid."
        )

    unique_universe: list[str] = []
    seen: set[str] = set()
    for symbol in universe:
        if symbol not in seen:
            seen.add(symbol)
            unique_universe.append(symbol)
    return unique_universe


def save_universe_metadata(
    output_path: Path,
    setup: dict[str, object],
    aligned_asset_frames: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Save the resolved peer-universe definition used by the strategy run."""
    aligned_asset_frames = aligned_asset_frames or {}
    rows: list[dict[str, object]] = []
    for position, asset_ticker in enumerate(setup["universe"], start=1):
        aligned_frame = aligned_asset_frames.get(asset_ticker)
        rows.append(
            {
                "anchor_ticker": setup["anchor_ticker"],
                "asset_class": setup["asset_class"],
                "asset_class_label": setup["asset_class_label"],
                "selection_source": setup["selection_source"],
                "selection_reason": setup["selection_reason"],
                "fallback_rule": setup["fallback_rule"],
                "universe_position": position,
                "universe_ticker": asset_ticker,
                "included_in_aligned_universe": asset_ticker in aligned_asset_frames,
                "aligned_rows": int(len(aligned_frame)) if aligned_frame is not None else 0,
            }
        )

    metadata_df = pd.DataFrame(rows)
    write_dataframe_artifact(
        metadata_df,
        output_path,
        producer="momentum_relative_strength_agent.save_universe_metadata",
        current_ticker=str(setup.get("anchor_ticker", "")).strip().upper(),
        research_grade=True,
        canonical_policy="always",
        parameters={
            "artifact_type": "relative_strength_universe",
            "selection_source": str(setup.get("selection_source", "")),
        },
    )


def load_aligned_universe_data(universe: list[str]) -> tuple[list[pd.Timestamp], dict[str, pd.DataFrame]]:
    """Load the same required columns for each asset and align them on a shared calendar.

    Daily-data simplification:
    We use the intersection of available dates across the universe. This avoids
    asynchronous ranking problems when some markets trade on holidays or weekends
    that others do not. It is conservative, easy to audit, and prevents
    accidentally ranking an asset using stale prices beside fresher prices from a
    different market.
    """
    asset_frames: dict[str, pd.DataFrame] = {}
    common_dates: set[pd.Timestamp] | None = None
    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "sma_50",
        "atr_14",
        "avg_volume_20",
        RELATIVE_STRENGTH_RETURN_COLUMN,
        "regime",
    ]

    for asset_ticker in universe:
        df = build_regime_dataframe_for_ticker(asset_ticker, save_output=True)
        if df.empty:
            print(f"Skipping {asset_ticker} in relative-strength universe because no usable regime rows were available.")
            continue
        df = df[required_columns].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        numeric_columns = [column for column in required_columns if column not in {"Date", "regime"}]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = (
            df.dropna(subset=required_columns)
            .sort_values("Date")
            .drop_duplicates(subset=["Date"], keep="last")
            .reset_index(drop=True)
        )
        asset_frames[asset_ticker] = df

        asset_date_set = set(df["Date"].tolist())
        if common_dates is None:
            common_dates = asset_date_set
        else:
            common_dates &= asset_date_set

    if len(asset_frames) < 2:
        raise ValueError(
            "The relative-strength strategy needs at least two usable assets after feature and regime preparation."
        )

    if not common_dates:
        raise ValueError("The relative-strength universe does not share enough common dates to trade.")

    ordered_common_dates = sorted(common_dates)
    if len(ordered_common_dates) < RELATIVE_STRENGTH_LOOKBACK_DAYS + 5:
        raise ValueError(
            "The aligned relative-strength universe is too short after date intersection. "
            "Try a smaller universe or a shorter lookback."
        )

    aligned_frames: dict[str, pd.DataFrame] = {}
    common_index = pd.Index(ordered_common_dates, name="Date")
    for asset_ticker, df in asset_frames.items():
        aligned_df = (
            df.set_index("Date")
            .reindex(common_index)
            .dropna()
            .reset_index()
            .sort_values("Date")
            .reset_index(drop=True)
        )
        aligned_df.attrs["ticker"] = asset_ticker
        aligned_frames[asset_ticker] = aligned_df

    return ordered_common_dates, aligned_frames


def select_top_asset(asset_frames: dict[str, pd.DataFrame], current_index: int) -> str | None:
    """Rank the universe by trailing return and return the strongest eligible asset."""
    ranking_rows: list[tuple[str, float]] = []

    for asset_ticker, asset_df in asset_frames.items():
        row = asset_df.iloc[current_index]
        trailing_return = float(row[RELATIVE_STRENGTH_RETURN_COLUMN])
        close_price = float(row["Close"])
        sma_50 = float(row["sma_50"])

        # We require both cross-sectional leadership and non-negative absolute
        # momentum so the strategy can step aside into cash when leadership is weak.
        if trailing_return <= RELATIVE_STRENGTH_ABSOLUTE_MOMENTUM_THRESHOLD:
            continue
        if close_price <= sma_50:
            continue

        ranking_rows.append((asset_ticker, trailing_return))

    if not ranking_rows:
        return None

    ranking_rows.sort(key=lambda item: item[1], reverse=True)
    return ranking_rows[0][0]


def build_relative_strength_curve(
    trade_df: pd.DataFrame,
    common_dates: list[pd.Timestamp],
    asset_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build the bar-based marked-to-market equity curve for the rotating top-asset strategy."""
    curve_df = pd.DataFrame({"Date": pd.to_datetime(common_dates)})
    curve_df["Close"] = np.nan
    curve_df["asset_ticker"] = pd.NA

    equity = np.full(len(curve_df), float(STARTING_CAPITAL), dtype=float)
    if trade_df.empty:
        curve_df["equity"] = equity
        curve_df["wealth_index"] = curve_df["equity"] / float(STARTING_CAPITAL)
        curve_df["bar_return"] = curve_df["wealth_index"].pct_change().fillna(0.0)
        curve_df["daily_return"] = curve_df["wealth_index"].pct_change().fillna(0.0)
        curve_df["cumulative_return"] = curve_df["wealth_index"] - 1.0
        curve_df["rolling_peak"] = curve_df["wealth_index"].cummax()
        curve_df["drawdown"] = (curve_df["wealth_index"] / curve_df["rolling_peak"]) - 1.0
        curve_df["timeframe_label"] = RESEARCH_TIMEFRAME_LABEL
        return curve_df

    date_to_index = pd.Series(curve_df.index.to_numpy(), index=curve_df["Date"])
    close_lookup = {
        asset_ticker: asset_df.set_index("Date")["Close"].reindex(curve_df["Date"]).to_numpy(dtype=float)
        for asset_ticker, asset_df in asset_frames.items()
    }

    current_capital = float(trade_df["capital_before"].iloc[0])
    fill_start_index = 0

    for _, trade_row in trade_df.iterrows():
        entry_index = date_to_index.get(pd.Timestamp(trade_row["entry_date"]))
        exit_index = date_to_index.get(pd.Timestamp(trade_row["exit_date"]))
        if pd.isna(entry_index) or pd.isna(exit_index):
            raise ValueError("Relative-strength trade dates do not align with the curve dates.")

        entry_index = int(entry_index)
        exit_index = int(exit_index)
        if exit_index <= entry_index:
            raise ValueError("Relative-strength trades must exit after they enter.")

        equity[fill_start_index:entry_index] = current_capital

        asset_ticker = str(trade_row["asset_ticker"]).strip().upper()
        asset_close_values = close_lookup[asset_ticker]
        cash_after_entry = float(trade_row["capital_before"]) - float(trade_row["capital_deployed"])
        shares = float(trade_row["shares"])

        equity[entry_index:exit_index] = cash_after_entry + (shares * asset_close_values[entry_index:exit_index])
        curve_df.loc[entry_index:exit_index - 1, "Close"] = asset_close_values[entry_index:exit_index]
        curve_df.loc[entry_index:exit_index - 1, "asset_ticker"] = asset_ticker

        current_capital = float(trade_row["capital_after"])
        fill_start_index = exit_index

    equity[fill_start_index:] = current_capital
    curve_df["equity"] = equity
    curve_df["wealth_index"] = curve_df["equity"] / float(STARTING_CAPITAL)
    curve_df["bar_return"] = curve_df["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    curve_df["daily_return"] = curve_df["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    curve_df["cumulative_return"] = curve_df["wealth_index"] - 1.0
    curve_df["rolling_peak"] = curve_df["wealth_index"].cummax()
    curve_df["drawdown"] = (curve_df["wealth_index"] / curve_df["rolling_peak"]) - 1.0
    curve_df["timeframe_label"] = RESEARCH_TIMEFRAME_LABEL
    return curve_df


def run_relative_strength_on_aligned_universe(
    common_dates: list[pd.Timestamp],
    asset_frames: dict[str, pd.DataFrame],
    *,
    anchor_ticker: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the relative-strength rotation logic on an already aligned universe.

    This helper is shared by:
    - the normal saved-file agent entrypoint
    - the in-memory walk-forward simulator path

    Bar-data simplification:
    Rebalance decisions are made using the completed close on a shared peer
    calendar and then executed at the next open of the selected asset.
    """
    primary_dates = pd.Series(pd.to_datetime(common_dates))
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("momentum_relative_strength", anchor_ticker)
    open_position: OpenPosition | None = None
    highest_close_since_entry: float | None = None
    current_asset_ticker: str | None = None
    signal_count = 0
    rejected_signal_count = 0

    for current_index in range(len(primary_dates) - 1):
        current_date = primary_dates.iloc[current_index]
        next_date = primary_dates.iloc[current_index + 1]
        rebalance_today = _is_rebalance_bar(
            current_index=current_index,
            current_date=current_date,
            next_date=next_date,
            frequency=RELATIVE_STRENGTH_REBALANCE_FREQUENCY,
        )
        target_asset = select_top_asset(asset_frames, current_index) if rebalance_today else None

        if open_position is not None and current_asset_ticker is not None:
            active_df = asset_frames[current_asset_ticker]
            row = active_df.iloc[current_index]
            next_row = active_df.iloc[current_index + 1]

            highest_close_since_entry = (
                float(row.Close)
                if highest_close_since_entry is None
                else max(highest_close_since_entry, float(row.Close))
            )
            trailing_stop = highest_close_since_entry - (
                RELATIVE_STRENGTH_TRAILING_STOP_ATR_MULTIPLIER * float(row.atr_14)
            )
            if open_position.stop_loss_used is None:
                open_position.stop_loss_used = float(trailing_stop)
            else:
                open_position.stop_loss_used = max(float(open_position.stop_loss_used), float(trailing_stop))

            exit_reason = _first_bar_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            if exit_reason is None and rebalance_today and target_asset != current_asset_ticker:
                exit_reason = "rebalance_rotation" if target_asset is not None else "relative_strength_to_cash"

            if exit_reason is not None:
                trade_record = close_position_from_signal(
                    position=open_position,
                    next_row=next_row,
                    exit_index=current_index + 1,
                    rng=execution_rng,
                    ticker=current_asset_ticker,
                    exit_reason=exit_reason,
                )
                trades.append(trade_record)
                capital = float(trade_record["capital_after"])
                open_position = None
                current_asset_ticker = None
                highest_close_since_entry = None

        if open_position is None and rebalance_today and target_asset is not None:
            if (current_index + 1) >= (len(primary_dates) - 1):
                rejected_signal_count += 1
                continue

            target_df = asset_frames[target_asset]
            row = target_df.iloc[current_index]
            next_row = target_df.iloc[current_index + 1]
            initial_stop = float(row.Close - (RELATIVE_STRENGTH_TRAILING_STOP_ATR_MULTIPLIER * float(row.atr_14)))
            signal_count += 1

            if not _entry_is_still_valid(float(next_row.Open), initial_stop, None):
                rejected_signal_count += 1
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="momentum_relative_strength",
                ticker=target_asset,
                stop_loss_used=initial_stop,
                take_profit_used=None,
                entry_reason="relative_strength_leader_rotation",
            )
            if candidate_position is not None:
                open_position = candidate_position
                current_asset_ticker = target_asset
                highest_close_since_entry = float(row.Close)
            else:
                rejected_signal_count += 1

    if open_position is not None and current_asset_ticker is not None:
        final_df = asset_frames[current_asset_ticker]
        trade_record = close_position_from_signal(
            position=open_position,
            next_row=final_df.iloc[-1],
            exit_index=len(primary_dates) - 1,
            rng=execution_rng,
            ticker=current_asset_ticker,
            exit_reason="end_of_sample",
        )
        trades.append(trade_record)
        capital = float(trade_record["capital_after"])
        open_position = None
        current_asset_ticker = None
        highest_close_since_entry = None

    trades_df = _finalize_trades(trades)
    trades_df.attrs["signal_count"] = signal_count
    trades_df.attrs["rejected_signal_count"] = rejected_signal_count
    curve_df = build_relative_strength_curve(trades_df, common_dates, asset_frames)
    return trades_df, curve_df


def run_relative_strength_strategy_for_market_df(
    market_df: pd.DataFrame,
    anchor_ticker: str | None = None,
) -> pd.DataFrame:
    """Run the relative-strength strategy on the date window of one supplied fold.

    The walk-forward framework passes one fold of the anchor ticker's market
    data. We use that fold's dates as the allowed calendar window, then align
    the peer universe to that same window so the cross-asset strategy can be
    evaluated inside each fold without future leakage.
    """
    resolved_ticker = (
        str(anchor_ticker).strip().upper()
        if anchor_ticker
        else str(market_df.attrs.get("ticker", ticker)).strip().upper()
    )
    universe = resolve_universe(resolved_ticker)
    common_dates, asset_frames = load_aligned_universe_data(universe)

    allowed_dates = set(pd.to_datetime(market_df["Date"], errors="coerce").dropna().tolist())
    filtered_common_dates = [date for date in common_dates if date in allowed_dates]
    if len(filtered_common_dates) < RELATIVE_STRENGTH_LOOKBACK_DAYS + 5:
        return pd.DataFrame()

    filtered_frames: dict[str, pd.DataFrame] = {}
    for asset_ticker, asset_df in asset_frames.items():
        filtered_df = asset_df.loc[asset_df["Date"].isin(filtered_common_dates)].copy()
        filtered_df = filtered_df.sort_values("Date").reset_index(drop=True)
        if len(filtered_df) != len(filtered_common_dates):
            continue
        filtered_df.attrs["ticker"] = asset_ticker
        filtered_frames[asset_ticker] = filtered_df

    if len(filtered_frames) < 2:
        return pd.DataFrame()

    trades_df, _ = run_relative_strength_on_aligned_universe(
        filtered_common_dates,
        filtered_frames,
        anchor_ticker=resolved_ticker,
    )
    return trades_df


def main() -> None:
    """Run the relative-strength rotation strategy and save both trades and equity curve."""
    # Re-read TICKER at runtime so imported in-process uses stay aligned with the
    # current environment, not just the value captured at module import time.
    current_ticker = os.environ.get("TICKER", ticker).strip().upper()
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    curve_output_path = data_clean_dir / f"{current_ticker}_momentum_relative_strength_curve.csv"
    metadata_output_path = data_clean_dir / f"{current_ticker}_momentum_relative_strength_universe.csv"

    setup = resolve_relative_strength_setup(current_ticker)
    universe = resolve_universe(current_ticker)
    common_dates, asset_frames = load_aligned_universe_data(universe)
    anchor_market_path = data_clean_dir / f"{current_ticker}_regimes.csv"
    anchor_market_df = load_regime_data(
        anchor_market_path,
        required_columns=["Date", "Open", "High", "Low", "Close", "avg_volume_20", "regime"],
    )
    allowed_dates = set(pd.to_datetime(anchor_market_df["Date"], errors="coerce").dropna().tolist())
    filtered_common_dates = [date for date in common_dates if date in allowed_dates]

    filtered_frames: dict[str, pd.DataFrame] = {}
    for asset_ticker, asset_df in asset_frames.items():
        filtered_df = asset_df.loc[asset_df["Date"].isin(filtered_common_dates)].copy()
        filtered_df = filtered_df.sort_values("Date").reset_index(drop=True)
        if len(filtered_df) != len(filtered_common_dates):
            continue
        filtered_df.attrs["ticker"] = asset_ticker
        filtered_frames[asset_ticker] = filtered_df

    if len(filtered_common_dates) >= RELATIVE_STRENGTH_LOOKBACK_DAYS + 5 and len(filtered_frames) >= 2:
        trades_df, curve_df = run_relative_strength_on_aligned_universe(
            filtered_common_dates,
            filtered_frames,
            anchor_ticker=current_ticker,
        )
    else:
        trades_df = _finalize_trades([])
        anchor_dates = sorted(allowed_dates)
        curve_df = build_relative_strength_curve(
            trades_df,
            anchor_dates,
            filtered_frames,
        )

    save_trade_outputs(
        current_ticker=current_ticker,
        agent_name="momentum_relative_strength",
        trades_df=trades_df,
        output_dir=data_clean_dir,
    )
    save_curve_csv(curve_df, curve_output_path)
    save_universe_metadata(metadata_output_path, setup, aligned_asset_frames=filtered_frames)

    summary_text = {
        "asset_class": setup["asset_class_label"],
        "selection_source": setup["selection_source"],
        "universe_size": len(universe),
        "calendar_rows": len(filtered_common_dates),
        "total_trades": len(trades_df),
        "final_cumulative_return": (
            float(curve_df["cumulative_return"].iloc[-1]) if not curve_df.empty else 0.0
        ),
        "annualized_sharpe": float(
            calculate_annualized_sharpe_from_daily_returns(
                curve_df["daily_return"],
                dates=curve_df["Date"],
            )
        ),
        "timeframe_label": RESEARCH_TIMEFRAME_LABEL,
        "data_interval": RESEARCH_INTERVAL,
    }
    print(summary_text)
    print("Relative-strength universe:", ", ".join(universe))
    print("\nFirst 10 trades:")
    print(trades_df.head(10))


if __name__ == "__main__":
    main()
