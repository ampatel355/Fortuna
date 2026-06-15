"""Shared in-memory strategy runners used by the single-ticker pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import random

import pandas as pd

try:
    from execution_model import (
        STARTING_CAPITAL,
        TRADE_LOG_COLUMNS,
        OpenPosition,
        build_execution_rng,
        close_position_from_signal,
        open_position_from_signal,
    )
    from strategy_config import (
        BREAKOUT_ADX_MIN,
        BREAKOUT_CLOSE_BUFFER,
        BREAKOUT_FAILED_BREAKOUT_BUFFER,
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        BREAKOUT_MOMENTUM_THRESHOLD,
        BREAKOUT_NO_VOLUME_RANGE_EXPANSION_RATIO,
        BREAKOUT_REWARD_TO_RISK,
        BREAKOUT_STOP_ATR_BUFFER,
        BREAKOUT_TIME_STOP_BARS,
        BREAKOUT_VOLUME_MULTIPLIER,
        ADX_TREND_FOLLOWING_ADX_MIN,
        ADX_TREND_FOLLOWING_ATR_RATIO_MAX,
        ADX_TREND_FOLLOWING_MAX_HOLDING_BARS,
        ADX_TREND_FOLLOWING_STOP_ATR_MULTIPLIER,
        ADX_TREND_FOLLOWING_TARGET_ATR_MULTIPLIER,
        MEAN_REVERSION_ADX_MAX,
        MEAN_REVERSION_ATR_RATIO_MAX,
        MEAN_REVERSION_BB_WIDTH_RATIO_MAX,
        MEAN_REVERSION_LOWER_BAND_BUFFER,
        MEAN_REVERSION_MA_ALIGNMENT_THRESHOLD,
        MEAN_REVERSION_RSI_ENTRY_MAX,
        MEAN_REVERSION_RSI_EXIT_MIN,
        MEAN_REVERSION_STOP_ATR_BUFFER,
        MEAN_REVERSION_STOP_LOW_COLUMN,
        MEAN_REVERSION_TREND_DISTANCE_THRESHOLD,
        RANDOM_ENTRY_PROBABILITY,
        RANDOM_HOLDING_PERIOD_MAX_BARS,
        RANDOM_HOLDING_PERIOD_MIN_BARS,
        TREND_PULLBACK_ADX_MIN,
        TREND_PULLBACK_RSI_MAX,
        TREND_PULLBACK_RSI_MIN,
        TREND_PULLBACK_STOP_ATR_BUFFER,
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
        VALIDATION_MOMENTUM_ADX_MIN,
        VALIDATION_MOMENTUM_ATR_RATIO_MAX,
        VALIDATION_MOMENTUM_MAX_HOLDING_BARS,
        VALIDATION_MOMENTUM_MIN_CAPITAL_FRACTION,
        VALIDATION_MOMENTUM_RSI_MAX,
        VALIDATION_MOMENTUM_RSI_MIN,
        VALIDATION_MOMENTUM_STOP_ATR_MULTIPLIER,
        VALIDATION_MOMENTUM_TARGET_ATR_MULTIPLIER,
        VALIDATION_MOMENTUM_TRAILING_STOP_ATR_MULTIPLIER,
        VALIDATION_MOMENTUM_VOL_TARGET_RATIO,
        VALIDATION_MOMENTUM_VOLATILITY_EXIT_RATIO,
        UPTREND_OVERSOLD_MAX_HOLDING_BARS,
        UPTREND_OVERSOLD_RSI_MAX,
        UPTREND_OVERSOLD_STOP_ATR_MULTIPLIER,
        UPTREND_OVERSOLD_TARGET_ATR_MULTIPLIER,
        UPTREND_OVERSOLD_ZSCORE_MAX,
        VOLATILITY_SQUEEZE_ADX_MIN,
        VOLATILITY_SQUEEZE_BB_WIDTH_RATIO_MAX,
        VOLATILITY_SQUEEZE_MAX_HOLDING_BARS,
        VOLATILITY_SQUEEZE_STOP_ATR_MULTIPLIER,
        VOLATILITY_SQUEEZE_TARGET_ATR_MULTIPLIER,
        VOLATILITY_SQUEEZE_VOLUME_RATIO_MIN,
        CONNORS_RSI2_ENTRY_MAX,
        CONNORS_RSI2_EXIT_MIN,
        CONNORS_RSI2_EXIT_SMA_BARS,
        CONNORS_RSI2_MAX_HOLDING_BARS,
        CONNORS_RSI2_REQUIRE_SMA200_FILTER,
        CONNORS_RSI2_RSI_LOOKBACK_BARS,
        CONNORS_RSI2_STOP_ATR_MULTIPLIER,
        DONCHIAN_ADX_MIN,
        DONCHIAN_BREAKOUT_HIGH_COLUMN,
        DONCHIAN_BREAKOUT_LOOKBACK_DAYS,
        DONCHIAN_BREAKOUT_STOP_LOW_COLUMN,
        DONCHIAN_MAX_HOLDING_BARS,
        DONCHIAN_STOP_LOOKBACK_DAYS,
        DONCHIAN_STOP_ATR_MULTIPLIER,
        DONCHIAN_TARGET_ATR_MULTIPLIER,
        TURN_OF_MONTH_MAX_HOLDING_BARS,
        TURN_OF_MONTH_REQUIRE_SMA200_FILTER,
        TURN_OF_MONTH_STOP_ATR_MULTIPLIER,
        MEDIUM_RETURN_COLUMN,
        RELATIVE_STRENGTH_RETURN_COLUMN,
    )
    from timeframe_config import timeframe_output_suffix
except ModuleNotFoundError:
    from Code.execution_model import (
        STARTING_CAPITAL,
        TRADE_LOG_COLUMNS,
        OpenPosition,
        build_execution_rng,
        close_position_from_signal,
        open_position_from_signal,
    )
    from Code.strategy_config import (
        BREAKOUT_ADX_MIN,
        BREAKOUT_CLOSE_BUFFER,
        BREAKOUT_FAILED_BREAKOUT_BUFFER,
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        BREAKOUT_MOMENTUM_THRESHOLD,
        BREAKOUT_NO_VOLUME_RANGE_EXPANSION_RATIO,
        BREAKOUT_REWARD_TO_RISK,
        BREAKOUT_STOP_ATR_BUFFER,
        BREAKOUT_TIME_STOP_BARS,
        BREAKOUT_VOLUME_MULTIPLIER,
        ADX_TREND_FOLLOWING_ADX_MIN,
        ADX_TREND_FOLLOWING_ATR_RATIO_MAX,
        ADX_TREND_FOLLOWING_MAX_HOLDING_BARS,
        ADX_TREND_FOLLOWING_STOP_ATR_MULTIPLIER,
        ADX_TREND_FOLLOWING_TARGET_ATR_MULTIPLIER,
        MEAN_REVERSION_ADX_MAX,
        MEAN_REVERSION_ATR_RATIO_MAX,
        MEAN_REVERSION_BB_WIDTH_RATIO_MAX,
        MEAN_REVERSION_LOWER_BAND_BUFFER,
        MEAN_REVERSION_MA_ALIGNMENT_THRESHOLD,
        MEAN_REVERSION_RSI_ENTRY_MAX,
        MEAN_REVERSION_RSI_EXIT_MIN,
        MEAN_REVERSION_STOP_ATR_BUFFER,
        MEAN_REVERSION_STOP_LOW_COLUMN,
        MEAN_REVERSION_TREND_DISTANCE_THRESHOLD,
        RANDOM_ENTRY_PROBABILITY,
        RANDOM_HOLDING_PERIOD_MAX_BARS,
        RANDOM_HOLDING_PERIOD_MIN_BARS,
        TREND_PULLBACK_ADX_MIN,
        TREND_PULLBACK_RSI_MAX,
        TREND_PULLBACK_RSI_MIN,
        TREND_PULLBACK_STOP_ATR_BUFFER,
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
        VALIDATION_MOMENTUM_ADX_MIN,
        VALIDATION_MOMENTUM_ATR_RATIO_MAX,
        VALIDATION_MOMENTUM_MAX_HOLDING_BARS,
        VALIDATION_MOMENTUM_MIN_CAPITAL_FRACTION,
        VALIDATION_MOMENTUM_RSI_MAX,
        VALIDATION_MOMENTUM_RSI_MIN,
        VALIDATION_MOMENTUM_STOP_ATR_MULTIPLIER,
        VALIDATION_MOMENTUM_TARGET_ATR_MULTIPLIER,
        VALIDATION_MOMENTUM_TRAILING_STOP_ATR_MULTIPLIER,
        VALIDATION_MOMENTUM_VOL_TARGET_RATIO,
        VALIDATION_MOMENTUM_VOLATILITY_EXIT_RATIO,
        UPTREND_OVERSOLD_MAX_HOLDING_BARS,
        UPTREND_OVERSOLD_RSI_MAX,
        UPTREND_OVERSOLD_STOP_ATR_MULTIPLIER,
        UPTREND_OVERSOLD_TARGET_ATR_MULTIPLIER,
        UPTREND_OVERSOLD_ZSCORE_MAX,
        VOLATILITY_SQUEEZE_ADX_MIN,
        VOLATILITY_SQUEEZE_BB_WIDTH_RATIO_MAX,
        VOLATILITY_SQUEEZE_MAX_HOLDING_BARS,
        VOLATILITY_SQUEEZE_STOP_ATR_MULTIPLIER,
        VOLATILITY_SQUEEZE_TARGET_ATR_MULTIPLIER,
        VOLATILITY_SQUEEZE_VOLUME_RATIO_MIN,
        CONNORS_RSI2_ENTRY_MAX,
        CONNORS_RSI2_EXIT_MIN,
        CONNORS_RSI2_EXIT_SMA_BARS,
        CONNORS_RSI2_MAX_HOLDING_BARS,
        CONNORS_RSI2_REQUIRE_SMA200_FILTER,
        CONNORS_RSI2_RSI_LOOKBACK_BARS,
        CONNORS_RSI2_STOP_ATR_MULTIPLIER,
        DONCHIAN_ADX_MIN,
        DONCHIAN_BREAKOUT_HIGH_COLUMN,
        DONCHIAN_BREAKOUT_LOOKBACK_DAYS,
        DONCHIAN_BREAKOUT_STOP_LOW_COLUMN,
        DONCHIAN_MAX_HOLDING_BARS,
        DONCHIAN_STOP_LOOKBACK_DAYS,
        DONCHIAN_STOP_ATR_MULTIPLIER,
        DONCHIAN_TARGET_ATR_MULTIPLIER,
        TURN_OF_MONTH_MAX_HOLDING_BARS,
        TURN_OF_MONTH_REQUIRE_SMA200_FILTER,
        TURN_OF_MONTH_STOP_ATR_MULTIPLIER,
        MEDIUM_RETURN_COLUMN,
        RELATIVE_STRENGTH_RETURN_COLUMN,
    )
    from Code.timeframe_config import timeframe_output_suffix


RANDOM_AGENT_REPRODUCIBLE = os.environ.get("RANDOM_AGENT_REPRODUCIBLE", "1") == "1"
RANDOM_AGENT_SEED = int(os.environ.get("RANDOM_AGENT_SEED", "42"))


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


def _prepare_market_df(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Validate and normalize a regime-tagged market DataFrame."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "Market data is missing the columns required for strategy execution: "
            + ", ".join(missing_columns)
        )

    prepared_df = df.copy()
    prepared_df["Date"] = pd.to_datetime(prepared_df["Date"], errors="coerce")
    numeric_columns = [column for column in required_columns if column not in {"Date", "regime"}]
    for column in numeric_columns:
        prepared_df[column] = pd.to_numeric(prepared_df[column], errors="coerce")
    prepared_df["regime"] = prepared_df["regime"].astype(str)

    return (
        prepared_df.dropna(subset=required_columns)
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _build_random_decision_generator(seed_override: int | None = None):
    """Create the random decision generator used by the random baseline."""
    if seed_override is not None:
        return random.Random(seed_override)
    if RANDOM_AGENT_REPRODUCIBLE:
        return random.Random(RANDOM_AGENT_SEED)
    return random.SystemRandom()


def _compute_rsi_2(close_series: pd.Series, lookback: int | None = None) -> pd.Series:
    """Return Connors-style short-horizon RSI using Wilder smoothing.

    On daily bars the default lookback is 2 (the classic Connors RSI(2)).
    On faster timeframes the lookback is scaled via CONNORS_RSI2_RSI_LOOKBACK_BARS
    so the indicator still covers the same calendar horizon.
    """
    period = lookback if lookback is not None else CONNORS_RSI2_RSI_LOOKBACK_BARS
    period = max(period, 1)
    alpha = 1.0 / period

    close_change = close_series.diff()
    gains = close_change.clip(lower=0.0)
    losses = -close_change.clip(upper=0.0)

    average_gain = gains.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    relative_strength = average_gain / average_loss.replace(0.0, pd.NA)
    rsi_2 = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi_2 = rsi_2.where(~((average_loss == 0.0) & average_gain.notna()), 100.0)
    rsi_2 = rsi_2.where(~((average_gain == 0.0) & (average_loss == 0.0)), 50.0)
    return pd.to_numeric(rsi_2, errors="coerce")


def _seed_with_ticker_component(base_seed: int | None, ticker: str | None) -> int | None:
    """Derive a reproducible ticker-specific seed from one optional base seed."""
    if base_seed is None:
        return None
    normalized_ticker = (ticker or "").strip().upper()
    if not normalized_ticker:
        return int(base_seed)
    ticker_component = sum((index + 1) * ord(char) for index, char in enumerate(normalized_ticker))
    return int(base_seed) + int(ticker_component)


def _finalize_trades(trades: list[dict[str, object]]) -> pd.DataFrame:
    """Convert collected trades into a consistently ordered DataFrame."""
    return pd.DataFrame(trades, columns=TRADE_LOG_COLUMNS)


def _resolve_strategy_ticker(market_df: pd.DataFrame, ticker: str | None = None) -> str | None:
    """Prefer an explicit ticker, then fall back to DataFrame metadata when available."""
    if ticker:
        return str(ticker).strip().upper()

    ticker_attr = market_df.attrs.get("ticker")
    if ticker_attr:
        return str(ticker_attr).strip().upper()

    return None


def _first_bar_exit_reason(
    row: pd.Series,
    stop_loss_used: float | None,
    take_profit_used: float | None,
) -> str | None:
    """Check stop/target breaches using only the completed current bar.

    This function is timeframe-agnostic — it works identically for daily,
    hourly, and 4-hour bars because it only inspects the bar's High and Low.

    Practical simplification for bar data:
    - Signals are generated after the close.
    - Entries and exits happen at the next bar's open.
    - If both stop and target are touched inside the same bar, we do not
      know which came first, so we choose the stop-loss path conservatively.
    """
    stop_hit = (
        stop_loss_used is not None
        and pd.notna(row["Low"])
        and float(row["Low"]) <= float(stop_loss_used)
    )
    target_hit = (
        take_profit_used is not None
        and pd.notna(row["High"])
        and float(row["High"]) >= float(take_profit_used)
    )

    if stop_hit and target_hit:
        return "stop_loss_same_bar_priority"
    if stop_hit:
        return "stop_loss"
    if target_hit:
        return "take_profit"
    return None


def _entry_is_still_valid(
    next_open: float,
    stop_loss_used: float | None,
    take_profit_used: float | None,
) -> bool:
    """Skip trades that gap through the planned stop or target before entry."""
    if not pd.notna(next_open) or float(next_open) <= 0:
        return False

    if stop_loss_used is not None and float(next_open) <= float(stop_loss_used):
        return False
    if take_profit_used is not None and float(next_open) >= float(take_profit_used):
        return False
    return True


def _safe_stop(value: float | int | None) -> float | None:
    """Convert optional numeric thresholds into clean floats."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _has_room_for_entry_and_exit(current_index: int, market_length: int) -> bool:
    """Return whether a new next-open entry still leaves one later open for exit."""
    entry_index = current_index + 1
    return entry_index < (market_length - 1)


def _close_terminal_position(
    *,
    trades: list[dict[str, object]],
    open_position: OpenPosition,
    final_row: pd.Series,
    final_exit_index: int,
    execution_rng,
    resolved_ticker: str | None,
) -> float:
    """Close one surviving position at the last available open in the sample."""
    trade_record = close_position_from_signal(
        position=open_position,
        next_row=final_row,
        exit_index=final_exit_index,
        rng=execution_rng,
        ticker=resolved_ticker,
        exit_reason="end_of_sample",
    )
    trades.append(trade_record)
    return float(trade_record["capital_after"])


def _is_month_end_rebalance(current_date: pd.Timestamp, next_date: pd.Timestamp) -> bool:
    """Return whether the signal day is the last trading day of its calendar month."""
    return (
        current_date.month != next_date.month
        or current_date.year != next_date.year
    )


def _is_rebalance_bar(
    *,
    current_index: int,
    current_date: pd.Timestamp,
    next_date: pd.Timestamp,
    frequency: str,
) -> bool:
    """Return whether the current closed bar should trigger a rebalance decision."""
    normalized_frequency = str(frequency).strip().lower()
    if normalized_frequency.startswith("month"):
        return current_date.to_period("M") != next_date.to_period("M")
    if normalized_frequency.startswith("week"):
        return current_date.to_period("W-FRI") != next_date.to_period("W-FRI")
    if normalized_frequency.startswith("every_") and normalized_frequency.endswith("_days"):
        try:
            day_count = int(normalized_frequency.removeprefix("every_").removesuffix("_days"))
        except ValueError:
            day_count = 1
        return ((current_index + 1) % max(day_count, 1)) == 0
    if normalized_frequency.startswith("every_") and normalized_frequency.endswith("_hours"):
        # Backward-compatible alias from the old intraday configuration.
        # When the pipeline runs on daily bars, every_24_hours maps to one bar.
        try:
            hour_count = int(normalized_frequency.removeprefix("every_").removesuffix("_hours"))
        except ValueError:
            hour_count = 24
        day_count = max(int(round(hour_count / 24.0)), 1)
        return ((current_index + 1) % day_count) == 0
    if normalized_frequency.startswith("every_") and normalized_frequency.endswith("_bars"):
        try:
            bar_count = int(normalized_frequency.removeprefix("every_").removesuffix("_bars"))
        except ValueError:
            bar_count = 1
        return ((current_index + 1) % max(bar_count, 1)) == 0
    return current_date.date() != next_date.date()


def run_trend_pullback_strategy(market_df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Buy pullbacks inside an established uptrend and exit on target, stop, or trend failure."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "ema_20",
            "sma_50",
            "sma_200",
            "adx_14",
            "rsi_14",
            "atr_14",
            TREND_PULLBACK_STOP_LOW_COLUMN,
            TREND_PULLBACK_TARGET_HIGH_COLUMN,
            "avg_volume_20",
            "regime",
        ],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("trend_pullback", resolved_ticker)
    open_position: OpenPosition | None = None
    signal_count = 0
    rejected_signal_count = 0

    for current_index in range(1, len(df) - 1):
        previous_previous_row = df.iloc[current_index - 2] if current_index >= 2 else df.iloc[current_index - 1]
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            # Trend filter: 50-bar SMA above 200-bar SMA confirms the broader
            # trend direction.  The old code checked sma_50 vs sma_200 AND
            # ma_50 vs ma_200, which were identical columns — removed the
            # duplicate.  ADX threshold is now a relaxed 15 (set in config).
            in_uptrend = (
                row.sma_50 > row.sma_200
                and row.Close > row.sma_50
                and row.adx_14 >= TREND_PULLBACK_ADX_MIN
            )
            # Pullback detection: price dips to or below the 20 EMA.
            touched_pullback_zone = (
                row.Low <= row.ema_20
                or row.Close <= row.ema_20
            )
            # RSI must not be extreme — confirms the pullback is a pause,
            # not a reversal.  Uses the wider 35-65 band from config.
            rsi_filter = (
                TREND_PULLBACK_RSI_MIN <= row.rsi_14 <= TREND_PULLBACK_RSI_MAX
            )
            # Bounce: current close above previous close shows buying resumed.
            bounce_confirmation = (
                touched_pullback_zone
                and row.Close >= previous_row.Close
            )
            entry_signal = in_uptrend and rsi_filter and bounce_confirmation

            if not entry_signal:
                continue
            signal_count += 1

            if not _has_room_for_entry_and_exit(current_index, len(df)):
                rejected_signal_count += 1
                continue

            stop_loss_used = _safe_stop(
                min(row[TREND_PULLBACK_STOP_LOW_COLUMN], row.Low) - (row.atr_14 * TREND_PULLBACK_STOP_ATR_BUFFER)
            )
            take_profit_used = _safe_stop(row[TREND_PULLBACK_TARGET_HIGH_COLUMN])
            if (
                take_profit_used is None
                or take_profit_used <= row.Close
            ) and stop_loss_used is not None:
                take_profit_used = float(row.Close + (2.0 * max(row.Close - stop_loss_used, 0.0)))

            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, take_profit_used):
                rejected_signal_count += 1
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="trend_pullback",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=take_profit_used,
                entry_reason="trend_up_pullback_bounce",
            )
            if candidate_position is not None:
                open_position = candidate_position
            else:
                rejected_signal_count += 1
        else:
            exit_reason = _first_bar_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            if exit_reason is None and row.Close < row.sma_200:
                exit_reason = "trend_filter_failed"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None

    if open_position is not None:
        capital = _close_terminal_position(
            trades=trades,
            open_position=open_position,
            final_row=df.iloc[-1],
            final_exit_index=len(df) - 1,
            execution_rng=execution_rng,
            resolved_ticker=resolved_ticker,
        )
        open_position = None

    trades_df = _finalize_trades(trades)
    trades_df.attrs["signal_count"] = signal_count
    trades_df.attrs["rejected_signal_count"] = rejected_signal_count
    return trades_df


def run_breakout_volume_momentum_strategy(
    market_df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Trade strong breakouts confirmed by broad participation and positive momentum."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "avg_volume_20",
            "volume_ratio_20",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "adx_14",
            "atr_14",
            BREAKOUT_HIGH_COLUMN,
            BREAKOUT_LOW_COLUMN,
            BREAKOUT_MOMENTUM_RETURN_COLUMN,
            "regime",
        ],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("breakout_volume_momentum", resolved_ticker)
    open_position: OpenPosition | None = None
    signal_count = 0
    rejected_signal_count = 0

    for current_index in range(1, len(df) - 1):
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            breakout_level = float(row[BREAKOUT_HIGH_COLUMN])
            # Primary breakout: close above the rolling high (now 120-bar ≈
            # 5 trading days, giving a meaningful resistance level).
            breakout_signal = (
                row.Close > breakout_level
                and previous_row.Close <= previous_row[BREAKOUT_HIGH_COLUMN]
            )
            # Near-breakout: bar pierced resistance and closed within a small
            # buffer of the level.
            near_breakout_close = (
                row.High >= breakout_level
                and row.Close >= breakout_level * (1.0 - BREAKOUT_CLOSE_BUFFER)
                and row.Close >= previous_row.Close
            )
            # Volume / range expansion confirmation.
            has_reliable_volume = pd.notna(row.avg_volume_20) and float(row.avg_volume_20) > 0
            if has_reliable_volume:
                participation_filter = row.volume_ratio_20 >= BREAKOUT_VOLUME_MULTIPLIER
            else:
                participation_filter = (
                    float(row.High - row.Low) >= (float(row.atr_14) * BREAKOUT_NO_VOLUME_RANGE_EXPANSION_RATIO)
                )
            # Momentum: MACD above its signal line (bullish momentum) OR
            # positive trailing return.  The old code required a strict MACD
            # crossover *in this bar*, which is extremely rare and was
            # filtering out most valid breakouts.
            momentum_filter = (
                row.macd_line > row.macd_signal
                or row[BREAKOUT_MOMENTUM_RETURN_COLUMN] >= BREAKOUT_MOMENTUM_THRESHOLD
            )
            entry_signal = (
                (breakout_signal or near_breakout_close)
                and participation_filter
                and momentum_filter
                and row.adx_14 >= BREAKOUT_ADX_MIN
            )

            if not entry_signal:
                continue
            signal_count += 1

            if not _has_room_for_entry_and_exit(current_index, len(df)):
                rejected_signal_count += 1
                continue

            stop_anchor = min(row.Low, row[BREAKOUT_LOW_COLUMN])
            stop_loss_used = _safe_stop(stop_anchor - (BREAKOUT_STOP_ATR_BUFFER * row.atr_14))
            if stop_loss_used is None:
                rejected_signal_count += 1
                continue

            signal_risk = max(row.Close - stop_loss_used, 0.0)
            if signal_risk <= 0:
                rejected_signal_count += 1
                continue

            take_profit_used = float(row.Close + (BREAKOUT_REWARD_TO_RISK * signal_risk))
            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, take_profit_used):
                rejected_signal_count += 1
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="breakout_volume_momentum",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=take_profit_used,
                entry_reason="breakout_volume_momentum",
            )
            if candidate_position is not None:
                open_position = candidate_position
            else:
                rejected_signal_count += 1
        else:
            exit_reason = _first_bar_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            holding_bars = current_index - open_position.entry_index
            failed_breakout = (
                row.Close < row[BREAKOUT_HIGH_COLUMN] * (1.0 - BREAKOUT_FAILED_BREAKOUT_BUFFER)
                and row.macd_hist < 0
            )
            timed_exit = holding_bars >= BREAKOUT_TIME_STOP_BARS

            if exit_reason is None and failed_breakout:
                exit_reason = "failed_breakout"
            if exit_reason is None and timed_exit:
                exit_reason = "time_stop"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None

    if open_position is not None:
        capital = _close_terminal_position(
            trades=trades,
            open_position=open_position,
            final_row=df.iloc[-1],
            final_exit_index=len(df) - 1,
            execution_rng=execution_rng,
            resolved_ticker=resolved_ticker,
        )
        open_position = None

    trades_df = _finalize_trades(trades)
    trades_df.attrs["signal_count"] = signal_count
    trades_df.attrs["rejected_signal_count"] = rejected_signal_count
    return trades_df


def run_mean_reversion_vol_filter_strategy(
    market_df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Fade oversold moves only when volatility and trend conditions stay tame."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "bollinger_mid",
            "bollinger_lower",
            "rsi_14",
            "adx_14",
            "atr_14",
            "atr_percent_ratio_60",
            "bollinger_width_ratio_60",
            "distance_from_ma_50",
            "ma_50_over_ma_200",
            MEAN_REVERSION_STOP_LOW_COLUMN,
            "avg_volume_20",
            "regime",
        ],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("mean_reversion_vol_filter", resolved_ticker)
    open_position: OpenPosition | None = None
    signal_count = 0
    rejected_signal_count = 0

    for current_index in range(1, len(df) - 1):
        previous_previous_row = df.iloc[current_index - 2] if current_index >= 2 else df.iloc[current_index - 1]
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            # Volatility filter: at least one of ATR ratio or Bollinger width
            # must indicate calm conditions.  Requiring both simultaneously
            # was too strict — they measure similar concepts and rejecting on
            # either alone was starving the strategy of entries.
            calm_volatility = (
                row.atr_percent_ratio_60 <= MEAN_REVERSION_ATR_RATIO_MAX
                or row.bollinger_width_ratio_60 <= MEAN_REVERSION_BB_WIDTH_RATIO_MAX
            )
            # Price must not be extremely far from the 50-bar mean, confirming
            # we're in a ranging environment.
            contained_trend = (
                abs(float(row.distance_from_ma_50)) <= MEAN_REVERSION_TREND_DISTANCE_THRESHOLD
            )
            regime_is_tradeable = str(row.regime).strip().lower() != "stressed"
            ranging_environment = (
                row.adx_14 < MEAN_REVERSION_ADX_MAX
                and calm_volatility
                and contained_trend
                and regime_is_tradeable
            )
            # Oversold: price near or below the lower Bollinger band with
            # RSI confirming the dip.  Only the current bar's RSI is checked
            # — requiring 3 consecutive oversold bars was too restrictive.
            oversold_entry = (
                row.Low <= row.bollinger_lower * (1.0 + MEAN_REVERSION_LOWER_BAND_BUFFER)
                and row.rsi_14 <= MEAN_REVERSION_RSI_ENTRY_MAX
            )
            entry_signal = ranging_environment and oversold_entry

            if not entry_signal:
                continue
            signal_count += 1

            if not _has_room_for_entry_and_exit(current_index, len(df)):
                rejected_signal_count += 1
                continue

            stop_loss_used = _safe_stop(
                min(row.Low, row[MEAN_REVERSION_STOP_LOW_COLUMN]) - (row.atr_14 * MEAN_REVERSION_STOP_ATR_BUFFER)
            )
            take_profit_used = _safe_stop(row.bollinger_mid)
            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, take_profit_used):
                rejected_signal_count += 1
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="mean_reversion_vol_filter",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=take_profit_used,
                entry_reason="oversold_mean_reversion",
            )
            if candidate_position is not None:
                open_position = candidate_position
            else:
                rejected_signal_count += 1
        else:
            exit_reason = _first_bar_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            if exit_reason is None and row.Close >= row.bollinger_mid:
                exit_reason = "mean_reversion_complete"
            if exit_reason is None and row.rsi_14 >= MEAN_REVERSION_RSI_EXIT_MIN:
                exit_reason = "rsi_recovery"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None

    if open_position is not None:
        capital = _close_terminal_position(
            trades=trades,
            open_position=open_position,
            final_row=df.iloc[-1],
            final_exit_index=len(df) - 1,
            execution_rng=execution_rng,
            resolved_ticker=resolved_ticker,
        )
        open_position = None

    trades_df = _finalize_trades(trades)
    trades_df.attrs["signal_count"] = signal_count
    trades_df.attrs["rejected_signal_count"] = rejected_signal_count
    return trades_df


def run_random_strategy(
    market_df: pd.DataFrame,
    decision_seed: int | None = None,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Run a pure random-timing control with random holding periods.

    This baseline is intentionally simpler than the structured strategies:
    - entry timing is random
    - exits are time-based only
    - there is no stop-loss, target, or indicator logic

    That keeps the baseline aligned with the Monte Carlo null model, which also
    tests whether random timing with the same trade count and holding-period
    profile could plausibly explain the realized result.
    """
    df = _prepare_market_df(
        market_df,
        ["Date", "Open", "High", "Low", "Close", "avg_volume_20", "regime"],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("random", resolved_ticker)
    decision_rng = _build_random_decision_generator(
        seed_override=_seed_with_ticker_component(decision_seed, resolved_ticker)
    )
    open_position: OpenPosition | None = None
    planned_exit_index: int | None = None
    signal_count = 0
    rejected_signal_count = 0

    for current_index in range(1, len(df) - 1):
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            if decision_rng.random() >= RANDOM_ENTRY_PROBABILITY:
                continue
            signal_count += 1

            if not _has_room_for_entry_and_exit(current_index, len(df)):
                rejected_signal_count += 1
                continue

            if not _entry_is_still_valid(float(next_row.Open), None, None):
                rejected_signal_count += 1
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="random",
                ticker=resolved_ticker,
                stop_loss_used=None,
                take_profit_used=None,
                entry_reason="random_timing_entry",
            )
            if candidate_position is not None:
                open_position = candidate_position
                planned_exit_index = open_position.entry_index + decision_rng.randint(
                    RANDOM_HOLDING_PERIOD_MIN_BARS,
                    RANDOM_HOLDING_PERIOD_MAX_BARS,
                )
            else:
                rejected_signal_count += 1
        else:
            exit_reason = None
            if planned_exit_index is not None and current_index >= planned_exit_index:
                exit_reason = "random_time_exit"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None
            planned_exit_index = None

    if open_position is not None:
        capital = _close_terminal_position(
            trades=trades,
            open_position=open_position,
            final_row=df.iloc[-1],
            final_exit_index=len(df) - 1,
            execution_rng=execution_rng,
            resolved_ticker=resolved_ticker,
        )
        open_position = None
        planned_exit_index = None

    trades_df = _finalize_trades(trades)
    trades_df.attrs["signal_count"] = signal_count
    trades_df.attrs["rejected_signal_count"] = rejected_signal_count
    return trades_df


# ---------------------------------------------------------------------------
# Validation Strategy (Adaptive Volatility-Managed Momentum, "AVM")
# ---------------------------------------------------------------------------
# Purpose:
#   A defensible, cross-asset validation strategy designed to be realistic:
#   it combines established trend and momentum structure with volatility-aware
#   position sizing and conservative risk exits.
#
# Entry logic (all required):
#   - Structural trend: Close above SMA-200 and SMA-50 >= SMA-200
#   - Momentum confirmation: trailing_return_20 > 0, trailing_return_60 > 0,
#     MACD line above MACD signal, and ADX above a low strength floor
#   - Quality / risk filter: RSI in a non-extreme range and ATR ratio below cap
#   - Trigger: either pullback reclaim (touch EMA-20 then recover) or a clean
#     20-bar continuation breakout.
#
# Exit stack:
#   - Initial stop-loss + take-profit based on ATR
#   - Trailing stop from highest close since entry
#   - Trend/momentum breakdown and volatility-shock exits
#   - Time stop after a fixed horizon


def run_trend_momentum_verification_strategy(
    market_df: pd.DataFrame,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Run the AVM validation strategy used to stress-test skill-vs-luck detection."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "avg_volume_20",
            "ema_20",
            "sma_50",
            "sma_200",
            "atr_14",
            "adx_14",
            "rsi_14",
            "macd_line",
            "macd_signal",
            "atr_percent_ratio_60",
            "volume_ratio_20",
            BREAKOUT_HIGH_COLUMN,
            BREAKOUT_LOW_COLUMN,
            MEDIUM_RETURN_COLUMN,
            RELATIVE_STRENGTH_RETURN_COLUMN,
            "regime",
        ],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("trend_momentum_verification", resolved_ticker)
    open_position: OpenPosition | None = None
    signal_count = 0
    rejected_signal_count = 0
    highest_close_since_entry = 0.0

    for current_index in range(1, len(df) - 1):
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            trend_up = row.Close > row.sma_200 and row.sma_50 >= row.sma_200
            momentum_positive = (
                row[MEDIUM_RETURN_COLUMN] > 0
                and row[RELATIVE_STRENGTH_RETURN_COLUMN] > 0
                and row.macd_line > row.macd_signal
                and row.adx_14 >= VALIDATION_MOMENTUM_ADX_MIN
            )
            rsi_in_range = VALIDATION_MOMENTUM_RSI_MIN <= row.rsi_14 <= VALIDATION_MOMENTUM_RSI_MAX
            controlled_volatility = row.atr_percent_ratio_60 <= VALIDATION_MOMENTUM_ATR_RATIO_MAX
            regime_is_tradeable = str(row.regime).strip().lower() != "stressed"
            pullback_reclaim = (
                row.Low <= row.ema_20
                and row.Close >= row.ema_20
                and row.Close >= previous_row.Close
            )
            volume_confirmation = pd.isna(row.volume_ratio_20) or float(row.volume_ratio_20) >= 0.9
            breakout_continuation = (
                row.Close > row[BREAKOUT_HIGH_COLUMN]
                and previous_row.Close <= previous_row[BREAKOUT_HIGH_COLUMN]
                and volume_confirmation
            )
            trigger_signal = pullback_reclaim or breakout_continuation
            entry_signal = (
                trend_up
                and momentum_positive
                and rsi_in_range
                and controlled_volatility
                and regime_is_tradeable
                and trigger_signal
            )

            if not entry_signal:
                continue
            signal_count += 1

            if not _has_room_for_entry_and_exit(current_index, len(df)):
                rejected_signal_count += 1
                continue

            initial_stop_from_structure = _safe_stop(row[BREAKOUT_LOW_COLUMN])
            initial_stop_from_atr = _safe_stop(
                row.Close - (VALIDATION_MOMENTUM_STOP_ATR_MULTIPLIER * row.atr_14)
            )
            if initial_stop_from_structure is None:
                stop_loss_used = initial_stop_from_atr
            elif initial_stop_from_atr is None:
                stop_loss_used = initial_stop_from_structure
            else:
                stop_loss_used = min(initial_stop_from_structure, initial_stop_from_atr)
            take_profit_used = _safe_stop(
                row.Close + (VALIDATION_MOMENTUM_TARGET_ATR_MULTIPLIER * row.atr_14)
            )

            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, take_profit_used):
                rejected_signal_count += 1
                continue

            raw_atr_ratio = float(row.atr_percent_ratio_60) if pd.notna(row.atr_percent_ratio_60) else 1.0
            stabilized_atr_ratio = max(raw_atr_ratio, 0.25)
            capital_fraction_override = float(
                max(
                    VALIDATION_MOMENTUM_MIN_CAPITAL_FRACTION,
                    min(VALIDATION_MOMENTUM_VOL_TARGET_RATIO / stabilized_atr_ratio, 1.0),
                )
            )

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="trend_momentum_verification",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=take_profit_used,
                capital_fraction_override=capital_fraction_override,
                entry_reason="trend_momentum_verification",
            )
            if candidate_position is not None:
                open_position = candidate_position
                highest_close_since_entry = float(row.Close)
            else:
                rejected_signal_count += 1
        else:
            if float(row.Close) > highest_close_since_entry:
                highest_close_since_entry = float(row.Close)

            exit_reason = _first_bar_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )

            if exit_reason is None:
                trailing_stop = highest_close_since_entry - (
                    VALIDATION_MOMENTUM_TRAILING_STOP_ATR_MULTIPLIER * row.atr_14
                )
                if row.Low <= trailing_stop:
                    exit_reason = "trailing_stop"

            if exit_reason is None and (
                row.atr_percent_ratio_60 >= VALIDATION_MOMENTUM_VOLATILITY_EXIT_RATIO
                and row.Close < row.ema_20
            ):
                exit_reason = "volatility_shock_exit"

            if exit_reason is None and (
                row.Close < row.sma_200
                or (row.trailing_return_20 <= 0 and row.macd_line < row.macd_signal)
            ):
                exit_reason = "trend_momentum_breakdown"

            if exit_reason is None:
                holding_bars = current_index - open_position.entry_index
                if holding_bars >= VALIDATION_MOMENTUM_MAX_HOLDING_BARS:
                    exit_reason = "time_stop"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None
            highest_close_since_entry = 0.0

    if open_position is not None:
        capital = _close_terminal_position(
            trades=trades,
            open_position=open_position,
            final_row=df.iloc[-1],
            final_exit_index=len(df) - 1,
            execution_rng=execution_rng,
            resolved_ticker=resolved_ticker,
        )
        open_position = None

    trades_df = _finalize_trades(trades)
    trades_df.attrs["signal_count"] = signal_count
    trades_df.attrs["rejected_signal_count"] = rejected_signal_count
    return trades_df


def run_adx_trend_following_strategy(
    market_df: pd.DataFrame,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Trade directional trends using Wilder-style ADX/+DI confirmation."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "ema_20",
            "sma_50",
            "sma_200",
            "adx_14",
            "plus_di_14",
            "minus_di_14",
            "atr_14",
            "atr_percent_ratio_60",
            "trailing_return_20",
            BREAKOUT_LOW_COLUMN,
            "regime",
        ],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("adx_trend_following", resolved_ticker)
    open_position: OpenPosition | None = None
    signal_count = 0
    rejected_signal_count = 0

    for current_index in range(1, len(df) - 1):
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            trend_filter = (
                row.Close > row.sma_200
                and row.sma_50 >= row.sma_200
                and row.trailing_return_20 > 0
            )
            directional_strength = (
                row.plus_di_14 > row.minus_di_14
                and row.adx_14 >= ADX_TREND_FOLLOWING_ADX_MIN
                and row.adx_14 >= previous_row.adx_14
            )
            volatility_filter = row.atr_percent_ratio_60 <= ADX_TREND_FOLLOWING_ATR_RATIO_MAX
            regime_filter = str(row.regime).strip().lower() != "stressed"
            entry_signal = trend_filter and directional_strength and volatility_filter and regime_filter

            if not entry_signal:
                continue
            signal_count += 1

            if not _has_room_for_entry_and_exit(current_index, len(df)):
                rejected_signal_count += 1
                continue

            stop_loss_used = _safe_stop(
                min(row[BREAKOUT_LOW_COLUMN], row.Close - (ADX_TREND_FOLLOWING_STOP_ATR_MULTIPLIER * row.atr_14))
            )
            take_profit_used = _safe_stop(
                row.Close + (ADX_TREND_FOLLOWING_TARGET_ATR_MULTIPLIER * row.atr_14)
            )
            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, take_profit_used):
                rejected_signal_count += 1
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="adx_trend_following",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=take_profit_used,
                entry_reason="adx_di_trend_following",
            )
            if candidate_position is not None:
                open_position = candidate_position
            else:
                rejected_signal_count += 1
        else:
            exit_reason = _first_bar_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            if exit_reason is None and (
                row.Close < row.ema_20
                and row.minus_di_14 > row.plus_di_14
            ):
                exit_reason = "trend_reversal"
            if exit_reason is None:
                holding_bars = current_index - open_position.entry_index
                if holding_bars >= ADX_TREND_FOLLOWING_MAX_HOLDING_BARS:
                    exit_reason = "time_stop"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None

    if open_position is not None:
        _close_terminal_position(
            trades=trades,
            open_position=open_position,
            final_row=df.iloc[-1],
            final_exit_index=len(df) - 1,
            execution_rng=execution_rng,
            resolved_ticker=resolved_ticker,
        )

    trades_df = _finalize_trades(trades)
    trades_df.attrs["signal_count"] = signal_count
    trades_df.attrs["rejected_signal_count"] = rejected_signal_count
    return trades_df


def run_uptrend_oversold_reversion_strategy(
    market_df: pd.DataFrame,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Buy short-term oversold dips only inside established uptrends."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "ema_20",
            "sma_50",
            "sma_200",
            "bollinger_lower",
            "rsi_14",
            "zscore_20",
            "atr_14",
            "regime",
        ],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("uptrend_oversold_reversion", resolved_ticker)
    open_position: OpenPosition | None = None
    signal_count = 0
    rejected_signal_count = 0

    for current_index in range(1, len(df) - 1):
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            trend_filter = (
                row.Close > row.sma_200
                and row.sma_50 >= row.sma_200
            )
            oversold_signal = (
                row.zscore_20 <= UPTREND_OVERSOLD_ZSCORE_MAX
                and (
                    row.Low <= row.bollinger_lower
                    or row.rsi_14 <= UPTREND_OVERSOLD_RSI_MAX
                )
            )
            rebound_confirmation = row.Close >= previous_row.Close
            regime_filter = str(row.regime).strip().lower() != "stressed"
            entry_signal = trend_filter and oversold_signal and rebound_confirmation and regime_filter

            if not entry_signal:
                continue
            signal_count += 1

            if not _has_room_for_entry_and_exit(current_index, len(df)):
                rejected_signal_count += 1
                continue

            stop_loss_used = _safe_stop(
                min(row.Low, row.Close - (UPTREND_OVERSOLD_STOP_ATR_MULTIPLIER * row.atr_14))
            )
            take_profit_used = _safe_stop(
                row.Close + (UPTREND_OVERSOLD_TARGET_ATR_MULTIPLIER * row.atr_14)
            )
            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, take_profit_used):
                rejected_signal_count += 1
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="uptrend_oversold_reversion",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=take_profit_used,
                entry_reason="uptrend_oversold_reversion",
            )
            if candidate_position is not None:
                open_position = candidate_position
            else:
                rejected_signal_count += 1
        else:
            exit_reason = _first_bar_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            if exit_reason is None and row.Close >= row.ema_20:
                exit_reason = "mean_reversion_complete"
            if exit_reason is None:
                holding_bars = current_index - open_position.entry_index
                if holding_bars >= UPTREND_OVERSOLD_MAX_HOLDING_BARS:
                    exit_reason = "time_stop"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None

    if open_position is not None:
        _close_terminal_position(
            trades=trades,
            open_position=open_position,
            final_row=df.iloc[-1],
            final_exit_index=len(df) - 1,
            execution_rng=execution_rng,
            resolved_ticker=resolved_ticker,
        )

    trades_df = _finalize_trades(trades)
    trades_df.attrs["signal_count"] = signal_count
    trades_df.attrs["rejected_signal_count"] = rejected_signal_count
    return trades_df


def run_volatility_squeeze_breakout_strategy(
    market_df: pd.DataFrame,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Trade breakouts that emerge from low-volatility compression regimes."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "ema_20",
            "sma_200",
            "adx_14",
            "atr_14",
            "macd_line",
            "macd_signal",
            "volume_ratio_20",
            "bollinger_width_ratio_60",
            BREAKOUT_HIGH_COLUMN,
            BREAKOUT_LOW_COLUMN,
            "regime",
        ],
    )
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("volatility_squeeze_breakout", resolved_ticker)
    open_position: OpenPosition | None = None
    signal_count = 0
    rejected_signal_count = 0
    highest_close_since_entry = 0.0

    for current_index in range(1, len(df) - 1):
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            compression = row.bollinger_width_ratio_60 <= VOLATILITY_SQUEEZE_BB_WIDTH_RATIO_MAX
            breakout = (
                row.Close > row[BREAKOUT_HIGH_COLUMN]
                and previous_row.Close <= previous_row[BREAKOUT_HIGH_COLUMN]
            )
            momentum_filter = (
                row.macd_line > row.macd_signal
                and row.adx_14 >= VOLATILITY_SQUEEZE_ADX_MIN
            )
            volume_filter = (
                pd.isna(row.volume_ratio_20)
                or float(row.volume_ratio_20) >= VOLATILITY_SQUEEZE_VOLUME_RATIO_MIN
            )
            trend_filter = row.Close > row.sma_200
            regime_filter = str(row.regime).strip().lower() != "stressed"
            entry_signal = (
                compression
                and breakout
                and momentum_filter
                and volume_filter
                and trend_filter
                and regime_filter
            )

            if not entry_signal:
                continue
            signal_count += 1

            if not _has_room_for_entry_and_exit(current_index, len(df)):
                rejected_signal_count += 1
                continue

            stop_loss_used = _safe_stop(
                min(
                    row[BREAKOUT_LOW_COLUMN],
                    row.Close - (VOLATILITY_SQUEEZE_STOP_ATR_MULTIPLIER * row.atr_14),
                )
            )
            take_profit_used = _safe_stop(
                row.Close + (VOLATILITY_SQUEEZE_TARGET_ATR_MULTIPLIER * row.atr_14)
            )
            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, take_profit_used):
                rejected_signal_count += 1
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="volatility_squeeze_breakout",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=take_profit_used,
                entry_reason="squeeze_breakout",
            )
            if candidate_position is not None:
                open_position = candidate_position
                highest_close_since_entry = float(row.Close)
            else:
                rejected_signal_count += 1
        else:
            if float(row.Close) > highest_close_since_entry:
                highest_close_since_entry = float(row.Close)

            exit_reason = _first_bar_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            if exit_reason is None:
                trailing_stop = highest_close_since_entry - (
                    VOLATILITY_SQUEEZE_STOP_ATR_MULTIPLIER * row.atr_14
                )
                if row.Low <= trailing_stop:
                    exit_reason = "trailing_stop"
            if exit_reason is None and row.Close < row.ema_20 and row.macd_line < row.macd_signal:
                exit_reason = "breakout_failed"
            if exit_reason is None:
                holding_bars = current_index - open_position.entry_index
                if holding_bars >= VOLATILITY_SQUEEZE_MAX_HOLDING_BARS:
                    exit_reason = "time_stop"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None
            highest_close_since_entry = 0.0

    if open_position is not None:
        _close_terminal_position(
            trades=trades,
            open_position=open_position,
            final_row=df.iloc[-1],
            final_exit_index=len(df) - 1,
            execution_rng=execution_rng,
            resolved_ticker=resolved_ticker,
        )

    trades_df = _finalize_trades(trades)
    trades_df.attrs["signal_count"] = signal_count
    trades_df.attrs["rejected_signal_count"] = rejected_signal_count
    return trades_df


def run_connors_rsi2_pullback_strategy(
    market_df: pd.DataFrame,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Buy short-term RSI(2) washouts only inside higher-timeframe uptrends."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "sma_50",
            "sma_200",
            "atr_14",
            "regime",
        ],
    )
    df["rsi_2"] = _compute_rsi_2(df["Close"])
    df["sma_5"] = pd.to_numeric(
        df["Close"].rolling(window=CONNORS_RSI2_EXIT_SMA_BARS).mean(),
        errors="coerce",
    )
    df = df.dropna(subset=["rsi_2", "sma_5"]).reset_index(drop=True)

    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("connors_rsi2_pullback", resolved_ticker)
    open_position: OpenPosition | None = None
    signal_count = 0
    rejected_signal_count = 0

    for current_index in range(1, len(df) - 1):
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            trend_filter = (
                row.sma_50 >= row.sma_200
                and (
                    (not CONNORS_RSI2_REQUIRE_SMA200_FILTER)
                    or row.Close > row.sma_200
                )
            )
            oversold_signal = row.rsi_2 <= CONNORS_RSI2_ENTRY_MAX
            dip_confirmation = row.Close <= previous_row.Close
            regime_filter = str(row.regime).strip().lower() != "stressed"
            entry_signal = trend_filter and oversold_signal and dip_confirmation and regime_filter

            if not entry_signal:
                continue
            signal_count += 1

            if not _has_room_for_entry_and_exit(current_index, len(df)):
                rejected_signal_count += 1
                continue

            stop_loss_used = _safe_stop(
                min(
                    row.Low,
                    row.Close - (CONNORS_RSI2_STOP_ATR_MULTIPLIER * row.atr_14),
                )
            )
            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, None):
                rejected_signal_count += 1
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="connors_rsi2_pullback",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=None,
                entry_reason="connors_rsi2_pullback",
            )
            if candidate_position is None:
                rejected_signal_count += 1
                continue
            open_position = candidate_position
        else:
            exit_reason = _first_bar_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            if exit_reason is None and row.rsi_2 >= CONNORS_RSI2_EXIT_MIN:
                exit_reason = "rsi2_recovery"
            if exit_reason is None and row.Close >= row.sma_5:
                exit_reason = "mean_reversion_complete"
            if exit_reason is None and row.Close < row.sma_200:
                exit_reason = "long_trend_break"
            if exit_reason is None:
                holding_bars = current_index - open_position.entry_index
                if holding_bars >= CONNORS_RSI2_MAX_HOLDING_BARS:
                    exit_reason = "time_stop"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None

    if open_position is not None:
        _close_terminal_position(
            trades=trades,
            open_position=open_position,
            final_row=df.iloc[-1],
            final_exit_index=len(df) - 1,
            execution_rng=execution_rng,
            resolved_ticker=resolved_ticker,
        )

    trades_df = _finalize_trades(trades)
    trades_df.attrs["signal_count"] = signal_count
    trades_df.attrs["rejected_signal_count"] = rejected_signal_count
    return trades_df


def run_donchian_trend_reentry_strategy(
    market_df: pd.DataFrame,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Trade medium-term Donchian breakouts with trend and momentum filters."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "ema_20",
            "sma_50",
            "sma_200",
            "adx_14",
            "atr_14",
            "macd_line",
            "macd_signal",
            "regime",
        ],
    )
    df[DONCHIAN_BREAKOUT_HIGH_COLUMN] = pd.to_numeric(
        df["High"].rolling(window=DONCHIAN_BREAKOUT_LOOKBACK_DAYS).max().shift(1),
        errors="coerce",
    )
    df[DONCHIAN_BREAKOUT_STOP_LOW_COLUMN] = pd.to_numeric(
        df["Low"].rolling(window=DONCHIAN_STOP_LOOKBACK_DAYS).min().shift(1),
        errors="coerce",
    )
    df = df.dropna(
        subset=[DONCHIAN_BREAKOUT_HIGH_COLUMN, DONCHIAN_BREAKOUT_STOP_LOW_COLUMN]
    ).reset_index(drop=True)
    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("donchian_trend_reentry", resolved_ticker)
    open_position: OpenPosition | None = None
    signal_count = 0
    rejected_signal_count = 0

    for current_index in range(1, len(df) - 1):
        previous_row = df.iloc[current_index - 1]
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            trend_filter = row.Close > row.sma_200 and row.sma_50 >= row.sma_200
            breakout_trigger = (
                row.Close > row[DONCHIAN_BREAKOUT_HIGH_COLUMN]
                and previous_row.Close <= previous_row[DONCHIAN_BREAKOUT_HIGH_COLUMN]
            )
            momentum_filter = (
                row.macd_line >= row.macd_signal
                and row.adx_14 >= DONCHIAN_ADX_MIN
            )
            regime_filter = str(row.regime).strip().lower() != "stressed"
            entry_signal = trend_filter and breakout_trigger and momentum_filter and regime_filter

            if not entry_signal:
                continue
            signal_count += 1

            if not _has_room_for_entry_and_exit(current_index, len(df)):
                rejected_signal_count += 1
                continue

            stop_loss_used = _safe_stop(
                min(
                    row[DONCHIAN_BREAKOUT_STOP_LOW_COLUMN],
                    row.Close - (DONCHIAN_STOP_ATR_MULTIPLIER * row.atr_14),
                )
            )
            take_profit_used = _safe_stop(
                row.Close + (DONCHIAN_TARGET_ATR_MULTIPLIER * row.atr_14)
            )
            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, take_profit_used):
                rejected_signal_count += 1
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="donchian_trend_reentry",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=take_profit_used,
                entry_reason="donchian_breakout_reentry",
            )
            if candidate_position is None:
                rejected_signal_count += 1
                continue
            open_position = candidate_position
        else:
            exit_reason = _first_bar_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            if exit_reason is None and row.Close < row.sma_50:
                exit_reason = "trend_support_break"
            if exit_reason is None and row.Close < row.ema_20 and row.macd_line < row.macd_signal:
                exit_reason = "momentum_fade"
            if exit_reason is None:
                holding_bars = current_index - open_position.entry_index
                if holding_bars >= DONCHIAN_MAX_HOLDING_BARS:
                    exit_reason = "time_stop"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None

    if open_position is not None:
        _close_terminal_position(
            trades=trades,
            open_position=open_position,
            final_row=df.iloc[-1],
            final_exit_index=len(df) - 1,
            execution_rng=execution_rng,
            resolved_ticker=resolved_ticker,
        )

    trades_df = _finalize_trades(trades)
    trades_df.attrs["signal_count"] = signal_count
    trades_df.attrs["rejected_signal_count"] = rejected_signal_count
    return trades_df


def run_turn_of_month_seasonality_strategy(
    market_df: pd.DataFrame,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Trade a fixed window after month transitions with conservative risk control."""
    df = _prepare_market_df(
        market_df,
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "sma_200",
            "atr_14",
            "regime",
        ],
    )
    month_bucket = df["Date"].dt.to_period("M")
    df["is_month_end"] = month_bucket != month_bucket.shift(-1)
    df = df.dropna(subset=["is_month_end"]).reset_index(drop=True)

    resolved_ticker = _resolve_strategy_ticker(market_df, ticker)
    trades: list[dict[str, object]] = []
    capital = STARTING_CAPITAL
    execution_rng = build_execution_rng("turn_of_month_seasonality", resolved_ticker)
    open_position: OpenPosition | None = None
    signal_count = 0
    rejected_signal_count = 0

    for current_index in range(1, len(df) - 1):
        row = df.iloc[current_index]
        next_row = df.iloc[current_index + 1]

        if open_position is None:
            trend_filter = (
                (not TURN_OF_MONTH_REQUIRE_SMA200_FILTER)
                or row.Close > row.sma_200
            )
            regime_filter = str(row.regime).strip().lower() != "stressed"
            entry_signal = bool(row.is_month_end) and trend_filter and regime_filter

            if not entry_signal:
                continue
            signal_count += 1

            if not _has_room_for_entry_and_exit(current_index, len(df)):
                rejected_signal_count += 1
                continue

            stop_loss_used = _safe_stop(
                row.Close - (TURN_OF_MONTH_STOP_ATR_MULTIPLIER * row.atr_14)
            )
            if not _entry_is_still_valid(float(next_row.Open), stop_loss_used, None):
                rejected_signal_count += 1
                continue

            candidate_position = open_position_from_signal(
                signal_row=row,
                next_row=next_row,
                capital_before=capital,
                regime_at_entry=str(row.regime),
                entry_index=current_index + 1,
                rng=execution_rng,
                strategy_name="turn_of_month_seasonality",
                ticker=resolved_ticker,
                stop_loss_used=stop_loss_used,
                take_profit_used=None,
                entry_reason="turn_of_month_window",
            )
            if candidate_position is None:
                rejected_signal_count += 1
                continue
            open_position = candidate_position
        else:
            exit_reason = _first_bar_exit_reason(
                row,
                open_position.stop_loss_used,
                open_position.take_profit_used,
            )
            if (
                exit_reason is None
                and TURN_OF_MONTH_REQUIRE_SMA200_FILTER
                and row.Close < row.sma_200
            ):
                exit_reason = "trend_filter_break"
            if exit_reason is None:
                holding_bars = current_index - open_position.entry_index
                if holding_bars >= TURN_OF_MONTH_MAX_HOLDING_BARS:
                    exit_reason = "calendar_window_complete"

            if exit_reason is None:
                continue

            trade_record = close_position_from_signal(
                position=open_position,
                next_row=next_row,
                exit_index=current_index + 1,
                rng=execution_rng,
                ticker=resolved_ticker,
                exit_reason=exit_reason,
            )
            trades.append(trade_record)
            capital = float(trade_record["capital_after"])
            open_position = None

    if open_position is not None:
        _close_terminal_position(
            trades=trades,
            open_position=open_position,
            final_row=df.iloc[-1],
            final_exit_index=len(df) - 1,
            execution_rng=execution_rng,
            resolved_ticker=resolved_ticker,
        )

    trades_df = _finalize_trades(trades)
    trades_df.attrs["signal_count"] = signal_count
    trades_df.attrs["rejected_signal_count"] = rejected_signal_count
    return trades_df


def run_strategy(
    agent_name: str,
    market_df: pd.DataFrame,
    random_decision_seed: int | None = None,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Dispatch one shared in-memory runner by strategy name."""
    if agent_name == "trend_pullback":
        return run_trend_pullback_strategy(market_df, ticker=ticker)
    if agent_name == "breakout_volume_momentum":
        return run_breakout_volume_momentum_strategy(market_df, ticker=ticker)
    if agent_name == "mean_reversion_vol_filter":
        return run_mean_reversion_vol_filter_strategy(market_df, ticker=ticker)
    if agent_name == "momentum_relative_strength":
        try:
            from momentum_relative_strength_agent import run_relative_strength_strategy_for_market_df
        except ModuleNotFoundError:
            from Code.momentum_relative_strength_agent import run_relative_strength_strategy_for_market_df

        resolved_ticker = ticker
        if resolved_ticker is None:
            resolved_ticker = str(market_df.attrs.get("ticker", "")).strip().upper() or None
        return run_relative_strength_strategy_for_market_df(
            market_df,
            anchor_ticker=resolved_ticker,
        )
    if agent_name == "random":
        return run_random_strategy(
            market_df,
            decision_seed=random_decision_seed,
            ticker=ticker,
        )

    if agent_name == "trend_momentum_verification":
        return run_trend_momentum_verification_strategy(market_df, ticker=ticker)
    if agent_name == "adx_trend_following":
        return run_adx_trend_following_strategy(market_df, ticker=ticker)
    if agent_name == "uptrend_oversold_reversion":
        return run_uptrend_oversold_reversion_strategy(market_df, ticker=ticker)
    if agent_name == "volatility_squeeze_breakout":
        return run_volatility_squeeze_breakout_strategy(market_df, ticker=ticker)
    if agent_name == "connors_rsi2_pullback":
        return run_connors_rsi2_pullback_strategy(market_df, ticker=ticker)
    if agent_name == "donchian_trend_reentry":
        return run_donchian_trend_reentry_strategy(market_df, ticker=ticker)
    if agent_name == "turn_of_month_seasonality":
        return run_turn_of_month_seasonality_strategy(market_df, ticker=ticker)

    raise ValueError(f"Unsupported strategy name: {agent_name}")
