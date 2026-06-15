"""Shared strategy names, colors, and parameter defaults for the research pipeline.

When the pipeline runs on a non-daily timeframe (RESEARCH_INTERVAL != "1d"),
all bar-count parameters are automatically scaled so that indicators and
holding periods cover the same *calendar* duration as they would on daily data.
Threshold parameters (RSI levels, ATR multipliers, ratios, etc.) are left
unchanged because they are already scale-independent.

An explicit environment variable override always wins — if you set
``MA_20_BARS=20`` the pipeline will use exactly 20 bars regardless of the
active timeframe.  Without an override the default is computed as:

    scaled_value = max(1, round(daily_default * BARS_PER_TRADING_DAY))
"""

from __future__ import annotations

import os

try:
    from timeframe_config import (
        BARS_PER_TRADING_DAY,
        inverse_scale_daily_float,
        scale_daily_bars,
        scale_daily_signal_bars,
    )
except ModuleNotFoundError:
    from Code.timeframe_config import (
        BARS_PER_TRADING_DAY,
        inverse_scale_daily_float,
        scale_daily_bars,
        scale_daily_signal_bars,
    )


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a beginner-friendly helper."""
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a beginner-friendly helper."""
    return float(os.environ.get(name, str(default)))


def _env_text(name: str, default: str) -> str:
    """Read a text environment variable while keeping a readable default path."""
    return os.environ.get(name, default).strip()


def _env_int_scaled(name: str, daily_default: int) -> int:
    """Read an env var or auto-scale the daily default to the current timeframe.

    If the environment variable *name* is explicitly set, its value is used
    as-is (no scaling).  Otherwise the daily default is multiplied by
    BARS_PER_TRADING_DAY so indicator windows and holding periods cover the
    same calendar duration across timeframes.

    Use this for **structural** parameters: long MAs (100, 200), holding
    periods, time stops, regime history, and lookback windows for
    structural levels (rolling highs/lows for stop placement).
    """
    explicit = os.environ.get(name)
    if explicit is not None:
        return int(explicit)
    return scale_daily_bars(daily_default)


def _env_int_signal_scaled(name: str, daily_default: int) -> int:
    """Read an env var or signal-scale a daily default for intraday use.

    Uses ``sqrt(BARS_PER_TRADING_DAY)`` scaling so signal-generating
    indicators (RSI, ADX, ATR, Bollinger, MACD) stay responsive on faster
    timeframes while still being longer than their daily bar-count defaults.

    On daily data this is identical to ``_env_int_scaled``.
    """
    explicit = os.environ.get(name)
    if explicit is not None:
        return int(explicit)
    return scale_daily_signal_bars(daily_default)


def _env_float_inverse_scaled(name: str, daily_default: float) -> float:
    """Read an env var or inversely scale a daily probability.

    Used for per-bar entry probabilities: on faster timeframes the per-bar
    probability must decrease so that the *daily* trade frequency stays
    comparable.
    """
    explicit = os.environ.get(name)
    if explicit is not None:
        return float(explicit)
    return inverse_scale_daily_float(daily_default)


# ---------------------------------------------------------------------------
# Strategy ordering
# ---------------------------------------------------------------------------
#
# The order below drives:
# - the comparison table
# - the chart legends
# - the Monte Carlo / RCSI tables
# - the UI summaries
#
# Keep these names stable unless the entire pipeline is updated with them.
# The default stack prioritizes timing-style single-asset strategies.
# Momentum + Relative Strength is still available as an opt-in strategy via
# ENABLE_RELATIVE_STRENGTH=1 because it is primarily cross-asset selection.
ENABLE_RELATIVE_STRENGTH = _env_text("ENABLE_RELATIVE_STRENGTH", "0").lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
BASE_AGENT_ORDER = [
    "trend_pullback",
    "breakout_volume_momentum",
    "mean_reversion_vol_filter",
    "trend_momentum_verification",
    "adx_trend_following",
    "uptrend_oversold_reversion",
    "volatility_squeeze_breakout",
    "connors_rsi2_pullback",
    "donchian_trend_reentry",
    "turn_of_month_seasonality",
    "random",
]
AGENT_ORDER = BASE_AGENT_ORDER.copy()
if ENABLE_RELATIVE_STRENGTH:
    AGENT_ORDER.insert(3, "momentum_relative_strength")
BENCHMARK_NAME = "buy_and_hold"
COMPARISON_ORDER = AGENT_ORDER + [BENCHMARK_NAME]
EXPERIMENTAL_AGENT_ORDER = [
    "adx_trend_following",
    "uptrend_oversold_reversion",
    "volatility_squeeze_breakout",
]

AGENT_DISPLAY_NAMES = {
    "trend_pullback": "Trend + Pullback",
    "breakout_volume_momentum": "Breakout + Volume + Momentum",
    "mean_reversion_vol_filter": "Mean Reversion + Volatility Filter",
    "momentum_relative_strength": "Momentum + Relative Strength",
    "trend_momentum_verification": "Validation Strategy (Adaptive Volatility Momentum)",
    "random": "Random Baseline",
    "adx_trend_following": "ADX Trend Following",
    "uptrend_oversold_reversion": "Uptrend Oversold Reversion",
    "volatility_squeeze_breakout": "Volatility Squeeze Breakout",
    "connors_rsi2_pullback": "Connors RSI(2) Pullback",
    "donchian_trend_reentry": "Donchian Trend Reentry",
    "turn_of_month_seasonality": "Turn-of-Month Seasonality",
    BENCHMARK_NAME: "Buy and Hold",
}

AGENT_SHORT_DISPLAY_NAMES = {
    "trend_pullback": "Trend Pullback",
    "breakout_volume_momentum": "Breakout Vol+Mom",
    "mean_reversion_vol_filter": "Mean Rev Vol Filter",
    "momentum_relative_strength": "Rel Strength Mom",
    "trend_momentum_verification": "Validation AVM",
    "random": "Random",
    "adx_trend_following": "ADX Trend",
    "uptrend_oversold_reversion": "Oversold Reversion",
    "volatility_squeeze_breakout": "Squeeze Breakout",
    "connors_rsi2_pullback": "Connors RSI2",
    "donchian_trend_reentry": "Donchian Reentry",
    "turn_of_month_seasonality": "Turn-of-Month",
    BENCHMARK_NAME: "Buy & Hold",
}

AGENT_COLORS = {
    # Professional, high-contrast Tableau-style palette.
    "trend_pullback": "#1F77B4",
    "breakout_volume_momentum": "#FF7F0E",
    "mean_reversion_vol_filter": "#2CA02C",
    "momentum_relative_strength": "#D62728",
    "trend_momentum_verification": "#9467BD",
    "adx_trend_following": "#8C564B",
    "uptrend_oversold_reversion": "#E377C2",
    "volatility_squeeze_breakout": "#7F7F7F",
    "connors_rsi2_pullback": "#BCBD22",
    "donchian_trend_reentry": "#AEC7E8",
    "turn_of_month_seasonality": "#FF9896",
    "random": "#17BECF",
    BENCHMARK_NAME: "#2F2F2F",
}


# ---------------------------------------------------------------------------
# Feature-engineering defaults (auto-scaled for the active timeframe)
# ---------------------------------------------------------------------------
# On daily bars these are the classic chartist settings: 20-day EMA,
# 50/100/200-day SMAs, 14-day RSI/ATR/ADX, etc.
#
# Two scaling modes are used:
#   _env_int_scaled        → full calendar scaling (BARS_PER_TRADING_DAY)
#                             for structural trend filters and level lookbacks
#   _env_int_signal_scaled → sqrt scaling (SIGNAL_SCALING_FACTOR)
#                             for signal-generating indicators (RSI, ADX, ATR,
#                             Bollinger, MACD) so they stay responsive on
#                             intraday data
#
# Short-horizon MAs (20, 50) are signal-scaled because they drive entry/exit
# signals. Long-horizon MAs (100, 200) are fully scaled because they serve
# as structural trend filters.
MA_20_BARS = _env_int_signal_scaled("MA_20_BARS", 20)
MA_50_BARS = _env_int_signal_scaled("MA_50_BARS", 50)
MA_100_BARS = _env_int_scaled("MA_100_BARS", 100)
MA_200_BARS = _env_int_scaled("MA_200_BARS", 200)
PRICE_STD_LOOKBACK_BARS = _env_int_signal_scaled("PRICE_STD_LOOKBACK_BARS", 20)
SHORT_RETURN_LOOKBACK_BARS = _env_int_signal_scaled("SHORT_RETURN_LOOKBACK_BARS", 5)
MEDIUM_RETURN_LOOKBACK_BARS = _env_int_signal_scaled("MEDIUM_RETURN_LOOKBACK_BARS", 20)
RSI_LOOKBACK_DAYS = _env_int_signal_scaled("RSI_LOOKBACK_DAYS", 14)
ATR_LOOKBACK_DAYS = _env_int_signal_scaled("ATR_LOOKBACK_DAYS", 14)
BOLLINGER_LOOKBACK_DAYS = _env_int_signal_scaled("BOLLINGER_LOOKBACK_DAYS", 20)
BOLLINGER_STD_MULTIPLIER = _env_float("BOLLINGER_STD_MULTIPLIER", 2.0)
VOLUME_LOOKBACK_DAYS = _env_int_signal_scaled("VOLUME_LOOKBACK_DAYS", 20)
MACD_FAST_DAYS = _env_int_signal_scaled("MACD_FAST_DAYS", 12)
MACD_SLOW_DAYS = _env_int_signal_scaled("MACD_SLOW_DAYS", 26)
MACD_SIGNAL_DAYS = _env_int_signal_scaled("MACD_SIGNAL_DAYS", 9)
ADX_LOOKBACK_DAYS = _env_int_signal_scaled("ADX_LOOKBACK_DAYS", 14)


# ---------------------------------------------------------------------------
# Trend + Pullback parameters
# ---------------------------------------------------------------------------
# RSI band is wider than textbook (35-65 vs 40-60) to avoid filtering out
# valid pullbacks.  ADX 18 captures genuine trends while the rolling stop
# and target windows use 10-day and 20-day lookbacks — natural daily S/R
# levels.  ATR buffer of 1.0 gives stops enough room to survive normal
# daily noise without making risk too wide.
# Thresholds (RSI, ADX, ATR buffer) are scale-independent and NOT scaled.
TREND_PULLBACK_RSI_MIN = _env_float("TREND_PULLBACK_RSI_MIN", 35.0)
TREND_PULLBACK_RSI_MAX = _env_float("TREND_PULLBACK_RSI_MAX", 65.0)
TREND_PULLBACK_ADX_MIN = _env_float("TREND_PULLBACK_ADX_MIN", 18.0)
TREND_PULLBACK_STOP_LOOKBACK_DAYS = _env_int_signal_scaled("TREND_PULLBACK_STOP_LOOKBACK_DAYS", 10)
TREND_PULLBACK_TARGET_LOOKBACK_DAYS = _env_int_signal_scaled("TREND_PULLBACK_TARGET_LOOKBACK_DAYS", 20)
TREND_PULLBACK_STOP_ATR_BUFFER = _env_float("TREND_PULLBACK_STOP_ATR_BUFFER", 1.0)


# ---------------------------------------------------------------------------
# Breakout + Volume + Momentum parameters
# ---------------------------------------------------------------------------
# Bar-count parameters (lookbacks, time stops) are auto-scaled.
# Thresholds, ratios, and price buffers are scale-independent.
BREAKOUT_LOOKBACK_DAYS = _env_int_signal_scaled("BREAKOUT_LOOKBACK_DAYS", 20)
BREAKOUT_STOP_LOOKBACK_DAYS = _env_int_signal_scaled("BREAKOUT_STOP_LOOKBACK_DAYS", 10)
BREAKOUT_VOLUME_MULTIPLIER = _env_float("BREAKOUT_VOLUME_MULTIPLIER", 1.1)
BREAKOUT_MOMENTUM_LOOKBACK_DAYS = _env_int_signal_scaled("BREAKOUT_MOMENTUM_LOOKBACK_DAYS", 12)
BREAKOUT_MOMENTUM_THRESHOLD = _env_float("BREAKOUT_MOMENTUM_THRESHOLD", 0.005)
BREAKOUT_ADX_MIN = _env_float("BREAKOUT_ADX_MIN", 18.0)
BREAKOUT_REWARD_TO_RISK = _env_float("BREAKOUT_REWARD_TO_RISK", 2.0)
BREAKOUT_TIME_STOP_BARS = _env_int_scaled("BREAKOUT_TIME_STOP_BARS", 15)
BREAKOUT_CLOSE_BUFFER = _env_float("BREAKOUT_CLOSE_BUFFER", 0.002)
BREAKOUT_STOP_ATR_BUFFER = _env_float("BREAKOUT_STOP_ATR_BUFFER", 1.0)
BREAKOUT_FAILED_BREAKOUT_BUFFER = _env_float("BREAKOUT_FAILED_BREAKOUT_BUFFER", 0.005)
BREAKOUT_NO_VOLUME_RANGE_EXPANSION_RATIO = _env_float(
    "BREAKOUT_NO_VOLUME_RANGE_EXPANSION_RATIO",
    1.03,
)


# ---------------------------------------------------------------------------
# Mean Reversion + Volatility Filter parameters
# ---------------------------------------------------------------------------
# Bar-count lookbacks are auto-scaled.  Thresholds / ratios are not.
MEAN_REVERSION_ATR_FILTER_LOOKBACK_DAYS = _env_int_signal_scaled(
    "MEAN_REVERSION_ATR_FILTER_LOOKBACK_DAYS",
    20,
)
MEAN_REVERSION_BB_FILTER_LOOKBACK_DAYS = _env_int_signal_scaled(
    "MEAN_REVERSION_BB_FILTER_LOOKBACK_DAYS",
    20,
)
MEAN_REVERSION_STOP_LOOKBACK_DAYS = _env_int_signal_scaled("MEAN_REVERSION_STOP_LOOKBACK_DAYS", 5)
MEAN_REVERSION_STOP_ATR_BUFFER = _env_float("MEAN_REVERSION_STOP_ATR_BUFFER", 1.0)
MEAN_REVERSION_RSI_ENTRY_MAX = _env_float("MEAN_REVERSION_RSI_ENTRY_MAX", 35.0)
MEAN_REVERSION_RSI_EXIT_MIN = _env_float("MEAN_REVERSION_RSI_EXIT_MIN", 50.0)
MEAN_REVERSION_ADX_MAX = _env_float("MEAN_REVERSION_ADX_MAX", 25.0)
MEAN_REVERSION_ATR_RATIO_MAX = _env_float("MEAN_REVERSION_ATR_RATIO_MAX", 1.6)
MEAN_REVERSION_BB_WIDTH_RATIO_MAX = _env_float("MEAN_REVERSION_BB_WIDTH_RATIO_MAX", 1.6)
MEAN_REVERSION_TREND_DISTANCE_THRESHOLD = _env_float(
    "MEAN_REVERSION_TREND_DISTANCE_THRESHOLD",
    0.03,
)
MEAN_REVERSION_MA_ALIGNMENT_THRESHOLD = _env_float(
    "MEAN_REVERSION_MA_ALIGNMENT_THRESHOLD",
    0.025,
)
MEAN_REVERSION_LOWER_BAND_BUFFER = _env_float("MEAN_REVERSION_LOWER_BAND_BUFFER", 0.002)


# ---------------------------------------------------------------------------
# Momentum + Relative Strength parameters
# ---------------------------------------------------------------------------
#
# We keep the legacy MOMENTUM_LOOKBACK_DAYS environment variable as an alias so
# the current UI and any old automation do not break. The newer, clearer name is
# RELATIVE_STRENGTH_LOOKBACK_DAYS.  When neither env var is set the daily
# default of 60 is scaled to the active timeframe.
_rel_strength_explicit = os.environ.get(
    "RELATIVE_STRENGTH_LOOKBACK_DAYS",
    os.environ.get("MOMENTUM_LOOKBACK_DAYS"),
)
RELATIVE_STRENGTH_LOOKBACK_DAYS = (
    int(_rel_strength_explicit)
    if _rel_strength_explicit is not None
    else scale_daily_signal_bars(60)
)
MOMENTUM_LOOKBACK_DAYS = RELATIVE_STRENGTH_LOOKBACK_DAYS
RELATIVE_STRENGTH_TOP_N = _env_int("RELATIVE_STRENGTH_TOP_N", 1)
RELATIVE_STRENGTH_REBALANCE_FREQUENCY = _env_text(
    "RELATIVE_STRENGTH_REBALANCE_FREQUENCY",
    "daily",
).lower()
RELATIVE_STRENGTH_ABSOLUTE_MOMENTUM_THRESHOLD = _env_float(
    "RELATIVE_STRENGTH_ABSOLUTE_MOMENTUM_THRESHOLD",
    0.002,
)
RELATIVE_STRENGTH_TRAILING_STOP_ATR_MULTIPLIER = _env_float(
    "RELATIVE_STRENGTH_TRAILING_STOP_ATR_MULTIPLIER",
    3.5,
)
# When this environment variable is populated, it acts as an explicit manual
# override of the automatically selected same-asset-class universe.
RELATIVE_STRENGTH_MANUAL_UNIVERSE_TEXT = _env_text(
    "RELATIVE_STRENGTH_UNIVERSE",
    "",
)
RELATIVE_STRENGTH_FORCE_ASSET_CLASS = _env_text(
    "RELATIVE_STRENGTH_FORCE_ASSET_CLASS",
    "",
).lower()


# ---------------------------------------------------------------------------
# Random baseline parameters (auto-scaled for the active timeframe)
# ---------------------------------------------------------------------------
# Holding periods scale UP on faster timeframes so the random agent holds
# for the same *calendar* duration.  Entry probability scales DOWN so the
# number of trades per calendar day stays comparable.
RANDOM_HOLDING_PERIOD_MIN_BARS = _env_int_scaled("RANDOM_HOLDING_PERIOD_MIN_BARS", 4)
RANDOM_HOLDING_PERIOD_MAX_BARS = _env_int_scaled("RANDOM_HOLDING_PERIOD_MAX_BARS", 18)
RANDOM_ENTRY_PROBABILITY = _env_float_inverse_scaled("RANDOM_ENTRY_PROBABILITY", 0.07)


# ---------------------------------------------------------------------------
# Validation strategy (adaptive volatility-managed momentum) parameters
# ---------------------------------------------------------------------------
# Purpose: stress-test whether the framework can detect skill when there is a
# plausible edge. The design combines classic long-horizon momentum with
# volatility-aware sizing and risk controls.
VALIDATION_MOMENTUM_RSI_MIN = _env_float(
    "VALIDATION_MOMENTUM_RSI_MIN",
    _env_float("VALIDATION_TREND_RSI_MIN", 45.0),
)
VALIDATION_MOMENTUM_RSI_MAX = _env_float(
    "VALIDATION_MOMENTUM_RSI_MAX",
    _env_float("VALIDATION_TREND_RSI_MAX", 82.0),
)
VALIDATION_MOMENTUM_ADX_MIN = _env_float("VALIDATION_MOMENTUM_ADX_MIN", 10.0)
VALIDATION_MOMENTUM_ATR_RATIO_MAX = _env_float(
    "VALIDATION_MOMENTUM_ATR_RATIO_MAX",
    _env_float("VALIDATION_TREND_ATR_RATIO_MAX", 1.9),
)
VALIDATION_MOMENTUM_VOL_TARGET_RATIO = _env_float(
    "VALIDATION_MOMENTUM_VOL_TARGET_RATIO",
    1.1,
)
VALIDATION_MOMENTUM_MIN_CAPITAL_FRACTION = _env_float(
    "VALIDATION_MOMENTUM_MIN_CAPITAL_FRACTION",
    0.35,
)
VALIDATION_MOMENTUM_STOP_ATR_MULTIPLIER = _env_float(
    "VALIDATION_MOMENTUM_STOP_ATR_MULTIPLIER",
    _env_float("VALIDATION_TREND_STOP_ATR_MULTIPLIER", 2.2),
)
VALIDATION_MOMENTUM_TARGET_ATR_MULTIPLIER = _env_float(
    "VALIDATION_MOMENTUM_TARGET_ATR_MULTIPLIER",
    _env_float("VALIDATION_TREND_TARGET_ATR_MULTIPLIER", 6.0),
)
VALIDATION_MOMENTUM_TRAILING_STOP_ATR_MULTIPLIER = _env_float(
    "VALIDATION_MOMENTUM_TRAILING_STOP_ATR_MULTIPLIER",
    _env_float("VALIDATION_TREND_TRAILING_STOP_ATR_MULTIPLIER", 3.0),
)
VALIDATION_MOMENTUM_VOLATILITY_EXIT_RATIO = _env_float(
    "VALIDATION_MOMENTUM_VOLATILITY_EXIT_RATIO",
    2.6,
)
_val_max_hold_explicit = os.environ.get(
    "VALIDATION_MOMENTUM_MAX_HOLDING_BARS",
    os.environ.get("VALIDATION_TREND_MAX_HOLDING_BARS"),
)
VALIDATION_MOMENTUM_MAX_HOLDING_BARS = (
    int(_val_max_hold_explicit)
    if _val_max_hold_explicit is not None
    else scale_daily_bars(80)
)

# Backward-compatible aliases retained so any old imports/env references keep
# working while the implementation uses the new momentum-oriented naming.
VALIDATION_TREND_RSI_MIN = VALIDATION_MOMENTUM_RSI_MIN
VALIDATION_TREND_RSI_MAX = VALIDATION_MOMENTUM_RSI_MAX
VALIDATION_TREND_ATR_RATIO_MAX = VALIDATION_MOMENTUM_ATR_RATIO_MAX
VALIDATION_TREND_BB_WIDTH_RATIO_MAX = _env_float("VALIDATION_TREND_BB_WIDTH_RATIO_MAX", 2.5)
VALIDATION_TREND_STOP_ATR_MULTIPLIER = VALIDATION_MOMENTUM_STOP_ATR_MULTIPLIER
VALIDATION_TREND_TARGET_ATR_MULTIPLIER = VALIDATION_MOMENTUM_TARGET_ATR_MULTIPLIER
VALIDATION_TREND_TRAILING_STOP_ATR_MULTIPLIER = VALIDATION_MOMENTUM_TRAILING_STOP_ATR_MULTIPLIER
VALIDATION_TREND_MAX_HOLDING_BARS = VALIDATION_MOMENTUM_MAX_HOLDING_BARS


# ---------------------------------------------------------------------------
# Experimental strategy parameters (optional testing simulator suite)
# ---------------------------------------------------------------------------
# ADX Trend Following: classic directional trend logic with volatility guard.
ADX_TREND_FOLLOWING_ADX_MIN = _env_float("ADX_TREND_FOLLOWING_ADX_MIN", 18.0)
ADX_TREND_FOLLOWING_ATR_RATIO_MAX = _env_float("ADX_TREND_FOLLOWING_ATR_RATIO_MAX", 2.2)
ADX_TREND_FOLLOWING_STOP_ATR_MULTIPLIER = _env_float(
    "ADX_TREND_FOLLOWING_STOP_ATR_MULTIPLIER",
    2.0,
)
ADX_TREND_FOLLOWING_TARGET_ATR_MULTIPLIER = _env_float(
    "ADX_TREND_FOLLOWING_TARGET_ATR_MULTIPLIER",
    4.0,
)
ADX_TREND_FOLLOWING_MAX_HOLDING_BARS = _env_int_scaled(
    "ADX_TREND_FOLLOWING_MAX_HOLDING_BARS",
    90,
)

# Uptrend Oversold Reversion: short-term dip-buying only inside long uptrends.
UPTREND_OVERSOLD_ZSCORE_MAX = _env_float("UPTREND_OVERSOLD_ZSCORE_MAX", -1.2)
UPTREND_OVERSOLD_RSI_MAX = _env_float("UPTREND_OVERSOLD_RSI_MAX", 40.0)
UPTREND_OVERSOLD_STOP_ATR_MULTIPLIER = _env_float(
    "UPTREND_OVERSOLD_STOP_ATR_MULTIPLIER",
    1.5,
)
UPTREND_OVERSOLD_TARGET_ATR_MULTIPLIER = _env_float(
    "UPTREND_OVERSOLD_TARGET_ATR_MULTIPLIER",
    2.5,
)
UPTREND_OVERSOLD_MAX_HOLDING_BARS = _env_int_scaled(
    "UPTREND_OVERSOLD_MAX_HOLDING_BARS",
    20,
)

# Volatility Squeeze Breakout: breakout continuation after volatility compression.
VOLATILITY_SQUEEZE_BB_WIDTH_RATIO_MAX = _env_float(
    "VOLATILITY_SQUEEZE_BB_WIDTH_RATIO_MAX",
    0.85,
)
VOLATILITY_SQUEEZE_VOLUME_RATIO_MIN = _env_float(
    "VOLATILITY_SQUEEZE_VOLUME_RATIO_MIN",
    0.95,
)
VOLATILITY_SQUEEZE_ADX_MIN = _env_float("VOLATILITY_SQUEEZE_ADX_MIN", 14.0)
VOLATILITY_SQUEEZE_STOP_ATR_MULTIPLIER = _env_float(
    "VOLATILITY_SQUEEZE_STOP_ATR_MULTIPLIER",
    1.8,
)
VOLATILITY_SQUEEZE_TARGET_ATR_MULTIPLIER = _env_float(
    "VOLATILITY_SQUEEZE_TARGET_ATR_MULTIPLIER",
    3.5,
)
VOLATILITY_SQUEEZE_MAX_HOLDING_BARS = _env_int_scaled(
    "VOLATILITY_SQUEEZE_MAX_HOLDING_BARS",
    45,
)


# Connors RSI(2) Pullback: short-horizon dip buying inside structural uptrends.
# The RSI lookback and SMA exit window are part of the strategy's identity on
# daily bars (period=2, SMA=5).  On faster timeframes they are scaled so the
# signal still covers the same calendar horizon (~2 days / ~5 days).
CONNORS_RSI2_RSI_LOOKBACK_BARS = _env_int_signal_scaled("CONNORS_RSI2_RSI_LOOKBACK_BARS", 2)
CONNORS_RSI2_EXIT_SMA_BARS = _env_int_signal_scaled("CONNORS_RSI2_EXIT_SMA_BARS", 5)
CONNORS_RSI2_ENTRY_MAX = _env_float("CONNORS_RSI2_ENTRY_MAX", 5.0)
CONNORS_RSI2_EXIT_MIN = _env_float("CONNORS_RSI2_EXIT_MIN", 70.0)
CONNORS_RSI2_STOP_ATR_MULTIPLIER = _env_float(
    "CONNORS_RSI2_STOP_ATR_MULTIPLIER",
    1.5,
)
CONNORS_RSI2_MAX_HOLDING_BARS = _env_int_scaled("CONNORS_RSI2_MAX_HOLDING_BARS", 8)
CONNORS_RSI2_REQUIRE_SMA200_FILTER = _env_text(
    "CONNORS_RSI2_REQUIRE_SMA200_FILTER",
    "1",
).lower() in {"1", "true", "yes", "y", "on"}


# Donchian Trend Reentry: trend breakouts with medium-term channel anchors.
DONCHIAN_BREAKOUT_LOOKBACK_DAYS = _env_int_signal_scaled("DONCHIAN_BREAKOUT_LOOKBACK_DAYS", 55)
DONCHIAN_STOP_LOOKBACK_DAYS = _env_int_signal_scaled("DONCHIAN_STOP_LOOKBACK_DAYS", 20)
DONCHIAN_ADX_MIN = _env_float("DONCHIAN_ADX_MIN", 15.0)
DONCHIAN_STOP_ATR_MULTIPLIER = _env_float("DONCHIAN_STOP_ATR_MULTIPLIER", 2.0)
DONCHIAN_TARGET_ATR_MULTIPLIER = _env_float("DONCHIAN_TARGET_ATR_MULTIPLIER", 4.0)
DONCHIAN_MAX_HOLDING_BARS = _env_int_scaled("DONCHIAN_MAX_HOLDING_BARS", 80)


# Turn-of-Month seasonality: short holding window around month transitions.
TURN_OF_MONTH_MAX_HOLDING_BARS = _env_int_scaled("TURN_OF_MONTH_MAX_HOLDING_BARS", 4)
TURN_OF_MONTH_STOP_ATR_MULTIPLIER = _env_float(
    "TURN_OF_MONTH_STOP_ATR_MULTIPLIER",
    2.0,
)
TURN_OF_MONTH_REQUIRE_SMA200_FILTER = _env_text(
    "TURN_OF_MONTH_REQUIRE_SMA200_FILTER",
    "1",
).lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Trade activity validation thresholds
# ---------------------------------------------------------------------------
# Total-trade and per-month counts are calendar-based so they don't need
# timeframe scaling.  The minimum average holding bars does need scaling.
MIN_ACCEPTABLE_TOTAL_TRADES = _env_int("MIN_ACCEPTABLE_TOTAL_TRADES", 20)
UNDERACTIVE_TRADES_PER_MONTH = _env_float("UNDERACTIVE_TRADES_PER_MONTH", 0.75)
OVERACTIVE_TRADES_PER_MONTH = _env_float("OVERACTIVE_TRADES_PER_MONTH", 35.0)
OVERACTIVE_MIN_AVG_HOLDING_BARS = _env_float(
    "OVERACTIVE_MIN_AVG_HOLDING_BARS",
    max(2.0, 2.0 * BARS_PER_TRADING_DAY),
)


# ---------------------------------------------------------------------------
# Shared feature column names
# ---------------------------------------------------------------------------
RELATIVE_STRENGTH_RETURN_COLUMN = f"trailing_return_{RELATIVE_STRENGTH_LOOKBACK_DAYS}"
# Legacy compatibility alias for the preserved compiled walk-forward module.
# The refactor renamed the relative-strength return column to make the intent
# clearer, but the preserved bytecode implementation still imports the older
# symbol name.
MOMENTUM_RETURN_COLUMN = RELATIVE_STRENGTH_RETURN_COLUMN
BREAKOUT_MOMENTUM_RETURN_COLUMN = f"trailing_return_{BREAKOUT_MOMENTUM_LOOKBACK_DAYS}"
BREAKOUT_HIGH_COLUMN = f"rolling_high_{BREAKOUT_LOOKBACK_DAYS}_prev"
BREAKOUT_LOW_COLUMN = f"rolling_low_{BREAKOUT_STOP_LOOKBACK_DAYS}_prev"
TREND_PULLBACK_STOP_LOW_COLUMN = f"rolling_low_{TREND_PULLBACK_STOP_LOOKBACK_DAYS}_prev"
TREND_PULLBACK_TARGET_HIGH_COLUMN = f"rolling_high_{TREND_PULLBACK_TARGET_LOOKBACK_DAYS}_prev"
MEAN_REVERSION_STOP_LOW_COLUMN = f"rolling_low_{MEAN_REVERSION_STOP_LOOKBACK_DAYS}_prev"
DONCHIAN_BREAKOUT_HIGH_COLUMN = f"rolling_high_{DONCHIAN_BREAKOUT_LOOKBACK_DAYS}_prev"
DONCHIAN_BREAKOUT_STOP_LOW_COLUMN = f"rolling_low_{DONCHIAN_STOP_LOOKBACK_DAYS}_prev"
SHORT_RETURN_COLUMN = f"trailing_return_{SHORT_RETURN_LOOKBACK_BARS}"
MEDIUM_RETURN_COLUMN = f"trailing_return_{MEDIUM_RETURN_LOOKBACK_BARS}"


def format_strategy_name(agent_name: str, *, short: bool = False) -> str:
    """Return either the full or short display label for one strategy."""
    mapping = AGENT_SHORT_DISPLAY_NAMES if short else AGENT_DISPLAY_NAMES
    return mapping.get(agent_name, agent_name.replace("_", " ").title())
