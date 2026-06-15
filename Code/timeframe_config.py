"""Shared timeframe helpers for the research pipeline.

These helpers centralize timeframe assumptions so the rest of the code can
stay readable:
- data downloads use the same interval and Yahoo-valid history window
- annualization uses observed bar density instead of one hard-coded constant
- parameter scaling converts daily-calibrated values to bar-equivalent counts
- charts and UI text can display the active timeframe consistently

**Timeframe aliases**: the pipeline accepts ``"daily"``, ``"1d"``, ``"hourly"``,
``"1h"``, ``"4h"``, ``"4hour"`` etc. — all normalized to a canonical interval
(``"1d"``, ``"1h"``, ``"4h"``).

**4-hour data**: Yahoo Finance does not support a native ``"4h"`` interval.
The data loader must download ``"1h"`` bars and resample to 4-hour OHLCV.
The helpers ``yahoo_download_interval()`` and ``yahoo_download_period()``
provide the correct download parameters for any configured interval.
"""

from __future__ import annotations

import os

import pandas as pd


def _env_text(name: str, default: str) -> str:
    """Read a text environment variable with a clean fallback."""
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a clean fallback."""
    return float(os.environ.get(name, str(default)))


# ---------------------------------------------------------------------------
# Alias normalization — accept friendly names and resolve to canonical forms
# ---------------------------------------------------------------------------
_INTERVAL_ALIASES: dict[str, str] = {
    "daily": "1d",
    "1d": "1d",
    "d": "1d",
    "day": "1d",
    "hourly": "1h",
    "1h": "1h",
    "1hour": "1h",
    "60m": "1h",
    "60min": "1h",
    "4h": "4h",
    "4hour": "4h",
    "4hr": "4h",
    "4-hour": "4h",
    "4-hr": "4h",
}


def normalize_interval(raw: str) -> str:
    """Resolve a user-facing interval alias to its canonical form.

    Accepts ``"daily"``, ``"1d"``, ``"hourly"``, ``"1h"``, ``"4h"``, ``"4hour"``,
    etc. and returns one of ``"1d"``, ``"1h"``, ``"4h"``, or the original string
    lowered if it is not in the alias table.
    """
    return _INTERVAL_ALIASES.get(raw.strip().lower(), raw.strip().lower())


_raw_interval = _env_text("RESEARCH_INTERVAL", "1d")
RESEARCH_INTERVAL: str = normalize_interval(_raw_interval)
RESEARCH_TIMEFRAME_LABEL = _env_text("RESEARCH_TIMEFRAME_LABEL", "Daily")

# ---------------------------------------------------------------------------
# Yahoo Finance download parameters
# ---------------------------------------------------------------------------
# Yahoo Finance limits most intraday intervals, while daily/weekly/monthly bars
# can usually request full history.
#
# IMPORTANT: Yahoo does NOT support "4h" directly. For 4h the pipeline must
# download "1h" data and resample. The mapping below is only for intervals
# Yahoo actually accepts.
YAHOO_MAX_PERIOD_BY_INTERVAL = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "90m": "60d",
    "1h": "730d",
    "1d": "max",
    "5d": "max",
    "1wk": "max",
    "1mo": "max",
    "3mo": "max",
}

# Map a research interval to the interval Yahoo Finance should actually
# download.  Most intervals are downloaded as-is; "4h" must be sourced from
# "1h" and resampled downstream.
_YAHOO_DOWNLOAD_INTERVAL_MAP: dict[str, str] = {
    "4h": "1h",
}


def yahoo_download_interval(interval: str | None = None) -> str:
    """Return the yfinance-compatible interval to actually download.

    For most intervals this is the interval itself.  For ``"4h"`` it returns
    ``"1h"`` because Yahoo does not offer a 4-hour bar natively.
    """
    canonical = normalize_interval(interval) if interval else RESEARCH_INTERVAL
    return _YAHOO_DOWNLOAD_INTERVAL_MAP.get(canonical, canonical)


def yahoo_download_period(interval: str | None = None) -> str:
    """Return the maximum Yahoo Finance period for the download interval."""
    dl_interval = yahoo_download_interval(interval)
    return YAHOO_MAX_PERIOD_BY_INTERVAL.get(dl_interval, "730d")


def requires_resampling(interval: str | None = None) -> bool:
    """Return whether the configured interval requires post-download resampling."""
    canonical = normalize_interval(interval) if interval else RESEARCH_INTERVAL
    return canonical in _YAHOO_DOWNLOAD_INTERVAL_MAP


DEFAULT_YAHOO_PERIOD = yahoo_download_period()

# ---------------------------------------------------------------------------
# Timeframe scaling
# ---------------------------------------------------------------------------
# How many bars fit in one equity trading day (~6.5 hours).  These are
# conservative equity-market estimates.  For 24-hour markets (crypto, forex)
# the realized density will be higher, but parameter scaling uses these as
# safe defaults.  Override with the BARS_PER_TRADING_DAY environment variable
# if you want tighter control for a specific asset class.
_ESTIMATED_BARS_PER_TRADING_DAY = {
    "1m": 390.0,
    "2m": 195.0,
    "5m": 78.0,
    "15m": 26.0,
    "30m": 13.0,
    "60m": 7.0,
    "1h": 7.0,
    "90m": 5.0,
    "4h": 2.0,
    "1d": 1.0,
    "5d": 0.2,
    "1wk": 0.2,
    "1mo": 1.0 / 21.0,
    "3mo": 1.0 / 63.0,
}

BARS_PER_TRADING_DAY: float = _env_float(
    "BARS_PER_TRADING_DAY",
    _ESTIMATED_BARS_PER_TRADING_DAY.get(RESEARCH_INTERVAL, 1.0),
)

# The human-readable label auto-detects when the user hasn't set one
# explicitly.
_AUTO_TIMEFRAME_LABELS = {
    "1m": "1-Min",
    "2m": "2-Min",
    "5m": "5-Min",
    "15m": "15-Min",
    "30m": "30-Min",
    "60m": "Hourly",
    "1h": "Hourly",
    "90m": "90-Min",
    "4h": "4-Hour",
    "1d": "Daily",
    "5d": "Weekly(5d)",
    "1wk": "Weekly",
    "1mo": "Monthly",
    "3mo": "Quarterly",
}

if not os.environ.get("RESEARCH_TIMEFRAME_LABEL"):
    RESEARCH_TIMEFRAME_LABEL = _AUTO_TIMEFRAME_LABELS.get(
        RESEARCH_INTERVAL, RESEARCH_TIMEFRAME_LABEL
    )


def scale_daily_bars(daily_count: int) -> int:
    """Convert a daily-calibrated bar count to the current timeframe.

    The idea: if the daily pipeline uses a 20-bar SMA (≈20 trading days),
    the equivalent on hourly data is 20 * 7 = 140 bars — still covering
    ~20 trading days of price history.

    Use this for **structural** parameters: long trend filters (MA_100,
    MA_200), holding periods, time stops, and regime history requirements
    where the intent is to cover the same calendar duration.

    The function respects a minimum of 1 bar so indicators never degenerate.
    """
    return max(1, round(daily_count * BARS_PER_TRADING_DAY))


# ---------------------------------------------------------------------------
# Signal-indicator scaling
# ---------------------------------------------------------------------------
# For signal-generating indicators (RSI, ADX, ATR, Bollinger, MACD) a full
# BARS_PER_TRADING_DAY multiplier makes the indicator too smooth on intraday
# data and kills trade generation.  Using sqrt(BARS_PER_TRADING_DAY) is a
# standard quant-finance compromise: it moderately extends the lookback so
# the indicator captures multi-bar patterns while staying responsive enough
# to produce actionable signals.
#
# Justification:
#   - sqrt preserves the rough *information content* per window (since
#     variance scales with N, standard deviation with sqrt(N))
#   - On 1h (7 bars/day): sqrt(7)≈2.65 → 14-day RSI becomes ~37 bars
#     (~5.3 trading days) — still much longer than the daily 14-bar window
#     but responsive enough to generate entries
#   - On 4h (2 bars/day): sqrt(2)≈1.41 → 14-day RSI becomes ~20 bars
#     (~10 trading days) — close to daily but slightly adapted
#   - On 1d (1 bar/day): sqrt(1)=1.0 → no change from daily
import math as _math

SIGNAL_SCALING_FACTOR: float = _env_float(
    "SIGNAL_SCALING_FACTOR",
    _math.sqrt(BARS_PER_TRADING_DAY) if BARS_PER_TRADING_DAY > 1 else 1.0,
)


def scale_daily_signal_bars(daily_count: int) -> int:
    """Scale signal-indicator windows for intraday data using sqrt scaling.

    Signal-generating indicators (RSI, ADX, ATR, Bollinger, MACD) need to
    remain responsive on faster timeframes.  Instead of the full calendar-
    equivalent multiplier, this function uses ``sqrt(BARS_PER_TRADING_DAY)``
    so the indicators stay actionable while still being longer than their
    daily bar-count equivalents.

    Use ``scale_daily_bars()`` for structural parameters (long MAs, holding
    periods, time stops) that need full calendar-duration coverage.
    """
    return max(1, round(daily_count * SIGNAL_SCALING_FACTOR))


def scale_daily_float(daily_value: float) -> float:
    """Scale a float-valued daily parameter to the current timeframe."""
    return daily_value * BARS_PER_TRADING_DAY


def inverse_scale_daily_float(daily_value: float) -> float:
    """Inversely scale a daily probability to the current timeframe.

    If daily entry probability is 0.07, on hourly (7 bars/day) the
    per-bar probability should be ~0.07 / 7 = 0.01 so the *daily*
    expected trade count stays similar.
    """
    if BARS_PER_TRADING_DAY <= 0:
        return daily_value
    return daily_value / BARS_PER_TRADING_DAY


def timeframe_output_suffix() -> str:
    """Return a suffix like '_1h' for non-daily timeframes, or '' for daily.

    This keeps daily output paths backward-compatible while separating
    intraday results into their own directories.
    """
    if RESEARCH_INTERVAL == "1d":
        return ""
    return f"_{RESEARCH_INTERVAL}"

# These are only fallbacks. The preferred path is to infer realized bar density
# from timestamps because equities, crypto, forex, and futures trade on
# different calendars.
DEFAULT_BARS_PER_YEAR_BY_INTERVAL = {
    "1h": 24.0 * 365.25,
    "60m": 24.0 * 365.25,
    "4h": 6.0 * 365.25,
    "1d": 252.0,
}
DEFAULT_BARS_PER_YEAR = float(
    DEFAULT_BARS_PER_YEAR_BY_INTERVAL.get(RESEARCH_INTERVAL, 252.0)
)


def normalize_timestamp_series(date_series: pd.Series) -> pd.Series:
    """Normalize timestamps into timezone-naive UTC-like datetimes."""
    parsed = pd.to_datetime(date_series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def interval_is_intraday(interval: str | None = None) -> bool:
    """Return whether one interval should behave like intraday data."""
    normalized_interval = (interval or RESEARCH_INTERVAL).strip().lower()
    return normalized_interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "4h"}


def infer_intraday_like_frequency(date_series: pd.Series) -> bool:
    """Return whether a timestamp series behaves like intraday data."""
    clean_dates = normalize_timestamp_series(pd.Series(date_series)).dropna().drop_duplicates()
    if len(clean_dates) <= 1:
        return False

    if clean_dates.dt.hour.nunique() > 1:
        return True
    if clean_dates.dt.minute.nunique() > 1:
        return True

    median_step = clean_dates.sort_values().diff().dropna().median()
    if pd.isna(median_step):
        return False

    return median_step < pd.Timedelta(days=1)


def interval_looks_compatible(
    date_series: pd.Series,
    interval: str | None = None,
) -> bool:
    """Return whether observed timestamps look compatible with one interval."""
    inferred_intraday = infer_intraday_like_frequency(date_series)
    expected_intraday = interval_is_intraday(interval)
    return inferred_intraday if expected_intraday else (not inferred_intraday)


def infer_bars_per_year(date_series: pd.Series) -> float:
    """Estimate the annualized bar count from observed timestamps.

    We use the realized timestamp density rather than a one-size-fits-all
    constant. That keeps annualized Sharpe and realized-vol scaling sensible for
    equities, crypto, forex, and futures on the same shared framework.
    """
    clean_dates = normalize_timestamp_series(pd.Series(date_series)).dropna().drop_duplicates()
    if len(clean_dates) <= 1:
        return DEFAULT_BARS_PER_YEAR

    ordered_dates = clean_dates.sort_values().reset_index(drop=True)
    total_span_days = (
        ordered_dates.iloc[-1] - ordered_dates.iloc[0]
    ).total_seconds() / 86400.0
    if total_span_days <= 0:
        return DEFAULT_BARS_PER_YEAR

    bars_per_day = (len(ordered_dates) - 1) / total_span_days
    estimated_bars_per_year = bars_per_day * 365.25
    if estimated_bars_per_year <= 0:
        return DEFAULT_BARS_PER_YEAR

    return float(estimated_bars_per_year)


def infer_months_covered(date_series: pd.Series) -> float:
    """Estimate how many calendar months a timestamp series covers."""
    clean_dates = normalize_timestamp_series(pd.Series(date_series)).dropna()
    if clean_dates.empty:
        return 0.0

    span_days = (
        clean_dates.max() - clean_dates.min()
    ).total_seconds() / 86400.0
    if span_days <= 0:
        return 0.0

    return float(span_days / 30.4375)


def timeframe_title_suffix() -> str:
    """Return a compact timeframe label for chart titles and UI text."""
    return f"({RESEARCH_TIMEFRAME_LABEL})"
