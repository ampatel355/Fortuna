"""Build the technical feature table used by the research pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from data_loader import main as download_raw_data
    from strategy_config import (
        ADX_LOOKBACK_DAYS,
        ATR_LOOKBACK_DAYS,
        BOLLINGER_LOOKBACK_DAYS,
        BOLLINGER_STD_MULTIPLIER,
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOOKBACK_DAYS,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_LOOKBACK_DAYS,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        BREAKOUT_STOP_LOOKBACK_DAYS,
        DONCHIAN_BREAKOUT_HIGH_COLUMN,
        DONCHIAN_BREAKOUT_LOOKBACK_DAYS,
        DONCHIAN_BREAKOUT_STOP_LOW_COLUMN,
        DONCHIAN_STOP_LOOKBACK_DAYS,
        MACD_FAST_DAYS,
        MACD_SIGNAL_DAYS,
        MACD_SLOW_DAYS,
        MA_100_BARS,
        MA_200_BARS,
        MA_20_BARS,
        MA_50_BARS,
        MEAN_REVERSION_ATR_FILTER_LOOKBACK_DAYS,
        MEAN_REVERSION_BB_FILTER_LOOKBACK_DAYS,
        MEAN_REVERSION_STOP_LOOKBACK_DAYS,
        MOMENTUM_LOOKBACK_DAYS,
        PRICE_STD_LOOKBACK_BARS,
        RELATIVE_STRENGTH_RETURN_COLUMN,
        RSI_LOOKBACK_DAYS,
        SHORT_RETURN_LOOKBACK_BARS,
        MEDIUM_RETURN_LOOKBACK_BARS,
        TREND_PULLBACK_STOP_LOOKBACK_DAYS,
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
        TREND_PULLBACK_TARGET_LOOKBACK_DAYS,
        VOLUME_LOOKBACK_DAYS,
    )
    from timeframe_config import (
        RESEARCH_INTERVAL,
        RESEARCH_TIMEFRAME_LABEL,
        infer_bars_per_year,
        interval_looks_compatible,
        normalize_timestamp_series,
        normalize_interval,
        scale_daily_bars,
        timeframe_output_suffix,
    )
except ModuleNotFoundError:
    from Code.data_loader import main as download_raw_data
    from Code.strategy_config import (
        ADX_LOOKBACK_DAYS,
        ATR_LOOKBACK_DAYS,
        BOLLINGER_LOOKBACK_DAYS,
        BOLLINGER_STD_MULTIPLIER,
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOOKBACK_DAYS,
        BREAKOUT_LOW_COLUMN,
        BREAKOUT_MOMENTUM_LOOKBACK_DAYS,
        BREAKOUT_MOMENTUM_RETURN_COLUMN,
        BREAKOUT_STOP_LOOKBACK_DAYS,
        DONCHIAN_BREAKOUT_HIGH_COLUMN,
        DONCHIAN_BREAKOUT_LOOKBACK_DAYS,
        DONCHIAN_BREAKOUT_STOP_LOW_COLUMN,
        DONCHIAN_STOP_LOOKBACK_DAYS,
        MACD_FAST_DAYS,
        MACD_SIGNAL_DAYS,
        MACD_SLOW_DAYS,
        MA_100_BARS,
        MA_200_BARS,
        MA_20_BARS,
        MA_50_BARS,
        MEAN_REVERSION_ATR_FILTER_LOOKBACK_DAYS,
        MEAN_REVERSION_BB_FILTER_LOOKBACK_DAYS,
        MEAN_REVERSION_STOP_LOOKBACK_DAYS,
        MOMENTUM_LOOKBACK_DAYS,
        PRICE_STD_LOOKBACK_BARS,
        RELATIVE_STRENGTH_RETURN_COLUMN,
        RSI_LOOKBACK_DAYS,
        SHORT_RETURN_LOOKBACK_BARS,
        MEDIUM_RETURN_LOOKBACK_BARS,
        TREND_PULLBACK_STOP_LOOKBACK_DAYS,
        TREND_PULLBACK_STOP_LOW_COLUMN,
        TREND_PULLBACK_TARGET_HIGH_COLUMN,
        TREND_PULLBACK_TARGET_LOOKBACK_DAYS,
        VOLUME_LOOKBACK_DAYS,
    )
    from Code.timeframe_config import (
        RESEARCH_INTERVAL,
        RESEARCH_TIMEFRAME_LABEL,
        infer_bars_per_year,
        interval_looks_compatible,
        normalize_timestamp_series,
        normalize_interval,
        scale_daily_bars,
        timeframe_output_suffix,
    )


ticker = os.environ.get("TICKER", "SPY")


def resolve_data_raw_root(project_root: Path) -> Path:
    """Return the project's raw-data root, preferring the uppercase path."""
    lowercase_dir = project_root / "data_raw"
    uppercase_dir = project_root / "Data_Raw"

    if uppercase_dir.exists():
        return uppercase_dir
    if lowercase_dir.exists():
        return lowercase_dir

    uppercase_dir.mkdir(parents=True, exist_ok=True)
    return uppercase_dir


def resolve_data_raw_dir(project_root: Path, interval: str | None = None) -> Path:
    """Return the interval-specific raw-data folder."""
    canonical_interval = normalize_interval(interval or RESEARCH_INTERVAL)
    return resolve_data_raw_root(project_root) / canonical_interval


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


def load_raw_prices(input_path: Path, current_ticker: str | None = None) -> pd.DataFrame:
    """Load raw prices, downloading them first when the CSV does not exist yet."""
    current_ticker = current_ticker or ticker
    if not input_path.exists():
        download_raw_data(current_ticker)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find or create the raw price file: {input_path}"
        )

    df = pd.read_csv(input_path)
    df["Date"] = normalize_timestamp_series(df["Date"])
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    interval_is_compatible = (
        df.get("data_interval", pd.Series(dtype="object"))
        .astype(str)
        .str.lower()
        .eq(RESEARCH_INTERVAL)
        .any()
        if "data_interval" in df.columns
        else interval_looks_compatible(df["Date"], RESEARCH_INTERVAL)
    )
    if not interval_is_compatible:
        download_raw_data(current_ticker)
        df = pd.read_csv(input_path)
        df["Date"] = normalize_timestamp_series(df["Date"])
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return (
        df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date", ascending=True)
        .reset_index(drop=True)
    )


def calculate_rsi(close_series: pd.Series, lookback_days: int) -> pd.Series:
    """Calculate Wilder-style RSI from closing prices."""
    price_change = close_series.diff()
    gains = price_change.clip(lower=0.0)
    losses = -price_change.clip(upper=0.0)

    average_gain = gains.ewm(alpha=1 / lookback_days, adjust=False, min_periods=lookback_days).mean()
    average_loss = losses.ewm(alpha=1 / lookback_days, adjust=False, min_periods=lookback_days).mean()

    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))

    # When losses are zero, RSI should read 100 rather than NaN.
    rsi = rsi.where(~((average_loss == 0.0) & average_gain.notna()), 100.0)
    rsi = rsi.where(~((average_gain == 0.0) & (average_loss == 0.0)), 50.0)
    return rsi


def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate True Range using the standard high/low and prior close definition."""
    previous_close = df["Close"].shift(1)
    high_low = df["High"] - df["Low"]
    high_previous_close = (df["High"] - previous_close).abs()
    low_previous_close = (df["Low"] - previous_close).abs()
    return pd.concat([high_low, high_previous_close, low_previous_close], axis=1).max(axis=1)


def calculate_atr(df: pd.DataFrame, lookback_days: int) -> pd.Series:
    """Calculate Wilder-style ATR from True Range."""
    true_range = calculate_true_range(df)
    return true_range.ewm(alpha=1 / lookback_days, adjust=False, min_periods=lookback_days).mean()


def calculate_macd(close_series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, its signal line, and the MACD histogram."""
    ema_fast = close_series.ewm(span=MACD_FAST_DAYS, adjust=False, min_periods=MACD_FAST_DAYS).mean()
    ema_slow = close_series.ewm(span=MACD_SLOW_DAYS, adjust=False, min_periods=MACD_SLOW_DAYS).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(
        span=MACD_SIGNAL_DAYS,
        adjust=False,
        min_periods=MACD_SIGNAL_DAYS,
    ).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calculate_adx(df: pd.DataFrame, lookback_days: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate the standard Wilder ADX, +DI, and -DI indicator set."""
    high_diff = df["High"].diff()
    low_diff = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (high_diff > low_diff) & (high_diff > 0),
            high_diff,
            0.0,
        ),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where(
            (low_diff > high_diff) & (low_diff > 0),
            low_diff,
            0.0,
        ),
        index=df.index,
        dtype=float,
    )

    atr = calculate_atr(df, lookback_days)
    plus_dm_smoothed = plus_dm.ewm(
        alpha=1 / lookback_days,
        adjust=False,
        min_periods=lookback_days,
    ).mean()
    minus_dm_smoothed = minus_dm.ewm(
        alpha=1 / lookback_days,
        adjust=False,
        min_periods=lookback_days,
    ).mean()

    plus_di = 100.0 * (plus_dm_smoothed / atr.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm_smoothed / atr.replace(0.0, np.nan))
    dx = 100.0 * (
        (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0.0, np.nan)
    )
    adx = dx.ewm(
        alpha=1 / lookback_days,
        adjust=False,
        min_periods=lookback_days,
    ).mean()
    return adx, plus_di, minus_di


def build_feature_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Add the full feature set used by the active strategy stack."""
    df = raw_df.copy()
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)
    bars_per_year = infer_bars_per_year(df["Date"])
    df["timeframe_label"] = RESEARCH_TIMEFRAME_LABEL
    df["data_interval"] = RESEARCH_INTERVAL

    # One-bar percentage return from close-to-close movement. We keep the
    # legacy daily_return alias so the rest of the pipeline can migrate without
    # breaking every downstream file at once.
    df["bar_return"] = df["Close"].pct_change()
    df["daily_return"] = df["bar_return"]

    # Use the standard chartist moving-average set:
    # - 20 EMA for faster pullback guidance
    # - 20 SMA for Bollinger Bands
    # - 50 / 100 / 200 SMA for broader trend structure
    df["ema_20"] = df["Close"].ewm(span=MA_20_BARS, adjust=False, min_periods=MA_20_BARS).mean()
    df["sma_20"] = df["Close"].rolling(window=MA_20_BARS).mean()
    df["sma_50"] = df["Close"].rolling(window=MA_50_BARS).mean()
    df["sma_100"] = df["Close"].rolling(window=MA_100_BARS).mean()
    df["sma_200"] = df["Close"].rolling(window=MA_200_BARS).mean()

    # Legacy aliases are preserved so existing downstream files and plots keep
    # working, but they now map to the standard indicator types above.
    df["ma_20"] = df["ema_20"]
    df["ma_50"] = df["sma_50"]
    df["ma_100"] = df["sma_100"]
    df["ma_200"] = df["sma_200"]

    # Rolling volatility from close-to-close bar returns.
    df["std_20"] = df["bar_return"].rolling(window=PRICE_STD_LOOKBACK_BARS).std(ddof=0)
    df["price_std_20"] = df["Close"].rolling(window=PRICE_STD_LOOKBACK_BARS).std(ddof=0)

    # Average volume helps the breakout strategy separate broad participation
    # from weak price pops on thin trading activity.
    df["avg_volume_20"] = df["Volume"].rolling(window=VOLUME_LOOKBACK_DAYS).mean()
    df["volume_ratio_20"] = np.where(
        df["avg_volume_20"] > 0,
        df["Volume"] / df["avg_volume_20"],
        np.where(df["Volume"] == 0, 1.0, np.nan),
    )

    # RSI captures the balance between recent up-closes and down-closes.
    df["rsi_14"] = calculate_rsi(df["Close"], RSI_LOOKBACK_DAYS)

    # ATR is a bar-based proxy for realized trading range and short-term noise.
    df["true_range"] = calculate_true_range(df)
    df["atr_14"] = calculate_atr(df, ATR_LOOKBACK_DAYS)
    df["atr_percent"] = np.where(df["Close"] > 0, df["atr_14"] / df["Close"], np.nan)

    # ADX and directional indicators are standard tools for separating trend
    # environments from ranging environments.
    df["adx_14"], df["plus_di_14"], df["minus_di_14"] = calculate_adx(df, ADX_LOOKBACK_DAYS)

    # Bollinger Bands define the mean-reversion extremes and the band width acts
    # as a volatility expansion / contraction filter.
    df["bollinger_mid"] = df["sma_20"]
    df["bollinger_std"] = df["Close"].rolling(window=BOLLINGER_LOOKBACK_DAYS).std(ddof=0)
    df["bollinger_upper"] = df["bollinger_mid"] + (BOLLINGER_STD_MULTIPLIER * df["bollinger_std"])
    df["bollinger_lower"] = df["bollinger_mid"] - (BOLLINGER_STD_MULTIPLIER * df["bollinger_std"])
    df["bollinger_width"] = np.where(
        df["bollinger_mid"] != 0,
        (df["bollinger_upper"] - df["bollinger_lower"]) / df["bollinger_mid"],
        np.nan,
    )

    # MACD gives the breakout strategy a trend-acceleration confirmation.
    df["macd_line"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["Close"])

    # Rolling highs and lows are always shifted by one bar for signal thresholds.
    # That keeps each signal grounded in information known before the next trade.
    df[f"rolling_high_{BREAKOUT_LOOKBACK_DAYS}_prev"] = (
        df["High"].rolling(window=BREAKOUT_LOOKBACK_DAYS).max().shift(1)
    )
    df[f"rolling_low_{BREAKOUT_STOP_LOOKBACK_DAYS}_prev"] = (
        df["Low"].rolling(window=BREAKOUT_STOP_LOOKBACK_DAYS).min().shift(1)
    )
    df[f"rolling_low_{TREND_PULLBACK_STOP_LOOKBACK_DAYS}_prev"] = (
        df["Low"].rolling(window=TREND_PULLBACK_STOP_LOOKBACK_DAYS).min().shift(1)
    )
    df[f"rolling_high_{TREND_PULLBACK_TARGET_LOOKBACK_DAYS}_prev"] = (
        df["High"].rolling(window=TREND_PULLBACK_TARGET_LOOKBACK_DAYS).max().shift(1)
    )
    df[f"rolling_low_{MEAN_REVERSION_STOP_LOOKBACK_DAYS}_prev"] = (
        df["Low"].rolling(window=MEAN_REVERSION_STOP_LOOKBACK_DAYS).min().shift(1)
    )

    # Donchian channel columns for the Donchian Trend Reentry strategy.
    # Only create if the lookback differs from an already-created column.
    donchian_high_col = f"rolling_high_{DONCHIAN_BREAKOUT_LOOKBACK_DAYS}_prev"
    if donchian_high_col not in df.columns:
        df[donchian_high_col] = (
            df["High"].rolling(window=DONCHIAN_BREAKOUT_LOOKBACK_DAYS).max().shift(1)
        )
    donchian_low_col = f"rolling_low_{DONCHIAN_STOP_LOOKBACK_DAYS}_prev"
    if donchian_low_col not in df.columns:
        df[donchian_low_col] = (
            df["Low"].rolling(window=DONCHIAN_STOP_LOOKBACK_DAYS).min().shift(1)
        )

    # Trailing return windows support both breakout confirmation and cross-asset
    # relative-strength ranking.
    df[BREAKOUT_MOMENTUM_RETURN_COLUMN] = (
        df["Close"] / df["Close"].shift(BREAKOUT_MOMENTUM_LOOKBACK_DAYS)
    ) - 1.0
    df[RELATIVE_STRENGTH_RETURN_COLUMN] = (
        df["Close"] / df["Close"].shift(MOMENTUM_LOOKBACK_DAYS)
    ) - 1.0
    df[f"trailing_return_{SHORT_RETURN_LOOKBACK_BARS}"] = (
        df["Close"] / df["Close"].shift(SHORT_RETURN_LOOKBACK_BARS)
    ) - 1.0
    # Legacy aliases so downstream code can use either the dynamic or fixed name.
    df["trailing_return_5"] = df[f"trailing_return_{SHORT_RETURN_LOOKBACK_BARS}"]
    df[f"trailing_return_{MEDIUM_RETURN_LOOKBACK_BARS}"] = (
        df["Close"] / df["Close"].shift(MEDIUM_RETURN_LOOKBACK_BARS)
    ) - 1.0
    df["trailing_return_20"] = df[f"trailing_return_{MEDIUM_RETURN_LOOKBACK_BARS}"]
    df["trailing_return_60"] = df[RELATIVE_STRENGTH_RETURN_COLUMN]
    annual_momentum_lookback_bars = scale_daily_bars(252)
    df[f"trailing_return_{annual_momentum_lookback_bars}"] = (
        df["Close"] / df["Close"].shift(annual_momentum_lookback_bars)
    ) - 1.0
    df["trailing_return_252"] = df[f"trailing_return_{annual_momentum_lookback_bars}"]

    # Legacy aliases are kept so the rest of the project can continue to load
    # shared names without caring which strategy uses them.
    df[BREAKOUT_HIGH_COLUMN] = df[f"rolling_high_{BREAKOUT_LOOKBACK_DAYS}_prev"]
    df[BREAKOUT_LOW_COLUMN] = df[f"rolling_low_{BREAKOUT_STOP_LOOKBACK_DAYS}_prev"]
    df[TREND_PULLBACK_STOP_LOW_COLUMN] = df[f"rolling_low_{TREND_PULLBACK_STOP_LOOKBACK_DAYS}_prev"]
    df[TREND_PULLBACK_TARGET_HIGH_COLUMN] = df[f"rolling_high_{TREND_PULLBACK_TARGET_LOOKBACK_DAYS}_prev"]
    df["rolling_high_20_prev"] = df[BREAKOUT_HIGH_COLUMN]
    df["rolling_low_10_prev"] = df[BREAKOUT_LOW_COLUMN]
    df["rolling_high_55_prev"] = df[DONCHIAN_BREAKOUT_HIGH_COLUMN]
    df["rolling_low_20_prev"] = df[donchian_low_col]

    # Relative filters compare today's volatility with each asset's own recent
    # history, which is more robust than using one absolute threshold across
    # equities, futures, crypto, and forex.
    df["atr_percent_median_60"] = df["atr_percent"].rolling(window=MEAN_REVERSION_ATR_FILTER_LOOKBACK_DAYS).median()
    df["atr_percent_ratio_60"] = np.where(
        df["atr_percent_median_60"] > 0,
        df["atr_percent"] / df["atr_percent_median_60"],
        np.nan,
    )
    df["bollinger_width_median_60"] = df["bollinger_width"].rolling(
        window=MEAN_REVERSION_BB_FILTER_LOOKBACK_DAYS
    ).median()
    df["bollinger_width_ratio_60"] = np.where(
        df["bollinger_width_median_60"] > 0,
        df["bollinger_width"] / df["bollinger_width_median_60"],
        np.nan,
    )

    # Trend distance features make it easier to express "calm, non-trending"
    # environments for the mean-reversion strategy.
    df["distance_from_ma_20"] = np.where(df["ema_20"] > 0, (df["Close"] / df["ema_20"]) - 1.0, np.nan)
    df["distance_from_ma_50"] = np.where(df["ma_50"] > 0, (df["Close"] / df["ma_50"]) - 1.0, np.nan)
    df["ma_50_over_ma_200"] = np.where(df["ma_200"] > 0, (df["ma_50"] / df["ma_200"]) - 1.0, np.nan)

    # zscore_20 is preserved because several parts of the old project already
    # know how to display or analyze it.
    df["zscore_20"] = np.where(
        df["price_std_20"] > 0,
        (df["Close"] - df["sma_20"]) / df["price_std_20"],
        np.nan,
    )

    df["bars_per_year"] = bars_per_year

    # The first rows cannot support the longest lookbacks, so we drop them only
    # after every feature has been created.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


def build_features_for_ticker(
    current_ticker: str,
    *,
    project_root: Path | None = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """Load one ticker, build the feature table, and optionally save it."""
    project_root = project_root or Path(__file__).resolve().parents[1]
    raw_dir = resolve_data_raw_dir(project_root, RESEARCH_INTERVAL)
    clean_dir = resolve_data_clean_dir(project_root)
    input_path = raw_dir / f"{current_ticker}.csv"
    output_path = clean_dir / f"{current_ticker}_features.csv"

    raw_df = load_raw_prices(input_path, current_ticker=current_ticker)
    feature_df = build_feature_dataframe(raw_df)

    if save_output:
        clean_dir.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(output_path, index=False)

    return feature_df


def main() -> None:
    """Create and save the feature table for the active ticker."""
    feature_df = build_features_for_ticker(ticker)
    print(feature_df.head(10))


if __name__ == "__main__":
    main()
