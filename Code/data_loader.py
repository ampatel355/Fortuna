"""Download and clean historical market data for one ticker.

Timeframe handling
------------------
- **Daily (1d)**: downloaded directly from Yahoo Finance with ``period="max"``.
- **Hourly (1h)**: downloaded directly with ``interval="1h"`` (Yahoo limits
  intraday history to ~730 calendar days).
- **4-hour (4h)**: Yahoo Finance does **not** support a native 4h interval.
  The loader downloads 1h bars and resamples them into 4-hour OHLCV candles:
      Open  = first,  High = max,  Low = min,  Close = last,  Volume = sum.
  Incomplete trailing bars (fewer than 4 source bars) are dropped.
"""

import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

try:
    from timeframe_config import (
        RESEARCH_INTERVAL,
        RESEARCH_TIMEFRAME_LABEL,
        interval_looks_compatible,
        normalize_interval,
        normalize_timestamp_series,
        requires_resampling,
        yahoo_download_interval,
        yahoo_download_period,
    )
except ModuleNotFoundError:
    from Code.timeframe_config import (
        RESEARCH_INTERVAL,
        RESEARCH_TIMEFRAME_LABEL,
        interval_looks_compatible,
        normalize_interval,
        normalize_timestamp_series,
        requires_resampling,
        yahoo_download_interval,
        yahoo_download_period,
    )

# Read the active ticker from the environment, or fall back to SPY.
configured_ticker = os.environ.get("TICKER", "SPY")
ALLOW_STALE_RAW_FALLBACK = os.environ.get("ALLOW_STALE_RAW_FALLBACK", "0") == "1"
RAW_DATA_MAX_STALENESS_HOURS = float(os.environ.get("RAW_DATA_MAX_STALENESS_HOURS", "48"))

# ---------------------------------------------------------------------------
# 4h resampling constants
# ---------------------------------------------------------------------------
# The number of source bars (1h) that make up one 4h candle.
_4H_SOURCE_BARS = 4
# OHLCV resampling rules used when building 4h bars from 1h data.
_OHLCV_RESAMPLE_RULES: dict[str, str] = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}


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
    """Return the raw-data subfolder for one research interval."""
    canonical_interval = normalize_interval(interval or RESEARCH_INTERVAL)
    return resolve_data_raw_root(project_root) / canonical_interval


def _fallback_file_is_fresh_enough(existing_df: pd.DataFrame) -> bool:
    """Return whether a saved raw file is fresh enough to reuse on download failure."""
    if "downloaded_at_utc" not in existing_df.columns:
        return False

    downloaded_at = pd.to_datetime(existing_df["downloaded_at_utc"], errors="coerce", utc=True)
    downloaded_at = downloaded_at.dropna()
    if downloaded_at.empty:
        return False

    file_age_hours = (
        datetime.now(timezone.utc) - downloaded_at.iloc[-1].to_pydatetime()
    ).total_seconds() / 3600.0
    return file_age_hours <= RAW_DATA_MAX_STALENESS_HOURS


def _resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-hour OHLCV bars into 4-hour candles.

    Rather than snapping to midnight-based calendar bins, we resample within
    each continuous trading session and aggregate consecutive 1h bars in blocks
    of four. That keeps equities aligned to their actual session structure
    while still handling 24-hour markets correctly.
    """
    if df_1h.empty:
        return df_1h.copy()

    work = df_1h.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if work.empty:
        return work

    time_gap = work["Date"].diff()
    session_start = time_gap.isna() | (time_gap > pd.Timedelta(minutes=90))
    work["_session_id"] = session_start.cumsum() - 1
    work["_chunk_id"] = work.groupby("_session_id").cumcount() // _4H_SOURCE_BARS

    resampled = (
        work.groupby(["_session_id", "_chunk_id"], sort=True)
        .agg(
            Date=("Date", "last"),
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
            _source_bar_count=("Close", "count"),
        )
        .reset_index(drop=True)
    )
    resampled = resampled.loc[resampled["_source_bar_count"] >= 2].copy()
    resampled = resampled.drop(columns=["_source_bar_count"], errors="ignore")
    return resampled.dropna(subset=["Date", "Open", "High", "Low", "Close"]).reset_index(drop=True)


def _validate_data_quality(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Run a basic data-quality audit and fix recoverable issues.

    Checks performed:
    - Duplicate timestamps → keep last
    - Chronological sort
    - Missing OHLC → drop row
    - Negative or zero prices → drop row
    - Volume < 0 → set to 0
    """
    df = df.copy()

    # Duplicates
    n_dupes = df.duplicated(subset=["Date"], keep="last").sum()
    if n_dupes > 0:
        print(f"  [{ticker}] Removed {n_dupes} duplicate timestamp(s).")
        df = df.drop_duplicates(subset=["Date"], keep="last")

    # Sort chronologically
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)

    # Drop rows with missing OHLC
    ohlc_cols = ["Open", "High", "Low", "Close"]
    n_before = len(df)
    df = df.dropna(subset=ohlc_cols)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  [{ticker}] Dropped {n_dropped} row(s) with missing OHLC.")

    # Drop rows with non-positive prices
    price_mask = (df[ohlc_cols] > 0).all(axis=1)
    n_bad_price = (~price_mask).sum()
    if n_bad_price > 0:
        print(f"  [{ticker}] Dropped {n_bad_price} row(s) with non-positive prices.")
        df = df.loc[price_mask]

    # Clamp negative volume to zero
    if "Volume" in df.columns:
        neg_vol = (df["Volume"] < 0).sum()
        if neg_vol > 0:
            print(f"  [{ticker}] Clamped {neg_vol} negative volume value(s) to 0.")
            df["Volume"] = df["Volume"].clip(lower=0)

    return df.reset_index(drop=True)


def main(ticker: str | None = None) -> None:
    # Use the shared project ticker unless a specific ticker was passed in.
    ticker = ticker or configured_ticker

    # Find the project root so the script works from any current working directory.
    project_root = Path(__file__).resolve().parents[1]

    # Save raw inputs under an interval-specific folder so daily, hourly, and 4h
    # runs never overwrite each other.
    output_dir = resolve_data_raw_dir(project_root, RESEARCH_INTERVAL)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}.csv"
    legacy_output_path = resolve_data_raw_root(project_root) / f"{ticker}.csv"

    # Determine the actual Yahoo download interval and period.
    # For "4h", we download "1h" data and resample later.
    dl_interval = yahoo_download_interval()
    dl_period = yahoo_download_period()
    needs_resample = requires_resampling()

    if needs_resample:
        print(
            f"  [{ticker}] Downloading {dl_interval} data (will resample to "
            f"{RESEARCH_INTERVAL})."
        )

    # Download the longest Yahoo-valid history for the chosen interval.
    try:
        df = yf.download(
            ticker,
            period=dl_period,
            interval=dl_interval,
            auto_adjust=True,
            actions=False,
            progress=False,
        )
    except Exception:
        df = None

    if df is None or df.empty:
        fallback_paths = [output_path]
        if legacy_output_path != output_path:
            fallback_paths.append(legacy_output_path)

        for fallback_path in fallback_paths:
            if not fallback_path.exists():
                continue
            existing_df = pd.read_csv(fallback_path)
            is_interval_compatible = (
                "Date" in existing_df.columns
                and interval_looks_compatible(existing_df["Date"], RESEARCH_INTERVAL)
            )
            is_fresh_enough = ALLOW_STALE_RAW_FALLBACK or _fallback_file_is_fresh_enough(existing_df)
            if is_interval_compatible and is_fresh_enough:
                if fallback_path != output_path:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    existing_df.to_csv(output_path, index=False)
                    print(
                        f"Download failed for {ticker}. Migrated compatible legacy raw data "
                        f"from {fallback_path} to {output_path}."
                    )
                else:
                    print(
                        f"Download failed for {ticker}. Reusing the existing raw file at {output_path}."
                    )
                print(output_path)
                return

        searched_paths = ", ".join(str(path) for path in fallback_paths)
        raise RuntimeError(
            "The pipeline needs interval-compatible raw data before it can continue. "
            f"Checked: {searched_paths}"
        )

    # Move the date index into a normal column so it can be saved to CSV.
    df = df.reset_index()

    # yfinance may return a two-level column index even for one ticker, so flatten it.
    df.columns = [column[0] if isinstance(column, tuple) else column for column in df.columns]

    # Keep only the requested columns in the expected order.
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    columns_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[columns_to_keep]

    # Normalize timestamps so cross-asset alignment later in the research stack
    # does not depend on the source exchange timezone.
    df["Date"] = normalize_timestamp_series(df["Date"])

    # -----------------------------------------------------------------------
    # 4h resampling: aggregate 1h bars into 4-hour candles
    # -----------------------------------------------------------------------
    if needs_resample:
        n_source = len(df)
        df = _resample_to_4h(df)
        print(
            f"  [{ticker}] Resampled {n_source} x {dl_interval} bars → "
            f"{len(df)} x {RESEARCH_INTERVAL} bars."
        )

    # Basic data quality audit and cleanup.
    df = _validate_data_quality(df, ticker)

    # Remove rows with missing values, drop duplicates, and sort from oldest to
    # newest timestamp. We do not forward-fill missing bars because that would
    # invent prices and create fake trades.
    df = (
        df.dropna()
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date", ascending=True)
        .reset_index(drop=True)
    )
    df["data_interval"] = RESEARCH_INTERVAL
    df["timeframe_label"] = RESEARCH_TIMEFRAME_LABEL
    df["data_source"] = "yfinance"
    df["source_period_requested"] = dl_period
    df["download_interval"] = dl_interval
    df["source_download_interval"] = dl_interval
    df["downloaded_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    # Save the cleaned data to the requested CSV file.
    df.to_csv(output_path, index=False)

    # Print summary.
    print(f"  [{ticker}] Saved {len(df)} {RESEARCH_INTERVAL} bars to {output_path.name}")
    print(df.head(5))


if __name__ == "__main__":
    main()
