"""Helpers that rebuild missing per-strategy trade logs before analysis runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

try:
    from timeframe_config import RESEARCH_INTERVAL, timeframe_output_suffix
except ModuleNotFoundError:
    from Code.timeframe_config import RESEARCH_INTERVAL, timeframe_output_suffix


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


def _strategy_trade_creators():
    """Import only the core trade-generator entrypoints lazily."""
    try:
        from connors_rsi2_pullback_agent import main as create_connors_rsi2_trades
        from donchian_trend_reentry_agent import main as create_donchian_reentry_trades
        from adx_trend_following_agent import main as create_adx_trend_trades
        from breakout_volume_momentum_agent import main as create_breakout_trades
        from mean_reversion_vol_filter_agent import main as create_mean_reversion_trades
        from momentum_relative_strength_agent import main as create_relative_strength_trades
        from random_agent import main as create_random_trades
        from trend_momentum_verification_agent import main as create_validation_trades
        from trend_pullback_agent import main as create_trend_trades
        from turn_of_month_seasonality_agent import main as create_turn_of_month_trades
        from uptrend_oversold_reversion_agent import main as create_oversold_reversion_trades
        from volatility_squeeze_breakout_agent import main as create_squeeze_breakout_trades
    except ModuleNotFoundError:
        from Code.connors_rsi2_pullback_agent import main as create_connors_rsi2_trades
        from Code.donchian_trend_reentry_agent import main as create_donchian_reentry_trades
        from Code.adx_trend_following_agent import main as create_adx_trend_trades
        from Code.breakout_volume_momentum_agent import main as create_breakout_trades
        from Code.mean_reversion_vol_filter_agent import main as create_mean_reversion_trades
        from Code.momentum_relative_strength_agent import main as create_relative_strength_trades
        from Code.random_agent import main as create_random_trades
        from Code.trend_momentum_verification_agent import main as create_validation_trades
        from Code.trend_pullback_agent import main as create_trend_trades
        from Code.turn_of_month_seasonality_agent import main as create_turn_of_month_trades
        from Code.uptrend_oversold_reversion_agent import main as create_oversold_reversion_trades
        from Code.volatility_squeeze_breakout_agent import main as create_squeeze_breakout_trades

    return {
        "trend_pullback": create_trend_trades,
        "breakout_volume_momentum": create_breakout_trades,
        "mean_reversion_vol_filter": create_mean_reversion_trades,
        "momentum_relative_strength": create_relative_strength_trades,
        "trend_momentum_verification": create_validation_trades,
        "random": create_random_trades,
        "adx_trend_following": create_adx_trend_trades,
        "uptrend_oversold_reversion": create_oversold_reversion_trades,
        "volatility_squeeze_breakout": create_squeeze_breakout_trades,
        "connors_rsi2_pullback": create_connors_rsi2_trades,
        "donchian_trend_reentry": create_donchian_reentry_trades,
        "turn_of_month_seasonality": create_turn_of_month_trades,
    }


def _artifact_run_id(path: Path) -> str | None:
    """Return the sidecar run ID for one artifact when available."""
    metadata_path = path.with_suffix(f"{path.suffix}.meta.json")
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    run_id = str(metadata.get("run_id", "")).strip()
    return run_id or None


def _trade_file_is_compatible(
    trade_path: Path,
    market_path: Path,
    *,
    agent_name: str,
    current_ticker: str,
    data_clean_dir: Path,
    required_run_id: str | None = None,
) -> bool:
    """Check whether a saved trade log still aligns with the current market dates.

    This guards against stale artifacts after data-history changes. A trade file
    can exist on disk but still be unusable if its entry/exit dates fall outside
    the currently saved market history for the ticker.
    """
    if not trade_path.exists():
        return False
    if not market_path.exists():
        return True
    if required_run_id:
        trade_run_id = _artifact_run_id(trade_path)
        if trade_run_id != required_run_id:
            return False
    if trade_path.stat().st_mtime_ns < market_path.stat().st_mtime_ns:
        # If the market/regime file changed after the trade log was written, we
        # treat the trade artifact as stale and force regeneration so the trade
        # logic is re-evaluated on the latest bar history.
        return False

    try:
        trade_df = pd.read_csv(trade_path)
        market_df = pd.read_csv(market_path, usecols=["Date"])
        market_interval_df = pd.read_csv(market_path, nrows=5)
    except Exception:
        return False

    required_columns = {"entry_date", "exit_date"}
    if not required_columns.issubset(trade_df.columns):
        return False
    if "Date" not in market_df.columns:
        return False
    if "data_interval" in trade_df.columns and not trade_df["data_interval"].astype(str).str.lower().eq(RESEARCH_INTERVAL).all():
        return False
    if "data_interval" in market_interval_df.columns and not market_interval_df["data_interval"].astype(str).str.lower().eq(RESEARCH_INTERVAL).all():
        return False
    if trade_df.empty:
        return True

    market_dates = pd.to_datetime(market_df["Date"], errors="coerce").dropna()
    if market_dates.empty:
        return False
    market_date_set = set(market_dates)

    entry_dates = pd.to_datetime(trade_df["entry_date"], errors="coerce")
    exit_dates = pd.to_datetime(trade_df["exit_date"], errors="coerce")
    if entry_dates.isna().any() or exit_dates.isna().any():
        return False
    if "return" in trade_df.columns:
        normalized_returns = pd.to_numeric(trade_df["return"], errors="coerce")
        if (normalized_returns <= -1.0).any():
            return False

    if str(agent_name).strip().lower() == "momentum_relative_strength":
        if "asset_ticker" not in trade_df.columns:
            return False
        metadata_path = data_clean_dir / f"{current_ticker.upper()}_momentum_relative_strength_universe.csv"
        if not metadata_path.exists():
            return False
        try:
            metadata_df = pd.read_csv(metadata_path)
        except Exception:
            return False
        if "universe_ticker" not in metadata_df.columns:
            return False
        allowed_universe = {
            str(value).strip().upper()
            for value in metadata_df["universe_ticker"].dropna().tolist()
            if str(value).strip()
        }
        if not allowed_universe:
            return False
        traded_assets = {
            str(value).strip().upper()
            for value in trade_df["asset_ticker"].dropna().tolist()
            if str(value).strip()
        }
        if not traded_assets.issubset(allowed_universe):
            return False

    return bool(entry_dates.isin(market_date_set).all() and exit_dates.isin(market_date_set).all())


def ensure_trade_file_exists(current_ticker: str, agent_name: str) -> Path:
    """Rebuild one missing strategy trade file so downstream analysis stays consistent."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    trade_path = data_clean_dir / f"{current_ticker.upper()}_{agent_name}_trades.csv"
    market_path = data_clean_dir / f"{current_ticker.upper()}_regimes.csv"
    required_run_id = ""
    if os.environ.get("REQUIRE_TRADE_RUN_ID_MATCH", "0") == "1":
        required_run_id = os.environ.get("PIPELINE_RUN_ID", "").strip()

    if _trade_file_is_compatible(
        trade_path,
        market_path,
        agent_name=agent_name,
        current_ticker=current_ticker,
        data_clean_dir=data_clean_dir,
        required_run_id=required_run_id or None,
    ):
        return trade_path

    os.environ["TICKER"] = current_ticker.upper()
    creators = _strategy_trade_creators()
    if agent_name not in creators:
        raise ValueError(f"Unsupported strategy name for trade regeneration: {agent_name}")

    creators[agent_name]()
    if not trade_path.exists():
        raise FileNotFoundError(
            f"Trade file is still missing after regeneration attempt: {trade_path}"
        )
    if not _trade_file_is_compatible(
        trade_path,
        market_path,
        agent_name=agent_name,
        current_ticker=current_ticker,
        data_clean_dir=data_clean_dir,
        required_run_id=required_run_id or None,
    ):
        raise ValueError(
            "Regenerated trade file is still incompatible with the active ticker calendar: "
            f"{trade_path}"
        )

    return trade_path
