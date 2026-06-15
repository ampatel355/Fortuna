"""Run the adaptive volatility-managed momentum validation strategy."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from single_ticker_agent_common import load_regime_data, save_trade_outputs
    from strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        MEDIUM_RETURN_COLUMN,
        RELATIVE_STRENGTH_RETURN_COLUMN,
    )
    from strategy_simulator import resolve_data_clean_dir, run_strategy
except ModuleNotFoundError:
    from Code.single_ticker_agent_common import load_regime_data, save_trade_outputs
    from Code.strategy_config import (
        BREAKOUT_HIGH_COLUMN,
        BREAKOUT_LOW_COLUMN,
        MEDIUM_RETURN_COLUMN,
        RELATIVE_STRENGTH_RETURN_COLUMN,
    )
    from Code.strategy_simulator import resolve_data_clean_dir, run_strategy


ticker = os.environ.get("TICKER", "SPY")


def main() -> None:
    """Load regime-tagged data, run the strategy, and save the trade log."""
    current_ticker = os.environ.get("TICKER", ticker).strip().upper()
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    input_path = data_clean_dir / f"{current_ticker}_regimes.csv"
    required_columns = [
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
    ]
    df = load_regime_data(input_path, required_columns)
    df.attrs["ticker"] = current_ticker

    trades_df = run_strategy("trend_momentum_verification", df, ticker=current_ticker)
    save_trade_outputs(
        current_ticker=current_ticker,
        agent_name="trend_momentum_verification",
        trades_df=trades_df,
        output_dir=data_clean_dir,
    )

    print(f"Total number of trades: {len(trades_df)}")
    print("\nFirst 10 trades:")
    print(trades_df.head(10))


if __name__ == "__main__":
    main()
