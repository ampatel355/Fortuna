"""Run the random baseline strategy for the active ticker."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from single_ticker_agent_common import load_regime_data, save_trade_outputs
    from strategy_simulator import (
        RANDOM_AGENT_REPRODUCIBLE,
        RANDOM_AGENT_SEED,
        resolve_data_clean_dir,
        run_strategy,
    )
except ModuleNotFoundError:
    from Code.single_ticker_agent_common import load_regime_data, save_trade_outputs
    from Code.strategy_simulator import (
        RANDOM_AGENT_REPRODUCIBLE,
        RANDOM_AGENT_SEED,
        resolve_data_clean_dir,
        run_strategy,
    )


ticker = os.environ.get("TICKER", "SPY")


def main() -> None:
    """Load regime-tagged data, run the random baseline, and save the trade log."""
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
        "regime",
    ]
    df = load_regime_data(input_path, required_columns)
    df.attrs["ticker"] = current_ticker

    decision_seed = RANDOM_AGENT_SEED if RANDOM_AGENT_REPRODUCIBLE else None
    trades_df = run_strategy("random", df, random_decision_seed=decision_seed, ticker=current_ticker)
    save_trade_outputs(
        current_ticker=current_ticker,
        agent_name="random",
        trades_df=trades_df,
        output_dir=data_clean_dir,
    )

    print(f"Total number of trades: {len(trades_df)}")
    print(f"Random agent reproducible mode: {RANDOM_AGENT_REPRODUCIBLE}")
    print(f"Random agent seed used: {RANDOM_AGENT_SEED if RANDOM_AGENT_REPRODUCIBLE else 'None'}")
    print("\nFirst 10 trades:")
    print(trades_df.head(10))


if __name__ == "__main__":
    main()
