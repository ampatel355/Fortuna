"""Schedule-measure and leading-gap robustness for the GC=F panel.

Re-evaluates the FIXED realized trade logs under alternative null specifications
without re-running any backtest and without overwriting the canonical pipeline
outputs. Each invocation evaluates one measure (selected by environment) and
writes a measure-tagged CSV to Data_Clean/.

Measures (set before import via env):
  baseline       : gap-permutation Q (default).
  context_matched: MONTE_CARLO_CONTEXT_MATCHING=1.
  leading_gap_200: MONTE_CARLO_MIN_LEADING_INDEX=200 (exclude the universal
                   warm-up region a real strategy could not have traded in).

Usage:
  MEASURE_LABEL=baseline ./.venv/bin/python Code/schedule_measure_sensitivity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
os.environ.setdefault("TICKER", "GC=F")

import monte_carlo as mc
from strategy_config import AGENT_DISPLAY_NAMES, AGENT_ORDER

TICKER = "GC=F"
MEASURE_LABEL = os.environ.get("MEASURE_LABEL", "baseline")
DATA_CLEAN = PROJECT_ROOT / "Data_Clean"


def verdict(p: float, pct: float, z: float) -> str:
    if p <= 0.01 and pct >= 99 and z >= 2.33:
        return "Strong"
    if p <= 0.05 and pct >= 95 and z >= 1.65:
        return "Moderate"
    if p <= 0.20 and pct >= 80 and z >= 0.84:
        return "Weak"
    if pct < 50:
        return "Below null median"
    return "Random / luck"


def main() -> None:
    market_df = mc.load_market_data(DATA_CLEAN / f"{TICKER}_regimes.csv")
    child_sequences = np.random.SeedSequence(mc.SEED).spawn(len(AGENT_ORDER))
    rows: list[dict] = []
    for agent_name, child in zip(AGENT_ORDER, child_sequences):
        input_path = mc.ensure_trade_file_exists(TICKER, agent_name)
        trade_df = mc.load_trade_data(input_path, allow_empty=True)
        rng = mc.build_random_generator(
            reproducible=mc.REPRODUCIBLE,
            seed=int(child.generate_state(1, dtype=np.uint64)[0]),
        )
        if trade_df.empty:
            continue
        raw = trade_df["return"].to_numpy(dtype=float)
        adj = mc.adjust_trade_returns(
            raw_returns=raw, transaction_cost=mc.TRANSACTION_COST, input_path=input_path
        )
        actual = mc.calculate_cumulative_return_from_log_returns(mc.convert_to_log_returns(adj))
        sim, _ = mc.simulate_agent_null_cumulative_returns(
            agent_name=agent_name,
            current_ticker=TICKER,
            trade_df=trade_df,
            market_df=market_df,
            input_path=input_path,
            simulation_count=mc.NUMBER_OF_SIMULATIONS,
            rng=rng,
        )
        sim_arr = sim.to_numpy(dtype=float)
        p = mc.calculate_p_value(sim_arr, actual)
        pct = mc.calculate_actual_percentile(sim_arr, actual)
        mean_sim = float(np.mean(sim_arr))
        std_sim = float(np.std(sim_arr, ddof=0))
        rcsi = actual - mean_sim
        rcsi_z = rcsi / std_sim if std_sim > 0 else float("nan")
        rows.append(
            {
                "measure": MEASURE_LABEL,
                "agent": agent_name,
                "display": AGENT_DISPLAY_NAMES.get(agent_name, agent_name),
                "n_trades": int(len(trade_df)),
                "actual_cumulative_return": actual,
                "mean_simulated_return": mean_sim,
                "p_value": p,
                "percentile": pct,
                "RCSI": rcsi,
                "RCSI_z": rcsi_z,
                "verdict": verdict(p, pct, rcsi_z),
            }
        )
    df = pd.DataFrame(rows)
    df["bh_adjusted_p_value"] = mc.benjamini_hochberg_adjusted_p_values(df["p_value"])
    out = DATA_CLEAN / f"{TICKER}_sensitivity_{MEASURE_LABEL}.csv"
    df.to_csv(out, index=False)
    print(f"measure={MEASURE_LABEL}  simulations={mc.NUMBER_OF_SIMULATIONS}  context={mc.CONTEXT_MATCHING_ENABLED}  min_leading={mc.MIN_LEADING_INDEX}")
    print(df[["display", "n_trades", "actual_cumulative_return", "p_value", "percentile", "RCSI_z", "verdict"]].to_string(index=False))
    print(f"min p={df['p_value'].min():.4f}  any<=0.05: {(df['p_value']<=0.05).any()}  -> {out.name}")


if __name__ == "__main__":
    main()
