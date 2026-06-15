"""Positive control: does the test detect *genuine* timing skill on real prices?

A null-result paper must answer the referee's first question: "is your method
simply incapable of ever rejecting?" This script answers it on REAL GC=F open
prices (not synthetic toy paths) by constructing strategies whose entry timing
carries a *controlled, known* amount of genuine skill and then scoring them with
the paper's own structure-preserving timing null.

Construction (held fully faithful to the live pipeline):
  * Universe: real GC=F daily open prices from Data_Clean/GC=F_regimes.csv.
  * A strategy makes N long trades, each held d open-to-open bars, equal weight w.
  * Skill level alpha in [0, 1] is the probability that any given entry is placed
    in the top ORACLE_QUANTILE of the d-bar forward-return distribution available
    in its feasible window ("good timing"); with probability 1 - alpha the entry
    is placed uniformly at random ("luck"). alpha = 0 is a pure no-skill strategy;
    alpha = 1 is a consistently well-timed strategy. The oracle uses *future*
    information by construction -- this is an injected control, NOT a claim that
    any real gold trader has such foresight.
  * The realized schedule is scored against the SAME structure-preserving null the
    paper uses for every real strategy (mc.simulate_agent_null_cumulative_returns):
    durations and the realized gap multiset are preserved, only the *placement* is
    randomized, which destroys the oracle alignment. Skill therefore shows up as
    the realized return sitting in the upper tail of the null.

Both the realized statistic and the null are computed with execution-cost
simulation disabled so the comparison is deterministic and differs ONLY in
placement (skill vs. luck); this is set via the environment before import.

Outputs (never overwrites canonical pipeline artifacts):
  Data_Clean/GC=F_positive_control_power.csv          (one row per alpha x seed)
  Data_Clean/GC=F_positive_control_power_summary.csv  (one row per alpha)
  Charts/positive_control_skill_curve.png
  Charts/positive_control_example_histogram.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set null-model behaviour BEFORE importing monte_carlo (constants read at import).
os.environ.setdefault("TICKER", "GC=F")
os.environ.setdefault("MONTE_CARLO_SIMULATE_EXECUTION_COSTS", "0")  # symmetric, deterministic
os.environ.setdefault("MONTE_CARLO_CONTEXT_MATCHING", "0")
os.environ.setdefault("MONTE_CARLO_MIN_LEADING_INDEX", "0")
os.environ.setdefault("MONTE_CARLO_REPRODUCIBLE", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import monte_carlo as mc

TICKER = "GC=F"
DATA_CLEAN = PROJECT_ROOT / "Data_Clean"
CHARTS = PROJECT_ROOT / "Charts"

# ---- experiment configuration (env-overridable) ----------------------------
N_TRADES = int(os.environ.get("PC_N_TRADES", "30"))
HOLDING_BARS = int(os.environ.get("PC_HOLDING_BARS", "10"))
POSITION_FRACTION = float(os.environ.get("PC_POSITION_FRACTION", "0.05"))
ORACLE_QUANTILE = float(os.environ.get("PC_ORACLE_QUANTILE", "0.80"))  # top 20%
ALPHAS = [float(a) for a in os.environ.get("PC_ALPHAS", "0,0.2,0.4,0.6,0.8,1.0").split(",")]
SEEDS = list(range(int(os.environ.get("PC_SEEDS", "20"))))
N_SIM = mc.NUMBER_OF_SIMULATIONS
TXN = mc.TRANSACTION_COST


def classify(p: float, pct: float, z: float) -> str:
    """Canonical multi-metric verdict (matches Code/strategy_verdicts thresholds)."""
    if z >= 2.0 and p <= 0.05 and pct >= 95:
        return "Strong"
    if z >= 1.0 and p <= 0.05 and pct >= 85:
        return "Moderate"
    if z >= 0.5 and p <= 0.20 and pct >= 70:
        return "Weak"
    if -0.5 <= z <= 0.5 and 0.30 <= p <= 0.70:
        return "Random / luck"
    return "Below null median" if z < -0.5 else "Random / luck"


def build_skilled_schedule(
    open_prices: np.ndarray,
    forward_returns: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Place N non-overlapping long trades; a fraction ~alpha are well-timed.

    Two stages keep the baseline drift-neutral:
      1. *Unbiased* random non-overlapping placement -- draw N sorted points
         uniformly on [0, max_entry - N*d] and offset by k*d. This is the
         standard uniform draw over valid non-overlapping schedules (the same
         construction as the random-timing null) and spreads trades evenly across
         history for every alpha, so there is no back-loading toward the
         high-drift tail.
      2. For each trade independently flagged "skill" (probability alpha), move
         ONLY that trade to a top-ORACLE_QUANTILE forward-return index inside its
         local feasible interval (between its neighbours), leaving all other
         trades fixed. alpha = 0 therefore reduces exactly to the unbiased null
         draw; alpha = 1 times every entry well.
    """
    d = HOLDING_BARS
    max_entry = len(open_prices) - 1 - d  # last index with a valid d-bar exit
    span = max_entry - N_TRADES * d
    if span <= 0:
        raise ValueError("Not enough history for the requested trade count/duration.")

    base_points = np.sort(rng.integers(0, span + 1, size=N_TRADES))
    entry_idx = base_points + np.arange(N_TRADES, dtype=np.int64) * d  # non-overlapping

    skill_flags = rng.random(N_TRADES) < alpha
    for k in np.flatnonzero(skill_flags):
        low = (entry_idx[k - 1] + d) if k > 0 else 0
        high = (entry_idx[k + 1] - d) if k < N_TRADES - 1 else max_entry
        if high <= low:
            continue
        window = np.arange(low, high + 1)
        cut = float(np.quantile(forward_returns[window], ORACLE_QUANTILE))
        good = window[forward_returns[window] >= cut]
        entry_idx[k] = int(rng.choice(good)) if good.size > 0 else int(rng.choice(window))

    order = np.argsort(entry_idx)
    entry_idx = entry_idx[order]
    exit_idx = entry_idx + d
    return entry_idx, exit_idx


def realized_cumulative_return(open_prices: np.ndarray, entry_idx, exit_idx) -> float:
    """Score the realized schedule with the SAME math the null uses (exec costs off)."""
    gross = open_prices[exit_idx] / open_prices[entry_idx] - 1.0  # long, open-to-open
    position = gross - TXN
    adjusted = POSITION_FRACTION * position
    return float(np.expm1(np.log1p(adjusted).sum()))


def main() -> None:
    market_df = mc.load_market_data(DATA_CLEAN / f"{TICKER}_regimes.csv")
    calendar = market_df["Date"].to_numpy()
    open_prices = market_df["Open"].to_numpy(dtype=float)
    d = HOLDING_BARS
    # forward d-bar open-to-open return at each feasible index (oracle signal)
    forward_returns = np.full(len(open_prices), -np.inf)
    valid = np.arange(0, len(open_prices) - d)
    forward_returns[valid] = open_prices[valid + d] / open_prices[valid] - 1.0

    rows: list[dict] = []
    example = None  # (sim_array, actual) for the alpha=1 illustrative histogram
    for alpha in ALPHAS:
        for seed in SEEDS:
            rng = np.random.default_rng(10_000 + int(round(alpha * 100)) * 97 + seed)
            entry_idx, exit_idx = build_skilled_schedule(open_prices, forward_returns, alpha, rng)
            actual = realized_cumulative_return(open_prices, entry_idx, exit_idx)

            # Build a faithful trade_df and run the REAL structure-preserving null.
            trade_df = pd.DataFrame(
                {
                    "entry_date": calendar[entry_idx],
                    "exit_date": calendar[exit_idx],
                    "return": POSITION_FRACTION * (open_prices[exit_idx] / open_prices[entry_idx] - 1.0 - TXN),
                    "holding_bars": d,
                    "position_value_fraction": POSITION_FRACTION,
                    "direction": "long",
                    "regime_at_entry": "neutral",
                }
            )
            null_rng = mc.build_random_generator(reproducible=True, seed=20_000 + seed)
            sim, _ = mc.simulate_agent_null_cumulative_returns(
                agent_name="positive_control_oracle",
                current_ticker=TICKER,
                trade_df=trade_df,
                market_df=market_df,
                input_path=DATA_CLEAN / "positive_control_oracle_trades.csv",
                simulation_count=N_SIM,
                rng=null_rng,
            )
            sim_arr = sim.to_numpy(dtype=float)
            p = mc.calculate_p_value(sim_arr, actual)
            pct = mc.calculate_actual_percentile(sim_arr, actual)
            mean_sim = float(np.mean(sim_arr))
            std_sim = float(np.std(sim_arr, ddof=0))
            rcsi = actual - mean_sim
            z = rcsi / std_sim if std_sim > 0 else float("nan")
            rows.append(
                dict(alpha=alpha, seed=seed, actual=actual, mean_sim=mean_sim,
                     p_value=p, percentile=pct, RCSI=rcsi, RCSI_z=z,
                     verdict=classify(p, pct, z), n_sim=N_SIM)
            )
            if alpha == 1.0 and seed == 0:
                example = (sim_arr, actual, p, z)

    df = pd.DataFrame(rows)
    df.to_csv(DATA_CLEAN / f"{TICKER}_positive_control_power.csv", index=False)

    # ---- summary per alpha ----
    g = df.groupby("alpha")
    summary = pd.DataFrame({
        "alpha": g["p_value"].mean().index,
        "mean_p": g["p_value"].mean().values,
        "median_p": g["p_value"].median().values,
        "mean_percentile": g["percentile"].mean().values,
        "mean_RCSI_z": g["RCSI_z"].mean().values,
        "sd_RCSI_z": g["RCSI_z"].std(ddof=0).values,
        "frac_p_le_05": g.apply(lambda x: float((x["p_value"] <= 0.05).mean())).values,
        "frac_strong": g.apply(lambda x: float((x["verdict"] == "Strong").mean())).values,
    })
    summary.to_csv(DATA_CLEAN / f"{TICKER}_positive_control_power_summary.csv", index=False)

    print(f"config: N={N_TRADES} d={HOLDING_BARS} w={POSITION_FRACTION} oracle_q={ORACLE_QUANTILE} "
          f"n_sim={N_SIM} seeds={len(SEEDS)} exec_costs={mc.SIMULATE_EXECUTION_COSTS}")
    print(summary.to_string(index=False))

    _make_charts(summary, example)


def _make_charts(summary: pd.DataFrame, example) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHARTS.mkdir(exist_ok=True)
    a = summary["alpha"].to_numpy()

    # (1) skill-detection curve: mean p-value and mean RCSI_z vs injected skill
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.axhline(0.05, color="#b00020", ls="--", lw=1, alpha=0.7)
    ax1.text(a[0], 0.06, "p = 0.05", color="#b00020", fontsize=9, va="bottom")
    ax1.plot(a, summary["mean_p"], "-o", color="#1f4e79", lw=2, label="mean p-value")
    ax1.fill_between(a, summary["median_p"], summary["mean_p"], color="#1f4e79", alpha=0.12)
    ax1.set_xlabel("injected timing skill  $\\alpha$  (fraction of well-timed entries)")
    ax1.set_ylabel("one-sided p-value", color="#1f4e79")
    ax1.set_ylim(-0.02, 0.7)
    ax1.tick_params(axis="y", labelcolor="#1f4e79")

    ax2 = ax1.twinx()
    ax2.plot(a, summary["mean_RCSI_z"], "-s", color="#2e7d32", lw=2, label="mean RCSI$_z$")
    ax2.fill_between(a, summary["mean_RCSI_z"] - summary["sd_RCSI_z"],
                     summary["mean_RCSI_z"] + summary["sd_RCSI_z"], color="#2e7d32", alpha=0.12)
    ax2.axhline(2.0, color="#2e7d32", ls=":", lw=1, alpha=0.6)
    ax2.set_ylabel("RCSI$_z$ (std. devs above null mean)", color="#2e7d32")
    ax2.tick_params(axis="y", labelcolor="#2e7d32")
    ax1.set_title("Positive control: the test detects genuine timing skill on real GC=F prices")
    fig.tight_layout()
    fig.savefig(CHARTS / "positive_control_skill_curve.png", dpi=150)
    plt.close(fig)

    # (2) example null histogram with the realized (alpha=1) return marked
    if example is not None:
        sim_arr, actual, p, z = example
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.hist(sim_arr, bins=60, color="#9e9e9e", alpha=0.8,
                label="structure-preserving null (random placement)")
        ax.axvline(actual, color="#b00020", lw=2.5,
                   label=f"realized skilled return  (p = {p:.4f}, RCSI$_z$ = {z:.1f})")
        ax.set_xlabel("cumulative return")
        ax.set_ylabel("simulated schedules")
        ax.set_title("A genuinely well-timed strategy ($\\alpha=1$) lands far in the null's upper tail")
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(CHARTS / "positive_control_example_histogram.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
