"""Positive control under REALISTIC entry signals (not just perfect foresight).

The basic positive control (Code/positive_control_power.py) injects an oracle that
sees the future. A referee rightly asks whether the test detects *imperfect,
realistic* signals. This script answers that on the real GC=F price path by
driving entry placement with signals of controlled quality and several generating
processes, and measuring the test's power as a function of signal quality.

Signal quality is summarized by rho = Pearson correlation between the entry signal
and the d-bar forward open-to-open return. A strategy enters at its top-N
signal bars (mutually non-overlapping); the realized schedule is scored against
the paper's structure-preserving null. rho = 0 is no skill; rho -> 1 approaches
the oracle.

Two experiments:
  (A) Power vs. signal quality: a NOISY signal s = z(f) + lambda * noise whose
      correlation with the forward return is tuned to a target rho grid. Gives a
      smooth power curve and a minimum-detectable signal correlation (MDE).
  (B) Detection across signal-generating processes at a matched, modest target
      rho: noisy oracle, a CAUSAL momentum signal (trailing return, no foresight),
      an AR(1)-smoothed signal, and a regime-conditional signal (skill only in
      calm regimes). Shows detection does not rely on foresight or a particular
      mechanism.

Outputs (never overwrites canonical pipeline artifacts):
  Data_Clean/GC=F_signal_power_curve.csv
  Data_Clean/GC=F_signal_mechanisms.csv
  Charts/positive_control_signal_curve.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TICKER", "GC=F")
os.environ.setdefault("MONTE_CARLO_SIMULATE_EXECUTION_COSTS", "0")
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

N_TRADES = int(os.environ.get("PC_N_TRADES", "30"))
HOLDING_BARS = int(os.environ.get("PC_HOLDING_BARS", "10"))
POSITION_FRACTION = float(os.environ.get("PC_POSITION_FRACTION", "0.05"))
TARGET_RHOS = [float(x) for x in os.environ.get("PC_RHOS", "0,0.02,0.05,0.10,0.20,0.40").split(",")]
SEEDS = list(range(int(os.environ.get("PC_SIGNAL_SEEDS", "40"))))
MOMENTUM_K = int(os.environ.get("PC_MOMENTUM_K", "20"))
AR1_PHI = float(os.environ.get("PC_AR1_PHI", "0.9"))
MATCH_RHO = float(os.environ.get("PC_MATCH_RHO", "0.10"))
N_SIM = mc.NUMBER_OF_SIMULATIONS
TXN = mc.TRANSACTION_COST


def standardize(x: np.ndarray) -> np.ndarray:
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / sd if sd > 0 else x - mu


def place_top_n_by_signal(signal: np.ndarray, max_entry: int, d: int, n: int) -> np.ndarray:
    """Greedily place n non-overlapping (>= d apart) entries at the highest-signal bars."""
    order = np.argsort(-signal[: max_entry + 1])
    chosen: list[int] = []
    for idx in order:
        if len(chosen) >= n:
            break
        if all(abs(int(idx) - c) >= d for c in chosen):
            chosen.append(int(idx))
    if len(chosen) < n:
        raise ValueError("Could not place enough non-overlapping trades.")
    return np.array(sorted(chosen), dtype=np.int64)


def realized_cumulative_return(open_prices, entry_idx, exit_idx) -> float:
    gross = open_prices[exit_idx] / open_prices[entry_idx] - 1.0
    adjusted = POSITION_FRACTION * (gross - TXN)
    return float(np.expm1(np.log1p(adjusted).sum()))


def score_schedule(entry_idx, exit_idx, calendar, market_df, open_prices, seed):
    actual = realized_cumulative_return(open_prices, entry_idx, exit_idx)
    trade_df = pd.DataFrame({
        "entry_date": calendar[entry_idx], "exit_date": calendar[exit_idx],
        "return": POSITION_FRACTION * (open_prices[exit_idx] / open_prices[entry_idx] - 1.0 - TXN),
        "holding_bars": HOLDING_BARS, "position_value_fraction": POSITION_FRACTION,
        "direction": "long", "regime_at_entry": "neutral",
    })
    rng = mc.build_random_generator(reproducible=True, seed=30_000 + seed)
    sim, _ = mc.simulate_agent_null_cumulative_returns(
        agent_name="signal_positive_control", current_ticker=TICKER, trade_df=trade_df,
        market_df=market_df, input_path=DATA_CLEAN / "signal_pc_trades.csv",
        simulation_count=N_SIM, rng=rng,
    )
    s = sim.to_numpy(dtype=float)
    mean, sd = float(np.mean(s)), float(np.std(s, ddof=0))
    z = (actual - mean) / sd if sd > 0 else float("nan")
    return mc.calculate_p_value(s, actual), mc.calculate_actual_percentile(s, actual), z


# ---- signal generators (all return a per-bar score; causal ones use no future info) ----
def make_noisy_signal(fstd, target_rho, rng):
    """s = fstd + lambda*noise so that corr(s, f) ~= target_rho (target_rho=0 -> pure noise)."""
    if target_rho <= 0:
        return rng.standard_normal(fstd.shape)
    lam = float(np.sqrt(max(1.0 / target_rho**2 - 1.0, 0.0)))
    return fstd + lam * rng.standard_normal(fstd.shape)


def make_momentum_signal(open_prices, k):
    """Causal: trailing k-bar open-to-open return (no future information)."""
    s = np.full(open_prices.shape, -np.inf)
    s[k:] = open_prices[k:] / open_prices[:-k] - 1.0
    return s


def make_ar1_signal(fstd, phi, target_rho, rng):
    """An AR(1)-smoothed noisy signal: temporally autocorrelated, partially informative."""
    base = make_noisy_signal(fstd, target_rho, rng)
    out = np.empty_like(base)
    out[0] = base[0]
    for t in range(1, len(base)):
        out[t] = phi * out[t - 1] + np.sqrt(1 - phi**2) * base[t]
    return out


def make_regime_signal(fstd, regimes, rng):
    """Skill only inside calm regimes; pure noise elsewhere (regime-switching skill)."""
    s = rng.standard_normal(fstd.shape)
    calm = np.asarray(regimes == "calm")
    s[calm] = fstd[calm]
    return s


def run_signal(signal, max_entry, d, calendar, market_df, open_prices, fwd, seed):
    entry = place_top_n_by_signal(signal, max_entry, d, N_TRADES)
    exit_ = entry + d
    feas = np.isfinite(signal[: max_entry + 1]) & np.isfinite(fwd[: max_entry + 1])
    rho = float(np.corrcoef(signal[: max_entry + 1][feas], fwd[: max_entry + 1][feas])[0, 1])
    p, pct, z = score_schedule(entry, exit_, calendar, market_df, open_prices, seed)
    return rho, p, pct, z


def main() -> None:
    market_df = mc.load_market_data(DATA_CLEAN / f"{TICKER}_regimes.csv")
    calendar = market_df["Date"].to_numpy()
    open_prices = market_df["Open"].to_numpy(dtype=float)
    regimes = market_df["regime"].astype(str).str.lower().to_numpy() if "regime" in market_df else np.array([""] * len(open_prices))
    d = HOLDING_BARS
    max_entry = len(open_prices) - 1 - d
    fwd = np.full(len(open_prices), np.nan)
    fwd[: max_entry + 1] = open_prices[d : d + max_entry + 1] / open_prices[: max_entry + 1] - 1.0
    fstd = np.where(np.isfinite(fwd), standardize(np.nan_to_num(fwd, nan=0.0)), 0.0)

    # ---- (A) power vs signal quality (noisy signal) ----
    rows = []
    for target in TARGET_RHOS:
        for seed in SEEDS:
            rng = np.random.default_rng(40_000 + int(round(target * 1000)) * 131 + seed)
            s = make_noisy_signal(fstd, target, rng)
            rho, p, pct, z = run_signal(s, max_entry, d, calendar, market_df, open_prices, fwd, seed)
            rows.append(dict(target_rho=target, realized_rho=rho, p_value=p, percentile=pct, RCSI_z=z))
    df = pd.DataFrame(rows)
    g = df.groupby("target_rho")
    curve = pd.DataFrame({
        "target_rho": g.size().index,
        "mean_realized_rho": g["realized_rho"].mean().values,
        "mean_p": g["p_value"].mean().values,
        "median_p": g["p_value"].median().values,
        "reject_rate": g.apply(lambda x: float((x["p_value"] <= 0.05).mean())).values,
        "mean_RCSI_z": g["RCSI_z"].mean().values,
    })
    curve.to_csv(DATA_CLEAN / f"{TICKER}_signal_power_curve.csv", index=False)
    print("=== (A) power vs signal quality (noisy signal) ===")
    print(curve.round(4).to_string(index=False))

    # ---- (B) detection across generating processes at matched target rho ----
    mech_rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(50_000 + seed)
        gens = {
            "Noisy oracle": make_noisy_signal(fstd, MATCH_RHO, rng),
            "Momentum (causal)": make_momentum_signal(open_prices, MOMENTUM_K),
            "AR(1) smoothed": make_ar1_signal(fstd, AR1_PHI, MATCH_RHO, rng),
            "Regime-conditional": make_regime_signal(fstd, regimes, rng),
        }
        for name, sig in gens.items():
            rho, p, pct, z = run_signal(sig, max_entry, d, calendar, market_df, open_prices, fwd, seed)
            mech_rows.append(dict(mechanism=name, realized_rho=rho, p_value=p, percentile=pct, RCSI_z=z))
    mdf = pd.DataFrame(mech_rows)
    gm = mdf.groupby("mechanism")
    mech = pd.DataFrame({
        "mechanism": gm.size().index,
        "mean_realized_rho": gm["realized_rho"].mean().values,
        "mean_p": gm["p_value"].mean().values,
        "reject_rate": gm.apply(lambda x: float((x["p_value"] <= 0.05).mean())).values,
        "mean_RCSI_z": gm["RCSI_z"].mean().values,
    })
    # deterministic generators (momentum) have rho/placement fixed across seeds -> reject_rate is 0/1
    mech.to_csv(DATA_CLEAN / f"{TICKER}_signal_mechanisms.csv", index=False)
    print(f"\n=== (B) detection across mechanisms (target rho = {MATCH_RHO}, {len(SEEDS)} seeds) ===")
    print(mech.round(4).to_string(index=False))

    _chart(curve)
    print(f"\nconfig: N={N_TRADES} d={HOLDING_BARS} w={POSITION_FRACTION} n_sim={N_SIM} seeds={len(SEEDS)}")


def _chart(curve: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHARTS.mkdir(exist_ok=True)
    x = curve["mean_realized_rho"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(x, curve["reject_rate"], "-o", color="#1f4e79", lw=2)
    ax.axhline(0.80, color="#888", ls=":", lw=1)
    ax.axhline(0.05, color="#b00020", ls="--", lw=1)
    ax.set_xlabel("signal quality: corr(entry signal, forward return)")
    ax.set_ylabel("rejection rate at 5%")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Power vs. signal quality on real GC=F prices")
    fig.tight_layout()
    fig.savefig(CHARTS / "positive_control_signal_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
