"""Redraw the analysis figures with the shared publication style.

Reads the already-saved result CSVs (so the underlying numbers are unchanged and
faithful) and re-renders each figure with serif typography, clean spines, and a
muted palette. The single illustrative positive-control histogram is the only one
that re-runs computation, and only for one (alpha=1, seed=0) schedule.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TICKER", "GC=F")
os.environ.setdefault("MONTE_CARLO_SIMULATE_EXECUTION_COSTS", "0")
os.environ.setdefault("MONTE_CARLO_REPRODUCIBLE", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "Code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from pub_style import use_pub_style, BLUE, GREEN, RED, GREY, GOLD

D = ROOT / "Data_Clean"
C = ROOT / "Charts"
use_pub_style()


def skill_curve():
    s = pd.read_csv(D / "GC=F_positive_control_power_summary.csv")
    a = s["alpha"].to_numpy()
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.axhline(0.05, color=RED, ls="--", lw=1)
    ax1.text(a[0], 0.065, "$p=0.05$", color=RED, fontsize=9, va="bottom")
    ax1.plot(a, s["mean_p"], "-o", color=BLUE, label="mean $p$-value")
    ax1.set_xlabel(r"injected timing skill $\alpha$ (fraction of well-timed entries)")
    ax1.set_ylabel("one-sided $p$-value", color=BLUE)
    ax1.set_ylim(-0.02, 0.62); ax1.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(a, s["mean_RCSI_z"], "-s", color=GREEN, label=r"mean RCSI$_z$")
    ax2.fill_between(a, s["mean_RCSI_z"] - s["sd_RCSI_z"], s["mean_RCSI_z"] + s["sd_RCSI_z"],
                     color=GREEN, alpha=0.12)
    ax2.axhline(2.0, color=GREEN, ls=":", lw=1)
    ax2.set_ylabel(r"RCSI$_z$ (std.\ devs above null mean)", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax1.set_title("Positive control: detecting genuine timing skill on real GC=F prices")
    fig.savefig(C / "positive_control_skill_curve.png"); plt.close(fig)


def signal_curve():
    c = pd.read_csv(D / "GC=F_signal_power_curve.csv")
    x = c["mean_realized_rho"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.axhline(0.80, color=GREY, ls=":", lw=1); ax.text(x[-1], 0.81, "80% power", ha="right", fontsize=9, color="#555")
    ax.axhline(0.05, color=RED, ls="--", lw=1)
    ax.plot(x, c["reject_rate"], "-o", color=BLUE)
    ax.set_xlabel(r"signal quality: $\mathrm{corr}(\text{entry signal},\ \text{forward return})$")
    ax.set_ylabel("rejection rate at 5%")
    ax.set_ylim(-0.03, 1.04)
    ax.set_title("Power versus signal quality on real GC=F prices")
    fig.savefig(C / "positive_control_signal_curve.png"); plt.close(fig)


def cross_asset():
    P = pd.read_csv(D / "cross_asset_panel.csv").sort_values("p_value").reset_index(drop=True)
    m = len(P); mean_p = P["p_value"].mean(); n_nom = int((P["p_value"] <= 0.05).sum())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    bins = np.linspace(0, 1, 21)
    ax1.hist(P["p_value"], bins=bins, color=BLUE, edgecolor="white", alpha=0.9)
    ax1.axhline(m / 20, color=RED, ls="--", lw=1.3, label=f"if pure chance ({m/20:.1f}/bin)")
    ax1.set_xlabel("one-sided $p$-value"); ax1.set_ylabel(f"asset $\\times$ strategy tests (of {m})")
    ax1.set_title(f"$p$-values skew conservative (mean {mean_p:.2f})\n{n_nom} hits at 0.05 vs {0.05*m:.0f} expected")
    ax1.legend()
    k = P["rank"].to_numpy(); rand = P["is_random_baseline"].to_numpy()
    ax2.plot(k, P["p_value"], "o", ms=3.2, color="#555555", label="sorted $p$-values")
    ax2.plot(k[rand], P["p_value"].to_numpy()[rand], "o", ms=6.5, mfc="none", mec=RED, mew=1.5,
             label="random-entry baselines")
    ax2.plot(k, k / m * 0.05, "-", color=GREEN, lw=1.6, label="BH line (FDR 0.05)")
    xmax = max(40, int(0.15 * m)); ax2.set_xlim(0, xmax)
    ax2.set_ylim(0, float(P["p_value"].iloc[min(m - 1, xmax)]))
    ax2.set_xlabel("rank of $p$-value (smallest first)"); ax2.set_ylabel("$p$-value")
    ax2.set_title("No result crosses the BH line\n(0 discoveries at FDR 0.05)"); ax2.legend(loc="upper left")
    fig.savefig(C / "cross_asset_multiplicity.png"); plt.close(fig)


def synthetic_power():
    s = pd.read_csv(D / "synthetic_power_curve_summary.csv")
    s = s.sort_values("signal_strength")
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(s["signal_strength"] * 1e4, s["reject_rate_0_05"], "-o", color=GREEN)
    ax.axhline(0.05, color=RED, ls="--", lw=1)
    ax.set_xlabel("injected entry signal (basis points per event bar)")
    ax.set_ylabel("rejection rate at 5%")
    ax.set_ylim(-0.03, 1.04)
    ax.set_title("Synthetic power curve: deterministic entry signal")
    fig.savefig(C / "synthetic_power_curve.png"); plt.close(fig)


def synthetic_calibration():
    r = pd.read_csv(D / "synthetic_null_validation_runs.csv")
    s = pd.read_csv(D / "synthetic_null_validation_summary.csv")
    noskill = s[s["signal_strength"] == 0]["world"].tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    grid = np.linspace(0, 1, 200)
    for i, w in enumerate(noskill):
        pv = np.sort(r[r["world"] == w]["p_value"].to_numpy())
        if pv.size == 0:
            continue
        cdf = np.searchsorted(pv, grid, side="right") / pv.size
        ax1.plot(grid, cdf, lw=1.6, label=w)
    ax1.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.2, label="Uniform(0,1)")
    ax1.set_xlabel("$p$-value"); ax1.set_ylabel("empirical CDF")
    ax1.set_title("No-skill $p$-value calibration"); ax1.legend(fontsize=8)
    sub = s[(s["signal_strength"] == 0) & (s["world"].isin(noskill))]
    y = np.arange(len(sub))
    ax2.barh(y, sub["reject_rate_0_05"], color=BLUE, alpha=0.85,
             xerr=sub["mc_se_reject_0_05"], error_kw=dict(ecolor="#444", lw=1, capsize=3))
    ax2.axvline(0.05, color=RED, ls="--", lw=1.2)
    ax2.set_yticks(y); ax2.set_yticklabels(sub["world"], fontsize=8)
    ax2.set_xlabel("false positive rate at 5%"); ax2.set_title("Rejection at the nominal level")
    fig.savefig(C / "synthetic_type1_calibration.png"); plt.close(fig)


def example_histogram():
    """Re-run the single (alpha=1, seed=0) positive-control schedule for the illustrative null."""
    import monte_carlo as mc
    import positive_control_power as pc
    market_df = mc.load_market_data(D / "GC=F_regimes.csv")
    calendar = market_df["Date"].to_numpy()
    open_prices = market_df["Open"].to_numpy(dtype=float)
    d = pc.HOLDING_BARS
    fwd = np.full(len(open_prices), -np.inf)
    valid = np.arange(0, len(open_prices) - d)
    fwd[valid] = open_prices[valid + d] / open_prices[valid] - 1.0
    rng = np.random.default_rng(10_000 + 100 * 97 + 0)  # alpha=1, seed=0
    entry, exit_ = pc.build_skilled_schedule(open_prices, fwd, 1.0, rng)
    actual = pc.realized_cumulative_return(open_prices, entry, exit_)
    trade_df = pd.DataFrame({
        "entry_date": calendar[entry], "exit_date": calendar[exit_],
        "return": pc.POSITION_FRACTION * (open_prices[exit_] / open_prices[entry] - 1.0 - pc.TXN),
        "holding_bars": d, "position_value_fraction": pc.POSITION_FRACTION,
        "direction": "long", "regime_at_entry": "neutral"})
    sim, _ = mc.simulate_agent_null_cumulative_returns(
        agent_name="positive_control_oracle", current_ticker="GC=F", trade_df=trade_df,
        market_df=market_df, input_path=D / "pc.csv", simulation_count=mc.NUMBER_OF_SIMULATIONS,
        rng=mc.build_random_generator(reproducible=True, seed=20_000))
    s = sim.to_numpy(dtype=float)
    p = mc.calculate_p_value(s, actual); z = (actual - s.mean()) / s.std()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(s, bins=60, color=GREY, alpha=0.85, label="structure-preserving null (random placement)")
    ax.axvline(actual, color=RED, lw=2.4, label=f"realized skilled return ($p={p:.4f}$, RCSI$_z={z:.1f}$)")
    ax.set_xlabel("cumulative return"); ax.set_ylabel("simulated schedules")
    ax.set_title(r"A genuinely well-timed strategy ($\alpha=1$) lands far in the null's upper tail")
    ax.legend(loc="upper left")
    fig.savefig(C / "positive_control_example_histogram.png"); plt.close(fig)


if __name__ == "__main__":
    skill_curve(); signal_curve(); cross_asset(); synthetic_power(); synthetic_calibration()
    example_histogram()
    print("restyled: skill_curve, signal_curve, cross_asset, synthetic_power, synthetic_calibration, example_histogram")
