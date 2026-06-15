"""Regenerate the GC=F figure gallery at publication quality from canonical data.

Replaces the supplied raster figures (Figure_1..18) with clean, consistent,
serif-typeset charts produced directly from the canonical trade logs and the
structure-preserving null, using the same seed-42 spawn protocol and default
execution-cost model as the pipeline. The regenerated statistics are checked to
reproduce the published Monte Carlo distribution table; the script aborts on any
material mismatch so the figures can never silently drift from the text.

Outputs (Charts/): mc_<agent>.png x11, rcsi_by_strategy.png, pvalue_by_strategy.png,
regime_heatmap.png, rcsi_robustness.png, percentile_robustness.png, equity_curve.png.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TICKER", "GC=F")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "Code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import monte_carlo as mc
from strategy_config import AGENT_ORDER, AGENT_DISPLAY_NAMES
from pub_style import use_pub_style, BLUE, GREEN, RED, GREY, GOLD

use_pub_style()
D = ROOT / "Data_Clean"
C = ROOT / "Charts"
TICKER = "GC=F"

# published Monte Carlo distribution table (tab:mc) for the faithfulness check
TAB_MC = {  # display: (actual, median, p5, p95, p_value, percentile)
    "Trend + Pullback": (0.0832, 0.4013, -0.0611, 1.0725, 0.8700, 13.0),
    "ADX Trend Following": (0.8032, 0.3705, -0.0928, 1.0660, 0.1414, 85.9),
    "Volatility Squeeze Breakout": (0.0649, 0.0107, -0.0483, 0.0655, 0.0534, 94.7),
    "Connors RSI(2) Pullback": (0.1184, 0.0216, -0.1112, 0.1736, 0.1428, 85.7),
    "Random Baseline": (0.6011, 0.9381, 0.1276, 2.3314, 0.7103, 29.0),
}


def compute():
    market_df = mc.load_market_data(D / f"{TICKER}_regimes.csv")
    children = np.random.SeedSequence(mc.SEED).spawn(len(AGENT_ORDER))
    rows = []
    for agent, child in zip(AGENT_ORDER, children):
        input_path = mc.ensure_trade_file_exists(TICKER, agent)
        trade_df = mc.load_trade_data(input_path, allow_empty=True)
        if trade_df.empty:
            continue
        rng = mc.build_random_generator(reproducible=mc.REPRODUCIBLE,
                                        seed=int(child.generate_state(1, dtype=np.uint64)[0]))
        raw = trade_df["return"].to_numpy(dtype=float)
        adj = mc.adjust_trade_returns(raw, mc.TRANSACTION_COST, input_path)
        actual = mc.calculate_cumulative_return_from_log_returns(mc.convert_to_log_returns(adj))
        sim, _ = mc.simulate_agent_null_cumulative_returns(
            agent_name=agent, current_ticker=TICKER, trade_df=trade_df, market_df=market_df,
            input_path=input_path, simulation_count=mc.NUMBER_OF_SIMULATIONS, rng=rng)
        s = sim.to_numpy(dtype=float)
        rows.append(dict(agent=agent, display=AGENT_DISPLAY_NAMES.get(agent, agent),
                         actual=actual, sim=s, median=float(np.median(s)),
                         p5=float(np.percentile(s, 5)), p95=float(np.percentile(s, 95)),
                         mean=float(s.mean()),
                         p=mc.calculate_p_value(s, actual),
                         pct=mc.calculate_actual_percentile(s, actual),
                         n=int(len(trade_df))))
    return market_df, rows


def verify(rows):
    ok = True
    by = {r["display"]: r for r in rows}
    for disp, (a, med, p5, p95, p, pct) in TAB_MC.items():
        r = by.get(disp)
        if r is None:
            print(f"  MISSING {disp}"); ok = False; continue
        checks = [abs(r["actual"]-a) < 0.005, abs(r["median"]-med) < 0.02,
                  abs(r["p"]-p) < 0.02, abs(r["pct"]-pct) < 1.5]
        tag = "OK " if all(checks) else "XX "
        if not all(checks):
            print(f"  {tag}{disp}: actual {r['actual']:.4f}/{a} median {r['median']:.3f}/{med} "
                  f"p {r['p']:.3f}/{p} pct {r['pct']:.1f}/{pct}")
            ok = False
    print("GALLERY FAITHFULNESS:", "ALL PASS" if ok else "MISMATCH -- aborting figure write")
    return ok


def mc_histograms(rows):
    for r in rows:
        s = r["sim"]
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        ax.hist(s, bins=55, color=GREY, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvspan(r["p5"], r["p95"], color=BLUE, alpha=0.07)
        ax.axvline(r["median"], color="#555555", ls=":", lw=1.2, label="null median")
        col = GREEN if r["actual"] >= r["median"] else RED
        ax.axvline(r["actual"], color=col, lw=2.4,
                   label=f"realized ($p={r['p']:.3f}$, pct {r['pct']:.0f})")
        ax.set_xlabel("cumulative return"); ax.set_ylabel("simulated schedules")
        ax.set_title(f"{r['display']} ({r['n']} trades): realized vs. structure-preserving null")
        ax.legend(loc="upper right", fontsize=8.5)
        fig.savefig(C / f"mc_{r['agent']}.png"); plt.close(fig)


def bars(rows):
    rows_s = rows
    disp = [r["display"] for r in rows_s]
    x = np.arange(len(rows_s))
    # RCSI_z bar
    z = [(r["actual"] - r["mean"]) / r["sim"].std() for r in rows_s]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x, z, color=[GREEN if v >= 0 else RED for v in z], alpha=0.85)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(disp, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(r"standardized conditional excess (RCSI$_z$)")
    ax.set_title("Standardized conditional excess by strategy (GC=F, daily)")
    fig.savefig(C / "rcsi_by_strategy.png"); plt.close(fig)
    # p-value bar
    pv = [r["p"] for r in rows_s]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x, pv, color=BLUE, alpha=0.85)
    ax.axhline(0.05, color=RED, ls="--", lw=1.3, label="$p=0.05$")
    ax.set_xticks(x); ax.set_xticklabels(disp, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("one-sided Monte Carlo $p$-value")
    ax.set_title("Entry-placement $p$-values by strategy (none crosses 0.05)")
    ax.legend()
    fig.savefig(C / "pvalue_by_strategy.png"); plt.close(fig)


def robustness():
    f = D / "GC=F_monte_carlo_robustness_summary.csv"
    if not f.exists():
        print("  (robustness summary CSV absent; skipping robustness charts)"); return
    rb = pd.read_csv(f)
    name_col = "agent" if "agent" in rb.columns else rb.columns[0]
    def col(*cands):
        for c in cands:
            if c in rb.columns:
                return c
        return None
    rcsi_m = col("mean_RCSI", "RCSI_mean", "rcsi_mean")
    rcsi_s = col("std_RCSI", "RCSI_std", "rcsi_std")
    pct_m = col("mean_actual_percentile", "percentile_mean", "mean_percentile")
    pct_s = col("std_actual_percentile", "percentile_std", "std_percentile")
    disp = [AGENT_DISPLAY_NAMES.get(a, a) for a in rb[name_col]]
    x = np.arange(len(rb))
    if rcsi_m and rcsi_s:
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        ax.bar(x, rb[rcsi_m], yerr=rb[rcsi_s], color=BLUE, alpha=0.85,
               error_kw=dict(ecolor="#444", lw=1, capsize=3))
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(disp, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(r"conditional excess return (mean $\pm$ SD, 100 seeds)")
        ax.set_title("Seed robustness of the conditional excess return")
        fig.savefig(C / "rcsi_robustness.png"); plt.close(fig)
    if pct_m and pct_s:
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        ax.bar(x, rb[pct_m], yerr=rb[pct_s], color=GREEN, alpha=0.85,
               error_kw=dict(ecolor="#444", lw=1, capsize=3))
        ax.axhline(50, color=GREY, ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(disp, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(r"null percentile (mean $\pm$ SD, 100 seeds)")
        ax.set_title("Seed robustness of the null percentile")
        fig.savefig(C / "percentile_robustness.png"); plt.close(fig)


def regime_heatmap():
    """Trade-level return ratio (mean/std) per strategy x regime, from the trade logs."""
    order = ["calm", "neutral", "stressed"]
    data = {}
    counts = {}
    for agent in AGENT_ORDER:
        try:
            tp = mc.ensure_trade_file_exists(TICKER, agent)
            td = mc.load_trade_data(tp, allow_empty=True)
        except Exception:
            continue
        if td.empty or "regime_at_entry" not in td.columns:
            continue
        disp = AGENT_DISPLAY_NAMES.get(agent, agent)
        row = []; crow = []
        for reg in order:
            sub = td[td["regime_at_entry"].astype(str).str.lower() == reg]["return"].to_numpy(float)
            if len(sub) >= 5 and sub.std() > 0:
                row.append(sub.mean() / sub.std()); crow.append(len(sub))
            else:
                row.append(np.nan); crow.append(len(sub))
        data[disp] = row; counts[disp] = crow
    if not data:
        return
    M = np.array(list(data.values()))
    labels = list(data.keys())
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    vmax = np.nanmax(np.abs(M))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([r.capitalize() for r in order])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5)
    for i in range(len(labels)):
        for j in range(3):
            v = M[i, j]
            txt = "—" if np.isnan(v) else f"{v:.2f}\n(n={counts[labels[i]][j]})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="#111" if (np.isnan(v) or abs(v) < 0.6*vmax) else "white")
    ax.set_title("Trade-level return ratio by strategy and regime")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean / std per trade")
    fig.savefig(C / "regime_heatmap.png"); plt.close(fig)


def equity_curve(market_df):
    w = market_df[(market_df["Date"] >= pd.Timestamp("2002-03-04")) &
                  (market_df["Date"] <= pd.Timestamp("2026-04-02"))]
    px = w["Close"].to_numpy(float)
    bh = px / px[0]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(w["Date"], bh, color=GOLD, lw=1.6, label="GC=F buy-and-hold (close)")
    ax.set_ylabel("growth of \\$1"); ax.set_xlabel("")
    ax.set_title("GC=F price path, 2002--2026 (the fixed path all schedules are priced on)")
    ax.legend()
    fig.savefig(C / "equity_curve.png"); plt.close(fig)


def main():
    market_df, rows = compute()
    print("=== per-strategy stats (regenerated) ===")
    for r in rows:
        print(f"  {r['display']:<32} actual {r['actual']:+.4f}  median {r['median']:+.4f}  "
              f"p {r['p']:.4f}  pct {r['pct']:.1f}  n {r['n']}")
    if not verify(rows):
        sys.exit("Faithfulness check failed; not writing figures.")
    mc_histograms(rows); bars(rows); robustness(); regime_heatmap(); equity_curve(market_df)
    print(f"wrote {len(rows)} MC histograms + bars + robustness + heatmap + equity curve to Charts/")


if __name__ == "__main__":
    main()
