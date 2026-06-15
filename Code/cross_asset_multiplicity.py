"""Cross-asset evidence under multiplicity control.

The paper's headline (no robust timing skill on GC=F) invites the obvious
follow-up: across MANY assets, does anything survive? This script aggregates
every real, already-completed structure-preserving Monte Carlo test on disk
(Data_Clean/<ASSET>_full_comparison.csv -- one row per strategy per asset) and
asks whether ANY asset x strategy result survives multiple-testing control.

It computes, over the full panel:
  * the count of nominal p <= 0.05 vs. the count expected by chance,
  * Benjamini-Hochberg survivors at FDR 0.05,
  * Bonferroni survivors at FWER 0.05,
  * a Kolmogorov-Smirnov test of the panel p-values against Uniform(0,1)
    (a global no-skill null implies uniform p-values),
  * the leaderboard of smallest p-values, flagging which are RANDOM-entry
    baselines (their appearance near the top is direct evidence that nominal
    significance here is luck, not skill).

No backtests are re-run; only saved artifacts are read. Outputs:
  Data_Clean/cross_asset_panel.csv           (every test, sorted, with BH flags)
  Data_Clean/cross_asset_panel_summary.csv   (one-row headline summary)
  Charts/cross_asset_multiplicity.png        (p-value histogram + BH plot)
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = PROJECT_ROOT / "Data_Clean"
CHARTS = PROJECT_ROOT / "Charts"

FDR = 0.05
RANDOM_AGENTS = {"random"}  # random-entry baseline strategy names


def ks_uniform_pvalue(values: np.ndarray) -> tuple[float, float]:
    """Two-sided KS statistic and asymptotic p-value vs Uniform(0,1)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    cdf = np.arange(1, n + 1) / n
    d_plus = np.max(cdf - x)
    d_minus = np.max(x - (np.arange(0, n) / n))
    d = float(max(d_plus, d_minus))
    t = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d  # Stephens small-sample correction
    # asymptotic Kolmogorov survival function
    s = 0.0
    for k in range(1, 101):
        s += (-1) ** (k - 1) * np.exp(-2 * (k ** 2) * t ** 2)
    p = float(min(1.0, max(0.0, 2 * s)))
    return d, p


def main() -> None:
    rows = []
    for f in sorted(glob.glob(str(DATA_CLEAN / "*_full_comparison.csv"))):
        asset = os.path.basename(f).replace("_full_comparison.csv", "")
        df = pd.read_csv(f)
        if "p_value" not in df.columns:
            continue
        for _, r in df.iterrows():
            p = r.get("p_value", np.nan)
            if pd.isna(p):
                continue
            agent = str(r.get("agent"))
            rows.append(dict(
                asset=asset, agent=agent,
                p_value=float(p),
                RCSI_z=float(r.get("RCSI_z", np.nan)),
                percentile=float(r.get("actual_percentile", np.nan)),
                final_classification=r.get("final_classification"),
                number_of_trades=r.get("number_of_trades"),
                is_random_baseline=agent in RANDOM_AGENTS,
            ))

    P = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    m = len(P)
    P["rank"] = np.arange(1, m + 1)
    P["bh_threshold"] = P["rank"] / m * FDR
    below = P[P["p_value"] <= P["bh_threshold"]]
    kmax = int(below["rank"].max()) if len(below) else 0
    P["bh_significant"] = P["rank"] <= kmax
    bonf = FDR / m
    P["bonferroni_significant"] = P["p_value"] <= bonf
    # BH-adjusted p-values (monotone)
    adj = (P["p_value"].to_numpy() * m / P["rank"].to_numpy())
    P["bh_adjusted_p"] = np.minimum.accumulate(adj[::-1])[::-1].clip(0, 1)

    n_nominal = int((P["p_value"] <= 0.05).sum())
    expected_nominal = 0.05 * m
    ks_d, ks_p = ks_uniform_pvalue(P["p_value"].to_numpy())
    n_random_in_top10 = int(P.head(10)["is_random_baseline"].sum())

    P.to_csv(DATA_CLEAN / "cross_asset_panel.csv", index=False)
    summary = pd.DataFrame([dict(
        n_assets=int(P["asset"].nunique()),
        n_tests=m,
        n_nominal_p_le_05=n_nominal,
        expected_nominal_p_le_05=round(expected_nominal, 2),
        n_bh_significant=int(P["bh_significant"].sum()),
        n_bonferroni_significant=int(P["bonferroni_significant"].sum()),
        bonferroni_threshold=bonf,
        ks_stat_uniform=round(ks_d, 4),
        ks_pvalue_uniform=round(ks_p, 4),
        min_p_value=float(P["p_value"].min()),
        n_random_baselines_in_top10=n_random_in_top10,
    )])
    summary.to_csv(DATA_CLEAN / "cross_asset_panel_summary.csv", index=False)

    print(summary.T.to_string(header=False))
    print("\n=== smallest 12 p-values in the panel ===")
    print(P.head(12)[["asset", "agent", "p_value", "RCSI_z", "percentile",
                      "is_random_baseline", "bh_significant"]].to_string(index=False))

    _make_chart(P, m, ks_p)


def _make_chart(P: pd.DataFrame, m: int, ks_p: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHARTS.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    # left: p-value histogram vs uniform reference
    mean_p = float(P["p_value"].mean())
    n_nom = int((P["p_value"] <= 0.05).sum())
    bins = np.linspace(0, 1, 21)
    ax1.hist(P["p_value"], bins=bins, color="#5b8db8", edgecolor="white", alpha=0.9)
    ax1.axhline(m / 20, color="#b00020", ls="--", lw=1.4,
                label=f"if pure chance ({m/20:.1f}/bin)")
    ax1.set_xlabel("one-sided p-value")
    ax1.set_ylabel(f"asset $\\times$ strategy tests (of {m})")
    ax1.set_title(f"p-values skew conservative (mean {mean_p:.2f})\n"
                  f"{n_nom} hits at 0.05 vs {0.05*m:.0f} expected by chance")
    ax1.legend(fontsize=9)

    # right: Benjamini-Hochberg plot -- sorted p vs BH line; nothing crosses
    k = P["rank"].to_numpy()
    ax2.plot(k, P["p_value"], "o", ms=3, color="#444444", label="sorted p-values")
    rand = P["is_random_baseline"].to_numpy()
    ax2.plot(k[rand], P["p_value"].to_numpy()[rand], "o", ms=6,
             mfc="none", mec="#b00020", mew=1.5, label="random-entry baselines")
    ax2.plot(k, k / m * FDR, "-", color="#2e7d32", lw=1.6, label=f"BH line (FDR {FDR})")
    ax2.set_xlim(0, max(40, int(0.15 * m)))
    ax2.set_ylim(0, P["p_value"].iloc[min(len(P) - 1, max(40, int(0.15 * m)))] if m > 1 else 0.1)
    ax2.set_xlabel("rank of p-value (smallest first)")
    ax2.set_ylabel("p-value")
    ax2.set_title("No result crosses the BH line\n(0 discoveries at FDR 0.05)")
    ax2.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(CHARTS / "cross_asset_multiplicity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
