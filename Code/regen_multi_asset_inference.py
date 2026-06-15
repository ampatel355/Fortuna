"""Regenerate the multi-asset walk-forward inference figure for publication.

Identical chart to walk_forward_inference_plot.py but (1) sourced explicitly from
the phase-4 agent summary that backs Table tab:multi_asset_walk_forward in the
paper (verified to match the published values, 712 panels), (2) with the baked-in
caption/Run-ID footer removed (the LaTeX caption carries the explanation), and
(3) with the RCSI_z axis/legend labels typeset with a proper subscript.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pub_style import use_pub_style

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from plot_config import (
    AGENT_COLORS,
    BAR_EDGE_COLOR,
    ZERO_LINE_COLOR,
    apply_categorical_tick_labels,
    apply_clean_style,
    format_agent_name,
    save_chart,
    size_for_categories,
)

PHASE4_CSV = (
    CODE.parent
    / "Data_Clean"
    / "multi_asset_walk_forward_agent_summary__phase4_required_universe_20260526.csv"
)


def main() -> None:
    use_pub_style()
    df = pd.read_csv(PHASE4_CSV)
    df = df.dropna(subset=["mean_mean_actual_percentile", "mean_mean_RCSI_z"]).reset_index(drop=True)
    assert int(df["panel_count"].sum()) == 712, "panel count must match the published 712"

    x = np.arange(len(df))
    bar_colors = [AGENT_COLORS.get(a, "#355C7D") for a in df["agent"]]

    fig, ax = plt.subplots(figsize=size_for_categories(len(df), height=6.2))
    bars = ax.bar(
        x, df["mean_mean_actual_percentile"], color=bar_colors,
        edgecolor=BAR_EDGE_COLOR, linewidth=0.7, alpha=0.82, label="Mean Panel Percentile",
    )
    ax.axhline(50.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--")

    ax2 = ax.twinx()
    ax2.plot(
        x, df["mean_mean_RCSI_z"], color="#111111", linewidth=1.8, marker="o",
        markersize=5.2, label=r"Mean Panel RCSI$_z$",
    )
    ax2.axhline(0.0, color="#6B7280", linewidth=0.9, linestyle=":")

    apply_clean_style(
        ax, title="Multi-Asset Walk-Forward Monte Carlo Inference",
        x_label="Strategy", y_label="Mean Panel Percentile", show_y_grid=True,
    )
    ax2.set_ylabel(r"Mean Panel RCSI$_z$")
    apply_categorical_tick_labels(ax, [format_agent_name(a, short=True) for a in df["agent"]])
    ax.set_xticks(x)
    ax.set_ylim(0.0, max(100.0, float(df["mean_mean_actual_percentile"].max()) * 1.1))
    rcsi_abs_max = max(1.0, float(df["mean_mean_RCSI_z"].abs().max()) * 1.2)
    ax2.set_ylim(-rcsi_abs_max, rcsi_abs_max)

    for bar, p_value in zip(bars, df["mean_mean_p_value"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, float(bar.get_height()) + 1.6,
            f"p={float(p_value):.3f}", ha="center", va="bottom", fontsize=8.0, color="#374151",
        )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    # No add_figure_caption: the LaTeX \caption explains the figure; this removes
    # the duplicated caption and the leaked internal Run ID.
    save_chart(fig, "multi_asset_walk_forward_inference.png")
    print("regenerated Charts/multi_asset_walk_forward_inference.png (no Run ID, RCSI_z subscript)")


if __name__ == "__main__":
    main()
