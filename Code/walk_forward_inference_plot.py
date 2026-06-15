"""Display current-run multi-asset Monte Carlo inference comparisons."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from plot_config import (
    AGENT_COLORS,
    BAR_EDGE_COLOR,
    ZERO_LINE_COLOR,
    add_figure_caption,
    apply_categorical_tick_labels,
    apply_clean_style,
    format_agent_name,
    show_chart,
    save_chart,
    size_for_categories,
)

try:
    from walk_forward_plot_utils import (
        load_walk_forward_agent_summary,
        walk_forward_metadata,
    )
except ModuleNotFoundError:
    from Code.walk_forward_plot_utils import (
        load_walk_forward_agent_summary,
        walk_forward_metadata,
    )


def main() -> None:
    """Plot mean percentile and RCSI_z for the current walk-forward run."""
    summary_df, summary_path = load_walk_forward_agent_summary()
    metadata = walk_forward_metadata(summary_path)
    run_id = str(metadata.get("run_id", "")).strip()

    df = summary_df.copy()
    df = df.dropna(subset=["mean_mean_actual_percentile", "mean_mean_RCSI_z"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable walk-forward inference rows were available for plotting.")

    x_positions = np.arange(len(df))
    bar_colors = [AGENT_COLORS.get(agent, "#355C7D") for agent in df["agent"]]

    fig, ax = plt.subplots(figsize=size_for_categories(len(df), height=6.2))
    bars = ax.bar(
        x_positions,
        df["mean_mean_actual_percentile"],
        color=bar_colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.7,
        alpha=0.82,
        label="Mean Panel Percentile",
    )
    ax.axhline(50.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--")

    ax2 = ax.twinx()
    ax2.plot(
        x_positions,
        df["mean_mean_RCSI_z"],
        color="#111111",
        linewidth=1.8,
        marker="o",
        markersize=5.2,
        label="Mean Panel RCSI_z",
    )
    ax2.axhline(0.0, color="#6B7280", linewidth=0.9, linestyle=":")

    apply_clean_style(
        ax,
        title="Multi-Asset Walk-Forward Monte Carlo Inference",
        x_label="Strategy",
        y_label="Mean Panel Percentile",
        show_y_grid=True,
    )
    ax2.set_ylabel("Mean Panel RCSI_z")
    apply_categorical_tick_labels(
        ax,
        [format_agent_name(agent, short=True) for agent in df["agent"]],
    )
    ax.set_xticks(x_positions)
    ax.set_ylim(0.0, max(100.0, float(df["mean_mean_actual_percentile"].max()) * 1.1))

    rcsi_abs_max = max(1.0, float(df["mean_mean_RCSI_z"].abs().max()) * 1.2)
    ax2.set_ylim(-rcsi_abs_max, rcsi_abs_max)

    for bar, p_value in zip(bars, df["mean_mean_p_value"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(bar.get_height()) + 1.6,
            f"p={float(p_value):.3f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#374151",
        )

    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
    )

    add_figure_caption(
        fig,
        (
            "Bars show the mean panel percentile from the structure-preserving Monte Carlo null, "
            "while the black line shows mean panel RCSI_z. "
            "Text labels show mean panel p-values for the current run. "
            f"Run ID: {run_id}."
        ),
    )
    save_chart(fig, "multi_asset_walk_forward_inference.png")
    show_chart()


if __name__ == "__main__":
    main()
