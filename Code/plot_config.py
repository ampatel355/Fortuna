"""Shared professional styling helpers for all project charts."""

import os
from pathlib import Path
import textwrap
import warnings


# Every script reads the active ticker from the same environment variable.
ticker = os.environ.get("TICKER", "SPY")

# Matplotlib writes small config files when it starts.
# This keeps those files inside the project instead of a system folder.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib"),
)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd

try:
    from strategy_config import AGENT_COLORS, AGENT_DISPLAY_NAMES, AGENT_ORDER, format_strategy_name
    from timeframe_config import RESEARCH_TIMEFRAME_LABEL, timeframe_output_suffix, timeframe_title_suffix
except ModuleNotFoundError:
    from Code.strategy_config import AGENT_COLORS, AGENT_DISPLAY_NAMES, AGENT_ORDER, format_strategy_name
    from Code.timeframe_config import RESEARCH_TIMEFRAME_LABEL, timeframe_output_suffix, timeframe_title_suffix


# The master pipeline can turn off interactive popups while charts are being built.
show_plots = os.environ.get("SHOW_PLOTS", "1") == "1"
save_outputs = os.environ.get("SAVE_OUTPUTS", "0") == "1"

# Use a clean technical sans-serif style for better screen readability.
TECHNICAL_SANS_FONTS = [
    "DejaVu Sans",
    "Arial",
    "Helvetica",
    "Liberation Sans",
    "sans-serif",
]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": TECHNICAL_SANS_FONTS,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "axes.edgecolor": "#2E2E2E",
        "axes.linewidth": 1.0,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "grid.color": "#C9D2DC",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.5,
        "axes.titlepad": 8,
    }
)


# Shared figure sizing and text scale.
#
# We keep figures closer to square because the user wants paper-style visuals
# rather than stretched dashboard panels.
DEFAULT_FIGSIZE = (7.2, 7.2)
TITLE_SIZE = 12.4
SUBTITLE_SIZE = 8.6
LABEL_SIZE = 10.4
TICK_SIZE = 9.0
LEGEND_SIZE = 8.4
ANNOTATION_SIZE = 8.6
TABLE_HEADER_SIZE = 9.5
TABLE_BODY_SIZE = 9.0

# Professional high-contrast defaults.
REGIME_ORDER = ["calm", "neutral", "stressed"]
REGIME_DISPLAY_NAMES = ["Calm", "Neutral", "Stressed"]

ACTUAL_LINE_COLOR = "#D62728"
MEDIAN_LINE_COLOR = "#1F77B4"
ZERO_LINE_COLOR = "#4F4F4F"
GRID_COLOR = "#CCD5DF"
TEXT_COLOR = "#222222"
SPINE_COLOR = "#2E2E2E"
BACKGROUND_COLOR = "#FFFFFF"
NOTE_BOX_FACE_COLOR = "#FFFFFF"
BAR_EDGE_COLOR = "#2E2E2E"
CAPTION_COLOR = "#3C4A59"

# Diverging palette for heatmaps.
HEATMAP_COLORS = [
    "#2166AC",
    "#67A9CF",
    "#F7F7F7",
    "#EF8A62",
    "#B2182B",
]


def project_root() -> Path:
    """Return the root folder of the project."""
    return Path(__file__).resolve().parents[1]


def resolve_named_dir(lowercase_name: str, uppercase_name: str) -> Path:
    """Return a project folder, supporting either lower- or upper-case names."""
    root = project_root()
    uppercase_dir = root / uppercase_name
    lowercase_dir = root / lowercase_name

    if uppercase_dir.exists():
        return uppercase_dir
    if lowercase_dir.exists():
        return lowercase_dir

    uppercase_dir.mkdir(parents=True, exist_ok=True)
    return uppercase_dir


def data_clean_dir() -> Path:
    """Return the clean-data folder for the active timeframe.

    On daily data (the default) this returns ``Data_Clean`` for backward
    compatibility.  On other timeframes it returns e.g. ``Data_Clean_1h``
    so artifacts from different intervals never collide.
    """
    suffix = timeframe_output_suffix()
    return resolve_named_dir(f"data_clean{suffix}", f"Data_Clean{suffix}")


def charts_dir() -> Path:
    """Return the charts folder for the active timeframe."""
    suffix = timeframe_output_suffix()
    return resolve_named_dir(f"charts{suffix}", f"Charts{suffix}")


def format_agent_name(agent_name: str, short: bool = True) -> str:
    """Convert an internal agent name into a cleaner label for charts and UI text."""
    return format_strategy_name(agent_name, short=short)


def size_for_categories(
    count: int,
    *,
    min_width: float = 6.8,
    max_width: float = 8.6,
    height: float = 7.2,
    width_per_category: float = 0.44,
) -> tuple[float, float]:
    """Return a compact square-ish size for bar charts with many categories.

    The helper caps width so the figure remains paper-like instead of becoming
    an extremely wide dashboard panel.
    """
    width = min(max(min_width, 4.9 + count * width_per_category), max_width)
    side = max(width, height)
    return (side, side)


def size_for_heatmap(
    row_count: int,
    column_count: int,
    *,
    min_width: float = 7.0,
    max_width: float = 8.8,
    min_height: float = 7.0,
    max_height: float = 8.8,
) -> tuple[float, float]:
    """Return a balanced square-ish heatmap size that stays readable in print."""
    width = min(max(min_width, 4.8 + column_count * 0.65), max_width)
    height = min(max(min_height, 3.8 + row_count * 0.48), max_height)
    side = max(width, height)
    return (side, side)


def apply_categorical_tick_labels(
    ax,
    labels: list[str],
    *,
    axis: str = "x",
    fontsize: float | None = None,
) -> None:
    """Apply readable category labels with automatic rotation when names are long.

    The newer strategy names are more descriptive, which is useful in tables and
    papers, but can crowd smaller bar charts. This helper keeps saved figures and
    embedded app charts readable without forcing each plotting script to hand-tune
    the same rotation logic.
    """
    labels = [str(label) for label in labels]
    label_count = len(labels)
    longest_label = max((len(label) for label in labels), default=0)
    use_rotation = label_count >= 5 or longest_label >= 14
    rotation = 24 if use_rotation else 0
    horizontal_alignment = "right" if use_rotation else "center"
    resolved_fontsize = fontsize if fontsize is not None else (TICK_SIZE - 0.4 if use_rotation else TICK_SIZE)

    if axis == "y":
        ax.set_yticklabels(labels, fontsize=resolved_fontsize)
        for label in ax.get_yticklabels():
            label.set_rotation(rotation if label_count >= 6 or longest_label >= 16 else 0)
            label.set_ha("right" if label.get_rotation() else "right")
            label.set_va("center")
        return

    ax.set_xticklabels(labels, fontsize=resolved_fontsize)
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha(horizontal_alignment)
        label.set_va("top" if use_rotation else "center_baseline")


def load_csv_checked(input_path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a CSV file and confirm that it contains the needed columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"The input file is empty: {input_path}")

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns in {input_path}: {', '.join(missing_columns)}"
        )

    return df


def lighten_color(color: str, amount: float = 0.35) -> str:
    """Create a lighter shade of a color for fills and histogram bars."""
    red, green, blue = mcolors.to_rgb(color)
    new_red = 1 - (1 - red) * (1 - amount)
    new_green = 1 - (1 - green) * (1 - amount)
    new_blue = 1 - (1 - blue) * (1 - amount)
    return mcolors.to_hex((new_red, new_green, new_blue))


def histogram_color(agent_name: str) -> str:
    """Return a lighter histogram color for one strategy."""
    return lighten_color(AGENT_COLORS.get(agent_name, "#355C7D"), amount=0.22)


def fill_color(agent_name: str) -> str:
    """Return a lighter fill color for equity-curve shading."""
    return lighten_color(AGENT_COLORS.get(agent_name, "#355C7D"), amount=0.55)


def apply_bar_style(bars, linewidth: float = 0.6) -> None:
    """Give bar charts one consistent academic edge treatment."""
    for bar in bars:
        bar.set_edgecolor(BAR_EDGE_COLOR)
        bar.set_linewidth(linewidth)


def emphasize_tiny_bars(
    ax,
    bars,
    values,
    min_fraction_of_axis: float = 0.018,
) -> None:
    """Draw a thin cap line on near-zero bars so they remain visible."""
    y_min, y_max = ax.get_ylim()
    axis_span = abs(y_max - y_min)
    visibility_threshold = axis_span * min_fraction_of_axis

    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue

        numeric_value = float(value)
        if abs(numeric_value) >= visibility_threshold:
            continue

        ax.hlines(
            y=numeric_value,
            xmin=bar.get_x(),
            xmax=bar.get_x() + bar.get_width(),
            colors=[bar.get_facecolor()],
            linewidth=2.4,
            zorder=max(bar.get_zorder() + 0.5, 4),
        )


def format_large_number(value: float) -> str:
    """Format a numeric value in a readable way for axes and notes."""
    absolute_value = abs(value)

    if absolute_value < 1e-12:
        return "0"
    if absolute_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    if absolute_value >= 10:
        return f"{value:,.1f}"
    if absolute_value >= 1:
        return f"{value:,.2f}"
    if absolute_value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.4f}"


def format_precise_value(value: float) -> str:
    """Format value labels with a little more precision than axis ticks."""
    absolute_value = abs(value)

    if absolute_value >= 100:
        return f"{value:,.2f}"
    if absolute_value >= 1:
        return f"{value:,.2f}"
    if absolute_value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.4f}"


def axis_number_formatter():
    """Return a compact axis formatter."""
    return FuncFormatter(lambda value, _: format_large_number(value))


def apply_axis_number_format(ax, axis: str = "y") -> None:
    """Apply the shared numeric formatter to one axis."""
    formatter = axis_number_formatter()

    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)


def apply_clean_style(
    ax,
    title: str,
    x_label: str,
    y_label: str,
    show_y_grid: bool = True,
    add_legend: bool = False,
    legend_location: str = "upper center",
    legend_ncol: int = 1,
    legend_outside: bool = False,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    """Apply one consistent academic style to a matplotlib axis."""
    full_title = title
    if timeframe_title_suffix() not in full_title:
        full_title = f"{title} {timeframe_title_suffix()}"
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.figure.set_facecolor(BACKGROUND_COLOR)
    ax.set_title(
        full_title,
        fontsize=TITLE_SIZE,
        fontweight="normal",
        color=TEXT_COLOR,
        y=1.02,
        pad=0,
    )
    ax.title.set_wrap(True)
    ax.set_xlabel(x_label, fontsize=LABEL_SIZE, color=TEXT_COLOR)
    ax.set_ylabel(y_label, fontsize=LABEL_SIZE, color=TEXT_COLOR)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, colors=TEXT_COLOR)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

    if show_y_grid:
        ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.35, color=GRID_COLOR)
        ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    if add_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if legend_outside and legend_bbox_to_anchor is None:
                legend_bbox_to_anchor = (0.5, -0.18)
            ax.legend(
                handles,
                labels,
                frameon=False,
                fontsize=LEGEND_SIZE,
                loc=legend_location,
                ncol=legend_ncol,
                handlelength=1.8,
                borderaxespad=0.4,
                bbox_to_anchor=legend_bbox_to_anchor,
                columnspacing=1.0,
            )


def add_note_box(
    ax,
    text: str,
    x: float = 0.02,
    y: float = 0.98,
    ha: str = "left",
    va: str = "top",
) -> None:
    """Backward-compatible helper that now routes notes below the figure.

    The project used to place boxed text on top of the chart area. We keep the
    helper name to avoid breaking older scripts, but the visual treatment is now
    a clean caption below the figure instead of an overlay.
    """
    add_figure_caption(ax.figure, text)


def add_subtitle(ax, text: str) -> None:
    """Add a small subtitle-like note just below the title."""
    ax.text(
        0.5,
        1.002,
        text,
        transform=ax.transAxes,
        fontsize=SUBTITLE_SIZE,
        ha="center",
        va="bottom",
        color=CAPTION_COLOR,
    )


def add_figure_caption(
    fig,
    text: str,
    *,
    y: float = 0.02,
    x: float = 0.02,
    width: int = 118,
) -> None:
    """Add a compact caption below a chart instead of inside the plot area."""
    wrapped_text = textwrap.fill(" ".join(str(text).split()), width=width)
    setattr(fig, "_has_chart_caption", True)
    fig.text(
        x,
        y,
        wrapped_text,
        fontsize=SUBTITLE_SIZE,
        color=CAPTION_COLOR,
        ha="left",
        va="bottom",
    )


def create_placeholder_chart(
    title: str,
    output_filename: str,
    message: str,
    subtitle: str | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Path:
    """Create a clean placeholder chart when a figure has no valid data to plot."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.axis("off")

    ax.text(
        0.5,
        0.72,
        title,
        transform=ax.transAxes,
        fontsize=TITLE_SIZE,
        fontweight="normal",
        color=TEXT_COLOR,
        ha="center",
        va="center",
    )

    if subtitle:
        ax.text(
            0.5,
            0.62,
            subtitle,
            transform=ax.transAxes,
            fontsize=SUBTITLE_SIZE,
            color="#4B5563",
            ha="center",
            va="center",
        )

    ax.text(
        0.5,
        0.42,
        message,
        transform=ax.transAxes,
        fontsize=LABEL_SIZE,
        color=TEXT_COLOR,
        ha="center",
        va="center",
        linespacing=1.5,
    )

    output_path = save_chart(fig, output_filename)
    show_chart()
    return output_path


def save_chart(fig, filename: str) -> Path:
    """Save a chart when output persistence is enabled.

    By default the research runner now shows figures during interactive runs
    without writing hundreds of PNG files to disk. Saving is opt-in through
    SAVE_OUTPUTS=1.
    """
    output_path = charts_dir() / filename
    caption_present = bool(getattr(fig, "_has_chart_caption", False))
    bottom_margin = 0.15 if caption_present else 0.09

    for axis in fig.axes:
        if hasattr(axis, "set_anchor"):
            axis.set_anchor("C")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Keep content centered and reserve only the needed caption space.
        fig.tight_layout(rect=(0.07, bottom_margin, 0.97, 0.95), pad=1.0)
    if save_outputs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    return output_path


def show_chart() -> None:
    """Display the figure for interactive runs.

    These plot scripts run as short-lived subprocesses inside the workflow
    runner. A non-blocking ``plt.show(block=False)`` returns immediately and the
    subprocess exits, which closes the window before the user can see it. Using
    the default blocking show keeps the figure open until the user closes it,
    which is the expected behavior for an interactive research run.
    """
    if not show_plots:
        plt.close("all")
        return

    plt.show()
