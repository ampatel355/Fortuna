"""Shared publication figure style (serif, clean spines, muted palette)."""

from __future__ import annotations
import matplotlib as mpl

# muted, print-friendly palette
INK = "#1a1a1a"
BLUE = "#2f5e8f"
GREEN = "#2e6f4e"
RED = "#a4303f"
GREY = "#9aa0a6"
GOLD = "#b8860b"
PALETTE = [BLUE, GREEN, GOLD, RED, "#5b507a", GREY]


def use_pub_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": "#444444", "xtick.color": INK, "ytick.color": INK,
        "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
        "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.7,
        "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight",
        "lines.linewidth": 1.9, "lines.markersize": 5.5,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })
