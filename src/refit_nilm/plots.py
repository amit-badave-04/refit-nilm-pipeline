"""Shared figure style and save helper so every notebook produces consistent figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from . import config as C

PALETTE = {
    "aggregate": "#4C566A",
    "wm": "#D08770",
    "pred": "#5E81AC",
    "alt": "#A3BE8C",
    "alt2": "#B48EAD",
    "flag": "#BF616A",
    "grid": "#E5E9F0",
}


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "figure.figsize": (11, 3.6),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
        }
    )


def save(fig, name: str, folder: Path | None = None) -> Path:
    folder = folder or C.FIG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path
