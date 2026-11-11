"""Shared plotting style and output folders."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

APP_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = APP_DIR.parent / "documentation" / "Fig"
OUT_DIR = APP_DIR / "outputs"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CLASS_COLORS = {"N": "#1f4e79", "S": "#c77c02", "V": "#b85450", "F": "#2e7d32", "Q": "#7f7f7f"}


def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_json(obj, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
