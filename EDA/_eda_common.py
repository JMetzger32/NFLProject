"""Shared EDA infrastructure: headless plotting, output paths, section registry.

Every figure is written to disk (never shown), every figure has a CSV beside it
holding the numbers behind it, and every section's stdout is tee'd to a log.

Colors come from the validated reference palette. Categorical slots are assigned in
fixed order and never cycled; correlation-style plots use the diverging blue<->red
pair with a neutral gray midpoint. Slot 3 (aqua) sits below 3:1 on the light
surface, so charts using it ship direct labels or rely on the CSV table view.

Usage: imported by eda_1.py, eda_2.py, etc. Not run directly. Each driver calls
set_output_dir("EDA_N") once, before running any section, to target its own folder.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved, never displayed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = _BASE_DIR / "EDA_1"
DATA_DIR = OUT_DIR / "data"
LOG_DIR = OUT_DIR / "logs"


def set_output_dir(name: str, base: Path | None = None) -> None:
    """Point savefig/save_table/tee_stdout at <base>/<name>/ instead of the EDA_1
    default. Call once, before running any section — every helper below reads
    these module globals at call time, not at import time.

    `base` defaults to this file's directory (EDA/); the modeling drivers pass
    their own so this stays genuinely shared infrastructure rather than being
    forked per round.
    """
    global OUT_DIR, DATA_DIR, LOG_DIR
    OUT_DIR = (base or _BASE_DIR) / name
    DATA_DIR = OUT_DIR / "data"
    LOG_DIR = OUT_DIR / "logs"

# Reference palette, light surface.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots, fixed order. Never cycle past slot 3 in one chart.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
WIN_COLOR, LOSS_COLOR = SERIES[0], SERIES[1]

# Diverging pair for signed quantities (correlations, effect sizes).
DIVERGING_NEG = "#e34948"
DIVERGING_POS = "#2a78d6"
DIVERGING_MID = "#f0efec"

SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 10,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
    })


def despine(ax) -> None:
    """Recessive chrome: drop the top/right rules, keep a hairline baseline."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)


def new_fig(w: float = 9.0, h: float = 5.0):
    fig, ax = plt.subplots(figsize=(w, h))
    despine(ax)
    return fig, ax


def savefig(fig, name: str) -> str:
    """Save to EDA_1/<name>.png and return the bare filename for markdown embedding."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path.name}")
    return path.name


def save_table(df: pd.DataFrame, name: str) -> str:
    """Every figure ships the numbers behind it - this is also the a11y table view."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  data   -> data/{path.name}")
    return path.name


@contextmanager
def tee_stdout(name: str):
    """Mirror a section's stdout into EDA_1/logs/<name>.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{name}.log"

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    with open(path, "w") as fh:
        original = sys.stdout
        sys.stdout = _Tee(original, fh)
        try:
            yield
        finally:
            sys.stdout = original


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    fn: Callable


_SECTIONS: list[Section] = []


def section(key: str, title: str):
    def deco(fn):
        _SECTIONS.append(Section(key, title, fn))
        return fn

    return deco


def all_sections() -> list[Section]:
    return sorted(_SECTIONS, key=lambda s: s.key)


def run_sections(keys: list[str], ctx) -> list[str]:
    failures = []
    for sec in all_sections():
        if keys and sec.key not in keys:
            continue
        print(f"\n== {sec.key} — {sec.title}")
        start = time.time()
        try:
            with tee_stdout(f"{sec.key}_{_slug(sec.title)}"):
                print(f"== {sec.key} — {sec.title}")
                sec.fn(ctx)
            print(f"  done in {time.time() - start:.1f}s")
        except Exception as exc:  # keep going; one bad section shouldn't kill the run
            failures.append(sec.key)
            print(f"  FAILED: {exc}")
            import traceback

            traceback.print_exc()
    return failures


def _slug(title: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in title]
    return "".join(keep).strip("_").replace("__", "_")[:40]


# ---------------------------------------------------------------- statistics


def auc(scores: pd.Series, labels: pd.Series) -> float:
    """Rank-based AUC. Robust to NaN; returns nan when a class is missing."""
    d = pd.DataFrame({"s": scores, "y": labels}).dropna()
    if d["y"].nunique() < 2:
        return float("nan")
    ranks = d["s"].rank()
    pos, neg = d["y"] == 1, d["y"] == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bh_screen(pvalues: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Benjamini-Hochberg. Testing ~30 features invites false positives."""
    p = pvalues.dropna().sort_values()
    m = len(p)
    if m == 0:
        return pd.Series(dtype=bool)
    thresh = alpha * np.arange(1, m + 1) / m
    passing = p.values <= thresh
    cutoff = np.where(passing)[0].max() + 1 if passing.any() else 0
    keep = pd.Series(False, index=pvalues.index)
    keep.loc[p.index[:cutoff]] = True
    return keep


def binned_rate(
    df: pd.DataFrame, col: str, target: str = "won", bins: int = 10
) -> pd.DataFrame:
    """Decile a feature and report the target rate per bin, with Wilson error bars."""
    d = df[[col, target]].dropna()
    if d.empty:
        return pd.DataFrame()
    d = d.assign(bin=pd.qcut(d[col], bins, duplicates="drop"))
    g = d.groupby("bin", observed=True).agg(
        n=(target, "size"), rate=(target, "mean"), center=(col, "mean")
    ).reset_index()
    # Wilson interval - honest about small bins.
    z = 1.96
    n, p = g["n"], g["rate"]
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    g["lo"], g["hi"] = centre - half, centre + half
    g["bin"] = g["bin"].astype(str)
    return g
