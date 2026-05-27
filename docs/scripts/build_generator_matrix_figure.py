"""Build the SynHydro generator landscape comparison figure.

Renders a 2D grid of all stochastic generators with a nested method-family
classification on the Y axis and timescale on the X axis. The Y axis carries
three super-groups (Parametric / Hybrid / Non-parametric) and five sub-classes
(AR-family / HMM / Spectral / Bootstrap / k-NN). Super-group boundaries are
drawn as soft fades to convey the continuous mathematical mixing of methods.
Per-pill color bars encode single-site vs. multi-site capability. Generators
that natively support multiple timescales (ARFIMA, KNN-Bootstrap) are drawn as
single pills spanning the supported columns.

Saves vector (SVG) and raster (PNG, 300 dpi) outputs to docs/assets/images/.

Usage:
    python scripts/build_generator_matrix_figure.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from synhydro.plotting.config import COLORS, apply_plotting_style

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "assets" / "images"

COL_NAMES: Tuple[str, ...] = ("Daily", "Weekly", "Monthly", "Annual")
COL_CENTERS: Dict[str, float] = {
    name: (i + 0.5) / len(COL_NAMES) for i, name in enumerate(COL_NAMES)
}
COL_BOUNDS: Tuple[float, ...] = tuple(
    i / len(COL_NAMES) for i in range(len(COL_NAMES) + 1)
)
COL_INDEX: Dict[str, int] = {name: i for i, name in enumerate(COL_NAMES)}

ROW_ORDER: Tuple[str, ...] = ("AR", "HMM", "Spectral", "Bootstrap", "k-NN")
ROW_SLOTS: Dict[str, int] = {
    "AR": 4,
    "HMM": 1,
    "Spectral": 2,
    "Bootstrap": 1,
    "k-NN": 1,
}
ROW_WEIGHTS: Dict[str, float] = {
    "AR": 2.4,
    "HMM": 1.0,
    "Spectral": 1.5,
    "Bootstrap": 1.0,
    "k-NN": 1.25,
}
ROW_SUPER: Dict[str, str] = {
    "AR": "Parametric",
    "HMM": "Parametric",
    "Spectral": "Hybrid",
    "Bootstrap": "Hybrid",
    "k-NN": "Non-param.",
}
SUPER_ORDER: Tuple[str, ...] = ("Parametric", "Hybrid", "Non-param.")

PILL_HEIGHT = 0.058
PILL_HALF_WIDTH = 0.118
PILL_VGAP = 0.014
SPAN_INSET = 0.018
BAR_HEIGHT_FRAC = 0.22
PILL_LABEL_FONTSIZE = 7.5

COLOR_SINGLE = COLORS["observed"]
COLOR_MULTI = COLORS["ensemble_median"]
COLOR_EDGE = COLORS["observed"]
COLOR_TEXT = COLORS["observed"]
COLOR_DIVIDER = COLORS["grid"]


@dataclass
class GenSpec:
    """Placement specification for a single generator."""

    label: str
    row: str
    slot: float
    sites: str
    col: Optional[str] = None
    span: Optional[Tuple[str, str]] = None
    fontsize: Optional[float] = None


GENERATORS: Tuple[GenSpec, ...] = (
    GenSpec("ARFIMA", "AR", 0, "single", span=("Monthly", "Annual")),
    GenSpec("ThomasFiering", "AR", 1, "single", col="Monthly"),
    GenSpec("Matalas", "AR", 2, "multi", col="Monthly"),
    GenSpec("SMARTA", "AR", 2, "multi", col="Annual"),
    GenSpec("SPARTA", "AR", 3, "multi", col="Monthly"),
    GenSpec("MultiSiteHMM", "HMM", 0, "multi", col="Annual"),
    GenSpec("PhaseRandomization", "Spectral", 0, "single", col="Daily"),
    GenSpec(
        "MultisitePhaseRandomization", "Spectral", 1, "multi", col="Daily", fontsize=6
    ),
    GenSpec("WARM", "Spectral", 0.5, "single", col="Annual"),
    GenSpec("Kirsch", "Bootstrap", 0, "multi", span=("Weekly", "Monthly")),
    GenSpec("KNNBootstrap", "k-NN", 0, "multi", span=("Monthly", "Annual")),
)


def _row_bounds() -> Dict[str, Tuple[float, float]]:
    """Return ``(y0, y1)`` for each row in axis-data coordinates."""
    total = sum(ROW_WEIGHTS[r] for r in ROW_ORDER)
    cum = 0.0
    bounds: Dict[str, Tuple[float, float]] = {}
    for name in ROW_ORDER:
        h = ROW_WEIGHTS[name] / total
        y1 = 1.0 - cum
        y0 = y1 - h
        bounds[name] = (y0, y1)
        cum += h
    return bounds


def _super_bounds(
    row_bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    """Aggregate row bounds into super-group bounds."""
    bounds: Dict[str, Tuple[float, float]] = {}
    for super_name in SUPER_ORDER:
        rows = [r for r in ROW_ORDER if ROW_SUPER[r] == super_name]
        y_top = max(row_bounds[r][1] for r in rows)
        y_bot = min(row_bounds[r][0] for r in rows)
        bounds[super_name] = (y_bot, y_top)
    return bounds


def _slot_y_center(row_y0: float, row_y1: float, slot: float, n_slots: int) -> float:
    """Compute the y-center for a slot inside a row.

    ``slot`` is 0-indexed (top to bottom) and may be fractional to place a
    pill between two integer slots.
    """
    total_pill = n_slots * PILL_HEIGHT + max(n_slots - 1, 0) * PILL_VGAP
    free = (row_y1 - row_y0) - total_pill
    top = row_y1 - free / 2.0
    return top - PILL_HEIGHT / 2.0 - slot * (PILL_HEIGHT + PILL_VGAP)


def _pill_x_extent(spec: GenSpec) -> Tuple[float, float]:
    """Return ``(x_left, x_right)`` for a pill, handling spanning."""
    if spec.span is not None:
        first, last = spec.span
        i0 = COL_INDEX[first]
        i1 = COL_INDEX[last]
        x_left = COL_BOUNDS[i0] + SPAN_INSET
        x_right = COL_BOUNDS[i1 + 1] - SPAN_INSET
        return x_left, x_right
    if spec.col is None:
        raise ValueError(f"GenSpec {spec.label!r} has neither col nor span")
    center = COL_CENTERS[spec.col]
    return center - PILL_HALF_WIDTH, center + PILL_HALF_WIDTH


def _draw_pill(
    ax: plt.Axes,
    x_left: float,
    x_right: float,
    y_center: float,
    label: str,
    sites: str,
    fontsize: float,
) -> None:
    """Render a pill and its embedded site color bar."""
    width = x_right - x_left
    y0 = y_center - PILL_HEIGHT / 2.0

    pill = FancyBboxPatch(
        (x_left, y0),
        width,
        PILL_HEIGHT,
        boxstyle="round,pad=0,rounding_size=0.014",
        linewidth=1.0,
        edgecolor=COLOR_EDGE,
        facecolor="white",
        zorder=3,
        clip_on=False,
    )
    ax.add_patch(pill)

    bar_h = PILL_HEIGHT * BAR_HEIGHT_FRAC
    bar_inset = 0.0040
    bar_x = x_left + bar_inset
    bar_y = y0 + bar_inset
    bar_w = width - 2.0 * bar_inset

    bar_color = COLOR_SINGLE if sites == "single" else COLOR_MULTI
    ax.add_patch(
        FancyBboxPatch(
            (bar_x, bar_y),
            bar_w,
            bar_h,
            boxstyle=f"round,pad=0,rounding_size={bar_h / 2.0}",
            facecolor=bar_color,
            edgecolor="none",
            zorder=4,
            clip_on=False,
        )
    )

    text_y = y_center + bar_h / 2.0
    ax.text(
        (x_left + x_right) / 2.0,
        text_y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLOR_TEXT,
        zorder=5,
        clip_on=False,
    )


def _draw_row_divider(ax: plt.Axes, y: float, x0: float, x1: float) -> None:
    """Draw a faint dotted horizontal divider centered at ``y``."""
    ax.plot(
        [x0, x1],
        [y, y],
        color=COLOR_DIVIDER,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
        zorder=1,
        clip_on=False,
    )


def _draw_supergroup_bracket(
    ax: plt.Axes,
    y_bot: float,
    y_top: float,
    label: str,
    x_label: float,
    x_bracket: float,
) -> None:
    """Draw a vertical bracket and rotated label for a super-group."""
    pad = 0.012
    y0 = y_bot + pad
    y1 = y_top - pad
    ax.plot(
        [x_bracket, x_bracket],
        [y0, y1],
        color=COLOR_TEXT,
        linewidth=1.8,
        solid_capstyle="round",
        zorder=2,
        clip_on=False,
    )
    tick = 0.012
    ax.plot(
        [x_bracket, x_bracket + tick],
        [y0, y0],
        color=COLOR_TEXT,
        linewidth=1.4,
        zorder=2,
        clip_on=False,
    )
    ax.plot(
        [x_bracket, x_bracket + tick],
        [y1, y1],
        color=COLOR_TEXT,
        linewidth=1.4,
        zorder=2,
        clip_on=False,
    )
    ax.text(
        x_label,
        (y0 + y1) / 2.0,
        label,
        ha="center",
        va="center",
        fontsize=11,
        rotation=90,
        color=COLOR_TEXT,
        clip_on=False,
    )


def _draw_legend(ax: plt.Axes, y: float) -> None:
    """Draw two stacked legend rows centered horizontally below the grid."""
    swatch_w = 0.045
    swatch_h = 0.012
    text_gap = 0.010
    row_gap = 0.045
    center_x = 0.5

    rows = (
        (y + row_gap / 2.0, COLOR_SINGLE, "Single site generation only"),
        (y - row_gap / 2.0, COLOR_MULTI, "Multiple-site (MS) generation supported"),
    )

    text_objs = []
    for row_y, _, label in rows:
        t = ax.text(
            0.0,
            row_y,
            label,
            ha="left",
            va="center",
            fontsize=10,
            color=COLOR_TEXT,
            clip_on=False,
        )
        text_objs.append(t)

    ax.figure.canvas.draw()
    inv = ax.transData.inverted()

    for (row_y, color, _), t in zip(rows, text_objs):
        bbox = t.get_window_extent()
        x0, _y0 = inv.transform((bbox.x0, 0))
        x1, _y1 = inv.transform((bbox.x1, 0))
        text_width = x1 - x0
        total_width = swatch_w + text_gap + text_width
        swatch_x = center_x - total_width / 2.0
        t.set_x(swatch_x + swatch_w + text_gap)
        ax.add_patch(
            FancyBboxPatch(
                (swatch_x, row_y - swatch_h / 2.0),
                swatch_w,
                swatch_h,
                boxstyle=f"round,pad=0,rounding_size={swatch_h / 2.0}",
                facecolor=color,
                edgecolor="none",
                zorder=5,
                clip_on=False,
            )
        )


def build() -> Tuple[Path, Path]:
    """Render the generator landscape figure and write SVG and PNG outputs.

    Returns
    -------
    tuple of pathlib.Path
        ``(svg_path, png_path)`` for the two written artifacts.
    """
    apply_plotting_style()

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.set_xlim(-0.24, 1.04)
    ax.set_ylim(-0.20, 1.18)
    ax.set_aspect("auto")
    ax.grid(False)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    row_bounds = _row_bounds()
    super_bounds = _super_bounds(row_bounds)

    for xb in COL_BOUNDS:
        ax.plot(
            [xb, xb],
            [0.0, 1.0],
            color=COLOR_DIVIDER,
            linestyle="--",
            linewidth=0.7,
            alpha=0.55,
            zorder=1,
        )
    ax.plot(
        [0.0, 1.0],
        [0.0, 0.0],
        color=COLOR_DIVIDER,
        linestyle="--",
        linewidth=0.7,
        alpha=0.55,
        zorder=1,
    )
    ax.plot(
        [0.0, 1.0],
        [1.0, 1.0],
        color=COLOR_DIVIDER,
        linestyle="--",
        linewidth=0.7,
        alpha=0.55,
        zorder=1,
    )

    for name in ROW_ORDER[:-1]:
        y_border = row_bounds[name][0]
        _draw_row_divider(ax, y_border, 0.0, 1.0)

    for name in ROW_ORDER:
        y0, y1 = row_bounds[name]
        ax.text(
            -0.030,
            (y0 + y1) / 2.0,
            name,
            ha="center",
            va="center",
            fontsize=8,
            rotation=90,
            color=COLOR_TEXT,
            clip_on=False,
        )

    for super_name in SUPER_ORDER:
        y_bot, y_top = super_bounds[super_name]
        _draw_supergroup_bracket(
            ax, y_bot, y_top, super_name, x_label=-0.155, x_bracket=-0.100
        )

    for col in COL_NAMES:
        ax.text(
            COL_CENTERS[col],
            1.040,
            col,
            ha="center",
            va="bottom",
            fontsize=11,
            color=COLOR_TEXT,
            clip_on=False,
        )

    ax.text(
        0.5,
        1.115,
        "Timescale",
        ha="center",
        va="bottom",
        fontsize=12,
        color=COLOR_TEXT,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        -0.215,
        0.5,
        "Method family",
        ha="center",
        va="center",
        fontsize=12,
        rotation=90,
        color=COLOR_TEXT,
        fontweight="bold",
        clip_on=False,
    )

    for spec in GENERATORS:
        y0, y1 = row_bounds[spec.row]
        n_slots = ROW_SLOTS[spec.row]
        y_center = _slot_y_center(y0, y1, spec.slot, n_slots)
        x_left, x_right = _pill_x_extent(spec)
        fontsize = spec.fontsize if spec.fontsize is not None else PILL_LABEL_FONTSIZE
        _draw_pill(ax, x_left, x_right, y_center, spec.label, spec.sites, fontsize)

    _draw_legend(ax, y=-0.12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / "generator_matrix.svg"
    png_path = OUT_DIR / "generator_matrix.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("  wrote %s", svg_path.relative_to(REPO_ROOT))
    logger.info("  wrote %s", png_path.relative_to(REPO_ROOT))
    return svg_path, png_path


if __name__ == "__main__":
    build()
