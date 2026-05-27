"""
Build the small-multiples flow-duration-curve figure for the WaterProgramming
intro blog post.

This script consumes the pre-generated ensembles in
`experiments/model_diagnostics/outputs/<generator>/ensemble.h5`. It does not
fit or generate. To refresh the underlying ensembles (for example, to bump the
realization count), rerun the model-diagnostics experiment for each generator:

    venv/Scripts/python experiments/model_diagnostics/run_diagnostic.py \\
        --generator Kirsch --n_realizations 50 --n_years 30

Then re-run this script.

Run from the repo root:
    venv/Scripts/python docs/blog/figures/make_all_generators_fdc.py
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synhydro import Ensemble, load_example_data

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")
logger = logging.getLogger("make_all_generators_fdc")
logger.setLevel(logging.INFO)


PLOT_SITE = "USGS-01434000"
OUTPUTS_DIR = Path("experiments/model_diagnostics/outputs")
FAMILY_COLOR = {
    "Parametric": "#1f77b4",
    "Hybrid": "#9467bd",
    "Non-parametric": "#2ca02c",
}
ENSEMBLE_FILL = "#5B86C5"
ENSEMBLE_FILL_ALPHA = 0.30
OBSERVED_COLOR = "#111111"


@dataclass
class Method:
    folder: str
    display_name: str
    family: str
    scale: str  # 'Annual', 'Monthly', or 'Daily'
    expected_freq: str  # pandas freq for aggregating the historical reference


METHODS = [
    # Annual
    Method("SMARTA", "SMARTA", "Parametric", "Annual", "YS"),
    Method("HMM", "MultiSite-HMM", "Parametric", "Annual", "YS"),
    Method("WARM", "WARM", "Hybrid", "Annual", "YS"),
    # Monthly
    Method("ThomasFiering", "Thomas-Fiering", "Parametric", "Monthly", "MS"),
    Method("Matalas", "Matalas", "Parametric", "Monthly", "MS"),
    Method("ARFIMA", "ARFIMA", "Parametric", "Monthly", "MS"),
    Method("SPARTA", "SPARTA", "Parametric", "Monthly", "MS"),
    Method("Kirsch", "Kirsch", "Hybrid", "Monthly", "MS"),
    Method("KNNBootstrap", "KNN Bootstrap", "Non-parametric", "Monthly", "MS"),
    # Daily
    Method("PhaseRandomization", "Phase Randomization", "Hybrid", "Daily", "D"),
    Method(
        "MultisitePhaseRandomization", "Multisite Phase Rand.", "Hybrid", "Daily", "D"
    ),
]


def aggregate_obs(Q_daily: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "D":
        return Q_daily
    if target == "MS":
        return Q_daily.resample("MS").sum()
    if target == "YS":
        return Q_daily.resample("YS").sum()
    raise ValueError(f"unknown target freq {target}")


def load_method_data(
    method: Method, Q_daily: pd.DataFrame, site: str
) -> Optional[dict]:
    h5_path = OUTPUTS_DIR / method.folder / "ensemble.h5"
    if not h5_path.exists():
        logger.warning("%s: missing %s", method.folder, h5_path)
        return None
    try:
        ens = Ensemble.from_hdf5(str(h5_path))
    except Exception as exc:
        logger.warning("%s: failed to load: %s", method.folder, exc)
        return None

    syn_df = ens.data_by_site.get(site)
    if syn_df is None:
        first = ens.sites[0]
        logger.info("%s: site %s missing, using %s", method.folder, site, first)
        syn_df = ens.data_by_site[first]

    obs_at_freq = aggregate_obs(Q_daily, method.expected_freq)
    obs_series = (
        obs_at_freq[site] if site in obs_at_freq.columns else obs_at_freq.iloc[:, 0]
    )

    n_real = syn_df.shape[1]
    logger.info(
        "%s: loaded ensemble n_real=%d, n_obs=%d", method.folder, n_real, len(syn_df)
    )
    return {"syn": syn_df, "obs": obs_series.values, "n_real": n_real}


def fdc_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(values[~np.isnan(values)])[::-1]
    n = v.size
    if n == 0:
        return np.array([]), np.array([])
    ep = np.arange(1, n + 1) / (n + 1)
    return ep, v


def fdc_on_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ep, v = fdc_curve(values)
    if v.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, ep, v, left=v[0], right=v[-1])


def plot_panel(ax, panel_data: Optional[dict], method: Method, grid: np.ndarray):
    family_color = FAMILY_COLOR[method.family]
    ax.set_title(
        method.display_name, fontsize=10, color=family_color, fontweight="bold", pad=4
    )
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", alpha=0.2, linewidth=0.5)

    if panel_data is None:
        ax.text(
            0.5,
            0.5,
            "ensemble\nmissing",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="0.4",
        )
        return

    syn_df = panel_data["syn"]
    fdcs = np.stack([fdc_on_grid(syn_df[c].values, grid) for c in syn_df.columns])
    syn_min = np.nanmin(fdcs, axis=0)
    syn_max = np.nanmax(fdcs, axis=0)
    ax.fill_between(
        grid,
        syn_min,
        syn_max,
        color=ENSEMBLE_FILL,
        alpha=ENSEMBLE_FILL_ALPHA,
        linewidth=0,
        label="Synthetic range",
    )

    obs_ep, obs_v = fdc_curve(panel_data["obs"])
    ax.plot(obs_ep, obs_v, color=OBSERVED_COLOR, linewidth=1.8, label="Historical")


def build_layout(fig):
    """Place panels on a 3-row grid by timescale, centering partial rows."""
    gs = gridspec.GridSpec(
        3,
        6,
        figure=fig,
        left=0.07,
        right=0.985,
        top=0.88,
        bottom=0.10,
        wspace=0.20,
        hspace=0.45,
    )
    annual_cols = (1, 2, 3)  # 3 panels centered
    monthly_cols = (0, 1, 2, 3, 4, 5)  # 6 panels filling row
    daily_cols = (2, 3)  # 2 panels centered

    annual_axes = [fig.add_subplot(gs[0, c]) for c in annual_cols]
    monthly_axes = [fig.add_subplot(gs[1, c]) for c in monthly_cols]
    daily_axes = [fig.add_subplot(gs[2, c]) for c in daily_cols]

    # Share y within each row only -- different timescales naturally have
    # different flow magnitudes, so per-row scaling lets each row use its
    # vertical space efficiently.
    for row in (annual_axes, monthly_axes, daily_axes):
        base = row[0]
        for ax in row[1:]:
            ax.sharex(base)
            ax.sharey(base)

    return annual_axes, monthly_axes, daily_axes, gs


def tighten_row_ylim(axes_row, methods_row, panel_data, margin=0.20):
    """Set a tight log-y range for all panels in a row.

    The range covers all observed and synthetic values across the row, with a
    small margin in log space.
    """
    all_min = []
    all_max = []
    for m in methods_row:
        d = panel_data.get(m.folder)
        if d is None:
            continue
        syn = d["syn"].values
        obs = d["obs"]
        syn_pos = syn[(syn > 0) & np.isfinite(syn)]
        obs_pos = obs[(obs > 0) & np.isfinite(obs)]
        if syn_pos.size:
            all_min.append(syn_pos.min())
            all_max.append(syn_pos.max())
        if obs_pos.size:
            all_min.append(obs_pos.min())
            all_max.append(obs_pos.max())

    if not all_min or not all_max:
        return

    lo = min(all_min)
    hi = max(all_max)
    # Apply margin in log space
    log_lo = np.log10(lo)
    log_hi = np.log10(hi)
    span = log_hi - log_lo
    log_lo -= span * margin
    log_hi += span * margin
    new_lo = 10**log_lo
    new_hi = 10**log_hi
    for ax in axes_row:
        ax.set_ylim(new_lo, new_hi)


def main():
    out_path = Path(__file__).parent / "all_generators_fdc.png"

    logger.info("Loading example historical data ...")
    Q_daily = load_example_data("usgs_daily_streamflow_cms")
    logger.info("Historical: %d days, %d sites", *Q_daily.shape)

    panel_data = {m.folder: load_method_data(m, Q_daily, PLOT_SITE) for m in METHODS}

    fig = plt.figure(figsize=(14, 8.0))
    annual_axes, monthly_axes, daily_axes, gs = build_layout(fig)

    annual_methods = [m for m in METHODS if m.scale == "Annual"]
    monthly_methods = [m for m in METHODS if m.scale == "Monthly"]
    daily_methods = [m for m in METHODS if m.scale == "Daily"]

    grid = np.linspace(0.01, 0.99, 200)

    for ax, m in zip(annual_axes, annual_methods):
        plot_panel(ax, panel_data[m.folder], m, grid)
    for ax, m in zip(monthly_axes, monthly_methods):
        plot_panel(ax, panel_data[m.folder], m, grid)
    for ax, m in zip(daily_axes, daily_methods):
        plot_panel(ax, panel_data[m.folder], m, grid)

    tighten_row_ylim(annual_axes, annual_methods, panel_data)
    tighten_row_ylim(monthly_axes, monthly_methods, panel_data)
    tighten_row_ylim(daily_axes, daily_methods, panel_data)

    # Axis labels: y on first panel of each row, x on bottom panels
    for ax in daily_axes:
        ax.set_xlabel("Exceedance probability", fontsize=9)
    annual_axes[0].set_ylabel("Flow (cms)", fontsize=9)
    monthly_axes[0].set_ylabel("Flow (cms)", fontsize=9)
    daily_axes[0].set_ylabel("Flow (cms)", fontsize=9)

    # Row labels at the figure margin, vertically aligned with each row centre
    row_label_x = 0.015
    for label, axes_row in [
        ("Annual", annual_axes),
        ("Monthly", monthly_axes),
        ("Daily", daily_axes),
    ]:
        bb = axes_row[0].get_position()
        y_mid = (bb.y0 + bb.y1) / 2
        fig.text(
            row_label_x,
            y_mid,
            label,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            rotation=90,
            color="0.25",
        )

    n_real_sample = next((d["n_real"] for d in panel_data.values() if d is not None), 0)
    fig.suptitle(
        f"SynHydro generators: flow duration curves at {PLOT_SITE}"
        f"  (N = {n_real_sample} realizations per generator)",
        fontsize=13,
        y=0.965,
    )
    fig.text(
        0.5,
        0.925,
        "Panel titles colored by method family: "
        "parametric / hybrid / non-parametric.   "
        "Shaded band: range across synthetic realizations.   "
        "Bold line: historical record.",
        ha="center",
        va="center",
        fontsize=9,
        color="0.3",
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)


if __name__ == "__main__":
    main()
