"""
Alternative blog figure: log-flow KDE comparison, one panel per timescale.

Three stacked panels (Annual, Monthly, Daily). In each panel, every generator
that operates at that timescale is shown as a single KDE of log10(flow) over
the pooled realizations at gauge USGS-01434000, with the historical record
plotted as a bold dark KDE on top.

Consumes the pre-generated ensembles in
`experiments/model_diagnostics/outputs/<generator>/ensemble.h5`. To refresh,
rerun `experiments/model_diagnostics/run_diagnostic.py` per generator.

Run from the repo root:
    venv/Scripts/python docs/blog/figures/make_all_generators_kde.py
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from synhydro import Ensemble, load_example_data

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")
logger = logging.getLogger("make_all_generators_kde")
logger.setLevel(logging.INFO)


PLOT_SITE = "USGS-01434000"
OUTPUTS_DIR = Path("experiments/model_diagnostics/outputs")
OBSERVED_COLOR = "#111111"

# A categorical palette with enough distinct colors for the largest panel
# (six monthly generators).
GEN_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]


@dataclass
class Method:
    folder: str
    display_name: str
    scale: str  # 'Annual', 'Monthly', 'Daily'
    expected_freq: str  # 'YS', 'MS', or 'D'


METHODS = [
    # Annual
    Method("SMARTA", "SMARTA", "Annual", "YS"),
    Method("HMM", "MultiSite-HMM", "Annual", "YS"),
    Method("WARM", "WARM", "Annual", "YS"),
    # Monthly
    Method("ThomasFiering", "Thomas-Fiering", "Monthly", "MS"),
    Method("Matalas", "Matalas", "Monthly", "MS"),
    Method("ARFIMA", "ARFIMA", "Monthly", "MS"),
    Method("SPARTA", "SPARTA", "Monthly", "MS"),
    Method("Kirsch", "Kirsch", "Monthly", "MS"),
    Method("KNNBootstrap", "KNN Bootstrap", "Monthly", "MS"),
    # Daily
    Method("PhaseRandomization", "Phase Randomization", "Daily", "D"),
    Method("MultisitePhaseRandomization", "Multisite Phase Rand.", "Daily", "D"),
]


def aggregate_obs(Q_daily: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "D":
        return Q_daily
    if target == "MS":
        return Q_daily.resample("MS").sum()
    if target == "YS":
        return Q_daily.resample("YS").sum()
    raise ValueError(f"unknown target freq {target}")


def load_pooled_synthetic(method: Method, site: str) -> Optional[np.ndarray]:
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

    # Pool all realizations into one flat sample
    values = syn_df.values.flatten()
    return values


def kde_log_flow(values: np.ndarray, x_grid: np.ndarray) -> Optional[np.ndarray]:
    """Return KDE density estimated in log10(flow) space, evaluated on x_grid."""
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 3:
        return None
    logv = np.log10(values)
    try:
        kde = gaussian_kde(logv)
    except (np.linalg.LinAlgError, ValueError) as exc:
        logger.warning("KDE failed: %s", exc)
        return None
    return kde(x_grid)


def plot_panel(ax, scale: str, methods_row, panel_obs: np.ndarray, panel_syn: dict):
    # Compute combined log-flow range across this panel's data
    pooled = [panel_obs]
    for v in panel_syn.values():
        if v is not None:
            pooled.append(v)
    all_vals = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in pooled])
    log_all = np.log10(all_vals)
    lo = np.percentile(log_all, 0.5)
    hi = np.percentile(log_all, 99.5)
    span = hi - lo
    lo -= 0.10 * span
    hi += 0.10 * span
    x_grid = np.linspace(lo, hi, 400)

    # Synthetic KDEs first (background), historical last (foreground)
    for idx, m in enumerate(methods_row):
        v = panel_syn.get(m.folder)
        if v is None:
            continue
        dens = kde_log_flow(v, x_grid)
        if dens is None:
            continue
        color = GEN_PALETTE[idx % len(GEN_PALETTE)]
        ax.plot(
            x_grid, dens, color=color, linewidth=1.6, alpha=0.85, label=m.display_name
        )

    obs_dens = kde_log_flow(panel_obs, x_grid)
    if obs_dens is not None:
        ax.plot(
            x_grid,
            obs_dens,
            color=OBSERVED_COLOR,
            linewidth=2.4,
            label="Historical",
            zorder=10,
        )

    ax.set_title(scale, fontsize=12, fontweight="bold", loc="left", pad=4, color="0.20")
    ax.set_ylabel("Density", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, which="both", alpha=0.2, linewidth=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)

    leg = ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=1)
    leg.get_frame().set_alpha(0.85)


def main():
    out_path = Path(__file__).parent / "all_generators_kde.png"

    logger.info("Loading example historical data ...")
    Q_daily = load_example_data("usgs_daily_streamflow_cms")
    logger.info("Historical: %d days, %d sites", *Q_daily.shape)

    annual_methods = [m for m in METHODS if m.scale == "Annual"]
    monthly_methods = [m for m in METHODS if m.scale == "Monthly"]
    daily_methods = [m for m in METHODS if m.scale == "Daily"]

    Q_annual = aggregate_obs(Q_daily, "YS")[PLOT_SITE].values
    Q_monthly = aggregate_obs(Q_daily, "MS")[PLOT_SITE].values
    Q_daily_arr = Q_daily[PLOT_SITE].values

    syn_by_method = {m.folder: load_pooled_synthetic(m, PLOT_SITE) for m in METHODS}

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 11.0))
    plt.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.06, hspace=0.30)

    plot_panel(
        axes[0],
        "Annual",
        annual_methods,
        Q_annual,
        {m.folder: syn_by_method[m.folder] for m in annual_methods},
    )
    plot_panel(
        axes[1],
        "Monthly",
        monthly_methods,
        Q_monthly,
        {m.folder: syn_by_method[m.folder] for m in monthly_methods},
    )
    plot_panel(
        axes[2],
        "Daily",
        daily_methods,
        Q_daily_arr,
        {m.folder: syn_by_method[m.folder] for m in daily_methods},
    )

    axes[2].set_xlabel("log10(flow) [cms]", fontsize=10)

    fig.suptitle(
        f"SynHydro generators: log-flow KDE at {PLOT_SITE}",
        fontsize=14,
        y=0.975,
    )
    fig.text(
        0.5,
        0.952,
        "Pooled-realization density at each generator's native timescale; "
        "historical record shown as the bold dark curve.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="0.3",
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)


if __name__ == "__main__":
    main()
