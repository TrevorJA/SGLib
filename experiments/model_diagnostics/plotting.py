"""Diagnostic plotting orchestration for single-generator analysis.

Each `fig_*` function is a thin wrapper around a public `synhydro.plotting`
function: it pulls one site out of the ensemble, calls the public API, and
saves to the diagnostic output directory using the gen-key naming
convention.

The only exception is `fig_validation_summary`, which is a custom bar chart
of mean absolute relative error by metric category and is genuinely
diagnostic-specific.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from synhydro.core.ensemble import Ensemble
from synhydro.plotting import (
    COLORS,
    plot_autocorrelation,
    plot_cdf,
    plot_flow_duration_curve,
    plot_histogram,
    plot_seasonal_cycle,
    plot_spatial_correlation,
)

logger = logging.getLogger(__name__)


def fig_marginal_cdf(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    site_idx: int,
    output_dir: Path,
    generator_name: str = "",
) -> None:
    """Empirical CDF: ensemble vs observed.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic ensemble.
    Q_obs : pd.DataFrame
        Observed multi-site data at the same frequency as the ensemble.
    site_idx : int
        Index of site to plot.
    output_dir : Path
        Directory where the figure is saved.
    generator_name : str, default ''
        Generator key used in the figure title.
    """
    site = ensemble.site_names[site_idx]
    plot_cdf(
        ensemble,
        observed=Q_obs[site],
        site=site,
        title=f"Marginal CDF -- {generator_name} ({site})",
        filename=str(output_dir / "fig_01_marginal_cdf.png"),
    )
    plt.close("all")


def fig_marginal_pdf(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    site_idx: int,
    output_dir: Path,
    generator_name: str = "",
) -> None:
    """Histogram and KDE: observed vs ensemble.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic ensemble.
    Q_obs : pd.DataFrame
        Observed multi-site data at the same frequency as the ensemble.
    site_idx : int
        Index of site to plot.
    output_dir : Path
        Directory where the figure is saved.
    generator_name : str, default ''
        Generator key used in the figure title.
    """
    site = ensemble.site_names[site_idx]
    plot_histogram(
        ensemble,
        observed=Q_obs[site],
        site=site,
        title=f"Marginal PDF -- {generator_name} ({site})",
        filename=str(output_dir / "fig_01_marginal_pdf.png"),
    )
    plt.close("all")


def fig_seasonal_cycle(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    site_idx: int,
    output_dir: Path,
    generator_name: str = "",
) -> None:
    """Monthly mean seasonal cycle: ensemble band vs observed line.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic ensemble.
    Q_obs : pd.DataFrame
        Observed multi-site data at the same frequency as the ensemble.
    site_idx : int
        Index of site to plot.
    output_dir : Path
        Directory where the figure is saved.
    generator_name : str, default ''
        Generator key used in the figure title.
    """
    site = ensemble.site_names[site_idx]
    timestep = _ensemble_timestep(ensemble)
    plot_seasonal_cycle(
        ensemble,
        observed=Q_obs[site],
        site=site,
        statistic="mean",
        timestep=timestep,
        title=f"Seasonal cycle (means) -- {generator_name} ({site})",
        filename=str(output_dir / "fig_02_seasonal_cycle.png"),
    )
    plt.close("all")


def fig_seasonal_std(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    site_idx: int,
    output_dir: Path,
    generator_name: str = "",
) -> None:
    """Monthly std seasonal cycle: ensemble band vs observed line.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic ensemble.
    Q_obs : pd.DataFrame
        Observed multi-site data at the same frequency as the ensemble.
    site_idx : int
        Index of site to plot.
    output_dir : Path
        Directory where the figure is saved.
    generator_name : str, default ''
        Generator key used in the figure title.
    """
    site = ensemble.site_names[site_idx]
    timestep = _ensemble_timestep(ensemble)
    plot_seasonal_cycle(
        ensemble,
        observed=Q_obs[site],
        site=site,
        statistic="std",
        timestep=timestep,
        title=f"Seasonal variability -- {generator_name} ({site})",
        filename=str(output_dir / "fig_03_seasonal_std.png"),
    )
    plt.close("all")


def fig_acf(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    site_idx: int,
    output_dir: Path,
    generator_name: str = "",
    max_lag: int = 36,
) -> None:
    """Autocorrelation function with ensemble envelope.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic ensemble.
    Q_obs : pd.DataFrame
        Observed multi-site data at the same frequency as the ensemble.
    site_idx : int
        Index of site to plot.
    output_dir : Path
        Directory where the figure is saved.
    generator_name : str, default ''
        Generator key used in the figure title.
    max_lag : int, default 36
        Maximum lag in periods for the ACF.
    """
    site = ensemble.site_names[site_idx]
    timestep = _ensemble_timestep(ensemble)
    plot_autocorrelation(
        ensemble,
        observed=Q_obs[site],
        site=site,
        max_lag=max_lag,
        timestep=timestep,
        title=f"ACF -- {generator_name} ({site})",
        filename=str(output_dir / "fig_04_acf.png"),
    )
    plt.close("all")


def fig_fdc(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    site_idx: int,
    output_dir: Path,
    generator_name: str = "",
) -> None:
    """Flow duration curve with ensemble envelope.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic ensemble.
    Q_obs : pd.DataFrame
        Observed multi-site data at the same frequency as the ensemble.
    site_idx : int
        Index of site to plot.
    output_dir : Path
        Directory where the figure is saved.
    generator_name : str, default ''
        Generator key used in the figure title.
    """
    site = ensemble.site_names[site_idx]
    plot_flow_duration_curve(
        ensemble,
        observed=Q_obs[site],
        site=site,
        log_scale=True,
        title=f"Flow Duration Curve -- {generator_name} ({site})",
        filename=str(output_dir / "fig_05_fdc.png"),
    )
    plt.close("all")


def fig_cross_correlation(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    output_dir: Path,
    generator_name: str = "",
) -> None:
    """Side-by-side cross-site correlation: observed vs ensemble mean.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic ensemble.
    Q_obs : pd.DataFrame
        Observed multi-site data at the same frequency as the ensemble.
    output_dir : Path
        Directory where the figure is saved.
    generator_name : str, default ''
        Generator key used in the figure suptitle.
    """
    if len(ensemble.site_names) < 2:
        logger.info("Skipping cross-correlation figure (single site).")
        return

    timestep = _ensemble_timestep(ensemble)
    plot_spatial_correlation(
        ensemble,
        observed=Q_obs,
        timestep=timestep,
        method="pearson",
        title=f"Cross-site correlation -- {generator_name}",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        filename=str(output_dir / "fig_06_cross_correlation.png"),
    )
    plt.close("all")


def fig_validation_summary(
    validation_df: pd.DataFrame,
    output_dir: Path,
    generator_name: str = "",
) -> None:
    """Bar chart of mean absolute relative error by metric category.

    This is a custom diagnostic visualization with no equivalent in the
    public plotting module.

    Parameters
    ----------
    validation_df : pd.DataFrame
        Single-column frame indexed by metric-category name; values are
        the mean absolute relative error for that category.
    output_dir : Path
        Directory where the figure is saved.
    generator_name : str, default ''
        Generator key used in the figure title.
    """
    if validation_df is None or validation_df.empty:
        logger.info("Skipping validation summary figure (no data).")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    categories = validation_df.index.tolist()
    values = validation_df.values.ravel()

    bar_colors = [
        (
            COLORS["ensemble_median"]
            if v < 0.25
            else COLORS["drought_severe"] if v < 0.5 else COLORS["drought_extreme"]
        )
        for v in values
    ]
    ax.barh(categories, values, color=bar_colors, edgecolor="white", height=0.6)
    ax.axvline(0.25, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Mean absolute relative error (MARE)")
    ax.set_title(f"Validation summary -- {generator_name}")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "fig_07_validation_summary.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensemble_timestep(ensemble: Ensemble) -> str:
    """Map an ensemble's pandas frequency string to a plotting timestep.

    Returns 'daily', 'weekly', 'monthly', or 'annual'. Defaults to 'daily'
    when the frequency cannot be inferred.
    """
    freq = ensemble.frequency
    if freq is None:
        return "daily"
    base = freq.split("-")[0].upper()
    if base.startswith(("Y", "A")):
        return "annual"
    if base.startswith("Q"):
        return "monthly"
    if base.startswith("M"):
        return "monthly"
    if base.startswith("W"):
        return "weekly"
    return "daily"
