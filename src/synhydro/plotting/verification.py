"""
Verification panel plot for SynHydro.

Multi-panel comparison of per-period flow statistics between a synthetic
ensemble and an observed record, following the monthly-panel convention
of the synthetic streamflow literature.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from synhydro.core.ensemble import Ensemble
from synhydro.plotting.config import COLORS, STYLE, LAYOUT, LABELS
from synhydro.plotting._utils import (
    apply_default_styling,
    save_figure,
    get_site_data,
    resample_data,
    validate_ensemble_input,
    validate_observed_input,
    validate_timestep,
    warn_if_many_realizations,
)

logger = logging.getLogger(__name__)


def _set_box_color(bp, color):
    """Set boxplot colors using STYLE-driven median styling."""
    plt.setp(bp["boxes"], color=color, facecolor=color)
    plt.setp(bp["whiskers"], color=color, linestyle="solid")
    plt.setp(bp["caps"], color=color)
    plt.setp(
        bp["medians"],
        color=STYLE["boxplot_median_color"],
        linewidth=STYLE["boxplot_median_linewidth"],
    )


def _pivot_by_period(series: pd.Series, timestep: str, n_periods: int) -> np.ndarray:
    """Pivot a series to a (years, periods) array."""
    if timestep == "monthly":
        frame = series.to_frame()
        pivot = frame.pivot_table(
            index=frame.index.year, columns=frame.index.month, values=frame.columns[0]
        )
        return pivot.reindex(columns=range(1, 13)).dropna().values
    pivot = (
        series.groupby([series.index.year, series.index.isocalendar().week])
        .mean()
        .unstack()
        .reindex(columns=range(1, n_periods + 1))
    )
    return pivot.values


def _prepare_arrays(
    ensemble: Ensemble,
    observed: Optional[pd.Series],
    site: Optional[str],
    timestep: str,
    log_space: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], int, list, str]:
    """Build synthetic (realizations, years, periods) and observed
    (years, periods) arrays at the requested timestep."""
    site_data, site_name = get_site_data(ensemble, site)

    if timestep == "monthly":
        site_data_agg = resample_data(site_data, "monthly")
        n_periods = 12
        period_labels = LABELS["month_labels"]
    else:
        site_data_agg = resample_data(site_data, "weekly")
        n_periods = 52
        period_labels = list(range(1, 53))

    observed_pivot = None
    if observed is not None:
        obs_agg = resample_data(observed.to_frame(), timestep).iloc[:, 0]
        observed_pivot = _pivot_by_period(obs_agg, timestep, n_periods)

    synthetic = np.stack(
        [
            _pivot_by_period(site_data_agg[rid], timestep, n_periods)
            for rid in site_data_agg.columns
        ],
        axis=0,
    )

    if log_space:
        synthetic = np.log(np.clip(synthetic, a_min=1e-6, a_max=None))
        if observed_pivot is not None:
            observed_pivot = np.log(np.clip(observed_pivot, a_min=1e-6, a_max=None))

    return synthetic, observed_pivot, n_periods, period_labels, site_name


def _draw_observed_boxes(ax, data_2d, positions, n_periods):
    """Column-by-column observed boxplots, skipping all-NaN columns."""
    collected = {"boxes": [], "whiskers": [], "caps": [], "medians": []}
    for i in range(n_periods):
        column = data_2d[:, i]
        valid = column[~np.isnan(column)]
        if len(valid) == 0:
            continue
        bp = ax.boxplot(
            [valid],
            positions=[positions[i]],
            widths=0.25,
            sym="",
            patch_artist=True,
        )
        for key in collected:
            collected[key].extend(bp[key])
    _set_box_color(collected, COLORS["observed"])


def _panel_distributions(
    ax, synthetic, observed_pivot, positions_syn, positions_obs, n_periods, log_space
):
    """Panel 1: per-period flow distributions."""
    flat = synthetic.reshape((-1, n_periods))
    bp = ax.boxplot(
        flat, positions=positions_syn, widths=0.25, sym="", patch_artist=True
    )
    _set_box_color(bp, COLORS["ensemble_fill"])

    ax.plot([], c=COLORS["ensemble_fill"], label="Ensemble", linewidth=5)
    if observed_pivot is not None:
        _draw_observed_boxes(ax, observed_pivot, positions_obs, n_periods)
        ax.plot([], c=COLORS["observed"], label="Observed", linewidth=5)
        ax.legend(ncol=2, loc="upper right", fontsize=LAYOUT["legend_fontsize"])
    else:
        ax.legend(loc="upper right", fontsize=LAYOUT["legend_fontsize"])

    ax.set_ylabel("Log(Q)" if log_space else "Q", fontsize=LAYOUT["label_fontsize"])
    apply_default_styling(ax, legend=False, grid=True, hide_xticks=True)


def _panel_period_statistic(
    ax,
    synthetic,
    observed_resampled,
    positions_syn,
    positions_obs,
    n_periods,
    statistic,
    ylabel,
):
    """Panels 2 and 3: per-period statistic across years, per realization.

    The observed input is a seeded bootstrap resampling of observed
    years, giving a sampling distribution of the observed statistic
    comparable to the spread across realizations.
    """
    values = getattr(synthetic, statistic)(axis=1)
    bp = ax.boxplot(
        values, positions=positions_syn, widths=0.25, sym="", patch_artist=True
    )
    _set_box_color(bp, COLORS["ensemble_fill"])

    if observed_resampled is not None:
        _draw_observed_boxes(
            ax, getattr(observed_resampled, statistic)(axis=1), positions_obs, n_periods
        )

    ax.set_ylabel(ylabel, fontsize=LAYOUT["label_fontsize"])
    apply_default_styling(ax, legend=False, grid=True, hide_xticks=True)


def _panel_test_pvalues(
    ax, synthetic, observed_pivot, positions_syn, n_periods, test, ylabel
):
    """Panels 4 and 5: per-realization two-sample test p-values.

    Each realization is tested separately against the observed years for
    every period, so the p-value distribution does not shrink as the
    ensemble grows (pooling realizations before testing would).
    """
    if observed_pivot is None:
        ax.text(
            0.5,
            0.5,
            "Observed data not provided",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        apply_default_styling(ax, legend=False, grid=True, hide_xticks=True)
        ax.set_ylabel(ylabel, fontsize=LAYOUT["label_fontsize"])
        return

    n_realizations = synthetic.shape[0]
    pvalue_columns = []
    for i in range(n_periods):
        observed_valid = observed_pivot[:, i]
        observed_valid = observed_valid[~np.isnan(observed_valid)]
        pvals = []
        for r in range(n_realizations):
            syn_valid = synthetic[r, :, i]
            syn_valid = syn_valid[~np.isnan(syn_valid)]
            if len(observed_valid) > 2 and len(syn_valid) > 2:
                pvals.append(float(test(observed_valid, syn_valid)[1]))
        pvalue_columns.append(pvals if pvals else [np.nan])

    bp = ax.boxplot(
        pvalue_columns,
        positions=positions_syn + 0.15,
        widths=0.4,
        sym="",
        patch_artist=True,
    )
    _set_box_color(bp, COLORS["ensemble_fill"])
    ax.axhline(0.05, color="k", linewidth=1, linestyle="--")
    ax.set_xlim([0, n_periods + 1])
    ax.set_ylim([0, 1.05])
    ax.set_ylabel(ylabel, fontsize=LAYOUT["label_fontsize"])
    apply_default_styling(ax, legend=False, grid=True, hide_xticks=True)


def plot_verification_panel(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    timestep: str = "monthly",
    log_space: bool = False,
    seed: Optional[int] = None,
    figsize: Tuple[float, float] = LAYOUT["validation_figsize"],
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Multi-panel statistical verification plot.

    Five panels compare the ensemble against observed data per calendar
    period: (1) flow distributions, (2) period means, (3) period
    standard deviations, (4) Wilcoxon rank-sum p-values, and (5) Levene
    p-values. Statistics and tests are computed per realization; the
    observed statistic panels use a seeded bootstrap over observed
    years to show sampling uncertainty.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data.
    observed : pd.Series, optional
        Observed timeseries. Panels 4 and 5 require observed data and
        display a placeholder message when it is not provided.
    site : str, optional
        Site name to analyze. If None, uses the first site.
    timestep : str, default 'monthly'
        Temporal aggregation: 'monthly' or 'weekly'.
    log_space : bool, default False
        Compare statistics of log-transformed flows.
    seed : int, optional
        Seed for the bootstrap resampling of observed years in panels
        2 and 3.
    figsize : tuple, default from config
        Figure size (width, height) in inches.
    filename : str, optional
        Path to save the figure.
    dpi : int, default from config
        Resolution for the saved figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
        List of 5 axes objects.

    References
    ----------
    Herman, J.D., Zeff, H.B., Lamontagne, J.R., Reed, P.M., and
    Characklis, G.W. (2016). Synthetic drought scenario generation to
    support bottom-up water supply vulnerability assessments. Journal
    of Water Resources Planning and Management, 142(11), 04016050.

    Examples
    --------
    >>> fig, axes = plot_verification_panel(ensemble, observed=Q_obs)
    >>> fig, axes = plot_verification_panel(ensemble, Q_obs, log_space=True)
    """
    validate_ensemble_input(ensemble)
    validate_timestep(ensemble, timestep)
    observed = validate_observed_input(observed, required=False)
    warn_if_many_realizations(len(ensemble.realization_ids), context="verification")

    if timestep not in ["monthly", "weekly"]:
        raise ValueError(f"timestep must be 'monthly' or 'weekly', got '{timestep}'")

    synthetic, observed_pivot, n_periods, period_labels, site_name = _prepare_arrays(
        ensemble, observed, site, timestep, log_space
    )

    observed_resampled = None
    if observed_pivot is not None:
        rng = np.random.default_rng(seed)
        n_years = observed_pivot.shape[0]
        indices = rng.integers(0, n_years, size=(synthetic.shape[0], n_years))
        observed_resampled = observed_pivot[indices]

    fig, axes_arr = plt.subplots(5, 1, figsize=figsize, dpi=LAYOUT["default_dpi"])
    axes = list(axes_arr)

    positions_syn = np.arange(1, n_periods + 1) - 0.15
    positions_obs = np.arange(1, n_periods + 1) + 0.15

    _panel_distributions(
        axes[0],
        synthetic,
        observed_pivot,
        positions_syn,
        positions_obs,
        n_periods,
        log_space,
    )
    _panel_period_statistic(
        axes[1],
        synthetic,
        observed_resampled,
        positions_syn,
        positions_obs,
        n_periods,
        "mean",
        r"$\hat{\mu}_Q$",
    )
    _panel_period_statistic(
        axes[2],
        synthetic,
        observed_resampled,
        positions_syn,
        positions_obs,
        n_periods,
        "std",
        r"$\hat{\sigma}_Q$",
    )
    _panel_test_pvalues(
        axes[3],
        synthetic,
        observed_pivot,
        positions_syn,
        n_periods,
        stats.ranksums,
        "Rank-sum $p$",
    )
    _panel_test_pvalues(
        axes[4],
        synthetic,
        observed_pivot,
        positions_syn,
        n_periods,
        stats.levene,
        "Levene $p$",
    )

    # X-ticks on the bottom panel only
    ax = axes[4]
    if timestep == "monthly":
        ax.set_xticks(range(1, n_periods + 1))
        ax.set_xticklabels(period_labels, fontsize=LAYOUT["tick_fontsize"])
    else:
        ax.set_xticks(np.arange(0, n_periods + 1, 5))
        ax.set_xticklabels(
            np.arange(0, n_periods + 1, 5), fontsize=LAYOUT["tick_fontsize"]
        )

    space_text = ("Log space" if log_space else "Real space") + f" - {site_name}"
    fig.suptitle(
        f"Statistical Verification - {timestep.capitalize()}\n{space_text}",
        fontsize=LAYOUT["title_fontsize"] + 2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, axes
