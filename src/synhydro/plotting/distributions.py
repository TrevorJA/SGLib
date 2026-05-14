"""
Distribution plotting functions for SynHydro.

This module provides functions for plotting flow distributions, flow duration
curves, histograms, and monthly distribution comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
from scipy.stats import gaussian_kde
from synhydro.core.ensemble import Ensemble
from synhydro.plotting.config import COLORS, STYLE, LAYOUT, LABELS
from synhydro.plotting._utils import (
    setup_axes,
    apply_default_styling,
    save_figure,
    get_site_data,
    get_ylabel,
    filter_complete_years,
    resample_data,
    validate_ensemble_input,
    validate_observed_input,
)


def _is_annual_index(index: pd.DatetimeIndex) -> bool:
    """Return True when the index has at most one observation per calendar year."""
    if len(index) < 2:
        return True
    years = pd.Index(index).year
    return len(years) == len(pd.unique(years))


def _empirical_fdc(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical flow duration curve using Weibull plotting positions.

    Sorts the (non-NaN) values ascending and returns ``(p, sorted_vals)``
    where ``p[k] = (k - 0.5) / N`` for ``k = 1..N``. This places one point
    per data value and avoids the tail-flattening artifacts that arise when
    a fixed quantile grid is interpolated against a small sample.

    Parameters
    ----------
    values : np.ndarray
        1-D array of flow values, possibly containing NaN.

    Returns
    -------
    p : np.ndarray
        Non-exceedance probabilities, ascending in (0, 1).
    sorted_vals : np.ndarray
        Sorted flow values aligned with ``p``.
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.empty(0), np.empty(0)
    sorted_vals = np.sort(arr)
    n = sorted_vals.size
    p = (np.arange(1, n + 1) - 0.5) / n
    return p, sorted_vals


def plot_flow_duration_curve(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    show_annual_range: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT["default_figsize"],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    log_scale: bool = True,
    units: str = "cms",
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot flow duration curve with ensemble uncertainty.

    Shows flow exceedance probabilities with ensemble range and total period FDC.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series, optional
        Observed timeseries for comparison
    site : str, optional
        Site name to plot. If None, uses first site in ensemble
    show_annual_range : bool, default True
        Show range of annual FDCs as shaded region
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default from config
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates
    xlabel : str, optional
        X-axis label. If None, uses 'Non-Exceedance Probability'
    ylabel : str, optional
        Y-axis label. If None, uses units
    legend : bool, default True
        Whether to display legend
    grid : bool, default True
        Whether to display grid
    log_scale : bool, default True
        Use logarithmic y-axis
    units : str, default 'cms'
        Flow units for y-axis label
    filename : str, optional
        Path to save figure
    dpi : int, default from config
        Resolution for saved figure
    **kwargs
        Forwarded to `ax.plot` and `ax.fill_between`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> fig, ax = plot_flow_duration_curve(ensemble, observed=Q_obs)
    >>> plot_flow_duration_curve(ensemble, show_annual_range=False, log_scale=False)
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    observed = validate_observed_input(observed, required=False)

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Filter complete years
    site_data = filter_complete_years(site_data)
    if observed is not None:
        observed = filter_complete_years(observed.to_frame()).iloc[:, 0]

    is_annual = _is_annual_index(site_data.index)

    # Total FDC for the ensemble plotted at empirical Weibull positions on
    # sorted values. Fixed-quantile interpolation (e.g. linspace 0..1) does
    # not work well at the tails of small samples, where many quantile
    # points map to the same min/max value and create artificial flat
    # horizontal segments. Empirical positions place one point per data
    # value and let matplotlib draw a faithful step-and-rise FDC.
    s_flat = site_data.values.flatten()
    s_pos, s_total_fdc = _empirical_fdc(s_flat)

    if show_annual_range:
        if is_annual:
            # Per-realization FDC, then envelope across realizations.
            # site_data shape is (n_periods, n_realizations); each column
            # is one realization at the same n_periods, so empirical FDC
            # positions are common across realizations.
            sorted_cols = np.sort(site_data.values, axis=0)
            valid = ~np.isnan(sorted_cols)
            # Per-column N (count of finite values) is uniform here
            # because all realizations have the same period count.
            n_per_real = int(valid[:, 0].sum())
            env_pos = (np.arange(1, n_per_real + 1) - 0.5) / n_per_real
            env_vals = sorted_cols[:n_per_real, :]
            s_fdc_min = np.nanmin(env_vals, axis=1)
            s_fdc_max = np.nanmax(env_vals, axis=1)
            envelope_label = "Ensemble Realization FDC Range"
        else:
            # Sub-annual data: per-year FDC at empirical within-year
            # positions, then min/max envelope across years. Pool values
            # across all realizations within each year.
            years = sorted(set(int(y) for y in site_data.index.year.unique()))
            per_year_vals = []
            for y in years:
                year_block = site_data.values[site_data.index.year == y]
                vals = np.sort(year_block.flatten())
                vals = vals[~np.isnan(vals)]
                if vals.size:
                    per_year_vals.append(vals)
            if per_year_vals:
                m = min(v.size for v in per_year_vals)
                env_pos = (np.arange(1, m + 1) - 0.5) / m
                env_arr = np.stack([v[:m] for v in per_year_vals], axis=1)
                s_fdc_min = np.nanmin(env_arr, axis=1)
                s_fdc_max = np.nanmax(env_arr, axis=1)
                envelope_label = "Ensemble Annual FDC Range"
            else:
                env_pos = None

        if env_pos is not None:
            ax.fill_between(
                env_pos,
                s_fdc_min,
                s_fdc_max,
                color=COLORS["ensemble_fill"],
                alpha=STYLE["fill_alpha"],
                label=envelope_label,
                **kwargs,
            )

    ax.plot(
        s_pos,
        s_total_fdc,
        color=COLORS["ensemble_median"],
        linewidth=STYLE["ensemble_linewidth"],
        label="Ensemble Total FDC",
        **kwargs,
    )

    if observed is not None:
        h_pos, h_total_fdc = _empirical_fdc(observed.values.flatten())

        # An observed series is one realization, so the inter-year
        # envelope is only meaningful for sub-annual data; for annual
        # observed data it would collapse to a horizontal rectangle.
        if show_annual_range and not _is_annual_index(observed.index):
            years = sorted(set(int(y) for y in observed.index.year.unique()))
            per_year_obs = []
            for y in years:
                vals = np.sort(observed.values[observed.index.year == y])
                vals = vals[~np.isnan(vals)]
                if vals.size:
                    per_year_obs.append(vals)
            if per_year_obs:
                m = min(v.size for v in per_year_obs)
                obs_env_pos = (np.arange(1, m + 1) - 0.5) / m
                obs_env_arr = np.stack([v[:m] for v in per_year_obs], axis=1)
                h_fdc_min = np.nanmin(obs_env_arr, axis=1)
                h_fdc_max = np.nanmax(obs_env_arr, axis=1)
                ax.fill_between(
                    obs_env_pos,
                    h_fdc_min,
                    h_fdc_max,
                    color=COLORS["observed"],
                    alpha=0.3,
                    label="Observed Annual FDC Range",
                    **kwargs,
                )

        ax.plot(
            h_pos,
            h_total_fdc,
            color=COLORS["observed"],
            linewidth=STYLE["observed_linewidth"],
            label="Observed Total FDC",
            **kwargs,
        )

    # Set labels
    if xlabel is None:
        xlabel = "Non-Exceedance Probability"
    if ylabel is None:
        ylabel = get_ylabel(units, log_scale)
    if title is None:
        title = f"Flow Duration Curve - {site_name}"

    # Apply styling
    apply_default_styling(ax, title, xlabel, ylabel, legend, grid, log_scale)

    # Tight layout
    if LAYOUT["tight_layout"]:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax


def plot_histogram(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    bins: Union[int, str] = "auto",
    density: bool = True,
    show_kde: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT["default_figsize"],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    log_x: bool = False,
    units: str = "cms",
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of flow values with optional KDE.

    Shows distribution of all flow values across ensemble and optionally
    overlays kernel density estimate.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series, optional
        Observed timeseries for comparison
    site : str, optional
        Site name to plot. If None, uses first site in ensemble
    bins : int or str, default 'auto'
        Number of bins or binning strategy ('auto', 'sturges', 'fd', etc.)
    density : bool, default True
        If True, plot probability density; else raw counts
    show_kde : bool, default True
        Overlay kernel density estimate
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default from config
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates
    xlabel : str, optional
        X-axis label. If None, uses units
    ylabel : str, optional
        Y-axis label. If None, uses 'Density' or 'Count'
    legend : bool, default True
        Whether to display legend
    grid : bool, default True
        Whether to display grid
    log_x : bool, default False
        Use logarithmic x-axis
    units : str, default 'cms'
        Flow units for x-axis label
    filename : str, optional
        Path to save figure
    dpi : int, default from config
        Resolution for saved figure
    **kwargs
        Forwarded to `ax.hist` and `ax.plot` (KDE overlay).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> plot_histogram(ensemble, observed=Q_obs, bins=50)
    >>> plot_histogram(ensemble, density=False, show_kde=False)
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    observed = validate_observed_input(observed, required=False)

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Flatten ensemble data
    ensemble_values = site_data.values.flatten()
    ensemble_values = ensemble_values[~np.isnan(ensemble_values)]

    # Plot ensemble histogram
    ax.hist(
        ensemble_values,
        bins=bins,
        density=density,
        color=COLORS["ensemble_fill"],
        alpha=0.6,
        edgecolor=COLORS["ensemble_median"],
        label="Ensemble",
        **kwargs,
    )

    # Plot observed histogram if provided
    if observed is not None:
        obs_values = observed.dropna().values
        ax.hist(
            obs_values,
            bins=bins,
            density=density,
            color=COLORS["observed"],
            alpha=0.5,
            edgecolor="black",
            linewidth=1.5,
            label="Observed",
            **kwargs,
        )

    # Add KDE if requested
    if show_kde:
        # Ensemble KDE
        kde_ens = gaussian_kde(ensemble_values)
        x_range = np.linspace(ensemble_values.min(), ensemble_values.max(), 200)
        ax.plot(
            x_range,
            kde_ens(x_range),
            color=COLORS["ensemble_median"],
            linewidth=STYLE["ensemble_linewidth"],
            label="Ensemble KDE",
        )

        # Observed KDE
        if observed is not None:
            obs_values = observed.dropna().values
            kde_obs = gaussian_kde(obs_values)
            x_range_obs = np.linspace(obs_values.min(), obs_values.max(), 200)
            ax.plot(
                x_range_obs,
                kde_obs(x_range_obs),
                color=COLORS["observed"],
                linewidth=STYLE["observed_linewidth"],
                label="Observed KDE",
            )

    # Set labels
    if xlabel is None:
        xlabel = LABELS["flow_units"].get(units, f"Flow ({units})")
    if ylabel is None:
        ylabel = "Density" if density else "Count"
    if title is None:
        title = f"Flow Distribution - {site_name}"

    # Apply styling (don't use log_scale for y-axis; apply to x-axis instead)
    apply_default_styling(ax, title, xlabel, ylabel, legend, grid, log_scale=False)
    if log_x:
        ax.set_xscale("log")

    # Tight layout
    if LAYOUT["tight_layout"]:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax


def plot_cdf(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    show_annual_range: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT["default_figsize"],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    log_scale: bool = False,
    units: str = "cms",
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot empirical cumulative distribution function with ensemble uncertainty.

    Shows the empirical CDF (cumulative non-exceedance probability) of flow
    values, with the ensemble band reflecting the spread of annual CDFs
    across realizations. Companion to `plot_flow_duration_curve`: same data,
    different plotting convention (CDF y-axis is cumulative probability,
    x-axis is flow value).

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data.
    observed : pd.Series, optional
        Observed timeseries for comparison.
    site : str, optional
        Site name to plot. If None, uses first site in ensemble.
    show_annual_range : bool, default True
        Show range of annual CDFs as a shaded region across the x-axis.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    figsize : tuple, default from config
        Figure size (width, height) in inches.
    title : str, optional
        Plot title. If None, auto-generates.
    xlabel : str, optional
        X-axis label. If None, uses units.
    ylabel : str, optional
        Y-axis label. If None, uses 'Cumulative Probability'.
    legend : bool, default True
        Whether to display legend.
    grid : bool, default True
        Whether to display grid.
    log_scale : bool, default False
        Use logarithmic x-axis (CDF y-axis is always 0-1).
    units : str, default 'cms'
        Flow units for x-axis label.
    filename : str, optional
        Path to save figure.
    dpi : int, default from config
        Resolution for saved figure.
    **kwargs
        Forwarded to `ax.plot` and `ax.fill_betweenx`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> fig, ax = plot_cdf(ensemble, observed=Q_obs)
    >>> plot_cdf(ensemble, show_annual_range=False, log_scale=True)
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    observed = validate_observed_input(observed, required=False)

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Filter complete years
    site_data = filter_complete_years(site_data)
    if observed is not None:
        observed = filter_complete_years(observed.to_frame()).iloc[:, 0]

    # Define cumulative non-exceedance probabilities
    nonexceedance = np.linspace(0.0001, 0.9999, 50)

    # Pooled ensemble CDF (quantiles of pooled flows give CDF inverse)
    s_total_cdf = np.nanquantile(site_data.values.flatten(), nonexceedance)

    # Annual CDFs across all realizations and years for the band
    if show_annual_range:
        s_annual_cdfs = (
            site_data.groupby(site_data.index.year)
            .quantile(nonexceedance)
            .unstack(level=0)
        )
        s_cdf_min = s_annual_cdfs.min(axis=1)
        s_cdf_max = s_annual_cdfs.max(axis=1)

        # Plot annual range as horizontal band (CDF y is fixed, x varies)
        ax.fill_betweenx(
            nonexceedance,
            s_cdf_min,
            s_cdf_max,
            color=COLORS["ensemble_fill"],
            alpha=STYLE["fill_alpha"],
            label="Ensemble Annual CDF Range",
            **kwargs,
        )

    # Plot pooled ensemble CDF
    ax.plot(
        s_total_cdf,
        nonexceedance,
        color=COLORS["ensemble_median"],
        linewidth=STYLE["ensemble_linewidth"],
        label="Ensemble Total CDF",
        **kwargs,
    )

    # Plot observed if provided
    if observed is not None:
        h_total_cdf = np.nanquantile(observed.values.flatten(), nonexceedance)

        if show_annual_range:
            h_annual_cdfs = (
                observed.groupby(observed.index.year)
                .quantile(nonexceedance)
                .unstack(level=0)
            )
            h_cdf_min = h_annual_cdfs.min(axis=1)
            h_cdf_max = h_annual_cdfs.max(axis=1)

            ax.fill_betweenx(
                nonexceedance,
                h_cdf_min,
                h_cdf_max,
                color=COLORS["observed"],
                alpha=0.3,
                label="Observed Annual CDF Range",
                **kwargs,
            )

        ax.plot(
            h_total_cdf,
            nonexceedance,
            color=COLORS["observed"],
            linewidth=STYLE["observed_linewidth"],
            label="Observed Total CDF",
            **kwargs,
        )

    # Set labels
    if xlabel is None:
        xlabel = LABELS["flow_units"].get(units, f"Flow ({units})")
    if ylabel is None:
        ylabel = "Cumulative Probability"
    if title is None:
        title = f"Empirical CDF - {site_name}"

    # Apply styling (log_scale here means x-axis since y is bounded 0-1)
    apply_default_styling(ax, title, xlabel, ylabel, legend, grid, log_scale=False)
    if log_scale:
        ax.set_xscale("log")

    # Tight layout
    if LAYOUT["tight_layout"]:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax


def plot_monthly_distributions(
    ensemble: Ensemble,
    observed: Optional[pd.Series] = None,
    site: Optional[str] = None,
    plot_type: str = "box",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT["wide_figsize"],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    log_scale: bool = False,
    units: str = "cms",
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot distribution of flows for each month.

    Shows monthly boxplots or violin plots comparing
    ensemble and observed distributions.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object containing synthetic data
    observed : pd.Series, optional
        Observed timeseries for comparison
    site : str, optional
        Site name to plot. If None, uses first site in ensemble
    plot_type : str, default 'box'
        Type of distribution plot: 'box' or 'violin'
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    figsize : tuple, default from config
        Figure size (width, height) in inches
    title : str, optional
        Plot title. If None, auto-generates
    xlabel : str, optional
        X-axis label. If None, uses 'Month'
    ylabel : str, optional
        Y-axis label. If None, uses units
    legend : bool, default True
        Whether to display legend
    grid : bool, default True
        Whether to display grid
    log_scale : bool, default False
        Use logarithmic y-axis
    units : str, default 'cms'
        Flow units for y-axis label
    filename : str, optional
        Path to save figure
    dpi : int, default from config
        Resolution for saved figure
    **kwargs
        Forwarded to `ax.boxplot` or `ax.violinplot`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> plot_monthly_distributions(ensemble, observed=Q_obs)
    >>> plot_monthly_distributions(ensemble, plot_type='violin')
    """
    # Validate inputs
    validate_ensemble_input(ensemble)
    observed = validate_observed_input(observed, required=False)

    if plot_type not in ["box", "violin"]:
        raise ValueError(f"plot_type must be 'box' or 'violin', got '{plot_type}'")

    # Setup axes
    fig, ax = setup_axes(ax, figsize)

    # Get site data
    site_data, site_name = get_site_data(ensemble, site)

    # Resample to monthly
    site_data_monthly = resample_data(site_data, "monthly")

    # Prepare data by month
    ensemble_by_month = []
    for month in range(1, 13):
        month_mask = site_data_monthly.index.month == month
        month_data = site_data_monthly[month_mask].values.flatten()
        month_data = month_data[~np.isnan(month_data)]
        ensemble_by_month.append(month_data)

    # Prepare observed data by month if provided
    obs_by_month = None
    if observed is not None:
        obs_monthly = resample_data(observed.to_frame(), "monthly").iloc[:, 0]
        obs_by_month = []
        for month in range(1, 13):
            month_mask = obs_monthly.index.month == month
            month_data = obs_monthly[month_mask].values
            month_data = month_data[~np.isnan(month_data)]
            obs_by_month.append(month_data)

    # Create positions for boxplots
    positions_ens = np.arange(1, 13) - 0.2
    positions_obs = np.arange(1, 13) + 0.2

    if plot_type == "box":
        # Ensemble boxplots
        bp_ens = ax.boxplot(
            ensemble_by_month,
            positions=positions_ens,
            widths=0.35,
            patch_artist=True,
            boxprops=dict(facecolor=COLORS["ensemble_fill"], alpha=0.7),
            medianprops=dict(color=COLORS["ensemble_median"], linewidth=2),
            whiskerprops=dict(color=COLORS["ensemble_median"]),
            capprops=dict(color=COLORS["ensemble_median"]),
            **kwargs,
        )

        # Observed boxplots
        if obs_by_month is not None:
            bp_obs = ax.boxplot(
                obs_by_month,
                positions=positions_obs,
                widths=0.35,
                patch_artist=True,
                boxprops=dict(facecolor=COLORS["observed"], alpha=0.5),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(color=COLORS["observed"]),
                capprops=dict(color=COLORS["observed"]),
                **kwargs,
            )

    elif plot_type == "violin":
        # Violin plots for ensemble
        parts_ens = ax.violinplot(
            ensemble_by_month,
            positions=positions_ens,
            widths=0.35,
            showmeans=False,
            showmedians=True,
        )
        for pc in parts_ens["bodies"]:
            pc.set_facecolor(COLORS["ensemble_fill"])
            pc.set_alpha(0.7)

        # Violin plots for observed
        if obs_by_month is not None:
            parts_obs = ax.violinplot(
                obs_by_month,
                positions=positions_obs,
                widths=0.35,
                showmeans=False,
                showmedians=True,
            )
            for pc in parts_obs["bodies"]:
                pc.set_facecolor(COLORS["observed"])
                pc.set_alpha(0.5)

    # Add legend manually
    if legend:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=COLORS["ensemble_fill"], alpha=0.7, label="Ensemble"),
        ]
        if obs_by_month is not None:
            legend_elements.append(
                Patch(facecolor=COLORS["observed"], alpha=0.5, label="Observed")
            )
        ax.legend(handles=legend_elements, fontsize=LAYOUT["legend_fontsize"])

    # Set x-axis labels to month names
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(LABELS["month_labels"])

    # Set labels
    if xlabel is None:
        xlabel = "Month"
    if ylabel is None:
        ylabel = get_ylabel(units, log_scale)
    if title is None:
        title = f"Monthly Flow Distributions - {site_name}"

    # Apply styling
    apply_default_styling(
        ax, title, xlabel, ylabel, legend=False, grid=grid, log_scale=log_scale
    )

    # Tight layout
    if LAYOUT["tight_layout"]:
        fig.tight_layout()

    # Save if filename provided
    if filename is not None:
        save_figure(fig, filename, dpi)

    return fig, ax
