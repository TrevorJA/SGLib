"""
Plots for verification and validation results.

Both functions consume the tidy frame produced by
``synhydro.verification.verify`` or ``synhydro.validation.validate``.
The presentation follows the standard convention of the synthetic
streamflow literature: the distribution of a statistic across
realizations shown against the observed value (Stedinger and Taylor,
1982).
"""

import logging
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from synhydro._evaluation import EvaluationResult, SKIP_COLUMNS
from synhydro.plotting.config import COLORS, STYLE, LAYOUT
from synhydro.plotting._utils import apply_default_styling, save_figure, setup_axes

logger = logging.getLogger(__name__)

FrameOrResult = Union[EvaluationResult, pd.DataFrame]


def _as_result(values: FrameOrResult) -> EvaluationResult:
    """Wrap a tidy frame in an EvaluationResult if needed."""
    if isinstance(values, EvaluationResult):
        return values
    if isinstance(values, pd.DataFrame):
        return EvaluationResult(
            values=values, skipped=pd.DataFrame(columns=SKIP_COLUMNS)
        )
    raise TypeError(
        f"Expected an EvaluationResult or tidy DataFrame, got {type(values)}."
    )


def plot_metric_distributions(
    result: FrameOrResult,
    metrics: Optional[List[str]] = None,
    sites: Optional[List[str]] = None,
    ncols: int = 3,
    annotate_rank: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Boxplots of scalar metrics across realizations with observed overlay.

    One panel per (metric, site): the distribution of the statistic
    across realizations as a boxplot, the observed value as a dashed
    line, and the observed value's percentile position within the
    ensemble as an annotation. Only scalar and matrix (site-pair)
    metrics are shown; curve metrics use :func:`plot_metric_curve`.

    Parameters
    ----------
    result : VerificationResult, ValidationResult, or pd.DataFrame
        Result object or tidy frame from ``verify()`` or ``validate()``.
    metrics : list of str, optional
        Metric names to plot. Defaults to all scalar metrics present.
    sites : list of str, optional
        Sites (or site-pair labels) to plot. Defaults to all present.
    ncols : int, default 3
        Number of panel columns.
    annotate_rank : bool, default True
        Annotate each panel with the observed value's percentile
        position within the synthetic distribution.
    figsize : tuple, optional
        Figure size; defaults to a size scaled to the panel grid.
    filename : str, optional
        Path to save the figure.
    dpi : int, default from config
        Resolution for the saved figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes

    References
    ----------
    Stedinger, J.R. and Taylor, M.R. (1982). Synthetic streamflow
    generation: 1. Model verification and validation. Water Resources
    Research, 18(4), 909-918.

    Examples
    --------
    >>> result = verify(ensemble, Q_obs, metrics=["marginal"])
    >>> fig, axes = plot_metric_distributions(result)
    """
    result = _as_result(result)
    summary = result.summary()
    frame = result.values

    scalar_rows = summary[summary["kind"].isin(["scalar", "matrix"])]
    if metrics is not None:
        scalar_rows = scalar_rows[scalar_rows["metric"].isin(metrics)]
    if sites is not None:
        scalar_rows = scalar_rows[scalar_rows["site"].isin(sites)]
    if scalar_rows.empty:
        raise ValueError(
            "No scalar or matrix metrics match the requested metrics/sites."
        )

    panels = list(scalar_rows[["metric", "site"]].itertuples(index=False))
    nrows = math.ceil(len(panels) / ncols)
    if figsize is None:
        figsize = (3.2 * ncols, 2.6 * nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=LAYOUT["default_dpi"], squeeze=False
    )

    for index, (metric, site) in enumerate(panels):
        ax = axes[index // ncols][index % ncols]
        values = frame[(frame["metric"] == metric) & (frame["site"] == site)][
            "value"
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        row = scalar_rows[
            (scalar_rows["metric"] == metric) & (scalar_rows["site"] == site)
        ].iloc[0]

        if len(values) > 0:
            bp = ax.boxplot([values], widths=0.5, sym="", patch_artist=True)
            plt.setp(
                bp["boxes"],
                color=COLORS["ensemble_fill"],
                facecolor=COLORS["ensemble_fill"],
            )
            plt.setp(bp["whiskers"], color=COLORS["ensemble_fill"])
            plt.setp(bp["caps"], color=COLORS["ensemble_fill"])
            plt.setp(
                bp["medians"],
                color=STYLE["boxplot_median_color"],
                linewidth=STYLE["boxplot_median_linewidth"],
            )

        observed = row["observed"]
        if np.isfinite(observed):
            ax.axhline(
                observed,
                color=COLORS["observed"],
                linestyle="--",
                linewidth=STYLE["observed_linewidth"],
            )
            if annotate_rank and np.isfinite(row["obs_percentile"]):
                ax.annotate(
                    f"obs at {row['obs_percentile']:.2f}",
                    xy=(0.97, 0.95),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=LAYOUT["tick_fontsize"],
                )

        ax.set_title(f"{metric}\n{site}", fontsize=LAYOUT["tick_fontsize"] + 1)
        ax.set_xticks([])
        apply_default_styling(ax, legend=False, grid=True)

    # Hide any unused panels
    for index in range(len(panels), nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)

    fig.tight_layout()
    if filename is not None:
        save_figure(fig, filename, dpi)
    return fig, axes


def plot_metric_curve(
    result: FrameOrResult,
    metric: str,
    site: str,
    ax: Optional[plt.Axes] = None,
    log_y: Optional[bool] = None,
    figsize: Tuple[float, float] = LAYOUT["default_figsize"],
    filename: Optional[str] = None,
    dpi: int = LAYOUT["save_dpi"],
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Ensemble band and observed overlay for a curve metric.

    Shows the 5th to 95th percentile band and median of the metric
    across realizations at each component (calendar month, lag,
    exceedance probability, or period band), with the observed curve
    overlaid.

    Parameters
    ----------
    result : VerificationResult, ValidationResult, or pd.DataFrame
        Result object or tidy frame from ``verify()`` or ``validate()``.
    metric : str
        Curve metric name (e.g. ``'monthly_mean'``, ``'acf'``,
        ``'fdc'``, ``'spectral_density'``).
    site : str
        Site to plot.
    ax : plt.Axes, optional
        Existing axes to draw on.
    log_y : bool, optional
        Log-scale the y axis. Defaults to True for ``'fdc'``.
    figsize : tuple, default from config
        Figure size when creating a new figure.
    filename : str, optional
        Path to save the figure.
    dpi : int, default from config
        Resolution for the saved figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> result = verify(ensemble, Q_obs, metrics=["fdc", "acf"])
    >>> fig, ax = plot_metric_curve(result, "fdc", "site_A")
    """
    result = _as_result(result)
    frame = result.values
    rows = frame[(frame["metric"] == metric) & (frame["site"] == site)]
    if rows.empty:
        raise ValueError(f"No values found for metric '{metric}' at site '{site}'.")
    if not (rows["kind"] == "curve").all() and not (rows["kind"] == "comparison").all():
        kinds = sorted(rows["kind"].unique())
        if "curve" not in kinds:
            raise ValueError(
                f"Metric '{metric}' has kind {kinds}; plot_metric_curve "
                f"requires a curve metric."
            )
    rows = rows[rows["component"].notna()]
    if rows.empty:
        raise ValueError(f"Metric '{metric}' has no per-component values.")

    if log_y is None:
        log_y = metric == "fdc"

    components = list(pd.unique(rows["component"]))
    numeric = all(
        isinstance(c, (int, float, np.integer, np.floating)) for c in components
    )
    positions = (
        np.array(components, dtype=float) if numeric else np.arange(len(components))
    )

    median_curve, q05_curve, q95_curve, observed_curve = [], [], [], []
    for component in components:
        group = rows[rows["component"] == component]
        values = group["value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) > 0:
            median_curve.append(np.median(values))
            q05_curve.append(np.percentile(values, 5))
            q95_curve.append(np.percentile(values, 95))
        else:
            median_curve.append(np.nan)
            q05_curve.append(np.nan)
            q95_curve.append(np.nan)
        observed = group["observed"].iloc[0]
        observed_curve.append(float(observed) if pd.notna(observed) else np.nan)

    fig, ax = setup_axes(ax, figsize)
    ax.fill_between(
        positions,
        q05_curve,
        q95_curve,
        color=COLORS["ensemble_fill"],
        alpha=STYLE["fill_alpha"],
        label="Ensemble 5-95%",
    )
    ax.plot(
        positions,
        median_curve,
        color=COLORS["ensemble_median"],
        linewidth=STYLE["ensemble_linewidth"],
        label="Ensemble median",
    )
    if np.any(np.isfinite(observed_curve)):
        ax.plot(
            positions,
            observed_curve,
            color=COLORS["observed"],
            linewidth=STYLE["observed_linewidth"],
            marker=STYLE["observed_marker"],
            markersize=STYLE["observed_markersize"],
            label="Observed",
        )

    if not numeric:
        ax.set_xticks(positions)
        ax.set_xticklabels(components, fontsize=LAYOUT["tick_fontsize"])

    ax.set_xlabel("Component", fontsize=LAYOUT["label_fontsize"])
    ax.set_ylabel(metric, fontsize=LAYOUT["label_fontsize"])
    ax.set_title(f"{metric} - {site}", fontsize=LAYOUT["title_fontsize"])
    apply_default_styling(ax, legend=True, grid=True, log_scale=log_y)

    if filename is not None:
        save_figure(fig, filename, dpi)
    return fig, ax
