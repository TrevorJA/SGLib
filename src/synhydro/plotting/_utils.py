"""
Internal utility functions for SynHydro plotting module.

These are private helper functions not exposed in the public API.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List
from synhydro.core.ensemble import Ensemble
from synhydro.plotting.config import COLORS, STYLE, LAYOUT, LABELS

logger = logging.getLogger(__name__)


def setup_axes(
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = LAYOUT["default_figsize"],
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Setup figure and axes for plotting.

    Parameters
    ----------
    ax : plt.Axes, optional
        Existing axes to use. If None, creates new figure.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=LAYOUT["default_dpi"])
    else:
        fig = ax.get_figure()
    return fig, ax


def apply_default_styling(
    ax: plt.Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool = True,
    grid: bool = True,
    log_scale: bool = False,
    hide_xticks: bool = False,
) -> None:
    """
    Apply consistent styling to an axes object.

    Parameters
    ----------
    ax : plt.Axes
        Axes to style.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    legend : bool
        Whether to show legend.
    grid : bool
        Whether to show grid.
    log_scale : bool
        Whether to use log scale on y-axis.
    hide_xticks : bool, default False
        If True, hide x-axis tick labels (used for inner panels of
        multi-panel plots that share the same x-axis).
    """
    if title is not None:
        ax.set_title(title, fontsize=LAYOUT["title_fontsize"])

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LAYOUT["label_fontsize"])

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LAYOUT["label_fontsize"])

    if grid:
        ax.grid(
            True,
            linestyle=STYLE["grid_linestyle"],
            linewidth=STYLE["grid_linewidth"],
            alpha=STYLE["grid_alpha"],
        )
    else:
        ax.grid(False)

    if log_scale:
        ax.set_yscale("log")

    if legend:
        ax.legend(fontsize=LAYOUT["legend_fontsize"], frameon=False)

    if hide_xticks:
        ax.set_xticks([])

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, filename: str, dpi: int = LAYOUT["save_dpi"]) -> None:
    """
    Save figure to file.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    filename : str
        Output filename.
    dpi : int
        Resolution for raster formats.
    """
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")


def get_site_data(
    ensemble: Ensemble, site: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Extract data for a single site from ensemble.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object.
    site : str, optional
        Site name. If None, uses first site.

    Returns
    -------
    data : pd.DataFrame
        Data for the site (time x realizations).
    site_name : str
        Name of the site.
    """
    if site is None:
        site = ensemble.site_names[0]
    elif site not in ensemble.site_names:
        raise ValueError(
            f"Site '{site}' not found in ensemble. "
            f"Available sites: {ensemble.site_names}"
        )

    data = ensemble.data_by_site[site]
    return data, site


def get_ylabel(units: str, log_scale: bool = False) -> str:
    """
    Get y-axis label based on units.

    Parameters
    ----------
    units : str
        Flow units.
    log_scale : bool
        Whether using log scale.

    Returns
    -------
    str
        Y-axis label.
    """
    base_label = LABELS["flow_units"].get(units, f"Streamflow ({units})")
    if log_scale:
        return f"log({base_label})"
    return base_label


def subset_date_range(
    data: pd.DataFrame, start_date: Optional[str] = None, end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Subset data by date range.

    Parameters
    ----------
    data : pd.DataFrame
        Data with DatetimeIndex.
    start_date : str, optional
        Start date.
    end_date : str, optional
        End date.

    Returns
    -------
    pd.DataFrame
        Subsetted data.
    """
    if start_date is not None:
        data = data[data.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        data = data[data.index <= pd.to_datetime(end_date)]
    return data


def filter_complete_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to include only complete years.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only complete years.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a pd.DatetimeIndex.")

    # Infer frequency
    freq = infer_datetime_frequency(df)

    min_periods_per_year = {"D": 365, "W": 52, "M": 12, "A": 1}

    df_index = df.index
    complete_years = []
    for year in df_index.year.unique():
        year_mask = df_index.year == year
        if year_mask.sum() >= min_periods_per_year[freq]:
            complete_years.append(year)

    return df[df.index.year.isin(complete_years)]


def infer_datetime_frequency(df: pd.DataFrame) -> str:
    """
    Infer the frequency of a pd.DatetimeIndex DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.

    Returns
    -------
    str
        Frequency string: 'D', 'W', 'ME', or 'YE'.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a pd.DatetimeIndex.")

    # Calculate time differences
    time_diffs = df.index.to_series().diff().dropna()

    # Determine most common frequency
    time_delta = time_diffs.value_counts().idxmax()

    # Infer frequency string
    if time_delta == pd.Timedelta(days=1):
        freq = "D"
    elif time_delta == pd.Timedelta(weeks=1):
        freq = "W"
    elif time_delta in [
        pd.Timedelta(days=28),
        pd.Timedelta(days=29),
        pd.Timedelta(days=30),
        pd.Timedelta(days=31),
    ]:
        freq = "ME"
    elif time_delta in [pd.Timedelta(days=365), pd.Timedelta(days=366)]:
        freq = "YE"
    else:
        raise ValueError(
            f"Unsupported frequency detected with time delta: {time_delta}."
        )

    return freq


def compute_ensemble_percentiles(
    data: pd.DataFrame, percentiles: List[float]
) -> pd.DataFrame:
    """
    Compute percentiles across ensemble members.

    Parameters
    ----------
    data : pd.DataFrame
        Data with realizations as columns.
    percentiles : List[float]
        Percentiles to compute (0-100).

    Returns
    -------
    pd.DataFrame
        DataFrame with percentile columns.
    """
    result = {}
    for p in percentiles:
        result[f"p{p}"] = data.quantile(p / 100, axis=1)
    return pd.DataFrame(result)


def format_date_axis(ax: plt.Axes, data: pd.DataFrame) -> None:
    """
    Format x-axis for date display.

    Parameters
    ----------
    ax : plt.Axes
        Axes object.
    data : pd.DataFrame
        Data with DatetimeIndex.
    """
    import matplotlib.dates as mdates

    # Determine appropriate date format based on data span
    date_range = (data.index.max() - data.index.min()).days

    if date_range <= 31:  # Less than a month
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    elif date_range <= 366:  # Less than a year
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    elif date_range <= 1825:  # Less than 5 years
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def resample_data(data: pd.DataFrame, timestep: str) -> pd.DataFrame:
    """
    Resample data to specified timestep.

    Parameters
    ----------
    data : pd.DataFrame
        Data with DatetimeIndex.
    timestep : str
        Timestep: 'daily', 'weekly', 'monthly', 'annual'.

    Returns
    -------
    pd.DataFrame
        Resampled data.
    """
    freq_map = {"daily": "D", "weekly": "W-SUN", "monthly": "MS", "annual": "AS"}

    if timestep not in freq_map:
        raise ValueError(
            f"Invalid timestep: {timestep}. " f"Must be one of {list(freq_map.keys())}"
        )

    freq = freq_map[timestep]
    return data.resample(freq).sum()


def get_temporal_grouper(data: pd.DataFrame, timestep: str) -> pd.Index:
    """
    Get grouper for temporal aggregation.

    Parameters
    ----------
    data : pd.DataFrame
        Data with DatetimeIndex.
    timestep : str
        Timestep: 'daily', 'weekly', 'monthly', 'annual'.

    Returns
    -------
    pd.Index
        Grouper index.
    """
    if timestep == "daily":
        return data.index.dayofyear
    elif timestep == "weekly":
        return pd.Index(data.index.isocalendar().week, dtype=int)
    elif timestep == "monthly":
        return data.index.month
    elif timestep == "annual":
        return data.index.year
    else:
        raise ValueError(f"Invalid timestep: {timestep}")


def validate_ensemble_input(ensemble: Ensemble) -> None:
    """
    Validate ensemble input.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble object to validate.

    Raises
    ------
    TypeError
        If ensemble is not an Ensemble object.
    ValueError
        If ensemble is empty.
    """
    if not isinstance(ensemble, Ensemble):
        raise TypeError(f"Expected Ensemble object, got {type(ensemble)}")

    if len(ensemble.realization_ids) == 0:
        raise ValueError("Ensemble has no realizations")

    if len(ensemble.site_names) == 0:
        raise ValueError("Ensemble has no sites")


def validate_observed_input(
    observed: Optional[Union[pd.Series, pd.DataFrame]], required: bool = False
) -> Optional[pd.Series]:
    """
    Validate observed data input.

    Parameters
    ----------
    observed : pd.Series, pd.DataFrame, or None
        Observed data.
    required : bool
        Whether observed data is required.

    Returns
    -------
    pd.Series or None
        Validated observed data as Series.

    Raises
    ------
    TypeError
        If observed is wrong type.
    ValueError
        If required but not provided.
    """
    if observed is None:
        if required:
            raise ValueError("Observed data is required for this plot")
        return None

    if isinstance(observed, pd.DataFrame):
        if observed.shape[1] == 1:
            observed = observed.iloc[:, 0]
        else:
            raise ValueError(
                "Observed DataFrame must have single column. "
                f"Got {observed.shape[1]} columns."
            )

    if not isinstance(observed, pd.Series):
        raise TypeError(
            f"Observed must be pd.Series or pd.DataFrame, " f"got {type(observed)}"
        )

    if not isinstance(observed.index, pd.DatetimeIndex):
        raise TypeError("Observed data must have DatetimeIndex")

    return observed


def warn_if_many_realizations(
    n: int, *, threshold: int = 1000, context: str = ""
) -> None:
    """Log a warning when the ensemble has more realizations than threshold.

    Functions that iterate per-realization or render per-realization traces
    can become slow for large N; this surfaces the cost in the log so the
    user knows what's happening.

    Parameters
    ----------
    n : int
        Number of realizations.
    threshold : int, default 1000
        Warn when n exceeds this value.
    context : str, optional
        Short label appended to the warning (e.g. "show_members").
    """
    if n > threshold:
        suffix = f" ({context})" if context else ""
        logger.warning(
            "Plotting %d realizations%s; this may be slow. "
            "Consider show_members=N or aggregating first.",
            n,
            suffix,
        )


def warn_if_few_realizations(
    n: int, *, threshold: int = 30, context: str = "percentile bands"
) -> None:
    """Log a warning when ensemble is too small for stable percentile estimates.

    Parameters
    ----------
    n : int
        Number of realizations.
    threshold : int, default 30
        Warn when n is below this value.
    context : str, default 'percentile bands'
        Short label for the warning.
    """
    if n < threshold:
        logger.warning(
            "Only %d realizations; %s may be unreliable. "
            "Recommend N >= %d for stable estimates.",
            n,
            context,
            threshold,
        )


_TIMESTEP_RANK = {"daily": 0, "weekly": 1, "monthly": 2, "annual": 3}
_FREQ_RANK = {
    "D": 0,
    "B": 0,
    "W": 1,
    "W-MON": 1,
    "W-SUN": 1,
    "M": 2,
    "MS": 2,
    "ME": 2,
    "Q": 2,
    "QS": 2,
    "Y": 3,
    "YS": 3,
    "A": 3,
    "AS": 3,
}


def validate_timestep(ensemble: Ensemble, timestep: str) -> None:
    """Raise if requested timestep is finer than ensemble's native frequency.

    Parameters
    ----------
    ensemble : Ensemble
        Ensemble to check.
    timestep : str
        One of 'daily', 'weekly', 'monthly', 'annual'.

    Raises
    ------
    ValueError
        If `timestep` is finer than `ensemble.frequency`. Coarser is fine
        (the function will resample).
    """
    if ensemble.frequency is None:
        return
    requested = _TIMESTEP_RANK.get(timestep)
    if requested is None:
        return
    base = ensemble.frequency.split("-")[0].upper()
    native = _FREQ_RANK.get(base)
    if native is None:
        return
    if requested < native:
        raise ValueError(
            f"Cannot plot at timestep={timestep!r}: ensemble is at "
            f"frequency={ensemble.frequency!r} which is coarser. "
            f"This would require disaggregation that the plotting module "
            f"does not perform."
        )
