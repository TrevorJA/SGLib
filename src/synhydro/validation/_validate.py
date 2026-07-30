"""
Validation orchestrator.
"""

import logging
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from synhydro.core.ensemble import Ensemble
from synhydro._evaluation import (
    resolve_frequency,
    check_observed_frequency,
    resolve_sites,
    VALUE_COLUMNS,
    SKIP_COLUMNS,
)
from synhydro.validation._result import ValidationResult
from synhydro.validation.metrics.threshold_drought import compute_threshold_drought
from synhydro.validation.metrics.ssi_drought import compute_ssi_drought

logger = logging.getLogger(__name__)

CATEGORIES = ("threshold_drought", "ssi_drought")


def _select_categories(
    metrics: Union[str, Iterable[str], None],
) -> list[str]:
    """Resolve a category selection, rejecting None and unknown names."""
    if metrics is None:
        raise ValueError(
            f"No metrics selected. Pass metrics='all' or a list of "
            f"categories from {list(CATEGORIES)}."
        )
    if isinstance(metrics, str):
        if metrics == "all":
            return list(CATEGORIES)
        metrics = [metrics]
    selected = []
    for item in metrics:
        if item not in CATEGORIES:
            raise ValueError(
                f"Unknown validation category '{item}'. Available "
                f"categories: {list(CATEGORIES)}."
            )
        if item not in selected:
            selected.append(item)
    if not selected:
        raise ValueError(
            f"Metric selection is empty. Pass metrics='all' or a list of "
            f"categories from {list(CATEGORIES)}."
        )
    return selected


def validate(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    metrics: Union[str, Iterable[str]],
    sites: Optional[list[str]] = None,
    frequency: Optional[str] = None,
    drought_threshold: Optional[float] = None,
    ssi_timescale: int = 12,
    ssi_dist: str = "gamma",
) -> ValidationResult:
    """
    Validate fit-for-purpose drought behavior of a synthetic ensemble.

    Each drought statistic is computed once on the observed record and
    once per realization; the result reports the observed value against
    the distribution across realizations, as in
    :func:`synhydro.verification.verify`.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic streamflow ensemble.
    Q_obs : pd.DataFrame
        Observed streamflow with a DatetimeIndex and sites as columns.
        Must have the same frequency as the ensemble.
    metrics : str or list of str
        ``'all'`` or a list of categories: ``'threshold_drought'``
        (run-theory events below a flow threshold) and
        ``'ssi_drought'`` (events on the Standardized Streamflow
        Index).
    sites : list of str, optional
        Subset of sites. Defaults to all sites shared by the ensemble
        and observed data.
    frequency : str, optional
        Frequency override; see :func:`synhydro.verification.verify`.
    drought_threshold : float, optional
        Flow threshold for threshold drought identification, applied
        to every site. If None, each site uses the 20th percentile of
        its observed flows.
    ssi_timescale : int, default 12
        SSI accumulation timescale in months.
    ssi_dist : str, default 'gamma'
        Distribution used for the SSI fit.

    Returns
    -------
    ValidationResult
        Tidy per-realization values, skipped-metric log, and metadata.

    Raises
    ------
    ValueError
        If no metrics are selected, no shared sites exist, or the
        observed frequency does not match the ensemble frequency.

    References
    ----------
    Stedinger, J.R. and Taylor, M.R. (1982). Synthetic streamflow
    generation: 1. Model verification and validation. Water Resources
    Research, 18(4), 909-918.

    Examples
    --------
    >>> result = validate(ensemble, Q_obs, metrics="all")
    >>> result.summary()
    """
    categories = _select_categories(metrics)
    resolved_sites = resolve_sites(ensemble, Q_obs, sites)

    first_realization = next(iter(ensemble.data_by_realization.values()))
    frequency_info = resolve_frequency(
        frequency, ensemble.frequency, first_realization.index
    )
    check_observed_frequency(Q_obs.index, frequency_info)

    logger.info(
        "Validating %d drought categories at %d sites (%s frequency, "
        "%d realizations).",
        len(categories),
        len(resolved_sites),
        frequency_info.base,
        len(ensemble.data_by_realization),
    )

    rows: list[tuple] = []
    skips: list[tuple] = []
    if "threshold_drought" in categories:
        new_rows, new_skips = compute_threshold_drought(
            ensemble, Q_obs, resolved_sites, drought_threshold
        )
        rows.extend(new_rows)
        skips.extend(new_skips)
    if "ssi_drought" in categories:
        new_rows, new_skips = compute_ssi_drought(
            ensemble,
            Q_obs,
            resolved_sites,
            ssi_timescale=ssi_timescale,
            ssi_dist=ssi_dist,
        )
        rows.extend(new_rows)
        skips.extend(new_skips)

    values = pd.DataFrame(rows, columns=VALUE_COLUMNS)
    skipped = pd.DataFrame(skips, columns=SKIP_COLUMNS)

    metadata = {
        "suite": "validation",
        "n_realizations": len(ensemble.data_by_realization),
        "n_sites": len(resolved_sites),
        "sites": list(resolved_sites),
        "base_frequency": frequency_info.base,
        "steps_per_year": frequency_info.steps_per_year,
        "n_obs_years": float(len(Q_obs) / frequency_info.steps_per_year),
        "options": {
            "drought_threshold": drought_threshold,
            "ssi_timescale": ssi_timescale,
            "ssi_dist": ssi_dist,
        },
        "reject_rate_metrics": [],
        "obs_site_median_flow": {
            site: float(np.nanmedian(Q_obs[site].to_numpy(dtype=float)))
            for site in resolved_sites
        },
    }

    return ValidationResult(values=values, skipped=skipped, metadata=metadata)
