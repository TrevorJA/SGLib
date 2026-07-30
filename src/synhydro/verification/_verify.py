"""
Verification orchestrator.
"""

import logging
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd

from synhydro.core.ensemble import Ensemble
from synhydro._evaluation import (
    MetricContext,
    resolve_frequency,
    check_observed_frequency,
    resolve_sites,
    run_metrics,
)
from synhydro.verification._registry import VERIFICATION_METRICS
from synhydro.verification._result import VerificationResult

logger = logging.getLogger(__name__)

_DEFAULT_ACF_LAGS = {"daily": 30, "weekly": 26, "monthly": 12, "annual": 10}


def verify(
    ensemble: Ensemble,
    observed: pd.DataFrame,
    metrics: Union[str, Iterable[Union[str, Callable]]],
    sites: Optional[list[str]] = None,
    frequency: Optional[str] = None,
    hurst_method: str = "rs",
    acf_lags: Optional[int] = None,
) -> VerificationResult:
    """
    Verify statistical property preservation of a synthetic ensemble.

    Each selected metric is computed once on the observed record and
    once on every realization. The result reports the observed
    statistic against the distribution of the statistic across
    realizations, including the observed value's rank position within
    the ensemble (Stedinger and Taylor, 1982).

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic streamflow ensemble.
    observed : pd.DataFrame
        Observed streamflow with a DatetimeIndex and sites as columns.
        Must have the same frequency as the ensemble.
    metrics : str, or list of str and callables
        Metric selection. ``'all'`` computes every registered metric.
        A list may mix metric names, category names (``'marginal'``,
        ``'temporal'``, ``'seasonal'``, ``'annual'``, ``'spatial'``,
        ``'fdc'``, ``'lmoments'``, ``'extremes'``, ``'spectral'``), and
        callables with signature ``f(x: pd.Series) -> float``.
    sites : list of str, optional
        Subset of sites to verify. Defaults to all sites shared by the
        ensemble and observed data.
    frequency : str, optional
        Frequency override (pandas alias or ``'daily'``, ``'weekly'``,
        ``'monthly'``, ``'annual'``). By default the frequency is taken
        from ensemble metadata, checked against the index, and inferred
        from the index if metadata is missing.
    hurst_method : {'rs', 'dfa'}, default 'rs'
        Hurst exponent estimation method.
    acf_lags : int, optional
        Maximum lag for the ``acf`` metric. Defaults to 30 for daily,
        26 for weekly, 12 for monthly, and 10 for annual data.

    Returns
    -------
    VerificationResult
        Tidy per-realization values, skipped-metric log, and metadata.
        Use ``.summary()`` for the per-metric comparison table and
        ``.category_summary()`` for the per-category rollup.

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
    >>> result = verify(ensemble, Q_obs, metrics="all")
    >>> result.summary()
    >>> result = verify(ensemble, Q_obs, metrics=["marginal", "acf"])
    """
    specs = VERIFICATION_METRICS.select(metrics)
    resolved_sites = resolve_sites(ensemble, observed, sites)

    first_realization = next(iter(ensemble.data_by_realization.values()))
    frequency_info = resolve_frequency(
        frequency, ensemble.frequency, first_realization.index
    )
    check_observed_frequency(observed.index, frequency_info)

    if acf_lags is None:
        acf_lags = _DEFAULT_ACF_LAGS[frequency_info.base]

    context = MetricContext(
        frequency=frequency_info,
        hurst_method=hurst_method,
        acf_lags=acf_lags,
    )

    logger.info(
        "Verifying %d metrics at %d sites (%s frequency, %d realizations).",
        len(specs),
        len(resolved_sites),
        frequency_info.base,
        len(ensemble.data_by_realization),
    )

    values, skipped = run_metrics(ensemble, observed, specs, context, resolved_sites)

    metadata = {
        "suite": "verification",
        "n_realizations": len(ensemble.data_by_realization),
        "n_sites": len(resolved_sites),
        "sites": list(resolved_sites),
        "base_frequency": frequency_info.base,
        "steps_per_year": frequency_info.steps_per_year,
        "n_obs_years": float(len(observed) / frequency_info.steps_per_year),
        "options": {"hurst_method": hurst_method, "acf_lags": acf_lags},
        "reject_rate_metrics": [
            spec.name for spec in specs if spec.summary_mode == "reject_rate"
        ],
        "obs_site_median_flow": {
            site: float(np.nanmedian(observed[site].to_numpy(dtype=float)))
            for site in resolved_sites
        },
    }

    return VerificationResult(values=values, skipped=skipped, metadata=metadata)
