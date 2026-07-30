"""
Verification suite: statistical property preservation.

Verification, in the sense of Stedinger and Taylor (1982), demonstrates
that generated flows reproduce the statistics the generator was designed
to reproduce: moments, correlations, and distributional shape. For
fit-for-purpose evaluation of characteristics not explicitly fit
(droughts), see :mod:`synhydro.validation`.

The suite has three layers:

1. Flat metric functions (``synhydro.verification.mean``,
   ``synhydro.verification.acf``, ...) directly callable on a series.
2. The :func:`verify` orchestrator, which evaluates selected metrics on
   the observed record and every realization and returns a
   :class:`VerificationResult`.
3. The tidy long-format DataFrame (``result.to_dataframe()``) used by
   plotting and the bootstrap tools.

References
----------
Stedinger, J.R. and Taylor, M.R. (1982). Synthetic streamflow
generation: 1. Model verification and validation. Water Resources
Research, 18(4), 909-918.
"""

from typing import Callable, Iterable, Optional

import pandas as pd

from synhydro.verification._registry import VERIFICATION_METRICS
from synhydro.verification._result import VerificationResult
from synhydro.verification._verify import verify
from synhydro.verification._testing import bootstrap_metric_ci, compare_methods
from synhydro.verification.metrics import (
    mean,
    std,
    cv,
    skewness,
    kurtosis,
    minimum,
    maximum,
    flow_q10,
    flow_q50,
    flow_q90,
    ks_statistic,
    lag1_autocorrelation,
    lag2_autocorrelation,
    acf,
    hurst,
    monthly_mean,
    monthly_std,
    monthly_skewness,
    monthly_maximum,
    monthly_minimum,
    monthly_lag1_correlation,
    monthly_ranksum_pvalue,
    monthly_levene_pvalue,
    annual_mean,
    annual_sd,
    annual_cv,
    annual_skewness,
    annual_lag1_autocorrelation,
    annual_minimum,
    annual_maximum,
    cross_correlation,
    cross_correlation_lag1,
    fdc,
    fdc_log_rmse,
    l_cv,
    l_skewness,
    l_kurtosis,
    annual_max_mean,
    annual_max_cv,
    gev_rp10,
    gev_rp50,
    gev_rp100,
    annual_min_mean,
    annual_min_cv,
    seven_day_min_mean,
    seven_day_min_cv,
    spectral_density,
    low_frequency_variance_fraction,
)

__all__ = [
    "verify",
    "VerificationResult",
    "register_metric",
    "list_metrics",
    "bootstrap_metric_ci",
    "compare_methods",
    # Marginal
    "mean",
    "std",
    "cv",
    "skewness",
    "kurtosis",
    "minimum",
    "maximum",
    "flow_q10",
    "flow_q50",
    "flow_q90",
    "ks_statistic",
    # Temporal
    "lag1_autocorrelation",
    "lag2_autocorrelation",
    "acf",
    "hurst",
    # Seasonal
    "monthly_mean",
    "monthly_std",
    "monthly_skewness",
    "monthly_maximum",
    "monthly_minimum",
    "monthly_lag1_correlation",
    "monthly_ranksum_pvalue",
    "monthly_levene_pvalue",
    # Annual
    "annual_mean",
    "annual_sd",
    "annual_cv",
    "annual_skewness",
    "annual_lag1_autocorrelation",
    "annual_minimum",
    "annual_maximum",
    # Spatial
    "cross_correlation",
    "cross_correlation_lag1",
    # Flow duration curve
    "fdc",
    "fdc_log_rmse",
    # L-moments
    "l_cv",
    "l_skewness",
    "l_kurtosis",
    # Extremes
    "annual_max_mean",
    "annual_max_cv",
    "gev_rp10",
    "gev_rp50",
    "gev_rp100",
    "annual_min_mean",
    "annual_min_cv",
    "seven_day_min_mean",
    "seven_day_min_cv",
    # Spectral
    "spectral_density",
    "low_frequency_variance_fraction",
]


def register_metric(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    category: str = "custom",
    kind: str = "scalar",
    needs: Iterable[str] = (),
    frequencies: Optional[Iterable[str]] = None,
    min_years: Optional[float] = None,
    units: str = "dimensionless",
    summary_mode: str = "distribution",
    citation: str = "",
    description: str = "",
) -> Callable:
    """
    Register a custom verification metric.

    Usable as a decorator or a plain call. Registered metrics are
    selectable by name in :func:`verify` and appear in
    :func:`list_metrics`.

    Parameters
    ----------
    func : Callable, optional
        Metric function ``f(x: pd.Series, **opts) -> float`` for scalar
        metrics; see :mod:`synhydro._evaluation._registry` kinds for
        curve, matrix, and comparison signatures.
    name : str, optional
        Metric name; defaults to the function name.
    category : str, default 'custom'
        Category label used for grouping and selection.
    kind : str, default 'scalar'
        Metric kind: 'scalar', 'curve', 'matrix', or 'comparison'.
    needs : iterable of str, optional
        Context attributes injected as keyword arguments
        (e.g. ``('steps_per_year',)``).
    frequencies : iterable of str, optional
        Base frequencies the metric supports; None means any.
    min_years : float, optional
        Minimum observed record length in years.
    units : str, default 'dimensionless'
        Units label for reporting.
    summary_mode : str, default 'distribution'
        ``'reject_rate'`` for p-value metrics.
    citation : str, optional
        Short citation shown in the metric inventory.
    description : str, optional
        One-line description; defaults to the first docstring line.

    Returns
    -------
    Callable
        The registered function, unchanged.

    Examples
    --------
    >>> from synhydro.verification import register_metric, verify
    >>> @register_metric(category="custom", units="flow")
    ... def q25(x):
    ...     return float(x.quantile(0.25))
    >>> result = verify(ensemble, Q_obs, metrics=["q25"])
    """
    return VERIFICATION_METRICS.register(
        func,
        name=name,
        category=category,
        kind=kind,
        needs=needs,
        frequencies=frequencies,
        min_years=min_years,
        units=units,
        summary_mode=summary_mode,
        citation=citation,
        description=description,
    )


def list_metrics() -> pd.DataFrame:
    """
    List all registered verification metrics.

    Returns
    -------
    pd.DataFrame
        One row per metric with columns ``name, category, kind, units,
        frequencies, min_years, citation, description``.
    """
    return VERIFICATION_METRICS.to_frame()
