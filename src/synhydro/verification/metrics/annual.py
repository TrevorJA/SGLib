"""
Annual aggregate verification metrics.

Statistics of calendar-year total flows. Annual standard deviation is
the classic detector of generators that reproduce sub-annual statistics
but fail to carry persistence up to interannual variability (Srinivas
and Srinivasan, 2005; Borgomeo et al., 2015).

References
----------
Stedinger, J.R. and Taylor, M.R. (1982). Synthetic streamflow
generation: 1. Model verification and validation. Water Resources
Research, 18(4), 909-918.

Srinivas, V.V. and Srinivasan, K. (2005). Hybrid moving block bootstrap
for stochastic simulation of multi-site multi-season streamflows.
Journal of Hydrology, 302(1-4), 307-330.
"""

import numpy as np
import pandas as pd

from synhydro._evaluation import sample_skewness
from synhydro._evaluation._stats import annual_aggregate
from synhydro.verification._registry import VERIFICATION_METRICS

_CITATION = "Stedinger and Taylor (1982); Srinivas and Srinivasan (2005)"
_MIN_YEARS = 10


def _annual_totals(x: pd.Series, steps_per_year: float) -> pd.Series:
    """Calendar-year totals with incomplete years removed."""
    return annual_aggregate(x, steps_per_year, how="sum")


@VERIFICATION_METRICS.register(
    category="annual",
    kind="scalar",
    units="flow_cumulative",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def annual_mean(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """Mean of calendar-year total flows."""
    totals = _annual_totals(x, steps_per_year)
    if len(totals) < 3:
        return np.nan
    return float(totals.mean())


@VERIFICATION_METRICS.register(
    category="annual",
    kind="scalar",
    units="flow_cumulative",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def annual_sd(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """Standard deviation of calendar-year total flows.

    Generators that do not carry sub-annual persistence up to the
    annual scale understate this statistic (Srinivas and Srinivasan,
    2005).
    """
    totals = _annual_totals(x, steps_per_year)
    if len(totals) < 3:
        return np.nan
    return float(totals.std(ddof=1))


@VERIFICATION_METRICS.register(
    category="annual",
    kind="scalar",
    units="dimensionless",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def annual_cv(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """Coefficient of variation of calendar-year total flows."""
    totals = _annual_totals(x, steps_per_year)
    if len(totals) < 3:
        return np.nan
    center = float(totals.mean())
    if abs(center) < 1e-10:
        return np.nan
    return float(totals.std(ddof=1) / center)


@VERIFICATION_METRICS.register(
    category="annual",
    kind="scalar",
    units="dimensionless",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def annual_skewness(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """Sample skewness of calendar-year total flows."""
    totals = _annual_totals(x, steps_per_year)
    if len(totals) < 3:
        return np.nan
    return sample_skewness(totals.to_numpy(dtype=float))


@VERIFICATION_METRICS.register(
    category="annual",
    kind="scalar",
    units="dimensionless",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def annual_lag1_autocorrelation(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """Lag-1 autocorrelation of calendar-year total flows."""
    totals = _annual_totals(x, steps_per_year)
    if len(totals) < 4:
        return np.nan
    return float(totals.autocorr(lag=1))


@VERIFICATION_METRICS.register(
    category="annual",
    kind="scalar",
    units="flow_cumulative",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def annual_minimum(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """Minimum calendar-year total flow (driest year)."""
    totals = _annual_totals(x, steps_per_year)
    if len(totals) < 3:
        return np.nan
    return float(totals.min())


@VERIFICATION_METRICS.register(
    category="annual",
    kind="scalar",
    units="flow_cumulative",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def annual_maximum(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """Maximum calendar-year total flow (wettest year)."""
    totals = _annual_totals(x, steps_per_year)
    if len(totals) < 3:
        return np.nan
    return float(totals.max())
