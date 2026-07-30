"""
Extreme flow verification metrics.

Annual maxima and minima statistics, GEV return-period quantiles fit to
annual maxima, and 7-day low flows for daily records. Comparing GEV
quantile estimates across the ensemble against the observed-record fit
follows Zaerpour et al. (2021).

References
----------
Stedinger, J.R., Vogel, R.M., and Foufoula-Georgiou, E. (1993).
Frequency analysis of extreme events. In Handbook of Hydrology, edited
by D.R. Maidment, McGraw-Hill, Chapter 18.

Zaerpour, M., Papalexiou, S.M., and Nazemi, A. (2021). Informing
stochastic streamflow generation by large-scale climate indices at
single and multiple sites. Advances in Water Resources, 156, 104037.
"""

import numpy as np
import pandas as pd
from scipy.stats import genextreme

from synhydro.core.statistics import fit_gev
from synhydro._evaluation._stats import annual_aggregate
from synhydro.verification._registry import VERIFICATION_METRICS

_CITATION = "Stedinger et al. (1993); Zaerpour et al. (2021)"
_MIN_YEARS = 10


def _annual_extreme(x: pd.Series, steps_per_year: float, how: str) -> pd.Series:
    """Annual max or min series with incomplete years removed."""
    return annual_aggregate(x, steps_per_year, how=how)


def _gev_return_level(
    x: pd.Series, steps_per_year: float, return_period: float
) -> float:
    """GEV return level for annual maxima at the given return period."""
    maxima = _annual_extreme(x, steps_per_year, "max")
    if len(maxima) < 5:
        return np.nan
    try:
        params = fit_gev(maxima.to_numpy(dtype=float), method="lmom")
    except ValueError:
        return np.nan
    return float(
        genextreme.isf(
            1.0 / return_period,
            params["shape"],
            loc=params["loc"],
            scale=params["scale"],
        )
    )


@VERIFICATION_METRICS.register(
    category="extremes",
    kind="scalar",
    units="flow",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def annual_max_mean(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """Mean of annual maximum flows."""
    maxima = _annual_extreme(x, steps_per_year, "max")
    if len(maxima) < 3:
        return np.nan
    return float(maxima.mean())


@VERIFICATION_METRICS.register(
    category="extremes",
    kind="scalar",
    units="dimensionless",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def annual_max_cv(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """Coefficient of variation of annual maximum flows."""
    maxima = _annual_extreme(x, steps_per_year, "max")
    if len(maxima) < 3:
        return np.nan
    center = float(maxima.mean())
    if abs(center) < 1e-10:
        return np.nan
    return float(maxima.std(ddof=1) / center)


@VERIFICATION_METRICS.register(
    category="extremes",
    kind="scalar",
    units="flow",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def gev_rp10(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """GEV 10-year return level of annual maximum flows (L-moment fit)."""
    return _gev_return_level(x, steps_per_year, 10.0)


@VERIFICATION_METRICS.register(
    category="extremes",
    kind="scalar",
    units="flow",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def gev_rp50(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """GEV 50-year return level of annual maximum flows (L-moment fit)."""
    return _gev_return_level(x, steps_per_year, 50.0)


@VERIFICATION_METRICS.register(
    category="extremes",
    kind="scalar",
    units="flow",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def gev_rp100(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """GEV 100-year return level of annual maximum flows (L-moment fit)."""
    return _gev_return_level(x, steps_per_year, 100.0)


@VERIFICATION_METRICS.register(
    category="extremes",
    kind="scalar",
    units="flow",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def annual_min_mean(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """Mean of annual minimum flows."""
    minima = _annual_extreme(x, steps_per_year, "min")
    if len(minima) < 3:
        return np.nan
    return float(minima.mean())


@VERIFICATION_METRICS.register(
    category="extremes",
    kind="scalar",
    units="dimensionless",
    needs=("steps_per_year",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def annual_min_cv(x: pd.Series, steps_per_year: float = 12.0) -> float:
    """Coefficient of variation of annual minimum flows."""
    minima = _annual_extreme(x, steps_per_year, "min")
    if len(minima) < 3:
        return np.nan
    center = float(minima.mean())
    if abs(center) < 1e-10:
        return np.nan
    return float(minima.std(ddof=1) / center)


@VERIFICATION_METRICS.register(
    category="extremes",
    kind="scalar",
    units="flow",
    needs=("steps_per_year",),
    frequencies=("daily",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def seven_day_min_mean(x: pd.Series, steps_per_year: float = 365.25) -> float:
    """Mean of annual 7-day minimum flows (daily records only)."""
    rolling = x.rolling(window=7, min_periods=7).mean()
    minima = annual_aggregate(rolling, steps_per_year, how="min", min_fraction=0.8)
    if len(minima) < 3:
        return np.nan
    return float(minima.mean())


@VERIFICATION_METRICS.register(
    category="extremes",
    kind="scalar",
    units="dimensionless",
    needs=("steps_per_year",),
    frequencies=("daily",),
    min_years=_MIN_YEARS,
    citation=_CITATION,
)
def seven_day_min_cv(x: pd.Series, steps_per_year: float = 365.25) -> float:
    """Coefficient of variation of annual 7-day minimum flows (daily only)."""
    rolling = x.rolling(window=7, min_periods=7).mean()
    minima = annual_aggregate(rolling, steps_per_year, how="min", min_fraction=0.8)
    if len(minima) < 3:
        return np.nan
    center = float(minima.mean())
    if abs(center) < 1e-10:
        return np.nan
    return float(minima.std(ddof=1) / center)
