"""
Temporal dependence verification metrics.

Short-term persistence (lag correlations, autocorrelation function) and
long-term persistence (Hurst exponent). Following Stedinger and Taylor
(1982), the Hurst exponent is estimated on annually aggregated flows:
sub-annual records mix seasonal persistence into the estimate and make
it unstable.

References
----------
Matalas, N.C. (1967). Mathematical assessment of synthetic hydrology.
Water Resources Research, 3(4), 937-945.

Koutsoyiannis, D. (2002). The Hurst phenomenon and fractional Gaussian
noise made easy. Hydrological Sciences Journal, 47(4), 573-595.

Stedinger, J.R. and Taylor, M.R. (1982). Synthetic streamflow
generation: 1. Model verification and validation. Water Resources
Research, 18(4), 909-918.
"""

import numpy as np
import pandas as pd

from synhydro.core.statistics import compute_hurst_exponent
from synhydro._evaluation._stats import annual_aggregate
from synhydro.verification._registry import VERIFICATION_METRICS


@VERIFICATION_METRICS.register(
    category="temporal",
    kind="scalar",
    units="dimensionless",
    citation="Matalas (1967)",
)
def lag1_autocorrelation(x: pd.Series) -> float:
    """Lag-1 autocorrelation of the full-record series."""
    if len(x) < 3:
        return np.nan
    return float(x.autocorr(lag=1))


@VERIFICATION_METRICS.register(
    category="temporal",
    kind="scalar",
    units="dimensionless",
    citation="Matalas (1967)",
)
def lag2_autocorrelation(x: pd.Series) -> float:
    """Lag-2 autocorrelation of the full-record series."""
    if len(x) < 4:
        return np.nan
    return float(x.autocorr(lag=2))


@VERIFICATION_METRICS.register(
    category="temporal",
    kind="curve",
    units="dimensionless",
    needs=("acf_lags",),
    citation="Salas et al. (1980); Kirsch et al. (2013)",
)
def acf(x: pd.Series, acf_lags: int = 12) -> pd.Series:
    """Autocorrelation function at lags 1 through acf_lags."""
    values = {}
    for lag in range(1, acf_lags + 1):
        values[lag] = float(x.autocorr(lag=lag)) if len(x) > lag + 1 else np.nan
    return pd.Series(values)


@VERIFICATION_METRICS.register(
    category="temporal",
    kind="scalar",
    units="dimensionless",
    needs=("hurst_method", "steps_per_year"),
    min_years=20,
    citation="Hurst (1951); Koutsoyiannis (2002)",
)
def hurst(
    x: pd.Series,
    hurst_method: str = "rs",
    steps_per_year: float = 12.0,
) -> float:
    """Hurst exponent of annually aggregated flows.

    Estimated on calendar-year totals because sub-annual estimation
    conflates seasonal persistence with long-range dependence
    (Stedinger and Taylor, 1982). Requires at least 20 years of data;
    estimates from records shorter than about 50 years remain noisy.
    """
    annual = annual_aggregate(x, steps_per_year, how="sum")
    n = len(annual)
    if n < 16:
        return np.nan
    try:
        result = compute_hurst_exponent(
            annual.to_numpy(dtype=float),
            method=hurst_method,
            min_window=4,
            max_window=max(n // 2, 8),
        )
    except ValueError:
        return np.nan
    return float(result["H"])
