"""
L-moment ratio verification metrics.

L-moment ratios are linear combinations of order statistics that
characterize distribution shape more robustly than product moments for
skewed hydrologic data and small samples (Hosking, 1990).

References
----------
Hosking, J.R.M. (1990). L-moments: Analysis and estimation of
distributions using linear combinations of order statistics. Journal of
the Royal Statistical Society, Series B, 52(1), 105-124.

Hosking, J.R.M. and Wallis, J.R. (1997). Regional Frequency Analysis:
An Approach Based on L-Moments. Cambridge University Press.
"""

import numpy as np
import pandas as pd

from synhydro.core.statistics import compute_lmoments
from synhydro.verification._registry import VERIFICATION_METRICS

_CITATION = "Hosking (1990)"
_MIN_LENGTH = 10


@VERIFICATION_METRICS.register(
    category="lmoments",
    kind="scalar",
    units="dimensionless",
    citation=_CITATION,
)
def l_cv(x: pd.Series) -> float:
    """L-coefficient of variation (tau-2 = L-scale / L-mean)."""
    values = x.to_numpy(dtype=float)
    if len(values) < _MIN_LENGTH:
        return np.nan
    l1, l2, _, _ = compute_lmoments(values)
    if abs(l1) < 1e-10:
        return np.nan
    return float(l2 / l1)


@VERIFICATION_METRICS.register(
    category="lmoments",
    kind="scalar",
    units="dimensionless",
    citation=_CITATION,
)
def l_skewness(x: pd.Series) -> float:
    """L-skewness ratio (tau-3)."""
    values = x.to_numpy(dtype=float)
    if len(values) < _MIN_LENGTH:
        return np.nan
    _, _, t3, _ = compute_lmoments(values)
    return float(t3)


@VERIFICATION_METRICS.register(
    category="lmoments",
    kind="scalar",
    units="dimensionless",
    citation=_CITATION,
)
def l_kurtosis(x: pd.Series) -> float:
    """L-kurtosis ratio (tau-4)."""
    values = x.to_numpy(dtype=float)
    if len(values) < _MIN_LENGTH:
        return np.nan
    _, _, _, t4 = compute_lmoments(values)
    return float(t4)
