"""
Marginal (whole-record distributional) verification metrics.

These metrics summarize the marginal distribution of flows at a single
site over the full record: moments, extremes of the sample, and flow
quantiles. Preservation of the first two moments is the oldest and most
universal verification target for synthetic streamflow generators
(Matalas, 1967).

References
----------
Matalas, N.C. (1967). Mathematical assessment of synthetic hydrology.
Water Resources Research, 3(4), 937-945.

Stedinger, J.R. and Taylor, M.R. (1982). Synthetic streamflow
generation: 1. Model verification and validation. Water Resources
Research, 18(4), 909-918.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from synhydro._evaluation import sample_skewness, sample_kurtosis
from synhydro.verification._registry import VERIFICATION_METRICS

_CITATION = "Matalas (1967); Stedinger and Taylor (1982)"


@VERIFICATION_METRICS.register(
    category="marginal", kind="scalar", units="flow", citation=_CITATION
)
def mean(x: pd.Series) -> float:
    """Mean flow over the full record."""
    return float(np.mean(x.to_numpy(dtype=float)))


@VERIFICATION_METRICS.register(
    category="marginal", kind="scalar", units="flow", citation=_CITATION
)
def std(x: pd.Series) -> float:
    """Flow standard deviation (sample, ddof=1)."""
    return float(np.std(x.to_numpy(dtype=float), ddof=1))


@VERIFICATION_METRICS.register(
    category="marginal", kind="scalar", units="dimensionless", citation=_CITATION
)
def cv(x: pd.Series) -> float:
    """Coefficient of variation (std / mean)."""
    values = x.to_numpy(dtype=float)
    center = float(np.mean(values))
    if abs(center) < 1e-10:
        return np.nan
    return float(np.std(values, ddof=1) / center)


@VERIFICATION_METRICS.register(
    category="marginal", kind="scalar", units="dimensionless", citation=_CITATION
)
def skewness(x: pd.Series) -> float:
    """Sample skewness (bias-corrected G1 estimator)."""
    return sample_skewness(x.to_numpy(dtype=float))


@VERIFICATION_METRICS.register(
    category="marginal", kind="scalar", units="dimensionless", citation=_CITATION
)
def kurtosis(x: pd.Series) -> float:
    """Excess kurtosis (Fisher definition, bias-corrected)."""
    return sample_kurtosis(x.to_numpy(dtype=float))


@VERIFICATION_METRICS.register(
    category="marginal", kind="scalar", units="flow", citation=_CITATION
)
def minimum(x: pd.Series) -> float:
    """Minimum flow over the full record."""
    return float(np.min(x.to_numpy(dtype=float)))


@VERIFICATION_METRICS.register(
    category="marginal", kind="scalar", units="flow", citation=_CITATION
)
def maximum(x: pd.Series) -> float:
    """Maximum flow over the full record."""
    return float(np.max(x.to_numpy(dtype=float)))


@VERIFICATION_METRICS.register(
    category="marginal", kind="scalar", units="flow", citation=_CITATION
)
def flow_q10(x: pd.Series) -> float:
    """10th percentile of the flow distribution."""
    return float(np.percentile(x.to_numpy(dtype=float), 10))


@VERIFICATION_METRICS.register(
    category="marginal", kind="scalar", units="flow", citation=_CITATION
)
def flow_q50(x: pd.Series) -> float:
    """Median flow."""
    return float(np.percentile(x.to_numpy(dtype=float), 50))


@VERIFICATION_METRICS.register(
    category="marginal", kind="scalar", units="flow", citation=_CITATION
)
def flow_q90(x: pd.Series) -> float:
    """90th percentile of the flow distribution."""
    return float(np.percentile(x.to_numpy(dtype=float), 90))


@VERIFICATION_METRICS.register(
    category="marginal",
    kind="comparison",
    units="dimensionless",
    citation="Kolmogorov-Smirnov two-sample statistic",
)
def ks_statistic(x: pd.Series, reference: pd.Series) -> float:
    """Two-sample Kolmogorov-Smirnov distance from the observed distribution.

    The maximum absolute difference between the empirical CDFs of one
    realization and the observed record. Zero indicates identical
    empirical distributions. The p-value is deliberately not reported:
    with long records the test rejects for trivial differences, and
    non-rejection is not evidence of equality.
    """
    result = sp_stats.ks_2samp(reference.to_numpy(dtype=float), x.to_numpy(dtype=float))
    return float(result.statistic)
