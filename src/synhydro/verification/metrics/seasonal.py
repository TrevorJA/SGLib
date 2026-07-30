"""
Seasonal (per-calendar-month) verification metrics.

Monthly-panel statistics are the dominant presentation convention of the
nonparametric generation literature: a statistic computed per calendar
month on every realization, compared against the observed monthly value
(Lall and Sharma, 1996; Nowak et al., 2010).

All statistics here are computed per realization; realizations are never
pooled before testing. Pooling inflates the synthetic sample size, so
hypothesis-test p-values would shrink toward zero as the ensemble grows,
a sample-size artifact rather than a quality signal.

References
----------
Lall, U. and Sharma, A. (1996). A nearest neighbor bootstrap for
resampling hydrologic time series. Water Resources Research, 32(3),
679-693.

Nowak, K., Prairie, J., Rajagopalan, B., and Lall, U. (2010). A
nonparametric stochastic approach for multisite disaggregation of
annual to daily streamflow. Water Resources Research, 46, W08529.

Herman, J.D., Zeff, H.B., Lamontagne, J.R., Reed, P.M., and
Characklis, G.W. (2016). Synthetic drought scenario generation to
support bottom-up water supply vulnerability assessments. Journal of
Water Resources Planning and Management, 142(11), 04016050.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from synhydro._evaluation import sample_skewness
from synhydro.verification._registry import VERIFICATION_METRICS

_SUBANNUAL = ("daily", "weekly", "monthly")
_CITATION = "Lall and Sharma (1996); Nowak et al. (2010)"
_TEST_CITATION = "Herman et al. (2016)"


def _monthly_series(x: pd.Series) -> pd.Series:
    """Aggregate to monthly means (identity for monthly input)."""
    return x.resample("MS").mean().dropna()


@VERIFICATION_METRICS.register(
    category="seasonal",
    kind="curve",
    units="flow",
    frequencies=_SUBANNUAL,
    citation=_CITATION,
)
def monthly_mean(x: pd.Series) -> pd.Series:
    """Mean flow per calendar month."""
    return x.groupby(x.index.month).mean()


@VERIFICATION_METRICS.register(
    category="seasonal",
    kind="curve",
    units="flow",
    frequencies=_SUBANNUAL,
    citation=_CITATION,
)
def monthly_std(x: pd.Series) -> pd.Series:
    """Flow standard deviation per calendar month (sample, ddof=1)."""
    return x.groupby(x.index.month).std(ddof=1)


@VERIFICATION_METRICS.register(
    category="seasonal",
    kind="curve",
    units="dimensionless",
    frequencies=_SUBANNUAL,
    citation=_CITATION,
)
def monthly_skewness(x: pd.Series) -> pd.Series:
    """Sample skewness per calendar month (bias-corrected G1)."""
    return x.groupby(x.index.month).apply(
        lambda values: sample_skewness(values.to_numpy(dtype=float))
    )


@VERIFICATION_METRICS.register(
    category="seasonal",
    kind="curve",
    units="flow",
    frequencies=_SUBANNUAL,
    citation=_CITATION,
)
def monthly_maximum(x: pd.Series) -> pd.Series:
    """Maximum flow per calendar month."""
    return x.groupby(x.index.month).max()


@VERIFICATION_METRICS.register(
    category="seasonal",
    kind="curve",
    units="flow",
    frequencies=_SUBANNUAL,
    citation=_CITATION,
)
def monthly_minimum(x: pd.Series) -> pd.Series:
    """Minimum flow per calendar month."""
    return x.groupby(x.index.month).min()


@VERIFICATION_METRICS.register(
    category="seasonal",
    kind="curve",
    units="dimensionless",
    frequencies=_SUBANNUAL,
    citation="Kirsch et al. (2013)",
)
def monthly_lag1_correlation(x: pd.Series) -> pd.Series:
    """Correlation of each month's flow with the previous month's flow.

    Computed on monthly mean flows across years: for calendar month m,
    the correlation between month-m values and month-(m-1) values. This
    seasonally varying month-to-month correlation is the central
    temporal check of Kirsch et al. (2013).
    """
    monthly = _monthly_series(x)
    previous = monthly.shift(1)
    values = {}
    for month in range(1, 13):
        mask = monthly.index.month == month
        current_m = monthly[mask]
        previous_m = previous[mask]
        valid = current_m.notna() & previous_m.notna()
        if valid.sum() < 3:
            values[month] = np.nan
        else:
            values[month] = float(
                np.corrcoef(current_m[valid], previous_m[valid])[0, 1]
            )
    return pd.Series(values)


@VERIFICATION_METRICS.register(
    category="seasonal",
    kind="comparison",
    units="pvalue",
    summary_mode="reject_rate",
    frequencies=_SUBANNUAL,
    citation=_TEST_CITATION,
)
def monthly_ranksum_pvalue(x: pd.Series, reference: pd.Series) -> pd.Series:
    """Wilcoxon rank-sum p-value per calendar month, one realization vs observed.

    Tests whether the realization's month-m values and the observed
    month-m values come from distributions with the same location
    (Herman et al., 2016). Summarized as the fraction of realizations
    rejecting at alpha = 0.05; under a perfect generator this rejection
    rate is near alpha.
    """
    return _monthly_test(x, reference, lambda a, b: sp_stats.ranksums(a, b).pvalue)


@VERIFICATION_METRICS.register(
    category="seasonal",
    kind="comparison",
    units="pvalue",
    summary_mode="reject_rate",
    frequencies=_SUBANNUAL,
    citation=_TEST_CITATION,
)
def monthly_levene_pvalue(x: pd.Series, reference: pd.Series) -> pd.Series:
    """Levene test p-value per calendar month, one realization vs observed.

    Tests whether the realization's month-m values and the observed
    month-m values have equal variance (Herman et al., 2016).
    Summarized as the fraction of realizations rejecting at
    alpha = 0.05.
    """
    return _monthly_test(x, reference, lambda a, b: sp_stats.levene(a, b).pvalue)


def _monthly_test(x: pd.Series, reference: pd.Series, test) -> pd.Series:
    """Apply a two-sample test month by month."""
    values = {}
    for month in range(1, 13):
        syn_m = x[x.index.month == month].to_numpy(dtype=float)
        obs_m = reference[reference.index.month == month].to_numpy(dtype=float)
        if len(syn_m) < 3 or len(obs_m) < 3:
            values[month] = np.nan
        else:
            values[month] = float(test(obs_m, syn_m))
    return pd.Series(values)
