"""
Spatial (cross-site) verification metrics.

Lag-0 cross-correlations between site pairs are the standard check of
multisite dependence preservation (Matalas, 1967). Lag-1 cross
correlations are directional and are not preserved by all multivariate
generation schemes, which makes them a useful additional diagnostic.

References
----------
Matalas, N.C. (1967). Mathematical assessment of synthetic hydrology.
Water Resources Research, 3(4), 937-945.

Tsoukalas, I., Efstratiadis, A., and Makropoulos, C. (2018). Stochastic
periodic autoregressive to anything (SPARTA): Modeling and simulation
of cyclostationary processes with arbitrary marginal distributions.
Water Resources Research, 54(1), 161-185.
"""

import numpy as np
import pandas as pd

from synhydro.verification._registry import VERIFICATION_METRICS
from synhydro._evaluation._runner import pair_label

_CITATION = "Matalas (1967); Tsoukalas et al. (2018)"


@VERIFICATION_METRICS.register(
    category="spatial",
    kind="matrix",
    units="dimensionless",
    citation=_CITATION,
)
def cross_correlation(frame: pd.DataFrame) -> pd.Series:
    """Lag-0 Pearson correlation for each unordered site pair."""
    corr = frame.corr()
    values = {}
    columns = list(frame.columns)
    for i, site_a in enumerate(columns):
        for site_b in columns[i + 1 :]:
            values[pair_label(site_a, site_b)] = float(corr.loc[site_a, site_b])
    return pd.Series(values)


@VERIFICATION_METRICS.register(
    category="spatial",
    kind="matrix",
    units="dimensionless",
    citation=_CITATION,
)
def cross_correlation_lag1(frame: pd.DataFrame) -> pd.Series:
    """Lag-1 cross-correlation for each ordered site pair.

    The correlation of flow at the first site with the previous
    timestep's flow at the second site. Pair labels are directional
    (``A->B`` correlates A at time t with B at time t-1).
    """
    values = {}
    columns = list(frame.columns)
    for site_a in columns:
        for site_b in columns:
            if site_a == site_b:
                continue
            current = frame[site_a]
            lagged = frame[site_b].shift(1)
            valid = current.notna() & lagged.notna()
            if valid.sum() < 3:
                value = np.nan
            else:
                value = float(np.corrcoef(current[valid], lagged[valid])[0, 1])
            values[f"{site_a}->{site_b}"] = value
    return pd.Series(values)
