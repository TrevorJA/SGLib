"""
Flow duration curve verification metrics.

The flow duration curve (FDC) summarizes the full marginal distribution
as flow versus exceedance probability and is a standard synthesis
verification target (Vogel and Fennessey, 1995).

References
----------
Vogel, R.M. and Fennessey, N.M. (1995). Flow duration curves II: A
review of applications in water resources planning. Water Resources
Bulletin, 31(6), 1029-1039.
"""

import numpy as np
import pandas as pd

from synhydro.verification._registry import VERIFICATION_METRICS

_CITATION = "Vogel and Fennessey (1995)"

EXCEEDANCE_GRID = (
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
)

_LOG_FLOOR = 1e-6


@VERIFICATION_METRICS.register(
    category="fdc",
    kind="curve",
    units="flow",
    citation=_CITATION,
)
def fdc(x: pd.Series) -> pd.Series:
    """Flow at fixed exceedance probabilities (flow duration curve).

    Components are exceedance probabilities: 0.01 is a high flow
    exceeded 1 percent of the time, 0.99 a low flow exceeded 99
    percent of the time.
    """
    values = x.to_numpy(dtype=float)
    if len(values) < 10:
        return pd.Series({prob: np.nan for prob in EXCEEDANCE_GRID}, dtype=float)
    quantiles = np.quantile(values, [1.0 - prob for prob in EXCEEDANCE_GRID])
    return pd.Series(dict(zip(EXCEEDANCE_GRID, quantiles)))


@VERIFICATION_METRICS.register(
    category="fdc",
    kind="comparison",
    units="dimensionless",
    citation=_CITATION,
)
def fdc_log_rmse(x: pd.Series, reference: pd.Series) -> float:
    """RMSE between log-space flow duration curves, one realization vs observed.

    Computed over the fixed exceedance grid with flows floored at 1e-6
    before taking logs. Zero indicates identical curves; log space
    weights low-flow and high-flow errors comparably.
    """
    syn_curve = fdc(x)
    obs_curve = fdc(reference)
    syn_log = np.log(np.clip(syn_curve.to_numpy(dtype=float), _LOG_FLOOR, None))
    obs_log = np.log(np.clip(obs_curve.to_numpy(dtype=float), _LOG_FLOOR, None))
    valid = np.isfinite(syn_log) & np.isfinite(obs_log)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((syn_log[valid] - obs_log[valid]) ** 2)))
