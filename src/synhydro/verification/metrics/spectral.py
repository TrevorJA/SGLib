"""
Spectral verification metrics.

The distribution of variance across period bands summarizes how much of
a series' variability lives at seasonal, interannual, and longer time
scales. Comparing synthetic and observed spectra follows the wavelet
autoregressive modeling lineage (Nowak et al., 2011).

References
----------
Nowak, K.C., Rajagopalan, B., and Zagona, E. (2011). Wavelet Auto-
Regressive Method (WARM) for multi-site streamflow simulation of data
with non-stationary spectra. Journal of Hydrology, 410(1-2), 1-12.
"""

import numpy as np
import pandas as pd
from scipy import signal

from synhydro.verification._registry import VERIFICATION_METRICS

_CITATION = "Nowak et al. (2011)"

# Period bands in years, from long to short. Labels are components.
PERIOD_BANDS = (
    (">8y", 8.0, np.inf),
    ("4-8y", 4.0, 8.0),
    ("2-4y", 2.0, 4.0),
    ("1-2y", 1.0, 2.0),
    ("0.5-1y", 0.5, 1.0),
    ("<0.5y", 0.0, 0.5),
)

_LOW_FREQUENCY_PERIOD_YEARS = 2.0


def _periodogram_years(
    x: pd.Series, steps_per_year: float
) -> tuple[np.ndarray, np.ndarray]:
    """Periodogram of the standardized series with frequency in cycles/year."""
    values = x.to_numpy(dtype=float)
    scale = np.std(values, ddof=1)
    if len(values) < 24 or scale < 1e-10:
        return np.array([]), np.array([])
    standardized = (values - np.mean(values)) / scale
    frequencies, psd = signal.periodogram(standardized, fs=steps_per_year)
    positive = frequencies > 0
    return frequencies[positive], psd[positive]


@VERIFICATION_METRICS.register(
    category="spectral",
    kind="curve",
    units="dimensionless",
    needs=("steps_per_year",),
    citation=_CITATION,
)
def spectral_density(x: pd.Series, steps_per_year: float = 12.0) -> pd.Series:
    """Fraction of spectral variance in fixed period bands.

    The periodogram of the standardized series is integrated over
    period bands (in years) and normalized to sum to one. Components
    are band labels ordered from the longest periods to the shortest.
    """
    frequencies, psd = _periodogram_years(x, steps_per_year)
    if len(frequencies) == 0:
        return pd.Series({label: np.nan for label, _, _ in PERIOD_BANDS})
    periods = 1.0 / frequencies
    total = float(np.sum(psd))
    values = {}
    for label, low, high in PERIOD_BANDS:
        in_band = (
            (periods > low) & (periods <= high)
            if np.isfinite(high)
            else (periods > low)
        )
        values[label] = float(np.sum(psd[in_band]) / total) if total > 0 else np.nan
    return pd.Series(values)


@VERIFICATION_METRICS.register(
    category="spectral",
    kind="scalar",
    units="dimensionless",
    needs=("steps_per_year",),
    citation=_CITATION,
)
def low_frequency_variance_fraction(
    x: pd.Series, steps_per_year: float = 12.0
) -> float:
    """Fraction of spectral variance at periods longer than 2 years.

    Low values indicate a series dominated by seasonal and shorter
    variability; generators that ignore interannual persistence
    understate this fraction.
    """
    frequencies, psd = _periodogram_years(x, steps_per_year)
    if len(frequencies) == 0:
        return np.nan
    periods = 1.0 / frequencies
    total = float(np.sum(psd))
    if total <= 0:
        return np.nan
    return float(np.sum(psd[periods > _LOW_FREQUENCY_PERIOD_YEARS]) / total)
