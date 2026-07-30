"""
Small statistical estimators shared by the evaluation suites.
"""

import numpy as np
from scipy import stats as sp_stats


def sample_skewness(x: np.ndarray) -> float:
    """
    Compute sample skewness (G1 formula).

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    float
        Sample skewness, or nan if n < 3 or variance is effectively zero.
    """
    n = len(x)
    if n < 3:
        return np.nan
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-10:
        return 0.0
    return float(n / ((n - 1) * (n - 2)) * np.sum(((x - m) / s) ** 3))


def sample_kurtosis(x: np.ndarray) -> float:
    """
    Compute excess kurtosis (Fisher definition, bias-corrected).

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    float
        Excess kurtosis, or nan if n < 4.
    """
    n = len(x)
    if n < 4:
        return np.nan
    return float(sp_stats.kurtosis(x, fisher=True, bias=False))


def extract_runs(values: np.ndarray, threshold: float) -> tuple[list[int], list[float]]:
    """
    Extract below-threshold run events from a series (theory of runs).

    A run is a maximal sequence of consecutive values below the
    threshold. Severity is the cumulative deficit over the run.

    Parameters
    ----------
    values : np.ndarray
        Series values.
    threshold : float
        Truncation level.

    Returns
    -------
    durations : list of int
        Duration of each run in timesteps.
    severities : list of float
        Cumulative deficit (threshold minus value, summed) of each run.

    References
    ----------
    Salas, J.D., Delleur, J.W., Yevjevich, V., and Lane, W.L. (1980).
    Applied Modeling of Hydrologic Time Series. Water Resources
    Publications.
    """
    below = values < threshold
    padded = np.concatenate(([False], below, [False]))
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends = np.where(padded[:-1] & ~padded[1:])[0]

    durations = (ends - starts).tolist()
    severities = [float(np.sum(threshold - values[s:e])) for s, e in zip(starts, ends)]
    return durations, severities
