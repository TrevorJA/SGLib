"""
Frequency normalization and inference for evaluation suites.

Ensemble metadata stores time resolution as a pandas frequency alias
(for example ``'MS'`` or ``'D'``) for most generators, but some store
plain words such as ``'monthly'``. Evaluation code must also survive
missing metadata by inferring the frequency from the DatetimeIndex.
This module normalizes all of those forms to a single FrequencyInfo.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrequencyInfo:
    """
    Normalized description of a timeseries frequency.

    Attributes
    ----------
    base : str
        One of ``'daily'``, ``'weekly'``, ``'monthly'``, ``'annual'``.
    pandas_alias : str
        Canonical pandas frequency alias for resampling.
    steps_per_year : float
        Average number of timesteps per year at this frequency.
    """

    base: str
    pandas_alias: str
    steps_per_year: float


DAILY = FrequencyInfo("daily", "D", 365.25)
WEEKLY = FrequencyInfo("weekly", "W-SUN", 52.18)
MONTHLY = FrequencyInfo("monthly", "MS", 12.0)
ANNUAL = FrequencyInfo("annual", "YS", 1.0)

_LITERALS = {
    "daily": DAILY,
    "day": DAILY,
    "weekly": WEEKLY,
    "week": WEEKLY,
    "monthly": MONTHLY,
    "month": MONTHLY,
    "annual": ANNUAL,
    "yearly": ANNUAL,
    "year": ANNUAL,
}

_ALIAS_ROOTS = {
    "D": DAILY,
    "B": DAILY,
    "W": WEEKLY,
    "M": MONTHLY,
    "MS": MONTHLY,
    "ME": MONTHLY,
    "Y": ANNUAL,
    "YS": ANNUAL,
    "YE": ANNUAL,
    "A": ANNUAL,
    "AS": ANNUAL,
}


def normalize_frequency(value: Optional[str]) -> Optional[FrequencyInfo]:
    """
    Normalize a frequency string to a FrequencyInfo.

    Accepts pandas frequency aliases (``'D'``, ``'W-SUN'``, ``'MS'``,
    ``'YS'``, etc.) and plain words (``'daily'``, ``'monthly'``, ...).
    Multiples such as ``'2D'`` are not supported and return None.

    Parameters
    ----------
    value : str or None
        Frequency string to normalize.

    Returns
    -------
    FrequencyInfo or None
        Normalized frequency, or None if the string is unrecognized.
    """
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    literal = _LITERALS.get(token.lower())
    if literal is not None:
        return literal
    root = token.upper().split("-")[0]
    return _ALIAS_ROOTS.get(root)


def infer_frequency(index: pd.DatetimeIndex) -> FrequencyInfo:
    """
    Infer the frequency of a DatetimeIndex.

    Tries ``pd.infer_freq`` first, then falls back to bucketing the
    median spacing between consecutive timestamps.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index to inspect. Needs at least 3 timestamps.

    Returns
    -------
    FrequencyInfo
        Inferred frequency.

    Raises
    ------
    ValueError
        If the index is too short or its spacing does not match a
        supported frequency (daily, weekly, monthly, annual).
    """
    if len(index) < 3:
        raise ValueError(
            f"Need at least 3 timestamps to infer frequency, found {len(index)}."
        )
    alias = pd.infer_freq(index)
    if alias is not None:
        info = normalize_frequency(alias)
        if info is not None:
            return info
    spacing_days = (
        np.diff(index.values).astype("timedelta64[s]").astype(float) / 86400.0
    )
    median_days = float(np.median(spacing_days))
    if 0.75 <= median_days <= 1.5:
        return DAILY
    if 6.0 <= median_days <= 8.0:
        return WEEKLY
    if 28.0 <= median_days <= 31.5:
        return MONTHLY
    if 350.0 <= median_days <= 380.0:
        return ANNUAL
    raise ValueError(
        f"Cannot infer a supported frequency from index "
        f"(median spacing {median_days:.1f} days). Supported frequencies "
        f"are daily, weekly, monthly, and annual."
    )


def resolve_frequency(
    explicit: Optional[str],
    metadata_value: Optional[str],
    index: pd.DatetimeIndex,
) -> FrequencyInfo:
    """
    Resolve the frequency for an evaluation run.

    Resolution order: explicit user argument, then ensemble metadata,
    then inference from the index. If metadata and index-inferred
    frequency disagree, the inferred frequency wins with a warning.

    Parameters
    ----------
    explicit : str or None
        User-supplied frequency override.
    metadata_value : str or None
        Frequency string from ensemble metadata (``time_resolution``).
    index : pd.DatetimeIndex
        Index of a representative realization.

    Returns
    -------
    FrequencyInfo
        Resolved frequency.

    Raises
    ------
    ValueError
        If an explicit value is unrecognized, or if no source yields a
        supported frequency.
    """
    if explicit is not None:
        info = normalize_frequency(explicit)
        if info is None:
            raise ValueError(
                f"Unrecognized frequency '{explicit}'. Use a pandas alias "
                f"('D', 'W', 'MS', 'YS') or a word ('daily', 'weekly', "
                f"'monthly', 'annual')."
            )
        return info

    inferred: Optional[FrequencyInfo] = None
    try:
        inferred = infer_frequency(index)
    except ValueError:
        pass

    from_metadata = normalize_frequency(metadata_value)
    if from_metadata is not None:
        if inferred is not None and inferred.base != from_metadata.base:
            logger.warning(
                "Ensemble metadata frequency '%s' (%s) disagrees with the "
                "index-inferred frequency (%s); using the inferred frequency.",
                metadata_value,
                from_metadata.base,
                inferred.base,
            )
            return inferred
        return from_metadata

    if inferred is not None:
        logger.info("Inferred ensemble frequency '%s' from index.", inferred.base)
        return inferred

    raise ValueError(
        "Could not determine ensemble frequency: metadata is missing or "
        "unrecognized and index inference failed. Pass frequency= explicitly."
    )


def check_observed_frequency(
    index: pd.DatetimeIndex,
    ensemble_info: FrequencyInfo,
) -> None:
    """
    Ensure observed data matches the ensemble frequency.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index of the observed data.
    ensemble_info : FrequencyInfo
        Resolved ensemble frequency.

    Raises
    ------
    ValueError
        If the observed frequency differs from the ensemble frequency.
    """
    observed_info = infer_frequency(index)
    if observed_info.base != ensemble_info.base:
        raise ValueError(
            f"Observed data frequency ({observed_info.base}) does not match "
            f"the ensemble frequency ({ensemble_info.base}). Resample one "
            f"side before evaluation."
        )
