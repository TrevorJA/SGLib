"""
Bootstrap uncertainty and generator comparison tools.

Both functions consume the tidy frame produced by
:func:`synhydro.verification.verify` (or
:func:`synhydro.validation.validate`): one row per (metric, site,
component, realization). Bootstrap resampling is over realizations,
treating each realization's statistic as an independent draw from the
generator's sampling distribution.

References
----------
Efron, B. and Tibshirani, R.J. (1993). An Introduction to the
Bootstrap. Chapman and Hall.

Vogel, R.M. and Shallcross, A.L. (1996). The moving blocks bootstrap
versus parametric time series models. Water Resources Research, 32(6),
1875-1882.
"""

import logging
from typing import Union

import numpy as np
import pandas as pd

from synhydro._evaluation import EvaluationResult

logger = logging.getLogger(__name__)

_GROUP_COLUMNS = ["category", "metric", "kind", "site", "component", "units"]
_NEAR_ZERO_FLOOR = 1e-10

FrameOrResult = Union[EvaluationResult, pd.DataFrame]


def _to_frame(values: FrameOrResult) -> pd.DataFrame:
    """Extract the tidy values frame from a result object or DataFrame."""
    if isinstance(values, EvaluationResult):
        return values.values
    if isinstance(values, pd.DataFrame):
        missing = [
            c
            for c in _GROUP_COLUMNS + ["realization", "value", "observed"]
            if c not in values.columns
        ]
        if missing:
            raise ValueError(f"Tidy frame is missing required columns: {missing}.")
        return values
    raise TypeError(
        f"Expected an EvaluationResult or tidy DataFrame, got {type(values)}."
    )


def _relative(value: float, observed: float, units: str) -> float:
    """Relative difference from observed, NaN for p-values and near-zero."""
    if units == "pvalue" or not np.isfinite(observed):
        return np.nan
    if abs(observed) < _NEAR_ZERO_FLOOR:
        return np.nan
    return float((value - observed) / abs(observed))


def bootstrap_metric_ci(
    values: FrameOrResult,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    statistic: str = "median",
    seed: Union[int, None] = None,
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for ensemble metric estimates.

    For each (metric, site, component) group, realizations are resampled
    with replacement n_bootstrap times; the chosen statistic (median or
    mean across realizations) is recomputed on each resample, and the
    confidence interval is taken from percentiles of the resampled
    statistics.

    Parameters
    ----------
    values : VerificationResult, ValidationResult, or pd.DataFrame
        Result object or tidy frame from ``verify()`` or ``validate()``.
    n_bootstrap : int, default 1000
        Number of bootstrap resamples.
    confidence_level : float, default 0.95
        Two-sided confidence level for the interval.
    statistic : {'median', 'mean'}, default 'median'
        Statistic of the per-realization values to bootstrap.
    seed : int, optional
        Seed for reproducible resampling.

    Returns
    -------
    pd.DataFrame
        One row per (category, metric, site, component) with columns
        ``observed, estimate, ci_lower, ci_upper, relative_diff,
        rd_ci_lower, rd_ci_upper, n_realizations``. Relative columns
        are NaN for p-value metrics and comparison-kind metrics (which
        have no observed value).

    References
    ----------
    Efron, B. and Tibshirani, R.J. (1993). An Introduction to the
    Bootstrap. Chapman and Hall.
    """
    if statistic not in ("median", "mean"):
        raise ValueError(f"Unknown statistic '{statistic}'. Use 'median' or 'mean'.")
    frame = _to_frame(values)
    stat_func = np.median if statistic == "median" else np.mean
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence_level) / 2.0

    records = []
    for keys, group in frame.groupby(_GROUP_COLUMNS, dropna=False, sort=False):
        category, metric, kind, site, component, units = keys
        vals = group["value"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        observed = group["observed"].iloc[0]
        observed = float(observed) if pd.notna(observed) else np.nan

        record = {
            "category": category,
            "metric": metric,
            "site": site,
            "component": component,
            "observed": observed,
            "estimate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "relative_diff": np.nan,
            "rd_ci_lower": np.nan,
            "rd_ci_upper": np.nan,
            "n_realizations": int(group["realization"].nunique()),
        }

        if len(vals) >= 2:
            indices = rng.integers(0, len(vals), size=(n_bootstrap, len(vals)))
            resampled = stat_func(vals[indices], axis=1)
            record["estimate"] = float(stat_func(vals))
            record["ci_lower"] = float(np.quantile(resampled, alpha))
            record["ci_upper"] = float(np.quantile(resampled, 1.0 - alpha))
            record["relative_diff"] = _relative(record["estimate"], observed, units)
            if np.isfinite(record["relative_diff"]):
                record["rd_ci_lower"] = _relative(record["ci_lower"], observed, units)
                record["rd_ci_upper"] = _relative(record["ci_upper"], observed, units)

        records.append(record)

    return pd.DataFrame(records)


def compare_methods(
    values_a: FrameOrResult,
    values_b: FrameOrResult,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: Union[int, None] = None,
) -> pd.DataFrame:
    """
    Compare two generators' metric errors with a bootstrap test.

    For each (metric, site, component) group shared by both results,
    the per-realization absolute error from observed is computed for
    each method, and the difference in mean absolute error (method A
    minus method B) is bootstrapped. When both ensembles have the same
    number of realizations, a paired bootstrap is used: one shared
    index draw resamples error pairs jointly. Otherwise the two error
    samples are resampled independently and ``paired`` is False.

    For comparison-kind metrics (which measure divergence from
    observed directly), the metric value itself is used as the error.

    Parameters
    ----------
    values_a, values_b : VerificationResult, ValidationResult, or pd.DataFrame
        Results from ``verify()`` or ``validate()`` on the same
        observed data. Run the orchestrator on each ensemble first.
    n_bootstrap : int, default 1000
        Number of bootstrap resamples.
    confidence_level : float, default 0.95
        Two-sided confidence level for the difference interval.
    seed : int, optional
        Seed for reproducible resampling.

    Returns
    -------
    pd.DataFrame
        One row per shared (category, metric, site, component) with
        columns ``method_a_mae, method_b_mae, diff_estimate,
        diff_ci_lower, diff_ci_upper, significant, better_method,
        paired``. ``significant`` is True when the confidence interval
        for the difference excludes zero; ``better_method`` is ``'a'``
        or ``'b'`` when significant, otherwise ``'none'``.

    References
    ----------
    Efron, B. and Tibshirani, R.J. (1993). An Introduction to the
    Bootstrap. Chapman and Hall.
    """
    # Fill the component column with a sentinel so scalar-metric group
    # keys (missing component) hash identically across both frames.
    sentinel = "__scalar__"
    frame_a = _to_frame(values_a).copy()
    frame_b = _to_frame(values_b).copy()
    for frame in (frame_a, frame_b):
        frame["component"] = frame["component"].fillna(sentinel)
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence_level) / 2.0

    groups_b = {
        keys: group for keys, group in frame_b.groupby(_GROUP_COLUMNS, sort=False)
    }

    warned_unpaired = False
    records = []
    for keys, group_a in frame_a.groupby(_GROUP_COLUMNS, sort=False):
        group_b = groups_b.get(keys)
        if group_b is None:
            continue
        category, metric, kind, site, component, units = keys
        if component == sentinel:
            component = pd.NA

        err_a = _errors(group_a)
        err_b = _errors(group_b)
        if len(err_a) < 2 or len(err_b) < 2:
            continue

        paired = len(err_a) == len(err_b)
        if paired:
            indices = rng.integers(0, len(err_a), size=(n_bootstrap, len(err_a)))
            diffs = np.mean(err_a[indices] - err_b[indices], axis=1)
        else:
            if not warned_unpaired:
                logger.warning(
                    "Ensembles have different realization counts (%d vs %d); "
                    "falling back to independent resampling.",
                    len(err_a),
                    len(err_b),
                )
                warned_unpaired = True
            idx_a = rng.integers(0, len(err_a), size=(n_bootstrap, len(err_a)))
            idx_b = rng.integers(0, len(err_b), size=(n_bootstrap, len(err_b)))
            diffs = np.mean(err_a[idx_a], axis=1) - np.mean(err_b[idx_b], axis=1)

        diff_lower = float(np.quantile(diffs, alpha))
        diff_upper = float(np.quantile(diffs, 1.0 - alpha))
        significant = bool(diff_lower > 0.0 or diff_upper < 0.0)
        diff_estimate = float(np.mean(err_a) - np.mean(err_b))
        if significant:
            better = "a" if diff_estimate < 0 else "b"
        else:
            better = "none"

        records.append(
            {
                "category": category,
                "metric": metric,
                "site": site,
                "component": component,
                "method_a_mae": float(np.mean(err_a)),
                "method_b_mae": float(np.mean(err_b)),
                "diff_estimate": diff_estimate,
                "diff_ci_lower": diff_lower,
                "diff_ci_upper": diff_upper,
                "significant": significant,
                "better_method": better,
                "paired": paired,
            }
        )

    return pd.DataFrame(records)


def _errors(group: pd.DataFrame) -> np.ndarray:
    """Per-realization absolute error from observed, ordered by realization.

    Comparison-kind metrics measure divergence from observed directly,
    so their value is the error.
    """
    ordered = group.sort_values("realization")
    vals = ordered["value"].to_numpy(dtype=float)
    observed = ordered["observed"].to_numpy(dtype=float)
    errors = np.where(np.isfinite(observed), np.abs(vals - observed), np.abs(vals))
    return errors[np.isfinite(errors)]
