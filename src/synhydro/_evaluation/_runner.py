"""
Generic evaluation engine shared by the verification and validation suites.

The runner evaluates each selected metric once on the observed record and
once on every ensemble realization, then emits a tidy long-format frame:
one row per (metric, site, component, realization). This implements the
standard reporting idiom of the synthetic streamflow literature: the
observed statistic is compared against the distribution of the statistic
across realizations (Stedinger and Taylor, 1982).
"""

import logging
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from synhydro.core.ensemble import Ensemble
from synhydro._evaluation._context import MetricContext
from synhydro._evaluation._registry import MetricSpec

logger = logging.getLogger(__name__)

VALUE_COLUMNS = [
    "category",
    "metric",
    "kind",
    "site",
    "component",
    "realization",
    "value",
    "observed",
    "units",
]

SKIP_COLUMNS = ["metric", "site", "reason"]

PAIR_SEPARATOR = "|"


def resolve_sites(
    ensemble: Ensemble,
    observed: pd.DataFrame,
    sites: Optional[list[str]] = None,
) -> list[str]:
    """
    Resolve the sites shared by the ensemble and the observed data.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic ensemble.
    observed : pd.DataFrame
        Observed data with sites as columns.
    sites : list of str, optional
        Requested subset. If None, all shared sites are used.

    Returns
    -------
    list of str
        Sites present in both the ensemble and the observed data.

    Raises
    ------
    ValueError
        If no shared sites exist or a requested site is unavailable.
    """
    shared = [s for s in ensemble.site_names if s in observed.columns]
    if not shared:
        raise ValueError(
            f"No shared sites between ensemble ({ensemble.site_names}) and "
            f"observed data ({list(observed.columns)})."
        )
    if sites is None:
        return shared
    missing = [s for s in sites if s not in shared]
    if missing:
        raise ValueError(
            f"Requested sites {missing} are not present in both the "
            f"ensemble and the observed data. Shared sites: {shared}."
        )
    return list(sites)


def run_metrics(
    ensemble: Ensemble,
    observed: pd.DataFrame,
    specs: list[MetricSpec],
    context: MetricContext,
    sites: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate metrics on observed data and every ensemble realization.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic ensemble.
    observed : pd.DataFrame
        Observed data with DatetimeIndex and sites as columns.
    specs : list of MetricSpec
        Metrics to evaluate.
    context : MetricContext
        Resolved run options (frequency, method choices).
    sites : list of str
        Sites to evaluate; must exist in both inputs.

    Returns
    -------
    values : pd.DataFrame
        Tidy frame with columns ``category, metric, kind, site,
        component, realization, value, observed, units``.
    skipped : pd.DataFrame
        Frame with columns ``metric, site, reason`` recording gated or
        failed evaluations.
    """
    rows: list[tuple] = []
    skips: list[tuple] = []

    n_obs_years = len(observed) / context.steps_per_year
    realizations = ensemble.data_by_realization
    obs_series = {site: observed[site].dropna() for site in sites}

    for spec in specs:
        if (
            spec.frequencies is not None
            and context.base_frequency not in spec.frequencies
        ):
            skips.append(
                (
                    spec.name,
                    "all",
                    f"requires {sorted(spec.frequencies)} data, ensemble is "
                    f"{context.base_frequency}",
                )
            )
            continue
        if spec.min_years is not None and n_obs_years < spec.min_years:
            skips.append(
                (
                    spec.name,
                    "all",
                    f"requires at least {spec.min_years} years of observed "
                    f"data, found {n_obs_years:.1f}",
                )
            )
            continue

        kwargs = {key: getattr(context, key) for key in spec.needs}

        if spec.kind == "matrix":
            _run_matrix(spec, kwargs, observed, realizations, sites, rows, skips)
        elif spec.kind == "comparison":
            _run_comparison(spec, kwargs, obs_series, realizations, sites, rows)
        else:
            _run_per_site(spec, kwargs, obs_series, realizations, sites, rows)

    values = pd.DataFrame(rows, columns=VALUE_COLUMNS)
    skipped = pd.DataFrame(skips, columns=SKIP_COLUMNS)
    return values, skipped


def _run_per_site(
    spec: MetricSpec,
    kwargs: dict,
    obs_series: dict[str, pd.Series],
    realizations: dict,
    sites: list[str],
    rows: list[tuple],
) -> None:
    """Evaluate a scalar or curve metric per site."""
    for site in sites:
        obs_result = spec.func(obs_series[site], **kwargs)
        for rid, frame in realizations.items():
            if site not in frame.columns:
                continue
            syn_result = spec.func(frame[site].dropna(), **kwargs)
            if spec.kind == "scalar":
                rows.append(
                    (
                        spec.category,
                        spec.name,
                        spec.kind,
                        site,
                        pd.NA,
                        rid,
                        float(syn_result),
                        float(obs_result),
                        spec.units,
                    )
                )
            else:
                for component, value in syn_result.items():
                    obs_value = (
                        float(obs_result.get(component, np.nan))
                        if obs_result is not None
                        else np.nan
                    )
                    rows.append(
                        (
                            spec.category,
                            spec.name,
                            spec.kind,
                            site,
                            component,
                            rid,
                            float(value),
                            obs_value,
                            spec.units,
                        )
                    )


def _run_comparison(
    spec: MetricSpec,
    kwargs: dict,
    obs_series: dict[str, pd.Series],
    realizations: dict,
    sites: list[str],
    rows: list[tuple],
) -> None:
    """Evaluate a comparison metric per site and realization."""
    for site in sites:
        reference = obs_series[site]
        for rid, frame in realizations.items():
            if site not in frame.columns:
                continue
            result = spec.func(frame[site].dropna(), reference, **kwargs)
            if isinstance(result, pd.Series):
                for component, value in result.items():
                    rows.append(
                        (
                            spec.category,
                            spec.name,
                            spec.kind,
                            site,
                            component,
                            rid,
                            float(value),
                            np.nan,
                            spec.units,
                        )
                    )
            else:
                rows.append(
                    (
                        spec.category,
                        spec.name,
                        spec.kind,
                        site,
                        pd.NA,
                        rid,
                        float(result),
                        np.nan,
                        spec.units,
                    )
                )


def _run_matrix(
    spec: MetricSpec,
    kwargs: dict,
    observed: pd.DataFrame,
    realizations: dict,
    sites: list[str],
    rows: list[tuple],
    skips: list[tuple],
) -> None:
    """Evaluate a matrix (site-pair) metric on aligned site frames."""
    if len(sites) < 2:
        skips.append((spec.name, "all", "requires at least 2 sites"))
        return
    obs_result = spec.func(observed[sites], **kwargs)
    for rid, frame in realizations.items():
        available = [s for s in sites if s in frame.columns]
        if len(available) < 2:
            continue
        syn_result = spec.func(frame[available], **kwargs)
        for label, value in syn_result.items():
            rows.append(
                (
                    spec.category,
                    spec.name,
                    spec.kind,
                    label,
                    pd.NA,
                    rid,
                    float(value),
                    float(obs_result.get(label, np.nan)),
                    spec.units,
                )
            )


def pair_label(site_a: str, site_b: str) -> str:
    """Return the canonical label for an unordered site pair."""
    return f"{site_a}{PAIR_SEPARATOR}{site_b}"


def site_pairs(sites: list[str]) -> list[tuple[str, str]]:
    """Return all unordered site pairs in column order."""
    return list(combinations(sites, 2))
