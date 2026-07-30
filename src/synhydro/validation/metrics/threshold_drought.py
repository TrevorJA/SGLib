"""
Threshold drought validation metrics.

Drought events are identified with the theory of runs: an event is a
maximal sequence of consecutive timesteps with flow below a truncation
level. Event duration is the run length, severity is the cumulative
deficit below the threshold, and frequency is the event count per
timestep.

References
----------
Yevjevich, V. (1967). An objective approach to definitions and
investigations of continental hydrologic droughts. Hydrology Paper 23,
Colorado State University.

Salas, J.D., Delleur, J.W., Yevjevich, V., and Lane, W.L. (1980).
Applied Modeling of Hydrologic Time Series. Water Resources
Publications.
"""

from typing import Optional

import numpy as np
import pandas as pd

from synhydro.core.ensemble import Ensemble
from synhydro._evaluation import extract_runs

CATEGORY = "threshold_drought"

METRIC_UNITS = {
    "mean_drought_duration": "timesteps",
    "mean_drought_severity": "flow_cumulative",
    "max_drought_duration": "timesteps",
    "max_drought_severity": "flow_cumulative",
    "drought_frequency": "per_timestep",
}

_CITATION = "Yevjevich (1967); Salas et al. (1980)"


def _event_statistics(values: np.ndarray, threshold: float) -> dict[str, float]:
    """Run-event statistics of a series below a fixed threshold."""
    durations, severities = extract_runs(values, threshold)
    return {
        "mean_drought_duration": float(np.mean(durations)) if durations else 0.0,
        "mean_drought_severity": float(np.mean(severities)) if severities else 0.0,
        "max_drought_duration": float(np.max(durations)) if durations else 0.0,
        "max_drought_severity": float(np.max(severities)) if severities else 0.0,
        "drought_frequency": (len(durations) / len(values) if len(values) > 0 else 0.0),
    }


def compute_threshold_drought(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
    threshold: Optional[float],
) -> tuple[list[tuple], list[tuple]]:
    """
    Compute threshold drought statistics per site and realization.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic ensemble.
    Q_obs : pd.DataFrame
        Observed flows with sites as columns.
    sites : list of str
        Sites to evaluate.
    threshold : float or None
        Truncation level applied to all sites. If None, each site uses
        the 20th percentile of its observed flows.

    Returns
    -------
    rows : list of tuple
        Tidy value rows (category, metric, kind, site, component,
        realization, value, observed, units).
    skips : list of tuple
        Skip records (metric, site, reason); always empty here.
    """
    rows: list[tuple] = []

    for site in sites:
        obs = Q_obs[site].dropna().to_numpy(dtype=float)
        site_threshold = (
            float(np.percentile(obs, 20)) if threshold is None else threshold
        )
        obs_stats = _event_statistics(obs, site_threshold)

        for rid, frame in ensemble.data_by_realization.items():
            if site not in frame.columns:
                continue
            syn = frame[site].dropna().to_numpy(dtype=float)
            syn_stats = _event_statistics(syn, site_threshold)
            for name, value in syn_stats.items():
                rows.append(
                    (
                        CATEGORY,
                        name,
                        "scalar",
                        site,
                        pd.NA,
                        rid,
                        float(value),
                        float(obs_stats[name]),
                        METRIC_UNITS[name],
                    )
                )

    return rows, []


def metric_inventory() -> list[dict]:
    """Inventory rows for list_metrics()."""
    descriptions = {
        "mean_drought_duration": "Mean below-threshold run length",
        "mean_drought_severity": "Mean cumulative deficit per event",
        "max_drought_duration": "Longest below-threshold run",
        "max_drought_severity": "Largest cumulative deficit",
        "drought_frequency": "Drought events per timestep",
    }
    return [
        {
            "name": name,
            "category": CATEGORY,
            "kind": "scalar",
            "units": units,
            "frequencies": "any",
            "min_years": None,
            "citation": _CITATION,
            "description": descriptions[name],
        }
        for name, units in METRIC_UNITS.items()
    ]
