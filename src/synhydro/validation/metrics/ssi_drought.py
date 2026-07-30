"""
SSI drought validation metrics.

The Standardized Streamflow Index (SSI) is fit on the observed record;
each synthetic realization is transformed with the same fitted
distributions. Drought events are identified on the SSI series: an
event starts when SSI drops to -1 or below and ends when SSI returns to
zero or above (McKee et al., 1993). Severity is the cumulative absolute
SSI over the event.

References
----------
McKee, T.B., Doesken, N.J., and Kleist, J. (1993). The relationship of
drought frequency and duration to time scales. Proc. 8th Conf. on
Applied Climatology, Anaheim, CA, 179-184.
"""

import logging

import numpy as np
import pandas as pd

from synhydro.core.ensemble import Ensemble
from synhydro.droughts.ssi import SSI

logger = logging.getLogger(__name__)

CATEGORY = "ssi_drought"

METRIC_UNITS = {
    "ssi_mean_drought_duration": "timesteps",
    "ssi_max_drought_duration": "timesteps",
    "ssi_mean_drought_severity": "dimensionless",
    "ssi_max_drought_severity": "dimensionless",
    "ssi_drought_frequency": "per_year",
}

_CITATION = "McKee et al. (1993)"


def _extract_ssi_droughts(
    ssi_values: np.ndarray,
    threshold: float = -1.0,
) -> dict[str, list]:
    """Extract drought events from SSI values.

    A drought event starts when SSI drops below the threshold and ends
    when it returns above 0.

    Parameters
    ----------
    ssi_values : np.ndarray
        SSI values.
    threshold : float, default -1.0
        SSI threshold to initiate a drought event.

    Returns
    -------
    dict
        'durations': list of drought durations in timesteps.
        'severities': list of cumulative SSI deficits (absolute value).
    """
    durations = []
    severities = []
    in_drought = False
    current_dur = 0
    current_sev = 0.0

    for val in ssi_values:
        if not np.isfinite(val):
            continue
        if in_drought:
            if val >= 0:
                # End of drought
                durations.append(current_dur)
                severities.append(current_sev)
                in_drought = False
                current_dur = 0
                current_sev = 0.0
            else:
                current_dur += 1
                current_sev += abs(val)
        else:
            if val <= threshold:
                in_drought = True
                current_dur = 1
                current_sev = abs(val)

    # Close any ongoing drought
    if in_drought and current_dur > 0:
        durations.append(current_dur)
        severities.append(current_sev)

    return {"durations": durations, "severities": severities}


def _event_statistics(ssi_values: np.ndarray, n_timesteps: int) -> dict[str, float]:
    """SSI drought event statistics. Frequency is events per year
    assuming 12 timesteps per year (SSI operates on monthly data)."""
    events = _extract_ssi_droughts(ssi_values)
    durations = events["durations"]
    severities = events["severities"]
    return {
        "ssi_mean_drought_duration": (float(np.mean(durations)) if durations else 0.0),
        "ssi_max_drought_duration": float(np.max(durations)) if durations else 0.0,
        "ssi_mean_drought_severity": (
            float(np.mean(severities)) if severities else 0.0
        ),
        "ssi_max_drought_severity": (float(np.max(severities)) if severities else 0.0),
        "ssi_drought_frequency": (
            len(durations) / (n_timesteps / 12.0) if n_timesteps > 0 else 0.0
        ),
    }


def compute_ssi_drought(
    ensemble: Ensemble,
    Q_obs: pd.DataFrame,
    sites: list[str],
    ssi_timescale: int = 12,
    ssi_dist: str = "gamma",
) -> tuple[list[tuple], list[tuple]]:
    """
    Compute SSI drought statistics per site and realization.

    The SSI is fit on the observed record at each site; realizations are
    transformed with the same fitted distributions so that synthetic
    droughts are measured on the observed climatology.

    Parameters
    ----------
    ensemble : Ensemble
        Synthetic ensemble.
    Q_obs : pd.DataFrame
        Observed flows with sites as columns.
    sites : list of str
        Sites to evaluate.
    ssi_timescale : int, default 12
        SSI accumulation timescale in months.
    ssi_dist : str, default 'gamma'
        Distribution used for the SSI fit.

    Returns
    -------
    rows : list of tuple
        Tidy value rows (category, metric, kind, site, component,
        realization, value, observed, units).
    skips : list of tuple
        Skip records (metric, site, reason) for sites where the SSI
        could not be fit or applied.
    """
    rows: list[tuple] = []
    skips: list[tuple] = []

    for site in sites:
        obs = Q_obs[site].dropna()
        if len(obs) < 36:
            skips.append(
                (
                    CATEGORY,
                    site,
                    "requires at least 36 observed timesteps for SSI fitting",
                )
            )
            continue

        try:
            ssi_calc = SSI(dist=ssi_dist, timescale=ssi_timescale)
            ssi_calc.fit(obs)
            obs_ssi = ssi_calc.get_training_ssi().dropna()
        except Exception:
            logger.debug("SSI fitting failed for site %s, skipping", site)
            skips.append((CATEGORY, site, "SSI fitting failed"))
            continue

        if len(obs_ssi) < 12:
            skips.append((CATEGORY, site, "fewer than 12 valid observed SSI values"))
            continue

        obs_stats = _event_statistics(obs_ssi.to_numpy(dtype=float), len(obs_ssi))

        site_rows: list[tuple] = []
        for rid, frame in ensemble.data_by_realization.items():
            if site not in frame.columns:
                continue
            syn = frame[site].dropna()
            if len(syn) < 36:
                continue
            try:
                syn_ssi = ssi_calc.transform(syn).dropna()
            except Exception:
                logger.debug(
                    "SSI transform failed for site %s realization %s", site, rid
                )
                continue
            if len(syn_ssi) < 12:
                continue
            syn_stats = _event_statistics(syn_ssi.to_numpy(dtype=float), len(syn_ssi))
            for name, value in syn_stats.items():
                site_rows.append(
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

        if site_rows:
            rows.extend(site_rows)
        else:
            skips.append((CATEGORY, site, "no realization produced a valid SSI series"))

    return rows, skips


def metric_inventory() -> list[dict]:
    """Inventory rows for list_metrics()."""
    descriptions = {
        "ssi_mean_drought_duration": "Mean SSI drought event length",
        "ssi_max_drought_duration": "Longest SSI drought event",
        "ssi_mean_drought_severity": "Mean cumulative absolute SSI per event",
        "ssi_max_drought_severity": "Largest cumulative absolute SSI",
        "ssi_drought_frequency": "SSI drought events per year",
    }
    return [
        {
            "name": name,
            "category": CATEGORY,
            "kind": "scalar",
            "units": units,
            "frequencies": "monthly",
            "min_years": 3,
            "citation": _CITATION,
            "description": descriptions[name],
        }
        for name, units in METRIC_UNITS.items()
    ]
