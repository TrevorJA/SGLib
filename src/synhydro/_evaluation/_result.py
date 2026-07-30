"""
Result container shared by the verification and validation suites.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

_SUMMARY_GROUP_COLUMNS = [
    "category",
    "metric",
    "kind",
    "site",
    "component",
    "units",
]

_NEAR_ZERO_FLOOR = 1e-10
_REJECT_ALPHA = 0.05
_FLOW_UNITS = frozenset({"flow", "flow_cumulative"})


@dataclass
class EvaluationResult:
    """
    Container for evaluation results in tidy long format.

    Attributes
    ----------
    values : pd.DataFrame
        One row per (metric, site, component, realization) with columns
        ``category, metric, kind, site, component, realization, value,
        observed, units``. The ``observed`` column repeats the observed
        statistic for the group; it is NaN for comparison-kind metrics,
        which have no observed value of their own.
    skipped : pd.DataFrame
        Metrics not evaluated, with columns ``metric, site, reason``.
    metadata : dict
        Run description: sites, realization count, frequency, options,
        and internal reporting hints.
    """

    values: pd.DataFrame
    skipped: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    _suite: str = "evaluation"

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the tidy per-realization values frame.

        Returns
        -------
        pd.DataFrame
            Copy of the tidy frame; one row per (metric, site,
            component, realization).
        """
        return self.values.copy()

    def summary(self) -> pd.DataFrame:
        """
        Summarize each metric's synthetic distribution against observed.

        For every (category, metric, site, component) group the summary
        reports the observed statistic, quantiles of the statistic
        across realizations, and the position of the observed value
        within the synthetic distribution (``obs_percentile``), the
        rank-based consistency check of Stedinger and Taylor (1982).

        Returns
        -------
        pd.DataFrame
            One row per (category, metric, site, component) with
            columns ``observed, syn_median, syn_mean, syn_q05, syn_q25,
            syn_q75, syn_q95, obs_percentile, in_90_band,
            relative_diff, reject_rate, n_realizations, units``.

        References
        ----------
        Stedinger, J.R. and Taylor, M.R. (1982). Synthetic streamflow
        generation: 1. Model verification and validation. Water
        Resources Research, 18(4), 909-918.
        """
        if self.values.empty:
            return pd.DataFrame(
                columns=_SUMMARY_GROUP_COLUMNS[:-1]
                + [
                    "observed",
                    "syn_median",
                    "syn_mean",
                    "syn_q05",
                    "syn_q25",
                    "syn_q75",
                    "syn_q95",
                    "obs_percentile",
                    "in_90_band",
                    "relative_diff",
                    "reject_rate",
                    "n_realizations",
                    "units",
                ]
            )

        reject_metrics = set(self.metadata.get("reject_rate_metrics", ()))
        site_median_flow = self.metadata.get("obs_site_median_flow", {})

        records = []
        grouped = self.values.groupby(_SUMMARY_GROUP_COLUMNS, dropna=False, sort=False)
        for keys, group in grouped:
            category, metric, kind, site, component, units = keys
            vals = group["value"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            observed = group["observed"].iloc[0]
            observed = float(observed) if pd.notna(observed) else np.nan

            record = {
                "category": category,
                "metric": metric,
                "kind": kind,
                "site": site,
                "component": component,
                "observed": observed,
                "syn_median": np.nan,
                "syn_mean": np.nan,
                "syn_q05": np.nan,
                "syn_q25": np.nan,
                "syn_q75": np.nan,
                "syn_q95": np.nan,
                "obs_percentile": np.nan,
                "in_90_band": pd.NA,
                "relative_diff": np.nan,
                "reject_rate": np.nan,
                "n_realizations": int(group["realization"].nunique()),
                "units": units,
            }

            if len(vals) > 0:
                record["syn_median"] = float(np.median(vals))
                record["syn_mean"] = float(np.mean(vals))
                q05, q25, q75, q95 = np.percentile(vals, [5, 25, 75, 95])
                record["syn_q05"] = float(q05)
                record["syn_q25"] = float(q25)
                record["syn_q75"] = float(q75)
                record["syn_q95"] = float(q95)

                if metric in reject_metrics:
                    record["reject_rate"] = float(np.mean(vals < _REJECT_ALPHA))
                elif np.isfinite(observed):
                    record["obs_percentile"] = _midrank_percentile(observed, vals)
                    record["in_90_band"] = bool(q05 <= observed <= q95)
                    record["relative_diff"] = _relative_diff(
                        record["syn_median"],
                        observed,
                        units,
                        site,
                        site_median_flow,
                    )

            records.append(record)

        return pd.DataFrame(records)

    def category_summary(self) -> pd.DataFrame:
        """
        Roll the summary up to one row per (category, site).

        All rollup quantities are unit-free, so metrics with different
        physical units are never averaged together. There is
        deliberately no single cross-category score.

        Returns
        -------
        pd.DataFrame
            One row per (category, site) with columns ``n_metrics,
            n_in_90_band, median_abs_relative_diff,
            median_obs_percentile_distance``.
        """
        summary = self.summary()
        if summary.empty:
            return pd.DataFrame(
                columns=[
                    "category",
                    "site",
                    "n_metrics",
                    "n_in_90_band",
                    "median_abs_relative_diff",
                    "median_obs_percentile_distance",
                ]
            )

        records = []
        for (category, site), group in summary.groupby(
            ["category", "site"], sort=False
        ):
            in_band = group["in_90_band"].dropna()
            rel = group["relative_diff"].to_numpy(dtype=float)
            rel = rel[np.isfinite(rel)]
            pct = group["obs_percentile"].to_numpy(dtype=float)
            pct = pct[np.isfinite(pct)]
            records.append(
                {
                    "category": category,
                    "site": site,
                    "n_metrics": int(group["metric"].nunique()),
                    "n_in_90_band": int(in_band.sum()),
                    "median_abs_relative_diff": (
                        float(np.median(np.abs(rel))) if len(rel) else np.nan
                    ),
                    "median_obs_percentile_distance": (
                        float(np.median(np.abs(pct - 0.5))) if len(pct) else np.nan
                    ),
                }
            )
        return pd.DataFrame(records)

    def _repr_html_(self) -> str:
        """Notebook display: styled summary table."""
        summary = self.summary()
        caption = (
            f"{type(self).__name__}: {self.metadata.get('n_realizations', '?')} "
            f"realizations, {self.metadata.get('n_sites', '?')} sites, "
            f"{self.metadata.get('base_frequency', '?')} frequency"
        )
        try:
            styler = (
                summary.style.background_gradient(
                    subset=["obs_percentile"], cmap="coolwarm", vmin=0.0, vmax=1.0
                )
                .format(precision=3, na_rep="")
                .set_caption(caption)
            )
            return styler.to_html()
        except Exception:
            return f"<p>{caption}</p>" + summary.to_html()

    def __repr__(self) -> str:
        n_metrics = self.values["metric"].nunique() if not self.values.empty else 0
        return (
            f"{type(self).__name__}(metrics={n_metrics}, "
            f"sites={self.metadata.get('n_sites', '?')}, "
            f"realizations={self.metadata.get('n_realizations', '?')}, "
            f"skipped={len(self.skipped)})"
        )


def _midrank_percentile(observed: float, synthetic: np.ndarray) -> float:
    """
    Midrank position of the observed value among synthetic values.

    Returns ``(n_less + 0.5 * n_equal + 0.5) / (n + 1)``, the plotting
    position of the observed statistic within the synthetic sample.
    Values near 0 or 1 indicate the observed statistic lies in the tail
    of the ensemble distribution.
    """
    n = len(synthetic)
    n_less = int(np.sum(synthetic < observed))
    n_equal = int(np.sum(synthetic == observed))
    return float((n_less + 0.5 * n_equal + 0.5) / (n + 1))


def _relative_diff(
    syn_median: float,
    observed: float,
    units: str,
    site: str,
    site_median_flow: dict,
) -> float:
    """
    Relative difference of the synthetic median from observed.

    Returns NaN when the observed value is too close to zero for a
    relative difference to be meaningful. For flow-unit metrics the
    floor is 1 percent of the site's median observed flow; otherwise a
    small absolute floor is used.
    """
    if units == "pvalue" or not np.isfinite(observed):
        return np.nan
    floor = _NEAR_ZERO_FLOOR
    if units in _FLOW_UNITS:
        median_flow = site_median_flow.get(site)
        if median_flow is not None and np.isfinite(median_flow) and median_flow > 0:
            floor = 0.01 * median_flow
    if abs(observed) < floor:
        return np.nan
    return float((syn_median - observed) / abs(observed))
