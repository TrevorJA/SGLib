"""
Evaluation context passed to metric functions.
"""

from dataclasses import dataclass

from synhydro._evaluation._frequency import FrequencyInfo


@dataclass(frozen=True)
class MetricContext:
    """
    Run-level options injected into metric functions.

    Metric functions declare which context attributes they need via the
    ``needs`` field of their MetricSpec; the runner injects those
    attributes as keyword arguments. All injectable options have
    defaults on the metric functions themselves so metrics remain
    directly callable outside an evaluation run.

    Attributes
    ----------
    frequency : FrequencyInfo
        Resolved ensemble frequency.
    hurst_method : str
        Hurst estimation method, ``'rs'`` or ``'dfa'``.
    acf_lags : int
        Maximum lag for autocorrelation function metrics.
    """

    frequency: FrequencyInfo
    hurst_method: str = "rs"
    acf_lags: int = 12

    @property
    def base_frequency(self) -> str:
        """Base frequency name ('daily', 'weekly', 'monthly', 'annual')."""
        return self.frequency.base

    @property
    def pandas_alias(self) -> str:
        """Canonical pandas frequency alias for resampling."""
        return self.frequency.pandas_alias

    @property
    def steps_per_year(self) -> float:
        """Average number of timesteps per year."""
        return self.frequency.steps_per_year
