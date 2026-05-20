import logging
from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Grouper, Series, Timedelta, Timestamp
from scipy.stats import norm
from spei import SI  # Import the original SI class
from spei._typing import ContinuousDist
from spei.utils import get_data_series, group_yearly_df, validate_series

# Import distribution utilities from dedicated module
from synhydro.droughts.distributions import get_distribution, validate_distribution

# Pandas version-aware frequency alias
_PANDAS_VERSION = tuple(int(x) for x in pd.__version__.split(".")[:2])
_MONTH_END_FREQ = "ME" if _PANDAS_VERSION >= (2, 2) else "M"

# Default CDF clip floor. Chosen so that abs(norm.ppf(epsilon)) == 4.5,
# which bounds SSI within +/- 4.5. This suppresses gamma-tail extrapolation
# artifacts when transforming flows below the fitted support.
_DEFAULT_CDF_EPSILON = float(norm.cdf(-4.5))  # approx 3.397673e-06


def _normalize_freq(freq: str | None) -> str | None:
    """Normalize frequency aliases for cross-pandas-version compatibility.

    Translates between 'M' and 'ME' month-end aliases depending on
    the installed pandas version (>= 2.2 uses 'ME', < 2.2 uses 'M').
    """
    if freq is None:
        return freq
    if _PANDAS_VERSION >= (2, 2) and freq == "M":
        return "ME"
    if _PANDAS_VERSION < (2, 2) and freq == "ME":
        return "M"
    return freq


def get_drought_metrics(ssi, end_drought_threshold_months=3):
    """
    Calculate drought metrics from standardized supply index (SSI) time series.

    Parameters
    ----------
    ssi : pd.Series
        Time series of standardized supply index values
    end_drought_threshold_months : int, default=3
        Number of consecutive days with SSI > 0 required to end a critical drought

    Returns
    -------
    pd.DataFrame
        DataFrame containing drought metrics for each identified drought event
    """
    import pandas as pd

    ## Get historic drought metrics
    drought_data = {}
    drought_counter = 0
    in_critical_drought = False
    drought_days = []
    positive_days_count = 0  # Track consecutive days with SSI > 0

    for ind in range(len(ssi)):
        if ssi.values[ind] < 0:
            drought_days.append(ind)
            positive_days_count = 0  # Reset counter when SSI < 0

            if ssi.values[ind] <= -1:
                in_critical_drought = True
        else:
            # Check if we're in a critical drought
            if in_critical_drought:
                positive_days_count += 1

                # Only end drought after N consecutive positive days
                if positive_days_count >= end_drought_threshold_months:
                    # Get date with max severity
                    max_severity_date = ssi.index[drought_days][
                        ssi.values[drought_days].argmin()
                    ]
                    max_severity_idx = drought_days[ssi.values[drought_days].argmin()]

                    # Calculate average severity during drought
                    avg_severity = ssi.values[drought_days].mean()

                    # Calculate recovery period (months from peak severity to end)
                    recovery_period = len(drought_days) - drought_days.index(
                        max_severity_idx
                    )

                    # Calculate standardized water surplus prior to drought start
                    start_idx = drought_days[0]

                    # 1-month prior surplus
                    prior_1m_start = max(0, start_idx - 1)
                    prior_1m_surplus = sum(
                        [v for v in ssi.values[prior_1m_start:start_idx] if v > 0]
                    )

                    # 3-month prior surplus
                    prior_3m_start = max(0, start_idx - 3)
                    prior_3m_surplus = sum(
                        [v for v in ssi.values[prior_3m_start:start_idx] if v > 0]
                    )

                    # 6-month prior surplus
                    prior_6m_start = max(0, start_idx - 6)
                    prior_6m_surplus = sum(
                        [v for v in ssi.values[prior_6m_start:start_idx] if v > 0]
                    )

                    drought_counter += 1
                    drought_data[drought_counter] = {
                        "start": ssi.index[drought_days[0]],
                        "end": ssi.index[drought_days[-1]],
                        "duration": len(drought_days),
                        "magnitude": sum(ssi.values[drought_days]),
                        "severity": min(ssi.values[drought_days]),
                        "avg_severity": avg_severity,
                        "max_severity_date": max_severity_date,
                        "recovery_period": recovery_period,
                        "prior_1m_surplus": prior_1m_surplus,
                        "prior_3m_surplus": prior_3m_surplus,
                        "prior_6m_surplus": prior_6m_surplus,
                    }

                    in_critical_drought = False
                    drought_days = []
                    positive_days_count = 0
            else:
                # Not in drought, reset drought_days
                drought_days = []
                positive_days_count = 0

    drought_metrics = pd.DataFrame(drought_data).transpose()
    if len(drought_metrics) > 0:
        for col in ["start", "end", "max_severity_date"]:
            if col in drought_metrics.columns:
                drought_metrics[col] = pd.to_datetime(drought_metrics[col])
        numeric_cols = [
            "duration",
            "magnitude",
            "severity",
            "avg_severity",
            "recovery_period",
            "prior_1m_surplus",
            "prior_3m_surplus",
            "prior_6m_surplus",
        ]
        for col in numeric_cols:
            if col in drought_metrics.columns:
                drought_metrics[col] = pd.to_numeric(drought_metrics[col])
    return drought_metrics


@dataclass
class SSI:
    """
    Independent SSI calculator that separates training and scoring phases.

    Uses the original spei.SI class internally for distribution fitting,
    then applies those fitted distributions to new data.

    Parameters
    ----------
    dist : str or ContinuousDist, default 'gamma'
        Probability distribution for SSI calculation.
        Can be string name: 'gamma', 'lognorm', 'pearson3', 'weibull', etc.
        Or scipy distribution object.
    timescale : int, default 12
        Rolling window size for temporal aggregation.
    fit_freq : str, optional
        Frequency for seasonal fitting ('M' for monthly, 'D' for daily).
        If None, fits single distribution to entire dataset.
    fit_window : int, default 0
        Moving window for distribution fitting.
    prob_zero : bool, default False
        Whether to handle zero probability separately.
    normal_scores_transform : bool, default False
        Whether to use normal scores transform instead of parametric fitting.
    agg_func : {'sum', 'mean'}, default 'sum'
        Aggregation function for rolling window.
    cdf_epsilon : float, keyword-only, default norm.cdf(-4.5) (approx 3.4e-6)
        CDF clip floor applied before the inverse-normal transform. The CDF
        is clipped to [cdf_epsilon, 1 - cdf_epsilon] so that saturated tail
        probabilities do not map to +/- inf via norm.ppf. This implicitly
        bounds the SSI series within +/- abs(norm.ppf(cdf_epsilon)):

            SSI_bound is approximately abs(norm.ppf(cdf_epsilon))

        Reasonable values:

            cdf_epsilon = 1e-3   -> SSI bound approx +/- 3.09 (literature standard)
            cdf_epsilon = 3e-5   -> SSI bound approx +/- 4.00
            cdf_epsilon = 3.4e-6 -> SSI bound approx +/- 4.50 (new default)

    Notes
    -----
    The default cdf_epsilon changed from an implicit floating-point safety
    value (1e-10, bounding SSI at approx +/- 6.36) to a value that bounds SSI
    within +/- 4.5. This is a behavior change: any monthly value that
    previously produced SSI < -4.5 or SSI > 4.5 will now be clipped to those
    bounds. The new default rejects gamma-tail extrapolation artifacts that
    arise when transforming flows below the fitted support, while still
    leaving room for genuine extremes. Pass an explicit cdf_epsilon to
    restore a different bound (e.g. cdf_epsilon=1e-3 for the SSI literature
    standard of +/- 3.09).

    Examples
    --------
    >>> # Basic usage with default gamma distribution
    >>> ssi = SSI()
    >>> ssi.fit(training_data)
    >>> ssi_values = ssi.transform(new_data)

    >>> # Using different distribution
    >>> ssi = SSI(dist='lognorm', timescale=6)
    >>> ssi.fit(training_data)
    """

    # Training parameters
    dist: Union[str, ContinuousDist] = "gamma"
    timescale: int = 12
    fit_freq: str | None = field(default=None)
    fit_window: int = field(default=0)
    prob_zero: bool = field(default=False)
    normal_scores_transform: bool = field(default=False)
    agg_func: Literal["sum", "mean"] = "sum"
    cdf_epsilon: float = field(default=_DEFAULT_CDF_EPSILON, kw_only=True)

    # Internal state
    _fitted_si: SI = field(init=False, repr=False, compare=False)
    _is_fitted: bool = field(default=False, init=False, repr=False, compare=False)
    _training_series: Series = field(init=False, repr=False, compare=False)
    _dist_obj: ContinuousDist = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Convert dist string to distribution object and normalize freq aliases."""
        self._dist_obj = get_distribution(self.dist)
        self.fit_freq = _normalize_freq(self.fit_freq)

    def fit(self, training_series: Series) -> "SSI":
        """
        Fit distributions using training data.

        Parameters
        ----------
        training_series : Series
            Time series data for fitting distributions

        Returns
        -------
        SSI
            Self for method chaining
        """
        self._training_series = training_series.copy()

        # Apply rolling aggregation if timescale is set
        if self.timescale > 0:
            training_series = (
                training_series.rolling(self.timescale, min_periods=self.timescale)
                .agg(self.agg_func)
                .dropna()
                .copy()
            )

        # Create and fit the original SI object
        self._fitted_si = SI(
            series=training_series,
            dist=self._dist_obj,
            timescale=0,
            fit_freq=self.fit_freq,
            fit_window=self.fit_window,
            prob_zero=self.prob_zero,
            normal_scores_transform=self.normal_scores_transform,
            agg_func=self.agg_func,
        )

        self._fitted_si.fit_distribution()
        self._is_fitted = True

        return self

    def transform(self, new_series: Series) -> Series:
        """
        Calculate SSI values for new data using fitted distributions.

        Parameters
        ----------
        new_series : Series
            New time series data to transform

        Returns
        -------
        Series
            SSI values for the new series
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform()")

        # Preprocess new series the same way as training data
        new_series = validate_series(new_series)

        if self.timescale > 0:
            new_series = (
                new_series.rolling(self.timescale, min_periods=self.timescale)
                .agg(self.agg_func)
                .dropna()
                .copy()
            )

        # Calculate CDF for new data using fitted distributions
        cdf = self._calculate_cdf_for_new_data(new_series)

        # Clip CDF tails to bound SSI within +/- abs(norm.ppf(cdf_epsilon));
        # default cdf_epsilon yields SSI in [-4.5, +4.5]. This suppresses
        # gamma-tail extrapolation artifacts when transforming flows below
        # the fitted support (e.g. synthetic traces from MOEA search).
        cdf_clipped = np.clip(cdf.values, self.cdf_epsilon, 1 - self.cdf_epsilon)

        # Convert to standard normal (SSI values)
        ssi = Series(
            norm.ppf(cdf_clipped, loc=0, scale=1), index=new_series.index, dtype=float
        )

        return ssi

    def fit_transform(
        self, training_series: Series, new_series: Series = None
    ) -> Series:
        """
        Fit on training data and transform new data in one step.

        Parameters
        ----------
        training_series : Series
            Data for fitting distributions
        new_series : Series, optional
            Data to transform. If None, transforms training_series.

        Returns
        -------
        Series
            SSI values
        """
        self.fit(training_series)

        if new_series is not None:
            return self.transform(new_series)
        else:
            # Transform training data using fitted distributions
            return self._fitted_si.norm_ppf()

    def _calculate_cdf_for_new_data(self, new_series: Series) -> Series:
        """
        Calculate CDF for new data using the fitted distributions.

        Parameters
        ----------
        new_series : Series
            New time series data

        Returns
        -------
        Series
            CDF values for new data
        """
        if self.normal_scores_transform:
            return self._calculate_cdf_nsf(new_series)
        else:
            return self._calculate_cdf_parametric(new_series)

    def _calculate_cdf_nsf(self, new_series: Series) -> Series:
        """Calculate CDF using Normal Scores Transform approach."""
        cdf = Series(np.nan, index=new_series.index, dtype=float)
        new_grouped = group_yearly_df(series=new_series)

        for date, grval in new_grouped.groupby(
            Grouper(freq=str(self._fitted_si.fit_freq))
        ):
            new_data = get_data_series(grval)

            # Find corresponding training data for this period
            if date in self._fitted_si._dist_dict:
                training_data = self._fitted_si._dist_dict[date].data.sort_values()
                n_train = len(training_data)

                # For each new data point, find its rank in training data
                for idx, val in new_data.items():
                    # Count how many training values are less than this new value
                    rank = (training_data < val).sum()
                    # Use Weibull plotting position
                    cdf.loc[idx] = (rank + 0.5) / n_train

        return cdf

    def _calculate_cdf_parametric(self, new_series: Series) -> Series:
        """Calculate CDF using fitted parametric distributions."""
        cdf = Series(np.nan, index=new_series.index, dtype=float)
        new_grouped = group_yearly_df(series=new_series)

        for date, grval in new_grouped.groupby(
            Grouper(freq=str(self._fitted_si.fit_freq))
        ):
            new_data = get_data_series(grval)

            if date in self._fitted_si._dist_dict:
                fitted_dist = self._fitted_si._dist_dict[date]

                # Construct parameters tuple: (shape_params..., loc, scale)
                if fitted_dist.pars is not None:
                    params = (*fitted_dist.pars, fitted_dist.loc, fitted_dist.scale)
                else:
                    params = (fitted_dist.loc, fitted_dist.scale)

                # Handle zero probability case
                if self.prob_zero and fitted_dist.p0 > 0:
                    zero_prob = fitted_dist.p0

                    for idx, val in new_data.items():
                        if val == 0:
                            cdf.loc[idx] = zero_prob
                        else:
                            # Get parametric CDF for non-zero values
                            dist_cdf = fitted_dist.dist.cdf(val, *params)
                            cdf.loc[idx] = zero_prob + (1 - zero_prob) * dist_cdf
                else:
                    # No zero handling needed
                    for idx, val in new_data.items():
                        cdf.loc[idx] = fitted_dist.dist.cdf(val, *params)

        return cdf

    def get_training_ssi(self) -> Series:
        """
        Get SSI values for the training data.

        Returns
        -------
        Series
            SSI values for training data
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before get_training_ssi()")

        return self._fitted_si.norm_ppf()

    @property
    def fitted_distributions(self) -> dict:
        """Access to fitted distributions (read-only)."""
        if not self._is_fitted:
            raise ValueError("Must call fit() before accessing fitted_distributions")
        return self._fitted_si._dist_dict.copy()
