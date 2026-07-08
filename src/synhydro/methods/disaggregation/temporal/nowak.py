"""
Nowak KNN proportion-vector temporal disaggregator (Nowak et al. 2010).

Disaggregates coarse-timestep streamflow to a finer timestep by resampling
historic proportion vectors with K-nearest neighbor matching on the coarse
flow total at an index gauge. Supports any pair of discrete timescales from
{annual, monthly, weekly} down to {monthly, weekly, daily} where the output
timestep is finer than the input timestep.

References
----------
Nowak, K., Prairie, J., Rajagopalan, B., & Lall, U. (2010).
A nonparametric stochastic approach for multisite disaggregation of
annual to daily streamflow. Water Resources Research, 46(8).

Lall, U., & Sharma, A. (1996). A nearest neighbor bootstrap for resampling
hydrologic time series. Water Resources Research, 32(3), 679-693.
"""

import calendar
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from synhydro.core.base import Disaggregator, DisaggregatorParams, FittedParams
from synhydro.core.ensemble import Ensemble
from synhydro.core.seeding import as_seed_sequence, realization_rng

_WEEKLY_FREQ = "W-SUN"

_TIMESTEP_RANK = {"annual": 0, "monthly": 1, "weekly": 2, "daily": 3}
_INPUT_TIMESTEPS = ("annual", "monthly", "weekly")
_OUTPUT_TIMESTEPS = ("monthly", "weekly", "daily")


@dataclass(frozen=True)
class _ScaleConfig:
    """
    Static facts about one (input_timestep, output_timestep) pair.

    Attributes
    ----------
    input_freq : str
        Pandas frequency string of the coarse (input) timestep.
    output_freq : str
        Pandas frequency string of the fine (output) timestep.
    period_labels : tuple
        Within-year labels of the coarse periods; each label keys one KNN
        candidate pool (calendar months 1-12, ISO weeks 1-52, or a single
        annual label).
    max_steps : int
        Maximum number of fine steps in one coarse period; used to pad the
        profile arrays (e.g. 31 days per month, 366 days per year).
    default_pool_shift : int
        Default for ``max_knn_pool_shift_timesteps``, chosen so the shifted
        pool windows stay small relative to the coarse period length.
    renormalize_truncated : bool
        Whether sampled proportion vectors truncated to a shorter target
        period are renormalized to sum to one. False only for the
        monthly-to-daily pair, whose established production behavior uses
        truncated proportions as-is (a small mass loss possible only when a
        29-day February profile is applied to a 28-day February).
    """

    input_freq: str
    output_freq: str
    period_labels: tuple
    max_steps: int
    default_pool_shift: int
    renormalize_truncated: bool


_SCALE_CONFIGS = {
    ("monthly", "daily"): _ScaleConfig("MS", "D", tuple(range(1, 13)), 31, 7, False),
    ("weekly", "daily"): _ScaleConfig("W-SUN", "D", tuple(range(1, 53)), 7, 2, True),
    ("monthly", "weekly"): _ScaleConfig("MS", "W-SUN", tuple(range(1, 13)), 5, 1, True),
    ("annual", "monthly"): _ScaleConfig("YS", "MS", (1,), 12, 0, True),
    ("annual", "weekly"): _ScaleConfig("YS", "W-SUN", (1,), 52, 2, True),
    ("annual", "daily"): _ScaleConfig("YS", "D", (1,), 366, 7, True),
}


def _normalize_freq(freq: Optional[str]) -> Optional[str]:
    """
    Normalize a pandas frequency alias to a canonical form.

    Weekly anchors (``'W'``, ``'W-MON'``, ...) normalize to ``'W-SUN'``,
    month aliases to ``'MS'``, and annual aliases to ``'YS'``.

    Parameters
    ----------
    freq : str or None
        Frequency alias to normalize.

    Returns
    -------
    str or None
        Canonical frequency string, or None if ``freq`` is None.
    """
    if freq is None:
        return None
    if freq.startswith("W"):
        return _WEEKLY_FREQ
    if freq in ("M", "ME") or freq.startswith("MS"):
        return "MS"
    if freq in ("A", "Y", "YE") or freq.startswith(("YS", "AS")):
        return "YS"
    return freq


class NowakDisaggregator(Disaggregator):
    """
    Temporal disaggregation via KNN resampling of historic proportion vectors,
    as described in Nowak et al. (2010).

    Supports both single-site and multisite disaggregation between discrete
    timescale pairs: any input timestep in {annual, monthly, weekly} to any
    finer output timestep in {monthly, weekly, daily}.

    For each coarse period in the synthetic data, finds the N historic periods
    whose total flow at the index gauge (sum of all sites) is most similar.
    One of the N candidates is sampled and its fine-timestep flow proportions
    are used to disaggregate the synthetic coarse flow at all sites, which
    guarantees summability of the fine flows to the coarse total
    (Nowak et al. 2010, Section 2.1).

    When the input timestep is monthly or weekly, candidate profiles are
    conditioned on the calendar period (month of year, or ISO week of year)
    and the pool is enlarged by shifting each historic window by up to
    ``max_knn_pool_shift_timesteps`` output timesteps in each direction. When
    the input timestep is annual there is a single unconditioned pool, exactly
    as in the original paper.

    Timescale conventions
    ---------------------
    - Weekly timesteps use ISO weeks anchored on Sundays (``'W-SUN'``). Years
      are treated as exactly 52 ISO weeks; ISO week 53 is folded into the
      week 52 pool on input and never generated on output, consistent with
      ``KirschGenerator``.
    - For monthly to weekly disaggregation, weeks do not nest inside months;
      each week is assigned to the calendar month containing its Sunday
      anchor. Disaggregated weekly flows sum to the synthetic monthly flow
      over the weeks assigned to that month.
    - Coarse periods with more fine steps than a sampled candidate profile
      (leap-year day, fifth week of a month) redistribute the missing
      proportion mass; the reverse case truncates the profile.

    References
    ----------
    Nowak, K., Prairie, J., Rajagopalan, B., & Lall, U. (2010).
    A nonparametric stochastic approach for multisite disaggregation of
    annual to daily streamflow. Water Resources Research, 46(8).

    Lall, U., & Sharma, A. (1996). A nearest neighbor bootstrap for
    resampling hydrologic time series. Water Resources Research, 32(3).
    """

    def __init__(
        self,
        *,
        input_timestep: str = "monthly",
        output_timestep: str = "daily",
        n_neighbors: int = 5,
        max_knn_pool_shift_timesteps: Optional[int] = None,
        boundary_blend_timesteps: int = 2,
        name: str = None,
        debug: bool = False,
    ):
        """
        Initialize the Nowak Disaggregator.

        Supports both single site (Series) and multi-site (DataFrame)
        disaggregation between any supported timescale pair.

        Parameters
        ----------
        input_timestep : {'annual', 'monthly', 'weekly'}, default='monthly'
            Timestep of the synthetic coarse flows to disaggregate.
        output_timestep : {'monthly', 'weekly', 'daily'}, default='daily'
            Timestep of the disaggregated output; must be finer than
            ``input_timestep``. Observed data passed to ``fit`` or
            ``preprocessing`` must be at this timestep.
        n_neighbors : int, default=5
            Number of K-nearest neighbors to consider for disaggregation.
        max_knn_pool_shift_timesteps : int, optional
            Maximum number of output timesteps that historic period windows
            are shifted (plus and minus) when building the KNN candidate
            pools. Larger values enlarge the pools but rotate the sampled
            profiles relative to the calendar. If None, a per-pair default
            is used: 7 for monthly to daily, 2 for weekly to daily, 1 for
            monthly to weekly, 0 for annual to monthly, 2 for annual to
            weekly, and 7 for annual to daily.
        boundary_blend_timesteps : int, default=2
            Number of output timesteps on each side of coarse-period
            boundaries to smooth with a centered rolling mean, reducing
            artificial discontinuities from independent per-period sampling.
            Coarse-period totals are preserved by rescaling. Set to 0 or
            None to disable.
        name : str, optional
            Name for this disaggregator instance.
        debug : bool, default=False
            Enable debug logging.

        Raises
        ------
        ValueError
            If either timestep is unrecognized or ``output_timestep`` is not
            finer than ``input_timestep``.
        """
        if input_timestep not in _INPUT_TIMESTEPS:
            raise ValueError(
                f"input_timestep must be one of {_INPUT_TIMESTEPS}, "
                f"got {input_timestep!r}"
            )
        if output_timestep not in _OUTPUT_TIMESTEPS:
            raise ValueError(
                f"output_timestep must be one of {_OUTPUT_TIMESTEPS}, "
                f"got {output_timestep!r}"
            )
        if _TIMESTEP_RANK[output_timestep] <= _TIMESTEP_RANK[input_timestep]:
            raise ValueError(
                f"output_timestep ({output_timestep!r}) must be finer than "
                f"input_timestep ({input_timestep!r})"
            )

        # Initialize base class
        super().__init__(name=name, debug=debug)

        # Resolve the timescale pair configuration
        self.input_timestep = input_timestep
        self.output_timestep = output_timestep
        self._scale = (input_timestep, output_timestep)
        self._config = _SCALE_CONFIGS[self._scale]

        # Store algorithm-specific parameters
        self.n_neighbors = n_neighbors
        if max_knn_pool_shift_timesteps is None:
            max_knn_pool_shift_timesteps = self._config.default_pool_shift
        self.max_knn_pool_shift_timesteps = max_knn_pool_shift_timesteps
        self.boundary_blend_timesteps = (
            boundary_blend_timesteps if boundary_blend_timesteps else 0
        )

        # Update init_params
        self.init_params.algorithm_params = {
            "method": "Nowak KNN Disaggregation",
            "input_timestep": input_timestep,
            "output_timestep": output_timestep,
            "n_neighbors": n_neighbors,
            "max_knn_pool_shift_timesteps": self.max_knn_pool_shift_timesteps,
            "boundary_blend_timesteps": self.boundary_blend_timesteps,
        }

        # dict containing trained KNN models for each period label
        self.knn_models = {}

    @property
    def input_frequency(self) -> str:
        """Pandas frequency string of the expected input ensemble."""
        return self._config.input_freq

    @property
    def output_frequency(self) -> str:
        """Pandas frequency string of the disaggregated output."""
        return self._config.output_freq

    # ------------------------------------------------------------------
    # Timescale helpers
    # ------------------------------------------------------------------

    def _shift_offset(self, k: int) -> pd.DateOffset:
        """
        Return a DateOffset of ``k`` output timesteps.

        Parameters
        ----------
        k : int
            Number of output timesteps.

        Returns
        -------
        pd.DateOffset
            Offset in days, weeks, or months depending on the output
            timestep.
        """
        if self._config.output_freq == "D":
            return pd.DateOffset(days=k)
        if self._config.output_freq == _WEEKLY_FREQ:
            return pd.DateOffset(weeks=k)
        return pd.DateOffset(months=k)

    def _coarse_labels(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Map coarse-timestep timestamps to within-year period labels.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Timestamps of the coarse (input) data.

        Returns
        -------
        np.ndarray
            Period label of each timestamp: calendar month (1-12), ISO week
            (1-52, week 53 folded into 52), or 1 for annual input.
        """
        if self.input_timestep == "monthly":
            return index.month
        if self.input_timestep == "weekly":
            return np.minimum(index.isocalendar().week.to_numpy().astype(int), 52)
        return np.ones(len(index), dtype=int)

    def _period_window(
        self, ts: pd.Timestamp
    ) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
        """
        Return the fine-timestep window covered by one synthetic coarse period.

        Parameters
        ----------
        ts : pd.Timestamp
            Timestamp of the coarse period (month start, W-SUN week end, or
            year start).

        Returns
        -------
        start : pd.Timestamp
            First timestamp of the window.
        end : pd.Timestamp
            Last timestamp of the window.
        expected_steps : int
            Number of output timesteps in the window.
        """
        if self._scale == ("monthly", "daily"):
            return (
                ts,
                ts + pd.offsets.MonthEnd(0),
                self._get_days_in_month(ts.year, ts.month),
            )
        if self._scale == ("weekly", "daily"):
            return ts - pd.Timedelta(days=6), ts, 7
        if self._scale == ("monthly", "weekly"):
            end = ts + pd.offsets.MonthEnd(0)
            return ts, end, self._count_sundays(ts.year, ts.month)
        if self._scale == ("annual", "monthly"):
            return ts, ts + pd.DateOffset(months=11), 12
        if self._scale == ("annual", "weekly"):
            return (
                pd.Timestamp.fromisocalendar(ts.year, 1, 7),
                pd.Timestamp.fromisocalendar(ts.year, 52, 7),
                52,
            )
        # ("annual", "daily")
        end = pd.Timestamp(year=ts.year, month=12, day=31)
        return ts, end, 366 if calendar.isleap(ts.year) else 365

    def _pool_window(self, year: int, label: int) -> Tuple[pd.Timestamp, int]:
        """
        Return the unshifted historic window for one (year, label) pool entry.

        Parameters
        ----------
        year : int
            Historic year (ISO year for weekly input or output windows that
            follow ISO weeks).
        label : int
            Within-year period label.

        Returns
        -------
        start : pd.Timestamp
            First timestamp of the unshifted window.
        expected_steps : int
            Number of output timesteps in the window.
        """
        if self._scale == ("monthly", "daily"):
            return (
                pd.Timestamp(year=year, month=label, day=1),
                self._get_days_in_month(year, label),
            )
        if self._scale == ("weekly", "daily"):
            return pd.Timestamp.fromisocalendar(year, label, 1), 7
        if self._scale == ("monthly", "weekly"):
            return self._first_sunday(year, label), self._count_sundays(year, label)
        if self._scale == ("annual", "monthly"):
            return pd.Timestamp(year=year, month=1, day=1), 12
        if self._scale == ("annual", "weekly"):
            return pd.Timestamp.fromisocalendar(year, 1, 7), 52
        # ("annual", "daily")
        return (
            pd.Timestamp(year=year, month=1, day=1),
            366 if calendar.isleap(year) else 365,
        )

    def _build_fine_index(self, coarse_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Build the output DatetimeIndex covering all synthetic coarse periods.

        Weekly output anchors are constructed per period (via ISO calendar
        arithmetic where applicable) rather than by a single date_range, so
        skipped ISO week 53 in the input never misaligns the output.

        Parameters
        ----------
        coarse_index : pd.DatetimeIndex
            Index of the synthetic coarse data.

        Returns
        -------
        pd.DatetimeIndex
            Output timestamps at the output frequency.
        """
        if self._scale == ("monthly", "daily"):
            return pd.date_range(
                start=coarse_index[0],
                end=coarse_index[-1] + pd.offsets.MonthEnd(0),
                freq="D",
            )
        if self._scale == ("weekly", "daily"):
            windows = [
                pd.date_range(ts - pd.Timedelta(days=6), ts, freq="D").values
                for ts in coarse_index
            ]
            return pd.DatetimeIndex(np.concatenate(windows))
        if self._scale == ("monthly", "weekly"):
            return pd.date_range(
                start=coarse_index[0],
                end=coarse_index[-1] + pd.offsets.MonthEnd(0),
                freq=_WEEKLY_FREQ,
            )
        if self._scale == ("annual", "monthly"):
            return pd.date_range(
                start=coarse_index[0],
                end=coarse_index[-1] + pd.DateOffset(months=11),
                freq="MS",
            )
        if self._scale == ("annual", "weekly"):
            dates = [
                pd.Timestamp.fromisocalendar(int(year), week, 7)
                for year in coarse_index.year
                for week in range(1, 53)
            ]
            return pd.DatetimeIndex(dates)
        # ("annual", "daily")
        return pd.date_range(
            start=coarse_index[0],
            end=pd.Timestamp(year=coarse_index[-1].year, month=12, day=31),
            freq="D",
        )

    def _needs_length_fix(self, label: int, expected_steps: int) -> bool:
        """
        Check whether a sampled profile may be shorter than the target period.

        True only for timescale pairs with variable period length, when the
        target period has the longer length: leap-year February (29 days),
        leap years (366 days), and five-Sunday months.

        Parameters
        ----------
        label : int
            Period label being disaggregated.
        expected_steps : int
            Output timesteps in the target period.

        Returns
        -------
        bool
            True if the sampled proportions may need missing-mass
            redistribution.
        """
        if self._scale == ("monthly", "daily"):
            return label == 2 and expected_steps == 29
        if self._scale == ("annual", "daily"):
            return expected_steps == 366
        if self._scale == ("monthly", "weekly"):
            return expected_steps == 5
        return False

    def _fine_to_coarse_keys(self, index: pd.DatetimeIndex):
        """
        Map fine-timestep output timestamps to coarse-period group keys.

        Used for boundary detection and total-preserving rescaling in
        ``_smooth_period_boundaries``.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Output timestamps.

        Returns
        -------
        pd.PeriodIndex or pd.Index
            One group key per timestamp; equal keys mark the same coarse
            period.
        """
        if self.input_timestep == "monthly":
            return index.to_period("M")
        if self.input_timestep == "weekly":
            return index.to_period(_WEEKLY_FREQ)
        # annual input
        if self.output_timestep == "weekly":
            # ISO year of each Sunday anchor; late-December ISO weeks can
            # have anchors in early January of the next calendar year
            return pd.Index(index.isocalendar().year.to_numpy().astype(int))
        return index.to_period("Y")

    @staticmethod
    def _first_sunday(year: int, month: int) -> pd.Timestamp:
        """
        Return the first Sunday of a calendar month.

        Parameters
        ----------
        year : int
            The year.
        month : int
            The month (1-12).

        Returns
        -------
        pd.Timestamp
            Date of the first Sunday in the month.
        """
        start = pd.Timestamp(year=year, month=month, day=1)
        return start + pd.Timedelta(days=(6 - start.dayofweek) % 7)

    @staticmethod
    def _count_sundays(year: int, month: int) -> int:
        """
        Count the Sundays in a calendar month.

        Parameters
        ----------
        year : int
            The year.
        month : int
            The month (1-12).

        Returns
        -------
        int
            Number of Sundays (4 or 5).
        """
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthEnd(0)
        return len(pd.date_range(start, end, freq=_WEEKLY_FREQ))

    @staticmethod
    def _get_days_in_month(year: int, month: int) -> int:
        """
        Get the actual number of days in a month for a specific year.

        Accounts for leap years (February has 29 days in leap years).

        Parameters
        ----------
        year : int
            The year.
        month : int
            The month (1-12).

        Returns
        -------
        int
            Number of days in the month.
        """
        return calendar.monthrange(year, month)[1]

    def _complete_years(self) -> np.ndarray:
        """
        Find historic years with complete coverage for pool construction.

        For calendar-year timescale pairs a year is complete when all 12
        calendar months are present in the observed index. For ISO-week
        pairs (weekly input, or annual to weekly) an ISO year is complete
        when its ISO weeks 1 through 52 lie inside the observed record.
        For monthly to weekly a year is complete when its first and last
        Sunday anchors lie inside the observed record, so every monthly
        pool window has all of its weeks.

        Returns
        -------
        np.ndarray
            Years usable for candidate pool construction.
        """
        index = self.Qh_index.index

        if self._scale == ("monthly", "weekly"):
            complete_years = []
            for year in index.year.unique():
                first_anchor = self._first_sunday(year, 1)
                dec_end = pd.Timestamp(year=year, month=12, day=31)
                last_anchor = dec_end - pd.Timedelta(days=(dec_end.dayofweek - 6) % 7)
                if first_anchor >= index[0] and last_anchor <= index[-1]:
                    complete_years.append(year)
                else:
                    self.logger.info(
                        f"Excluding year {year}: weekly anchors do not cover "
                        f"all months of the year"
                    )
            return np.array(complete_years)

        if self._scale in (("weekly", "daily"), ("annual", "weekly")):
            if self._scale == ("weekly", "daily"):
                first_day, last_day = 1, 7
            else:
                first_day, last_day = 7, 7
            candidate_years = np.unique(index.isocalendar().year.to_numpy())
            complete_years = []
            for year in candidate_years:
                year_start = pd.Timestamp.fromisocalendar(int(year), 1, first_day)
                year_end = pd.Timestamp.fromisocalendar(int(year), 52, last_day)
                if year_start >= index[0] and year_end <= index[-1]:
                    complete_years.append(int(year))
                else:
                    self.logger.info(
                        f"Excluding ISO year {year}: weeks 1-52 not fully "
                        f"contained in the observed record"
                    )
            return np.array(complete_years)

        # Calendar-year rule: all 12 months present
        all_years = index.year.unique()
        complete_years = []

        for year in all_years:
            year_data = self.Qh_index[index.year == year]
            months_present = year_data.index.month.unique()

            # Check if all 12 months are present
            if len(months_present) == 12:
                complete_years.append(year)
            else:
                self.logger.info(
                    f"Excluding year {year}: only {len(months_present)} months present"
                )

        return np.array(complete_years)

    # ------------------------------------------------------------------
    # Preprocessing and fitting
    # ------------------------------------------------------------------

    def preprocessing(
        self,
        Q_obs: Union[pd.Series, pd.DataFrame],
        *,
        sites: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Preprocess observed flow data at the output timestep.

        Validates input data and detects single-site vs multisite
        configuration.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed streamflow at the output (fine) timestep for the
            historic period, with a DatetimeIndex. If DataFrame, columns
            represent different sites.
        sites : list of str, optional
            Sites to use. If None, uses all columns.
        **kwargs
            Additional preprocessing parameters (currently unused).

        Raises
        ------
        ValueError
            If the observed data frequency contradicts the configured
            output timestep, or no complete years are found.
        """
        # Validate and store observed data
        Qh_fine = self._store_obs_data(Q_obs, sites)

        # Store validated data
        self.Qh_fine = Qh_fine

        # Set site_names for backward compatibility
        self.site_names = self._sites

        # Detect single-site vs multisite
        self.is_multisite = (
            isinstance(self.Qh_fine, pd.DataFrame) and self.Qh_fine.shape[1] > 1
        )

        if self.is_multisite:
            # Create index gauge as sum of all sites
            self.Qh_index = self.Qh_fine.sum(axis=1)
        else:
            # Convert to Series if single column DataFrame
            if isinstance(self.Qh_fine, pd.DataFrame):
                self.Qh_fine = self.Qh_fine.iloc[:, 0]
            self._sites = [self.Qh_fine.name if self.Qh_fine.name else "site_1"]
            self.site_names = self._sites
            self.Qh_index = self.Qh_fine

        # Cross-check the observed data frequency against the configured
        # output timestep when it can be inferred
        inferred = _normalize_freq(pd.infer_freq(self.Qh_index.index))
        if inferred is not None and inferred != self._config.output_freq:
            raise ValueError(
                f"{self.name} is configured for output_timestep="
                f"'{self.output_timestep}' ('{self._config.output_freq}') but "
                f"the observed data has frequency '{inferred}'. Observed data "
                f"must be at the output timestep."
            )

        # Get historic datetime stats and filter to complete years only
        self.historic_years = self._complete_years()
        self.n_historic_years = len(self.historic_years)

        if self.n_historic_years == 0:
            raise ValueError(
                "No complete years found in data. Nowak disaggregator requires at least one complete year."
            )

        # Update state
        self.update_state(preprocessed=True)
        self.logger.info(
            f"Preprocessing complete: {self.n_sites} sites, {self.n_historic_years} complete years, "
            f"{len(self.Qh_index)} observations at the output timestep"
        )

    def fit(
        self,
        Q_obs: Optional[Union[pd.Series, pd.DataFrame]] = None,
        *,
        sites: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Fit the Nowak Disaggregator to the data.

        Creates a dataset of candidate flow profiles for each coarse-period
        label, and trains one KNN model per label.

        If ``Q_obs`` is provided, ``preprocessing()`` is called automatically.
        If omitted, a prior call to ``preprocessing()`` is required.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            Observed data at the output timestep. If provided, runs
            preprocessing automatically.
        sites : list of str, optional
            Sites to use (only when Q_obs is provided).
        **kwargs
            Additional fitting parameters (currently unused).
        """
        # Auto-call preprocessing if Q_obs is provided
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)

        # Validate preprocessing
        self.validate_preprocessing()

        # Create the dataset of candidate flow profiles
        self.coarse_totals, self.flow_profiles = self._make_historic_profile_dataset()

        # Train KNN models for each period label
        for label in self._config.period_labels:
            self._train_knn_model(label)

        # Update state
        self.update_state(fitted=True)

        # Compute and store fitted parameters
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(
            f"Fitting complete: KNN models trained for "
            f"{len(self._config.period_labels)} period labels"
        )

    def _compute_fitted_params(self) -> FittedParams:
        """
        Extract and package fitted parameters from Nowak disaggregator.

        Returns
        -------
        FittedParams
            Dataclass containing all fitted parameters.
        """
        # Count parameters: one KNN model per period label with n_neighbors each
        n_params = len(self._config.period_labels) * self.n_neighbors

        # Get training period
        training_period = (
            str(self.Qh_index.index[0].date()),
            str(self.Qh_index.index[-1].date()),
        )

        # Package KNN model info
        fitted_models_info = {
            "knn_models": {
                label: "NearestNeighbors" for label in self._config.period_labels
            },
            "n_neighbors": self.n_neighbors,
            "max_knn_pool_shift_timesteps": self.max_knn_pool_shift_timesteps,
        }

        return FittedParams(
            means_=None,
            stds_=None,
            correlations_=None,
            distributions_={"type": "nonparametric", "method": "KNN sampling"},
            fitted_models_=fitted_models_info,
            n_parameters_=n_params,
            sample_size_=len(self.Qh_index),
            n_sites_=self.n_sites,
            training_period_=training_period,
        )

    def _make_historic_profile_dataset(self):
        """
        Create dataset of candidate flow profiles for each period label.

        For each label, we will have a dataset of coarse-period flow profiles
        for each year in the historic record, and for plus/minus
        ``max_knn_pool_shift_timesteps`` output timesteps around the period.

        This will generate both:
        - dataset of total coarse-period flows (index gauge), used to find KNN indices
        - dataset of fine-timestep flow proportions for each site, used to
          disaggregate coarse flows

        Format:
        coarse_totals : dict
            values are np.array of total flows (index gauge) for each year and shift
            (length = n_historic_years * (2*max_shift + 1))
        flow_profiles : dict
            For single site: values are np.array of flow proportions for each year and shift
            (shape = (n_historic_years * (2*max_shift + 1), max_steps))
            For multisite: values are np.array of flow proportions for each site, year and shift
            (shape = (n_historic_years * (2*max_shift + 1), max_steps, n_sites))
        """

        # Create a dict to hold coarse totals and fine profiles
        coarse_totals = {}
        flow_profiles = {}

        max_shift = self.max_knn_pool_shift_timesteps
        pad = self._shift_offset(max_shift)

        # Make a copy of data with wrap-around datetime to account for
        # +/- max_shift timestep shifts
        start_date = self.Qh_index.index[0]
        end_date = self.Qh_index.index[-1]
        wrap_start_date = start_date - pad
        wrap_end_date = end_date + pad

        # Create wrapped index gauge
        Qh_index_wrap = pd.Series(
            index=pd.date_range(
                start=wrap_start_date, end=wrap_end_date, freq=self._config.output_freq
            )
        )
        Qh_index_wrap = Qh_index_wrap.astype(float)

        Qh_index_wrap.loc[wrap_start_date:start_date] = self.Qh_index.loc[
            end_date - pad : end_date
        ]
        Qh_index_wrap.loc[start_date:end_date] = self.Qh_index.loc[start_date:end_date]
        Qh_index_wrap.loc[end_date:wrap_end_date] = self.Qh_index.loc[
            start_date : start_date + pad
        ]

        # forward and backward fill the NaN values
        Qh_index_wrap = Qh_index_wrap.ffill().bfill()

        # Create wrapped data for all sites
        if self.is_multisite:
            Qh_fine_wrap = pd.DataFrame(
                index=pd.date_range(
                    start=wrap_start_date,
                    end=wrap_end_date,
                    freq=self._config.output_freq,
                ),
                columns=self.site_names,
            )
            Qh_fine_wrap = Qh_fine_wrap.astype(float)

            Qh_fine_wrap.loc[wrap_start_date:start_date] = self.Qh_fine.loc[
                end_date - pad : end_date
            ]
            Qh_fine_wrap.loc[start_date:end_date] = self.Qh_fine.loc[
                start_date:end_date
            ]
            Qh_fine_wrap.loc[end_date:wrap_end_date] = self.Qh_fine.loc[
                start_date : start_date + pad
            ]

            # forward and backward fill the NaN values
            Qh_fine_wrap = Qh_fine_wrap.ffill().bfill()
        else:
            Qh_fine_wrap = Qh_index_wrap.copy()

        # Loop through each period label
        for label in self._config.period_labels:

            # Array of cumulative flow (index gauge)
            coarse_totals[label] = np.zeros(
                shape=(self.n_historic_years * (2 * max_shift + 1),)
            )

            # Array of flow proportions
            # Use maximum possible steps to accommodate variable period lengths
            max_steps = self._config.max_steps
            if self.is_multisite:
                flow_profiles[label] = np.zeros(
                    shape=(
                        self.n_historic_years * (2 * max_shift + 1),
                        max_steps,
                        self.n_sites,
                    )
                )
            else:
                flow_profiles[label] = np.zeros(
                    shape=(
                        self.n_historic_years * (2 * max_shift + 1),
                        max_steps,
                    )
                )

            # loop through time shifts
            for shift in range(-max_shift, max_shift + 1):

                # Loop through each year
                for y, year in enumerate(self.historic_years):

                    # Get the start and end dates for the period (accounting for shift)
                    base_start, expected_steps = self._pool_window(year, label)
                    start_date = base_start + self._shift_offset(shift)
                    end_date = start_date + self._shift_offset(expected_steps - 1)

                    # Get the flow data for the period (index gauge)
                    fine_index_data = Qh_index_wrap.loc[start_date:end_date]

                    # Validate that we have a complete period of data
                    actual_steps = len(fine_index_data)
                    if actual_steps != expected_steps:
                        raise ValueError(
                            f"Incomplete period data detected for label {label} year {year} with shift {shift}: "
                            f"extracted {actual_steps} steps but expected {expected_steps} steps "
                            f"(window: {start_date.date()} to {end_date.date()}). "
                            f"Temporal disaggregation requires complete data windows. "
                            f"Please ensure your input data has no gaps."
                        )

                    # Calculate the total coarse flow (index gauge)
                    total_coarse_flow = fine_index_data.sum()

                    # index for this pool entry
                    idx = y * (2 * max_shift + 1) + (shift + max_shift)

                    # Store the total coarse flow (index gauge)
                    coarse_totals[label][idx] = total_coarse_flow

                    # Store the flow proportions for each site
                    if self.is_multisite:
                        fine_site_data = Qh_fine_wrap.loc[start_date:end_date]
                        for s, site in enumerate(self.site_names):
                            # Use skipna=True to properly handle NaN values (e.g., from replaced zeros)
                            site_values = fine_site_data[site].values
                            site_total = fine_site_data[site].sum(skipna=True)

                            if site_total > 0 and not np.isnan(site_total):
                                # Calculate proportions, handling NaN values
                                proportions = site_values / site_total
                                # Replace any NaN proportions with uniform distribution over valid steps
                                if np.any(np.isnan(proportions)):
                                    n_valid = np.sum(~np.isnan(proportions))
                                    if n_valid > 0:
                                        # Redistribute NaN proportion mass uniformly over valid steps
                                        nan_mass = (
                                            np.sum(np.isnan(proportions)) / actual_steps
                                        )
                                        proportions = np.where(
                                            np.isnan(proportions), 0.0, proportions
                                        )
                                        proportions += nan_mass / n_valid
                                    else:
                                        # All proportions are NaN - use uniform
                                        proportions = (
                                            np.ones(actual_steps) / actual_steps
                                        )
                                flow_profiles[label][
                                    idx, :actual_steps, s
                                ] = proportions
                            else:
                                # Handle zero or NaN total flow case - use uniform distribution
                                flow_profiles[label][idx, :actual_steps, s] = (
                                    1.0 / actual_steps
                                )

                            # Ensure proportions are valid
                            flow_profiles[label][idx, :actual_steps, s] = np.clip(
                                flow_profiles[label][idx, :actual_steps, s], 0, 1
                            )
                            # Renormalize to ensure they sum to 1
                            prop_sum = flow_profiles[label][idx, :actual_steps, s].sum()
                            if prop_sum > 0 and not np.isnan(prop_sum):
                                flow_profiles[label][idx, :actual_steps, s] /= prop_sum
                            else:
                                # Fallback to uniform if sum is zero or NaN
                                flow_profiles[label][idx, :actual_steps, s] = (
                                    1.0 / actual_steps
                                )
                    else:
                        if total_coarse_flow > 0 and not np.isnan(total_coarse_flow):
                            # Calculate proportions, handling NaN values
                            proportions = fine_index_data.values / total_coarse_flow
                            # Replace any NaN proportions with uniform distribution over valid steps
                            if np.any(np.isnan(proportions)):
                                n_valid = np.sum(~np.isnan(proportions))
                                if n_valid > 0:
                                    # Redistribute NaN proportion mass uniformly over valid steps
                                    nan_mass = (
                                        np.sum(np.isnan(proportions)) / actual_steps
                                    )
                                    proportions = np.where(
                                        np.isnan(proportions), 0.0, proportions
                                    )
                                    proportions += nan_mass / n_valid
                                else:
                                    # All proportions are NaN - use uniform
                                    proportions = np.ones(actual_steps) / actual_steps
                            flow_profiles[label][idx, :actual_steps] = proportions
                        else:
                            # Handle zero or NaN flow case - use uniform distribution
                            flow_profiles[label][idx, :actual_steps] = (
                                1.0 / actual_steps
                            )

                        # limit flow proportions to [0, 1]
                        flow_profiles[label][idx, :actual_steps] = np.clip(
                            flow_profiles[label][idx, :actual_steps], 0, 1
                        )
                        # Renormalize to ensure they sum to 1
                        prop_sum = flow_profiles[label][idx, :actual_steps].sum()
                        if prop_sum > 0 and not np.isnan(prop_sum):
                            flow_profiles[label][idx, :actual_steps] /= prop_sum
                        else:
                            # Fallback to uniform if sum is zero or NaN
                            flow_profiles[label][idx, :actual_steps] = (
                                1.0 / actual_steps
                            )

        self.coarse_totals = coarse_totals
        self.flow_profiles = flow_profiles
        return coarse_totals, flow_profiles

    def _train_knn_model(self, label: int, n_neighbors: Optional[int] = None):
        """
        Train a KNN model for the given period label.

        KNN is based on the index gauge (sum of all sites) flows.

        Parameters
        ----------
        label : int
            The period label to train the model for.
        n_neighbors : int, optional
            The number of neighbors to use for the model.

        Returns
        -------
        knn : NearestNeighbors
            The trained KNN model.
        """

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Check if the model is already trained
        if label in self.knn_models:
            return self.knn_models[label]
        else:
            # Get the historic flows (index gauge) which share the label
            historic_flows = self.coarse_totals[label]

            # historic_flows is a 1D array of total flows for each year and shift
            # reshape to 2D array for KNN
            historic_flows = historic_flows.reshape(-1, 1)

            # Create the KNN model
            knn = NearestNeighbors(n_neighbors=n_neighbors)

            # Fit the model to the historic flows
            knn.fit(historic_flows)

            # Store the model in the dict
            self.knn_models[label] = knn

            return knn

    def find_knn_indices(
        self,
        Qs_coarse_array: np.ndarray,
        label: int,
        n_neighbors: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given coarse-period flow values, find the K nearest neighbors
        from the historic dataset.

        Parameters
        ----------
        Qs_coarse_array : np.ndarray
            The coarse-period flow values to disaggregate.
        label : int
            The period label which is being disaggregated.
        n_neighbors : int, optional
            The number of neighbors to find.

        Returns
        -------
        distances : np.ndarray
            The distances to the K nearest neighbors.
        indices : np.ndarray
            The indices of the K nearest neighbors in the historic dataset.
        """

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Qs_coarse_array is a 1D array of total flows for each period in the synthetic dataset
        # reshape to 2D array for KNN
        Qs_coarse_array = Qs_coarse_array.reshape(-1, 1)

        # get the KNN model for the label
        knn = self._train_knn_model(label, n_neighbors)

        # get the indices of the K nearest neighbors
        distances, indices = knn.kneighbors(Qs_coarse_array)

        return distances, indices

    def sample_knn_flows(
        self,
        Qs_coarse_array: np.ndarray,
        label: int,
        n_neighbors: Optional[int] = None,
        sample_method: str = "distance_weighted",
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Given coarse-period flow values, sample K nearest neighbors
        from the historic dataset.

        Parameters
        ----------
        Qs_coarse_array : np.ndarray
            The coarse-period flow values to disaggregate.
        label : int
            The period label which is being disaggregated.
        n_neighbors : int, optional
            The number of neighbors to sample.
        sample_method : str, default='distance_weighted'
            The sampling method to use: 'distance_weighted' (inverse
            distance) or 'lall_and_sharma_1996' (rank-based kernel from
            Lall and Sharma, 1996).
        rng : numpy.random.Generator, optional
            Random generator used for neighbor sampling.

        Returns
        -------
        sampled_indices : np.ndarray
            The sampled indices from the historic dataset.
        """

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if rng is None:
            rng = np.random.default_rng()

        # get the K nearest neighbors
        distances, indices = self.find_knn_indices(Qs_coarse_array, label, n_neighbors)

        # sample a single index
        if sample_method == "distance_weighted":
            sampled_indices = []
            for i in range(indices.shape[0]):
                # sample based on distance
                weights = 1 / (
                    distances[i, :] + 1e-10
                )  # Add small epsilon to avoid division by zero
                weights = weights / weights.sum()
                sampled_indices.append(rng.choice(indices[i, :].flatten(), p=weights))

        elif sample_method == "lall_and_sharma_1996":
            weights = []
            sampled_indices = []
            denom = np.array([1 / i for i in range(1, n_neighbors + 1)]).sum()
            for i in range(1, n_neighbors + 1):
                w = 1 / i / denom
                weights.append(w)
            weights = np.array(weights)

            for i in range(indices.shape[0]):
                sampled_indices.append(rng.choice(indices[i, :].flatten(), p=weights))
        else:
            raise ValueError(
                "Invalid sample method. Must be 'distance_weighted' or 'lall_and_sharma_1996'."
            )

        return np.array(sampled_indices)

    # ------------------------------------------------------------------
    # Disaggregation
    # ------------------------------------------------------------------

    def validate_input_ensemble(self, ensemble: Ensemble) -> None:
        """
        Validate that input ensemble is compatible with disaggregator.

        Checks temporal frequency (with alias normalization, e.g. any weekly
        anchor matches 'W-SUN') and site consistency.

        Parameters
        ----------
        ensemble : Ensemble
            Input ensemble to validate.

        Raises
        ------
        ValueError
            If ensemble is incompatible with disaggregator.
        """
        if not isinstance(ensemble, Ensemble):
            raise TypeError(f"Input must be an Ensemble object, got {type(ensemble)}")

        if _normalize_freq(ensemble.frequency) != self.input_frequency:
            raise ValueError(
                f"{self.name} expects input frequency '{self.input_frequency}', "
                f"but got '{ensemble.frequency}'"
            )

        # Check site consistency (if disaggregator has been fitted)
        if self.is_fitted and hasattr(self, "_sites"):
            ensemble_sites = ensemble.sites
            if set(ensemble_sites) != set(self._sites):
                missing_in_ensemble = set(self._sites) - set(ensemble_sites)
                extra_in_ensemble = set(ensemble_sites) - set(self._sites)

                error_msg = (
                    f"Site mismatch between fitted disaggregator and input ensemble."
                )
                if missing_in_ensemble:
                    error_msg += f"\n  Missing in ensemble: {missing_in_ensemble}"
                if extra_in_ensemble:
                    error_msg += f"\n  Extra in ensemble: {extra_in_ensemble}"

                raise ValueError(error_msg)

    def disaggregate(
        self,
        ensemble: Ensemble,
        n_neighbors: Optional[int] = None,
        sample_method: str = "distance_weighted",
        seed=None,
        **kwargs,
    ) -> Ensemble:
        """
        Disaggregate a coarse-timestep ensemble to the output timestep.

        Each realization is driven by its own independent RNG stream keyed to its
        GLOBAL realization index. The global index of a realization is taken to
        be its integer key in ``ensemble.data_by_realization`` (the convention
        used throughout SynHydro: ``KirschGenerator.generate`` keys its output by
        global index). Realization ``k`` uses the ``'disaggregation'`` sub-stream
        of the child seed for index ``k`` (see ``synhydro.core.seeding``); this
        is the counterpart of the ``'generation'`` sub-stream consumed by
        ``KirschGenerator.generate``, so when both stages receive the same master
        ``seed`` the generate-then-disaggregate handoff for realization ``k`` is
        reproducible end to end and independent of how the realization range is
        partitioned across calls or MPI ranks.

        Note
        ----
        Reproducibility is keyed to the ensemble's realization keys. If an
        ensemble is re-keyed or renumbered before disaggregation (e.g. by
        filtering and reindexing realizations), the output for a given
        physical trace changes accordingly. Preserve the original global indices
        as the realization keys to regenerate identical traces.

        Parameters
        ----------
        ensemble : Ensemble
            Streamflow ensemble at the input (coarse) timestep. Integer
            realization keys are interpreted as global realization indices.
        n_neighbors : int, optional
            Number of neighbors to use for disaggregation.
            If None, uses the value from initialization.
        sample_method : str, default='distance_weighted'
            Method to use for sampling the K nearest neighbors.
        seed : int or numpy.random.SeedSequence, optional
            Master seed. For realization with global index ``k``, the
            sampling stream is the ``'disaggregation'`` sub-stream of
            ``SeedSequence(seed).spawn(N)[k]``. A scalar seed is deterministic;
            None draws fresh OS entropy and is non-reproducible. The legacy
            global ``numpy.random`` state is never used.
        **kwargs
            Additional disaggregation parameters.

        Returns
        -------
        Ensemble
            Disaggregated streamflow ensemble at the output timestep.
        """
        # Validate fit
        self.validate_fit()

        # Validate input ensemble
        self.validate_input_ensemble(ensemble)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        master = as_seed_sequence(seed)

        # Disaggregate each realization with its own global-index-keyed stream
        fine_realization_dict = {}

        for realization_id, coarse_df in ensemble.data_by_realization.items():
            rng = realization_rng(master, realization_id, "disaggregation")

            # Disaggregate this realization
            fine_df = self._disaggregate_single_realization(
                coarse_df,
                n_neighbors=n_neighbors,
                sample_method=sample_method,
                rng=rng,
            )
            fine_realization_dict[realization_id] = fine_df

        # Create metadata for output ensemble
        from synhydro.core.ensemble import EnsembleMetadata

        first_fine = next(iter(fine_realization_dict.values()))
        metadata = EnsembleMetadata(
            generator_class=ensemble.metadata.generator_class,
            generator_params=ensemble.metadata.generator_params,
            n_realizations=len(fine_realization_dict),
            n_sites=len(self._sites),
            time_resolution=self.output_frequency,
            time_period=(
                str(first_fine.index[0].date()),
                str(first_fine.index[-1].date()),
            ),
        )

        # Create and return output ensemble
        fine_ensemble = Ensemble(fine_realization_dict, metadata=metadata)

        self.logger.info(
            f"Disaggregated {len(fine_realization_dict)} realizations from "
            f"{self.input_timestep} to {self.output_timestep}"
        )

        return fine_ensemble

    def _smooth_period_boundaries(self, Qs_fine, blend_steps: int):
        """
        Smooth output flows around coarse-period boundaries while preserving
        coarse-period totals.

        Applies a centered rolling mean over a small window around each period
        boundary to reduce artificial discontinuities from independent
        per-period KNN sampling. After smoothing, rescales each period's
        values so the period total matches the original (pre-smoothing) total.

        Parameters
        ----------
        Qs_fine : pd.DataFrame or pd.Series
            Output-timestep streamflow timeseries with DatetimeIndex.
        blend_steps : int
            Number of output timesteps on each side of each period boundary
            to smooth.

        Returns
        -------
        pd.DataFrame or pd.Series
            Smoothed flows with preserved coarse-period totals.
        """
        if blend_steps <= 0:
            return Qs_fine

        is_series = isinstance(Qs_fine, pd.Series)
        if is_series:
            df = Qs_fine.to_frame(name="_site")
        else:
            df = Qs_fine.copy()

        # Coarse-period group keys for boundary detection and rescaling
        period_keys = self._fine_to_coarse_keys(df.index)

        # Store original coarse-period totals per site
        original_period_totals = df.groupby(period_keys).sum()

        # Identify period-boundary indices (where the coarse period changes)
        boundary_mask = np.zeros(len(df), dtype=bool)
        for i in range(1, len(df)):
            if period_keys[i] != period_keys[i - 1]:
                # Mark steps within blend_steps of this boundary
                lo = max(0, i - blend_steps)
                hi = min(len(df), i + blend_steps)
                boundary_mask[lo:hi] = True

        # Apply smoothing column-by-column
        window = 2 * blend_steps + 1
        for col in df.columns:
            vals = df[col].values.astype(float)
            smoothed = (
                pd.Series(vals)
                .rolling(window=window, center=True, min_periods=1)
                .mean()
                .values
            )
            # Only replace values near boundaries
            vals[boundary_mask] = smoothed[boundary_mask]
            # Clip to non-negative (flows can't be negative)
            np.clip(vals, 0, None, out=vals)
            df[col] = vals

        # Rescale each period to preserve original coarse totals
        for period in original_period_totals.index:
            period_mask = period_keys == period
            for col in df.columns:
                current_sum = df.loc[period_mask, col].sum()
                target_sum = original_period_totals.loc[period, col]
                if current_sum > 0 and target_sum > 0:
                    df.loc[period_mask, col] *= target_sum / current_sum
                elif current_sum > 0 and target_sum == 0:
                    df.loc[period_mask, col] = 0.0

        if is_series:
            return df.iloc[:, 0]
        return df

    def _disaggregate_single_realization(
        self,
        Qs_coarse,
        n_neighbors=None,
        sample_method="distance_weighted",
        *,
        rng=None,
    ):
        """
        Disaggregate a single realization to the output timestep (internal method).

        Parameters
        ----------
        Qs_coarse : pd.Series or pd.DataFrame
            Coarse-timestep streamflow data for a single realization.
        n_neighbors : int, optional
            Number of neighbors to use.
        sample_method : str
            Sampling method for KNN.
        rng : numpy.random.Generator, optional
            Random generator used for neighbor sampling.

        Returns
        -------
        pd.DataFrame
            Output-timestep streamflow data for this realization.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Check if multisite consistency
        if self.is_multisite:
            if not isinstance(Qs_coarse, pd.DataFrame):
                raise ValueError(
                    "For multisite disaggregation, Qs_coarse must be a DataFrame."
                )
            if not all(col in self._sites for col in Qs_coarse.columns):
                raise ValueError(
                    "Qs_coarse columns must match the historic data columns."
                )
            # Create index gauge for synthetic data
            Qs_index = Qs_coarse.sum(axis=1)
        else:
            if isinstance(Qs_coarse, pd.DataFrame):
                if Qs_coarse.shape[1] != 1:
                    raise ValueError(
                        "For single site disaggregation, Qs_coarse must be a Series or single-column DataFrame."
                    )
                Qs_coarse = Qs_coarse.iloc[:, 0]
            Qs_index = Qs_coarse

        # Setup output
        fine_index = self._build_fine_index(Qs_coarse.index)

        if self.is_multisite:
            Qs_fine = pd.DataFrame(index=fine_index, columns=self._sites)
            Qs_fine = Qs_fine.astype(float)
        else:
            Qs_fine = pd.Series(index=fine_index)
            Qs_fine = Qs_fine.astype(float)

        Qs_fine[:] = np.nan

        # Within-year period labels of the synthetic coarse timestamps
        coarse_labels = self._coarse_labels(Qs_index.index)

        # loop through period labels
        for label in self._config.period_labels:

            period_mask = coarse_labels == label

            if not period_mask.any():
                continue

            # Get the coarse flow for the label (index gauge)
            Qs_index_array = Qs_index[period_mask].values

            # Get the K nearest neighbors
            sampled_indices = self.sample_knn_flows(
                Qs_index_array, label, n_neighbors, sample_method, rng=rng
            )

            # For each period, disaggregate the coarse flow using the sampled proportions
            period_dates = Qs_index.index[period_mask]

            for y, period_date in enumerate(period_dates):
                # Get the start and end dates and expected steps for the period
                start_date, end_date, expected_steps = self._period_window(period_date)

                # Get the flow proportions for the sampled period
                # The profiles are stored padded to max_steps, but we only want
                # the valid steps for this specific period
                sampled_idx = sampled_indices[y]

                if self.is_multisite:
                    proportions_for_period = self.flow_profiles[label][
                        sampled_idx, :expected_steps, :
                    ]
                else:
                    proportions_for_period = self.flow_profiles[label][
                        sampled_idx, :expected_steps
                    ]

                # Renormalize proportions truncated to a shorter target period
                # (e.g. a five-Sunday profile applied to a four-Sunday month).
                # Disabled for monthly-to-daily to preserve established
                # production behavior; see _ScaleConfig.renormalize_truncated.
                if self._config.renormalize_truncated:
                    if self.is_multisite:
                        col_sums = proportions_for_period.sum(axis=0)
                        col_sums = np.where(col_sums > 0, col_sums, 1.0)
                        proportions_for_period = proportions_for_period / col_sums
                    else:
                        prop_sum = proportions_for_period.sum()
                        if prop_sum > 0:
                            proportions_for_period = proportions_for_period / prop_sum

                # CRITICAL FIX: Handle period-length mismatch
                # If KNN selected a shorter candidate period (e.g. non-leap
                # February for a leap-year target, or a four-Sunday month for
                # a five-Sunday target), the trailing proportion will be zero.
                # We need to renormalize or fill missing steps. This can occur
                # because the flow_profiles array is pre-allocated with
                # max_steps, but only the actual steps for each historical
                # period are filled (steps beyond are left as zeros).
                if self.is_multisite:
                    # Check each site for zero proportions
                    for s in range(proportions_for_period.shape[1]):
                        site_props = proportions_for_period[:, s]
                        prop_sum = site_props.sum()
                        n_zeros = (site_props == 0.0).sum()

                        # If we have ANY zeros in a longer-than-candidate period,
                        # we likely sampled a shorter candidate. In this case, the
                        # proportions sum to 1.0 over the shorter length, but the
                        # trailing step is zero. We need to renormalize by scaling
                        # all proportions so they sum to 1.0 over all steps.
                        if n_zeros > 0 and self._needs_length_fix(
                            label, expected_steps
                        ):
                            # Simple approach: redistribute uniformly across ALL steps (including the zero step)
                            # This preserves the relative pattern while ensuring the trailing step gets a reasonable value
                            non_zero_mask = site_props > 0
                            n_non_zero = non_zero_mask.sum()
                            if (
                                n_non_zero == expected_steps - 1
                            ):  # Exactly one zero (trailing step)
                                # Scale down all existing proportions and give the trailing step the average
                                avg_prop = 1.0 / expected_steps
                                site_props[non_zero_mask] *= (
                                    1.0 - avg_prop
                                ) / prop_sum  # Scale to leave room for the trailing step
                                site_props[~non_zero_mask] = (
                                    avg_prop  # Assign average to the trailing step
                                )
                                proportions_for_period[:, s] = site_props
                            elif n_non_zero > 0:
                                # Multiple zeros - redistribute mass
                                missing_mass = 1.0 - prop_sum
                                site_props[non_zero_mask] += missing_mass / n_non_zero
                                proportions_for_period[:, s] = site_props
                        elif prop_sum == 0:
                            # All zeros - use uniform distribution (should rarely happen)
                            proportions_for_period[:, s] = 1.0 / expected_steps
                else:
                    prop_sum = proportions_for_period.sum()
                    n_zeros = (proportions_for_period == 0.0).sum()

                    # Same fix for single-site case
                    if n_zeros > 0 and self._needs_length_fix(label, expected_steps):
                        non_zero_mask = proportions_for_period > 0
                        n_non_zero = non_zero_mask.sum()
                        if n_non_zero == expected_steps - 1:
                            avg_prop = 1.0 / expected_steps
                            proportions_for_period[non_zero_mask] *= (
                                1.0 - avg_prop
                            ) / prop_sum
                            proportions_for_period[~non_zero_mask] = avg_prop
                        elif n_non_zero > 0:
                            missing_mass = 1.0 - prop_sum
                            proportions_for_period[non_zero_mask] += (
                                missing_mass / n_non_zero
                            )
                    elif prop_sum == 0:
                        # All zeros - use uniform distribution
                        proportions_for_period[:] = 1.0 / expected_steps

                # Disaggregate the coarse flow
                if self.is_multisite:
                    for s, site_name in enumerate(self._sites):
                        coarse_flow = Qs_coarse.loc[period_date, site_name]
                        Qs_fine.loc[start_date:end_date, site_name] = (
                            coarse_flow * proportions_for_period[:, s]
                        )
                else:
                    coarse_flow = Qs_coarse.loc[period_date]
                    Qs_fine.loc[start_date:end_date] = (
                        coarse_flow * proportions_for_period
                    )

        # Smooth period boundaries to reduce artificial discontinuities
        if self.boundary_blend_timesteps > 0:
            Qs_fine = self._smooth_period_boundaries(
                Qs_fine, self.boundary_blend_timesteps
            )

        # Ensure output is always a DataFrame for consistency with Ensemble class
        if isinstance(Qs_fine, pd.Series):
            Qs_fine = Qs_fine.to_frame(name=self._sites[0])

        return Qs_fine
