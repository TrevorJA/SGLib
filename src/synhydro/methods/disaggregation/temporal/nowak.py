import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from synhydro.core.base import Disaggregator, DisaggregatorParams, FittedParams
from synhydro.core.ensemble import Ensemble
from synhydro.core.seeding import as_seed_sequence, realization_rng


class NowakDisaggregator(Disaggregator):
    """
    Temporal disaggregation from monthly to daily as described in Nowak et al. (2010).

    Supports both single-site and multisite disaggregation from monthly to daily streamflows.

    For each month in synthetic data, finds the N historic monthly flow profiles
    which have similar total flow at the index gauge (sum of all sites).

    Then, randomly selects one of the N profiles and uses the daily flow proportions
    from that month to disaggregate the synthetic monthly flow at all sites.

    When disaggregating a month, only considers historic profiles from the same
    month of interest, with +/- max_month_shift days around each month.

    References
    ----------
    Nowak, K., Prairie, J., Rajagopalan, B., & Lall, U. (2010).
    A nonparametric stochastic approach for multisite disaggregation of
    annual to daily streamflow. Water Resources Research, 46(8).
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        max_month_shift: int = 7,
        blend_days: int = 2,
        name: str = None,
        debug: bool = False,
    ):
        """
        Initialize the Nowak Disaggregator.

        Supports both single site (Series) and multi-site (DataFrame) disaggregation.

        Parameters
        ----------
        n_neighbors : int, default=5
            Number of K-nearest neighbors to consider for disaggregation.
        max_month_shift : int, default=7
            Maximum number of days to shift around each month center
            when creating historic monthly flow profiles.
        blend_days : int or None, default=2
            Number of days on each side of month boundaries to smooth.
            Reduces artificial discontinuities at month transitions.
            Set to None or 0 to disable boundary smoothing.
        name : str, optional
            Name for this disaggregator instance.
        debug : bool, default=False
            Enable debug logging.
        """
        # Initialize base class
        super().__init__(name=name, debug=debug)

        # Store algorithm-specific parameters
        self.n_neighbors = n_neighbors
        self.max_month_shift = max_month_shift
        self.blend_days = blend_days if blend_days else 0

        # Update init_params
        self.init_params.algorithm_params = {
            "method": "Nowak KNN Disaggregation",
            "n_neighbors": n_neighbors,
            "max_month_shift": max_month_shift,
            "blend_days": self.blend_days,
        }

        # dict containing trained KNN models for each month
        self.knn_models = {}

        ## Utilities
        # Store default days per month (non-leap year)
        # Will be overridden with actual days during fit() based on data
        self.days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    @property
    def input_frequency(self) -> str:
        """Nowak disaggregator expects monthly input."""
        return "MS"  # Month Start

    @property
    def output_frequency(self) -> str:
        """Nowak disaggregator produces daily output."""
        return "D"  # Daily

    def preprocessing(self, Q_obs, *, sites=None, **kwargs):
        """
        Preprocess observed daily flow data.

        Validates input data and detects single-site vs multisite configuration.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Daily streamflow data for the historic period.
            Must have DatetimeIndex with daily frequency.
            If DataFrame, columns represent different sites.
        sites : list of str, optional
            Sites to use. If None, uses all columns.
        **kwargs
            Additional preprocessing parameters (currently unused).
        """
        # Validate and store observed data
        Qh_daily = self._store_obs_data(Q_obs, sites)

        # Store validated data
        self.Qh_daily = Qh_daily

        # Set site_names for backward compatibility
        self.site_names = self._sites

        # Detect single-site vs multisite
        self.is_multisite = (
            isinstance(self.Qh_daily, pd.DataFrame) and self.Qh_daily.shape[1] > 1
        )

        if self.is_multisite:
            # Create index gauge as sum of all sites
            self.Qh_index = self.Qh_daily.sum(axis=1)
        else:
            # Convert to Series if single column DataFrame
            if isinstance(self.Qh_daily, pd.DataFrame):
                self.Qh_daily = self.Qh_daily.iloc[:, 0]
            self._sites = [self.Qh_daily.name if self.Qh_daily.name else "site_1"]
            self.site_names = self._sites
            self.Qh_index = self.Qh_daily

        # Get historic datetime stats and filter to complete years only
        # A complete year must have data for all 12 months
        all_years = self.Qh_index.index.year.unique()
        complete_years = []

        for year in all_years:
            year_data = self.Qh_index[self.Qh_index.index.year == year]
            months_present = year_data.index.month.unique()

            # Check if all 12 months are present
            if len(months_present) == 12:
                complete_years.append(year)
            else:
                self.logger.info(
                    f"Excluding year {year}: only {len(months_present)} months present"
                )

        self.historic_years = np.array(complete_years)
        self.n_historic_years = len(self.historic_years)

        if self.n_historic_years == 0:
            raise ValueError(
                "No complete years found in data. Nowak disaggregator requires at least one complete year."
            )

        # Update state
        self.update_state(preprocessed=True)
        self.logger.info(
            f"Preprocessing complete: {self.n_sites} sites, {self.n_historic_years} complete years, "
            f"{len(self.Qh_index)} daily observations"
        )

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
        import calendar

        return calendar.monthrange(year, month)[1]

    def fit(self, Q_obs=None, *, sites=None, **kwargs):
        """
        Fit the Nowak Disaggregator to the data.

        Creates a dataset of candidate monthly flow profiles for each month,
        and trains KNN models for each month.

        If ``Q_obs`` is provided, ``preprocessing()`` is called automatically.
        If omitted, a prior call to ``preprocessing()`` is required.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            Observed data. If provided, runs preprocessing automatically.
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

        # Create the dataset of candidate monthly flow profiles
        self.monthly_cumulative_flows, self.daily_flow_profiles = (
            self._make_historic_monthly_profile_dataset()
        )

        # Train KNN models for each month
        for month in range(1, 13):
            self._train_knn_model(month)

        # Update state
        self.update_state(fitted=True)

        # Compute and store fitted parameters
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(f"Fitting complete: KNN models trained for 12 months")

    def _compute_fitted_params(self) -> FittedParams:
        """
        Extract and package fitted parameters from Nowak disaggregator.

        Returns
        -------
        FittedParams
            Dataclass containing all fitted parameters.
        """
        # Count parameters: 12 KNN models with n_neighbors each
        n_params = 12 * self.n_neighbors

        # Get training period
        training_period = (
            str(self.Qh_index.index[0].date()),
            str(self.Qh_index.index[-1].date()),
        )

        # Package KNN model info
        fitted_models_info = {
            "knn_models": {month: "NearestNeighbors" for month in range(1, 13)},
            "n_neighbors": self.n_neighbors,
            "max_month_shift": self.max_month_shift,
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

    def disaggregate(
        self,
        ensemble: Ensemble,
        n_neighbors=None,
        sample_method="distance_weighted",
        seed=None,
        **kwargs,
    ) -> Ensemble:
        """
        Disaggregate monthly ensemble to daily flows using the Nowak method.

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
        filtering and reindexing realizations), the daily output for a given
        physical trace changes accordingly. Preserve the original global indices
        as the realization keys to regenerate identical daily traces.

        Parameters
        ----------
        ensemble : Ensemble
            Monthly streamflow ensemble to disaggregate.
            Must have frequency 'MS' (monthly start). Integer realization keys
            are interpreted as global realization indices.
        n_neighbors : int, optional
            Number of neighbors to use for disaggregation.
            If None, uses the value from initialization.
        sample_method : str, default='distance_weighted'
            Method to use for sampling the K nearest neighbors.
        seed : int or numpy.random.SeedSequence, optional
            Master seed. For realization with global index ``k``, the daily
            sampling stream is the ``'disaggregation'`` sub-stream of
            ``SeedSequence(seed).spawn(N)[k]``. A scalar seed is deterministic;
            None draws fresh OS entropy and is non-reproducible. The legacy
            global ``numpy.random`` state is never used.
        **kwargs
            Additional disaggregation parameters.

        Returns
        -------
        Ensemble
            Disaggregated daily streamflow ensemble.
        """
        # Validate fit
        self.validate_fit()

        # Validate input ensemble
        self.validate_input_ensemble(ensemble)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        master = as_seed_sequence(seed)

        # Disaggregate each realization with its own global-index-keyed stream
        daily_realization_dict = {}

        for realization_id, monthly_df in ensemble.data_by_realization.items():
            rng = realization_rng(master, realization_id, "disaggregation")

            # Disaggregate this realization
            daily_df = self._disaggregate_single_realization(
                monthly_df,
                n_neighbors=n_neighbors,
                sample_method=sample_method,
                rng=rng,
            )
            daily_realization_dict[realization_id] = daily_df

        # Create metadata for daily ensemble
        from synhydro.core.ensemble import EnsembleMetadata

        first_daily = next(iter(daily_realization_dict.values()))
        metadata = EnsembleMetadata(
            generator_class=ensemble.metadata.generator_class,
            generator_params=ensemble.metadata.generator_params,
            n_realizations=len(daily_realization_dict),
            n_sites=len(self._sites),
            time_resolution=self.output_frequency,
            time_period=(
                str(first_daily.index[0].date()),
                str(first_daily.index[-1].date()),
            ),
        )

        # Create and return daily ensemble
        daily_ensemble = Ensemble(daily_realization_dict, metadata=metadata)

        self.logger.info(
            f"Disaggregated {len(daily_realization_dict)} realizations from monthly to daily"
        )

        return daily_ensemble

    def _smooth_month_boundaries(self, Qs_daily, blend_days: int):
        """
        Smooth daily flows around month boundaries while preserving monthly totals.

        Applies a centered rolling mean over a small window around each month
        boundary to reduce artificial discontinuities from independent monthly
        KNN sampling. After smoothing, rescales each month's daily values so
        the monthly total matches the original (pre-smoothing) total.

        Parameters
        ----------
        Qs_daily : pd.DataFrame or pd.Series
            Daily streamflow timeseries with DatetimeIndex.
        blend_days : int
            Number of days on each side of each month boundary to smooth.

        Returns
        -------
        pd.DataFrame or pd.Series
            Smoothed daily flows with preserved monthly totals.
        """
        if blend_days <= 0:
            return Qs_daily

        is_series = isinstance(Qs_daily, pd.Series)
        if is_series:
            df = Qs_daily.to_frame(name="_site")
        else:
            df = Qs_daily.copy()

        # Store original monthly totals per site
        original_monthly_totals = df.groupby(df.index.to_period("M")).sum()

        # Identify month-boundary indices (where calendar month changes)
        months = df.index.month
        boundary_mask = np.zeros(len(df), dtype=bool)
        for i in range(1, len(df)):
            if months[i] != months[i - 1]:
                # Mark days within blend_days of this boundary
                lo = max(0, i - blend_days)
                hi = min(len(df), i + blend_days)
                boundary_mask[lo:hi] = True

        # Apply smoothing column-by-column
        window = 2 * blend_days + 1
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

        # Rescale each month to preserve original monthly totals
        periods = df.index.to_period("M")
        for period in original_monthly_totals.index:
            period_mask = periods == period
            for col in df.columns:
                current_sum = df.loc[period_mask, col].sum()
                target_sum = original_monthly_totals.loc[period, col]
                if current_sum > 0 and target_sum > 0:
                    df.loc[period_mask, col] *= target_sum / current_sum
                elif current_sum > 0 and target_sum == 0:
                    df.loc[period_mask, col] = 0.0

        if is_series:
            return df.iloc[:, 0]
        return df

    def _disaggregate_single_realization(
        self,
        Qs_monthly,
        n_neighbors=None,
        sample_method="distance_weighted",
        *,
        rng=None,
    ):
        """
        Disaggregate a single realization from monthly to daily (internal method).

        This is the core disaggregation logic, refactored from disaggregate_monthly_flows.

        Parameters
        ----------
        Qs_monthly : pd.DataFrame
            Monthly streamflow data for a single realization.
        n_neighbors : int, optional
            Number of neighbors to use.
        sample_method : str
            Sampling method for KNN.

        Returns
        -------
        pd.DataFrame
            Daily streamflow data for this realization.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Check if multisite consistency
        if self.is_multisite:
            if not isinstance(Qs_monthly, pd.DataFrame):
                raise ValueError(
                    "For multisite disaggregation, Qs_monthly must be a DataFrame."
                )
            if not all(col in self._sites for col in Qs_monthly.columns):
                raise ValueError(
                    "Qs_monthly columns must match the historic data columns."
                )
            # Create index gauge for synthetic data
            Qs_monthly_index = Qs_monthly.sum(axis=1)
        else:
            if isinstance(Qs_monthly, pd.DataFrame):
                if Qs_monthly.shape[1] != 1:
                    raise ValueError(
                        "For single site disaggregation, Qs_monthly must be a Series or single-column DataFrame."
                    )
                Qs_monthly = Qs_monthly.iloc[:, 0]
            Qs_monthly_index = Qs_monthly

        # Setup output
        daily_index = pd.date_range(
            start=Qs_monthly.index[0],
            end=Qs_monthly.index[-1] + pd.offsets.MonthEnd(0),
            freq="D",
        )

        if self.is_multisite:
            Qs_daily = pd.DataFrame(index=daily_index, columns=self._sites)
            Qs_daily = Qs_daily.astype(float)
        else:
            Qs_daily = pd.Series(index=daily_index)
            Qs_daily = Qs_daily.astype(float)

        Qs_daily[:] = np.nan

        # loop through months
        for month in range(1, 13):

            monthly_mask = Qs_monthly_index.index.month == month

            if not monthly_mask.any():
                continue

            # Get the monthly flow for the month (index gauge)
            Qs_monthly_index_array = Qs_monthly_index[monthly_mask].values

            # Get the K nearest neighbors
            sampled_indices = self.sample_knn_monthly_flows(
                Qs_monthly_index_array, month, n_neighbors, sample_method, rng=rng
            )

            # For each year, disaggregate the Qs_monthly using the sampled daily flow proportions
            month_dates = Qs_monthly_index.index[monthly_mask]

            for y, month_date in enumerate(month_dates):
                # Get the start and end dates for the month
                start_date = month_date
                end_date = start_date + pd.offsets.MonthEnd(0)

                # Calculate expected days based on the actual year and month
                expected_days = self._get_days_in_month(
                    month_date.year, month_date.month
                )

                # Get the daily flow proportions for the sampled month
                # The profiles are stored with max 31 days, but we only want the valid days for this specific month
                sampled_idx = sampled_indices[y]

                if self.is_multisite:
                    daily_flow_proportions_for_month = self.daily_flow_profiles[month][
                        sampled_idx, :expected_days, :
                    ]
                else:
                    daily_flow_proportions_for_month = self.daily_flow_profiles[month][
                        sampled_idx, :expected_days
                    ]

                # CRITICAL FIX: Handle leap year mismatch
                # If KNN selected a non-leap February (28 days) but target is leap February (29 days),
                # the day 29 proportion will be zero. We need to renormalize or fill missing days.
                # This can occur because the daily_flow_profiles array is pre-allocated with 31 days,
                # but only the actual days for each historical month are filled (days beyond are left as zeros).
                if self.is_multisite:
                    # Check each site for zero proportions
                    for s in range(daily_flow_proportions_for_month.shape[1]):
                        site_props = daily_flow_proportions_for_month[:, s]
                        prop_sum = site_props.sum()
                        n_zeros = (site_props == 0.0).sum()

                        # If we have ANY zeros in a February with 29 days, we likely sampled a 28-day February
                        # In this case, the proportions sum to 1.0 over 28 days, but day 29 is zero
                        # We need to renormalize by scaling all proportions so they sum to 1.0 over all 29 days
                        if n_zeros > 0 and month == 2 and expected_days == 29:
                            # Simple approach: redistribute uniformly across ALL days (including the zero day)
                            # This preserves the relative pattern while ensuring day 29 gets a reasonable value
                            non_zero_mask = site_props > 0
                            n_non_zero = non_zero_mask.sum()
                            if (
                                n_non_zero == expected_days - 1
                            ):  # Exactly one zero (day 29)
                                # Scale down all existing proportions and give day 29 the average
                                avg_prop = 1.0 / expected_days
                                site_props[non_zero_mask] *= (
                                    1.0 - avg_prop
                                ) / prop_sum  # Scale to leave room for day 29
                                site_props[~non_zero_mask] = (
                                    avg_prop  # Assign average to day 29
                                )
                                daily_flow_proportions_for_month[:, s] = site_props
                            elif n_non_zero > 0:
                                # Multiple zeros - redistribute mass
                                missing_mass = 1.0 - prop_sum
                                site_props[non_zero_mask] += missing_mass / n_non_zero
                                daily_flow_proportions_for_month[:, s] = site_props
                        elif prop_sum == 0:
                            # All zeros - use uniform distribution (should rarely happen)
                            daily_flow_proportions_for_month[:, s] = 1.0 / expected_days
                else:
                    prop_sum = daily_flow_proportions_for_month.sum()
                    n_zeros = (daily_flow_proportions_for_month == 0.0).sum()

                    # Same fix for single-site case
                    if n_zeros > 0 and month == 2 and expected_days == 29:
                        non_zero_mask = daily_flow_proportions_for_month > 0
                        n_non_zero = non_zero_mask.sum()
                        if n_non_zero == expected_days - 1:
                            avg_prop = 1.0 / expected_days
                            daily_flow_proportions_for_month[non_zero_mask] *= (
                                1.0 - avg_prop
                            ) / prop_sum
                            daily_flow_proportions_for_month[~non_zero_mask] = avg_prop
                        elif n_non_zero > 0:
                            missing_mass = 1.0 - prop_sum
                            daily_flow_proportions_for_month[non_zero_mask] += (
                                missing_mass / n_non_zero
                            )
                    elif prop_sum == 0:
                        # All zeros - use uniform distribution
                        daily_flow_proportions_for_month[:] = 1.0 / expected_days

                # Disaggregate the monthly flow
                if self.is_multisite:
                    for s, site_name in enumerate(self._sites):
                        monthly_flow = Qs_monthly.loc[month_date, site_name]
                        Qs_daily.loc[start_date:end_date, site_name] = (
                            monthly_flow * daily_flow_proportions_for_month[:, s]
                        )
                else:
                    monthly_flow = Qs_monthly.loc[month_date]
                    Qs_daily.loc[start_date:end_date] = (
                        monthly_flow * daily_flow_proportions_for_month
                    )

        # Smooth month boundaries to reduce artificial discontinuities
        if self.blend_days > 0:
            Qs_daily = self._smooth_month_boundaries(Qs_daily, self.blend_days)

        # Ensure output is always a DataFrame for consistency with Ensemble class
        if isinstance(Qs_daily, pd.Series):
            Qs_daily = Qs_daily.to_frame(name=self._sites[0])

        return Qs_daily

    def _make_historic_monthly_profile_dataset(self):
        """
        Create dataset of candidate monthly flow profiles for each month.

        For each month, we will have a dataset of monthly flow profiles
        for each year in the historic record, and for +/- max_month_shift days around the month.

        This will generate both:
        - dataset of total monthly flows (index gauge), used to find KNN indices
        - dataset of daily flow proportions for each site, used to disaggregate monthly flows

        Format:
        monthly_cumulative_flows : dict
            values are np.array of total flows (index gauge) for each year and shift
            (length = n_historic_years * (2*max_month_shift + 1))
        daily_flow_profiles : dict
            For single site: values are np.array of daily flow proportions for each year and shift
            (shape = (n_historic_years * (2*max_month_shift + 1), n_days_in_month))
            For multisite: values are np.array of daily flow proportions for each site, year and shift
            (shape = (n_historic_years * (2*max_month_shift + 1), n_days_in_month, n_sites))
        """

        # Create a dict to hold monthly cumulative flows and daily profiles
        monthly_cumulative_flows = {}
        daily_flow_profiles = {}

        # Make a copy of data with wrap-around datetime to account +/- max_month_shift day shifts
        start_date = self.Qh_index.index[0]
        end_date = self.Qh_index.index[-1]
        wrap_start_date = start_date - pd.DateOffset(days=self.max_month_shift)
        wrap_end_date = end_date + pd.DateOffset(days=self.max_month_shift)

        # Create wrapped index gauge
        Qh_index_wrap = pd.Series(
            index=pd.date_range(start=wrap_start_date, end=wrap_end_date, freq="D")
        )
        Qh_index_wrap = Qh_index_wrap.astype(float)

        Qh_index_wrap.loc[wrap_start_date:start_date] = self.Qh_index.loc[
            end_date - pd.DateOffset(days=self.max_month_shift) : end_date
        ]
        Qh_index_wrap.loc[start_date:end_date] = self.Qh_index.loc[start_date:end_date]
        Qh_index_wrap.loc[end_date:wrap_end_date] = self.Qh_index.loc[
            start_date : start_date + pd.DateOffset(days=self.max_month_shift)
        ]

        # forward and backward fill the NaN values
        Qh_index_wrap = Qh_index_wrap.ffill().bfill()

        # Create wrapped data for all sites
        if self.is_multisite:
            Qh_daily_wrap = pd.DataFrame(
                index=pd.date_range(start=wrap_start_date, end=wrap_end_date, freq="D"),
                columns=self.site_names,
            )
            Qh_daily_wrap = Qh_daily_wrap.astype(float)

            Qh_daily_wrap.loc[wrap_start_date:start_date] = self.Qh_daily.loc[
                end_date - pd.DateOffset(days=self.max_month_shift) : end_date
            ]
            Qh_daily_wrap.loc[start_date:end_date] = self.Qh_daily.loc[
                start_date:end_date
            ]
            Qh_daily_wrap.loc[end_date:wrap_end_date] = self.Qh_daily.loc[
                start_date : start_date + pd.DateOffset(days=self.max_month_shift)
            ]

            # forward and backward fill the NaN values
            Qh_daily_wrap = Qh_daily_wrap.ffill().bfill()
        else:
            Qh_daily_wrap = Qh_index_wrap.copy()

        # Loop through each month
        for month in range(1, 13):

            # Array of cumulative flow (index gauge)
            monthly_cumulative_flows[month] = np.zeros(
                shape=(self.n_historic_years * (2 * self.max_month_shift + 1),)
            )

            # Array of daily flow proportions
            # Use maximum possible days (31) to accommodate all months including leap year February
            max_days = 31
            if self.is_multisite:
                daily_flow_profiles[month] = np.zeros(
                    shape=(
                        self.n_historic_years * (2 * self.max_month_shift + 1),
                        max_days,
                        self.n_sites,
                    )
                )
            else:
                daily_flow_profiles[month] = np.zeros(
                    shape=(
                        self.n_historic_years * (2 * self.max_month_shift + 1),
                        max_days,
                    )
                )

            # loop through time shifts
            for shift in range(-self.max_month_shift, self.max_month_shift + 1):

                # Loop through each year
                for y, year in enumerate(self.historic_years):

                    # Get the start and end dates for the 'month' (accounting for shift)
                    start_date = pd.Timestamp(
                        year=year, month=month, day=1
                    ) + pd.DateOffset(days=shift)

                    # Get actual number of days for the ORIGINAL month (not shifted month)
                    # This handles leap years - e.g., February 2024 has 29 days, February 2023 has 28 days
                    expected_days = self._get_days_in_month(year, month)
                    end_date = start_date + pd.DateOffset(days=expected_days - 1)

                    # Get the daily flow data for the month (index gauge)
                    daily_index_data = Qh_index_wrap.loc[start_date:end_date]

                    # Validate that we have a complete month of data
                    actual_days = len(daily_index_data)
                    if actual_days != expected_days:
                        raise ValueError(
                            f"Incomplete month data detected for month {month} year {year} with shift {shift}: "
                            f"extracted {actual_days} days but expected {expected_days} days "
                            f"(window: {start_date.date()} to {end_date.date()}). "
                            f"Temporal disaggregation requires complete data windows. "
                            f"Please ensure your input data has no gaps."
                        )

                    # Calculate the total monthly flow (index gauge)
                    total_monthly_flow = daily_index_data.sum()

                    # index for this month value
                    idx = y * (2 * self.max_month_shift + 1) + (
                        shift + self.max_month_shift
                    )

                    # Store the total monthly flow (index gauge)
                    monthly_cumulative_flows[month][idx] = total_monthly_flow

                    # Store the daily flow proportions for each site
                    if self.is_multisite:
                        daily_site_data = Qh_daily_wrap.loc[start_date:end_date]
                        for s, site in enumerate(self.site_names):
                            # Use skipna=True to properly handle NaN values (e.g., from replaced zeros)
                            site_values = daily_site_data[site].values
                            site_total = daily_site_data[site].sum(skipna=True)

                            if site_total > 0 and not np.isnan(site_total):
                                # Calculate proportions, handling NaN values
                                proportions = site_values / site_total
                                # Replace any NaN proportions with uniform distribution over valid days
                                if np.any(np.isnan(proportions)):
                                    n_valid = np.sum(~np.isnan(proportions))
                                    if n_valid > 0:
                                        # Redistribute NaN proportion mass uniformly over valid days
                                        nan_mass = (
                                            np.sum(np.isnan(proportions)) / actual_days
                                        )
                                        proportions = np.where(
                                            np.isnan(proportions), 0.0, proportions
                                        )
                                        proportions += nan_mass / n_valid
                                    else:
                                        # All proportions are NaN - use uniform
                                        proportions = np.ones(actual_days) / actual_days
                                daily_flow_profiles[month][
                                    idx, :actual_days, s
                                ] = proportions
                            else:
                                # Handle zero or NaN total flow case - use uniform distribution
                                daily_flow_profiles[month][idx, :actual_days, s] = (
                                    1.0 / actual_days
                                )

                            # Ensure proportions are valid
                            daily_flow_profiles[month][idx, :actual_days, s] = np.clip(
                                daily_flow_profiles[month][idx, :actual_days, s], 0, 1
                            )
                            # Renormalize to ensure they sum to 1
                            prop_sum = daily_flow_profiles[month][
                                idx, :actual_days, s
                            ].sum()
                            if prop_sum > 0 and not np.isnan(prop_sum):
                                daily_flow_profiles[month][
                                    idx, :actual_days, s
                                ] /= prop_sum
                            else:
                                # Fallback to uniform if sum is zero or NaN
                                daily_flow_profiles[month][idx, :actual_days, s] = (
                                    1.0 / actual_days
                                )
                    else:
                        if total_monthly_flow > 0 and not np.isnan(total_monthly_flow):
                            # Calculate proportions, handling NaN values
                            proportions = daily_index_data.values / total_monthly_flow
                            # Replace any NaN proportions with uniform distribution over valid days
                            if np.any(np.isnan(proportions)):
                                n_valid = np.sum(~np.isnan(proportions))
                                if n_valid > 0:
                                    # Redistribute NaN proportion mass uniformly over valid days
                                    nan_mass = (
                                        np.sum(np.isnan(proportions)) / actual_days
                                    )
                                    proportions = np.where(
                                        np.isnan(proportions), 0.0, proportions
                                    )
                                    proportions += nan_mass / n_valid
                                else:
                                    # All proportions are NaN - use uniform
                                    proportions = np.ones(actual_days) / actual_days
                            daily_flow_profiles[month][idx, :actual_days] = proportions
                        else:
                            # Handle zero or NaN flow case - use uniform distribution
                            daily_flow_profiles[month][idx, :actual_days] = (
                                1.0 / actual_days
                            )

                        # limit daily flow proportions to [0, 1]
                        daily_flow_profiles[month][idx, :actual_days] = np.clip(
                            daily_flow_profiles[month][idx, :actual_days], 0, 1
                        )
                        # Renormalize to ensure they sum to 1
                        prop_sum = daily_flow_profiles[month][idx, :actual_days].sum()
                        if prop_sum > 0 and not np.isnan(prop_sum):
                            daily_flow_profiles[month][idx, :actual_days] /= prop_sum
                        else:
                            # Fallback to uniform if sum is zero or NaN
                            daily_flow_profiles[month][idx, :actual_days] = (
                                1.0 / actual_days
                            )

        self.monthly_cumulative_flows = monthly_cumulative_flows
        self.daily_flow_profiles = daily_flow_profiles
        return monthly_cumulative_flows, daily_flow_profiles

    def _train_knn_model(self, month, n_neighbors=None):
        """
        Train a KNN model for the given month.

        KNN is based on the index gauge (sum of all sites) flows.

        Parameters
        ----------
        month : int
            The month to train the model for (1-12).
        n_neighbors : int
            The number of neighbors to use for the model.

        Returns
        -------
        knn : NearestNeighbors
            The trained KNN model.
        """

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Check if the model is already trained
        if month in self.knn_models:
            return self.knn_models[month]
        else:
            # Get the historic flows (index gauge) which are in the same month
            historic_flows = self.monthly_cumulative_flows[month]

            # historic_flows is a 1D array of total flows for each year and shift
            # reshape to 2D array for KNN
            historic_flows = historic_flows.reshape(-1, 1)

            # Create the KNN model
            knn = NearestNeighbors(n_neighbors=n_neighbors)

            # Fit the model to the historic flows
            knn.fit(historic_flows)

            # Store the model in the dict
            self.knn_models[month] = knn

            return knn

    def find_knn_indices(self, Qs_monthly_array, month, n_neighbors=None):
        """
        Given cumulative monthly flow values, find the K nearest neighbors
        from the historic dataset.

        Parameters
        ----------
        Qs_monthly_array : np.array
            The cumulative monthly flow values for the month to disaggregate.
        month : int
            The calendar month which is being disaggregated (1-12).
        n_neighbors : int
            The number of neighbors to find.

        Returns
        -------
        distances : np.array
            The distances to the K nearest neighbors.
        indices : np.array
            The indices of the K nearest neighbors in the historic dataset.
        """

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Qs_monthly_array is a 1D array of total flows for each month in the synthetic dataset
        # reshape to 2D array for KNN
        Qs_monthly_array = Qs_monthly_array.reshape(-1, 1)

        # get the KNN model for the month
        knn = self._train_knn_model(month, n_neighbors)

        # get the indices of the K nearest neighbors
        distances, indices = knn.kneighbors(Qs_monthly_array)

        return distances, indices

    def sample_knn_monthly_flows(
        self,
        Qs_monthly_array,
        month,
        n_neighbors=None,
        sample_method="distance_weighted",
        *,
        rng=None,
    ):
        """
        Given cumulative monthly flow values, sample K nearest neighbors
        from the historic dataset.

        Parameters
        ----------
        Qs_monthly_array : np.array
            The cumulative monthly flow values for the month to disaggregate.
        month : int
            The calendar month which is being disaggregated (1-12).
        n_neighbors : int
            The number of neighbors to sample.
        sample_method : str
            The sampling method to use.

        Returns
        -------
        sampled_indices : np.array
            The sampled indices from the historic dataset.
        """

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if rng is None:
            rng = np.random.default_rng()

        # get the K nearest neighbors
        distances, indices = self.find_knn_indices(Qs_monthly_array, month, n_neighbors)

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

    def disaggregate_monthly_flows(
        self, Qs_monthly, n_neighbors=None, sample_method="distance_weighted"
    ):
        """
        Disaggregate monthly to daily flows using the Nowak method.

        Parameters
        ----------
        Qs_monthly : pd.Series or pd.DataFrame
            Monthly streamflow data for the synthetic period.
            The index should be a datetime index.
            For multisite, should be DataFrame with same columns as historic data.
        n_neighbors : int
            The number of neighbors to use for disaggregation.
        sample_method : str
            The method to use for sampling the K nearest neighbors.

        Returns
        -------
        Qs_daily : pd.Series or pd.DataFrame
            Daily streamflow data for the synthetic period.
            The index will be a datetime index.
            For multisite, returns DataFrame with same columns as input.
        """

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Check if multisite consistency
        if self.is_multisite:
            if not isinstance(Qs_monthly, pd.DataFrame):
                raise ValueError(
                    "For multisite disaggregation, Qs_monthly must be a DataFrame."
                )
            if not all(col in self.site_names for col in Qs_monthly.columns):
                raise ValueError(
                    "Qs_monthly columns must match the historic data columns."
                )
            # Create index gauge for synthetic data
            Qs_monthly_index = Qs_monthly.sum(axis=1)
        else:
            if isinstance(Qs_monthly, pd.DataFrame):
                if Qs_monthly.shape[1] != 1:
                    raise ValueError(
                        "For single site disaggregation, Qs_monthly must be a Series or single-column DataFrame."
                    )
                Qs_monthly = Qs_monthly.iloc[:, 0]
            Qs_monthly_index = Qs_monthly

        # Setup output
        daily_index = pd.date_range(
            start=Qs_monthly.index[0],
            end=Qs_monthly.index[-1] + pd.offsets.MonthEnd(0),
            freq="D",
        )

        if self.is_multisite:
            Qs_daily = pd.DataFrame(index=daily_index, columns=self.site_names)
            Qs_daily = Qs_daily.astype(float)
        else:
            Qs_daily = pd.Series(index=daily_index)
            Qs_daily = Qs_daily.astype(float)

        Qs_daily[:] = np.nan

        # loop through months
        for month in range(1, 13):

            monthly_mask = Qs_monthly_index.index.month == month

            if not monthly_mask.any():
                continue

            # Get the monthly flow for the month (index gauge)
            Qs_monthly_index_array = Qs_monthly_index[monthly_mask].values

            # Get the K nearest neighbors
            sampled_indices = self.sample_knn_monthly_flows(
                Qs_monthly_index_array, month, n_neighbors, sample_method
            )

            # Get the daily flow proportions for the sampled indices
            if self.is_multisite:
                daily_flow_proportions = self.daily_flow_profiles[month][
                    sampled_indices, :, :
                ]  # shape: (n_years, n_days, n_sites)
            else:
                daily_flow_proportions = self.daily_flow_profiles[month][
                    sampled_indices, :
                ]  # shape: (n_years, n_days)

            # For each year, disaggregate the Qs_monthly using the sampled daily flow proportions
            month_dates = Qs_monthly_index.index[monthly_mask]

            for y, month_date in enumerate(month_dates):
                # Get the start and end dates for the month
                start_date = month_date
                end_date = start_date + pd.offsets.MonthEnd(0)

                # Get actual number of days in this month for this year
                expected_days = self._get_days_in_month(
                    start_date.year, start_date.month
                )

                # Extract only the actual days needed from the proportions array
                # (arrays are sized to 31 days max, but we only use the actual days)
                # After truncation, renormalize so proportions sum to 1.0
                # (ensures monthly total is preserved even when neighbor had
                # a different day count, e.g. leap vs non-leap February)
                if self.is_multisite:
                    daily_flow_proportions_for_month = daily_flow_proportions[
                        y, :expected_days, :
                    ]  # shape: (expected_days, n_sites)
                    col_sums = daily_flow_proportions_for_month.sum(axis=0)
                    col_sums = np.where(col_sums > 0, col_sums, 1.0)
                    daily_flow_proportions_for_month = (
                        daily_flow_proportions_for_month / col_sums
                    )
                else:
                    daily_flow_proportions_for_month = daily_flow_proportions[
                        y, :expected_days
                    ]  # shape: (expected_days,)
                    prop_sum = daily_flow_proportions_for_month.sum()
                    if prop_sum > 0:
                        daily_flow_proportions_for_month = (
                            daily_flow_proportions_for_month / prop_sum
                        )

                # Disaggregate the monthly flow using the daily flow proportions
                if self.is_multisite:
                    for s, site in enumerate(self.site_names):
                        site_monthly_flow = Qs_monthly.loc[month_date, site]
                        Qs_daily.loc[start_date:end_date, site] = (
                            site_monthly_flow * daily_flow_proportions_for_month[:, s]
                        )
                else:
                    monthly_flow = Qs_monthly_index_array[y]
                    Qs_daily.loc[start_date:end_date] = (
                        monthly_flow * daily_flow_proportions_for_month
                    )

                # Check for issues
                if self.is_multisite:
                    for site in self.site_names:
                        site_data = Qs_daily.loc[start_date:end_date, site]
                        if site_data.isnull().any() or (site_data < 0).any():
                            msg = f"Disaggregation failed for site {site}, month {month} and year {month_date.year}. "
                            msg += f"Qs_daily contains NaN or negative values."
                            raise ValueError(msg)
                else:
                    daily_data = Qs_daily.loc[start_date:end_date]
                    if daily_data.isnull().any() or (daily_data < 0).any():
                        msg = f"Disaggregation failed for month {month} and year {month_date.year}. "
                        msg += f"Qs_daily contains NaN or negative values."
                        raise ValueError(msg)

        # Smooth month boundaries to reduce artificial discontinuities
        if self.blend_days > 0:
            Qs_daily = self._smooth_month_boundaries(Qs_daily, self.blend_days)

        return Qs_daily
