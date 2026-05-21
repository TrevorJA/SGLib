"""
Kirsch Hybrid Bootstrap Generator (Kirsch et al. 2013)

Generates synthetic multi-site streamflow at monthly or weekly resolution by
bootstrapping standardized residuals and imposing fitted correlation structure
via Cholesky decomposition. A cross-year shifted matrix preserves the
year-to-year boundary correlation (December-January at monthly resolution, or
the equivalent week 26 boundary at weekly resolution).

The generator is hybrid (semi-parametric): observations are standardized by
fitted per-period mean and standard deviation (parametric pre-whitening) and a
Cholesky-decomposed intra-annual correlation matrix is imposed on synthetic
output (parametric correlation structure), while the new variability itself
comes from non-parametric bootstrap resampling of the standardized residuals.

Per Kirsch et al. (2013, eqs. 4-5), the correlation matrix and its Cholesky
factor are intra-annual (within-year) operators defined for a single site at a
time. Cross-site correlation is preserved through the shared bootstrap index
matrix M (Kirsch et al. 2013, p. 7), not through a joint multi-site correlation
matrix. This implementation therefore computes per-site Cholesky factors and
reuses one M across all sites.

Kirsch (2013) describes the method on weekly timesteps (52 columns, 26-week
halves). The algebra is identical for monthly timesteps (12 columns, 6-month
halves); both are supported here. The number of periods per year and the
half-year shift are derived from the input data frequency.

A normal score transform (NST) is applied in log space when
generate_using_log_flow=True. NST is a SynHydro-specific extension to Kirsch
(2013), which only z-score-standardizes (his eq. 3). NST eliminates bias in
the back-transformed marginal when standardized log-residuals are non-Gaussian.
Set generate_using_log_flow=False to disable both the log transform and NST
and run the algorithm closer to the published version.

References
----------
Kirsch, B.R., Characklis, G.W., and Zeff, H.B. (2013). Evaluating the impact
of alternative hydro-climate scenarios on transfer agreements: A practical
improvement for generating synthetic streamflows. Journal of Water Resources
Planning and Management, 139(4), 396-406.
https://doi.org/10.1061/(ASCE)WR.1943-5452.0000287
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from synhydro.core.base import Generator, FittedParams
from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.core.statistics import repair_correlation_matrix


_MONTHLY_FREQ = "MS"
_WEEKLY_FREQ = "W-SUN"
_PERIODS_PER_YEAR = {_MONTHLY_FREQ: 12, _WEEKLY_FREQ: 52}
_WEEKLY_ALIASES = {
    "W",
    "W-SUN",
    "W-MON",
    "W-TUE",
    "W-WED",
    "W-THU",
    "W-FRI",
    "W-SAT",
}


class KirschGenerator(Generator):
    """
    Kirsch hybrid bootstrap generator for monthly or weekly streamflow synthesis.

    Generates synthetic flows by bootstrap-resampling standardized residuals
    and imposing a fitted intra-annual correlation structure via Cholesky
    decomposition. Hybrid because parametric (mean, standard deviation,
    correlation) standardization wraps a non-parametric resampling core.

    The original Kirsch et al. (2013) paper specifies a weekly timestep with
    52 periods per year and a 26-week cross-year shift. This implementation
    additionally supports a monthly timestep (12 periods per year, 6-month
    cross-year shift); the algebra is identical and the period count is
    derived from the input data frequency at preprocessing time.

    References
    ----------
    Kirsch, B.R., Characklis, G.W., and Zeff, H.B. (2013). Evaluating the
    impact of alternative hydro-climate scenarios on transfer agreements.
    Journal of Water Resources Planning and Management, 139(4), 396-406.
    """

    supports_multisite = True
    supported_frequencies = ("MS", "W-SUN")

    def __init__(
        self,
        *,
        generate_using_log_flow: bool = True,
        matrix_repair_method: str = "spectral",
        name: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ):
        """
        Initialize Kirsch generator.

        Parameters
        ----------
        generate_using_log_flow : bool, default=True
            If True, generates in log-space for better handling of skewed distributions.
        matrix_repair_method : str, default='spectral'
            Method for repairing non-positive-definite correlation matrices.
        name : str, optional
            Name for this generator instance.
        debug : bool, default=False
            Enable debug logging.
        """
        super().__init__(name=name, debug=debug)

        self.generate_using_log_flow = generate_using_log_flow
        self.matrix_repair_method = matrix_repair_method

        self.init_params.algorithm_params = {
            "method": "Kirsch",
            "generate_using_log_flow": generate_using_log_flow,
            "matrix_repair_method": matrix_repair_method,
        }

        self.n_periods_per_year: Optional[int] = None
        self._target_frequency: Optional[str] = None
        self.U_site = {}
        self.U_prime_site = {}

    @property
    def output_frequency(self) -> str:
        """
        Return the temporal frequency of generated output.

        Set during preprocessing based on input data frequency and the
        ``timestep`` argument. Either ``'MS'`` (month start) or ``'W-SUN'``
        (week ending Sunday).
        """
        if self._target_frequency is None:
            raise ValueError("Run preprocessing() first to access output_frequency")
        return self._target_frequency

    @property
    def Q_obs_aggregated(self) -> pd.DataFrame:
        """
        Return observed per-period (monthly or weekly) flows as a DataFrame.

        Reconstructs a DatetimeIndex from the internal (year, period)
        MultiIndex according to ``self._target_frequency``. If the generator
        was fit with ``generate_using_log_flow=True``, the data is
        exponentiated back from log space.

        Returns
        -------
        pd.DataFrame
            Per-period aggregated observed flows indexed by DatetimeIndex at
            the target frequency.
        """
        if not hasattr(self, "Qm"):
            raise AttributeError(
                "Generator must be preprocessed before accessing Q_obs_aggregated"
            )

        if self.generate_using_log_flow:
            data = np.exp(self.Qm)
        else:
            data = self.Qm

        if isinstance(data.index, pd.MultiIndex):
            data = data.copy()
            if self._target_frequency == _MONTHLY_FREQ:
                dates = pd.to_datetime(
                    [f"{year}-{period:02d}-01" for year, period in data.index]
                )
            else:
                dates = pd.to_datetime(
                    [
                        pd.Timestamp.fromisocalendar(int(year), int(period), 7)
                        for year, period in data.index
                    ]
                )
            data.index = dates

        return data

    def _get_synthetic_index(self, n_years: int) -> pd.DatetimeIndex:
        """
        Build a DatetimeIndex for synthetic flows at the target frequency.

        For monthly output, returns ``n_years * 12`` consecutive month-starts.
        For weekly output, returns the Sundays of ISO weeks 1..52 of each
        synthetic year so that ordinal position ``k`` of year ``y`` lands on
        the Sunday of ISO week ``(k+1)`` of ``y``. This preserves the
        per-year week index used in fitting (Kirsch et al., 2013, p. 5) and
        prevents the 1.25-day-per-year calendar drift that a plain
        ``pd.date_range`` at W-SUN would introduce.

        Parameters
        ----------
        n_years : int
            Number of synthetic years to span.

        Returns
        -------
        pd.DatetimeIndex
            Index of length ``n_years * self.n_periods_per_year``.
        """
        start_year = int(self.Q.index.year.max()) + 1
        if self._target_frequency == _WEEKLY_FREQ:
            dates = [
                pd.Timestamp.fromisocalendar(y, w, 7)
                for y in range(start_year, start_year + n_years)
                for w in range(1, self.n_periods_per_year + 1)
            ]
            return pd.DatetimeIndex(dates)
        return pd.date_range(
            start=f"{start_year}-01-01",
            periods=n_years * self.n_periods_per_year,
            freq=self._target_frequency,
        )

    def preprocessing(
        self,
        Q_obs,
        *,
        sites=None,
        timestep: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Preprocess observed data for Kirsch generation.

        Detects the input data frequency and aggregates to the target
        per-period frequency (monthly or weekly). Daily input is aggregated
        according to ``timestep``; pre-aggregated input (already monthly or
        weekly) is used directly, and ``timestep`` is auto-detected when not
        given.

        Parameters
        ----------
        Q_obs : pd.DataFrame
            Observed historical flow data with DatetimeIndex. Accepts daily,
            monthly (MS), or weekly (W-SUN family) frequencies.
        sites : list, optional
            Sites to use. If None, uses all sites.
        timestep : {'monthly', 'weekly', None}, optional
            Target output resolution. If None, auto-detects from the input
            frequency; falls back to 'monthly' for daily input. Must agree
            with the input frequency when input is already pre-aggregated.
        **kwargs
            Additional preprocessing parameters.

        Raises
        ------
        ValueError
            If ``timestep`` contradicts the detected input frequency.
        """
        if timestep is not None and timestep not in ("monthly", "weekly"):
            raise ValueError(
                f"timestep must be 'monthly', 'weekly', or None, got {timestep!r}"
            )

        Q = self._store_obs_data(Q_obs, sites)

        input_freq = self._detect_input_frequency(Q)
        self._target_frequency = self._resolve_target_frequency(input_freq, timestep)
        self.n_periods_per_year = _PERIODS_PER_YEAR[self._target_frequency]

        if self._target_frequency == _MONTHLY_FREQ:
            self.Qm = self._aggregate_to_monthly(Q, input_freq)
        else:
            self.Qm = self._aggregate_to_weekly(Q, input_freq)

        if self.generate_using_log_flow:
            self.Qm = np.log(self.Qm.clip(lower=1e-6))

        self.Q = Q
        self.n_historic_years = Q.index.year.nunique()

        self.update_state(preprocessed=True)
        self.logger.info(
            "Preprocessing complete: %d sites, %d years, frequency=%s",
            self.n_sites,
            self.n_historic_years,
            self._target_frequency,
        )

    @staticmethod
    def _detect_input_frequency(Q: pd.DataFrame) -> Optional[str]:
        """
        Detect the frequency of an observed flow DataFrame.

        Parameters
        ----------
        Q : pd.DataFrame
            Observed flows with a DatetimeIndex.

        Returns
        -------
        str or None
            One of ``'D'``, ``'MS'``, ``'W-SUN'``, or None if not inferrable.
            Any weekly anchor (``'W'``, ``'W-MON'``, ...) is normalized to
            ``'W-SUN'``.
        """
        freq_str = pd.infer_freq(Q.index)
        if freq_str is None and Q.index.freq is not None:
            freq_str = Q.index.freq.freqstr
        if freq_str is None:
            return None
        if freq_str.startswith("W"):
            return _WEEKLY_FREQ
        if freq_str in ("MS", "M", "ME") or freq_str.startswith("MS"):
            return _MONTHLY_FREQ
        if freq_str == "D":
            return "D"
        return freq_str

    @staticmethod
    def _resolve_target_frequency(
        input_freq: Optional[str], timestep: Optional[str]
    ) -> str:
        """
        Pick the target output frequency given the input frequency and timestep.

        Parameters
        ----------
        input_freq : str or None
            Detected input frequency (``'D'``, ``'MS'``, ``'W-SUN'``, or None).
        timestep : {'monthly', 'weekly', None}
            User-supplied target. None means auto-detect.

        Returns
        -------
        str
            Either ``'MS'`` or ``'W-SUN'``.

        Raises
        ------
        ValueError
            If the requested ``timestep`` contradicts the input frequency.
        """
        if timestep == "monthly":
            requested = _MONTHLY_FREQ
        elif timestep == "weekly":
            requested = _WEEKLY_FREQ
        else:
            requested = None

        if input_freq == _MONTHLY_FREQ:
            if requested is not None and requested != _MONTHLY_FREQ:
                raise ValueError(
                    "Input data has monthly frequency but timestep='weekly' "
                    "was requested. Resampling monthly data to weekly is not "
                    "supported."
                )
            return _MONTHLY_FREQ

        if input_freq == _WEEKLY_FREQ:
            if requested is not None and requested != _WEEKLY_FREQ:
                raise ValueError(
                    "Input data has weekly frequency but timestep='monthly' "
                    "was requested. Resample to monthly outside the generator."
                )
            return _WEEKLY_FREQ

        return requested if requested is not None else _MONTHLY_FREQ

    @staticmethod
    def _aggregate_to_monthly(
        Q: pd.DataFrame, input_freq: Optional[str]
    ) -> pd.DataFrame:
        """
        Aggregate input flows to a (year, month) MultiIndex DataFrame.

        Parameters
        ----------
        Q : pd.DataFrame
            Observed flows, daily or monthly frequency.
        input_freq : str or None
            Detected input frequency; ``'MS'`` skips aggregation.

        Returns
        -------
        pd.DataFrame
            Monthly flows indexed by a ``(year, period)`` MultiIndex with
            level names ``('year', 'period')``.
        """
        if input_freq == _MONTHLY_FREQ:
            monthly = Q.copy()
            monthly.index = pd.MultiIndex.from_arrays(
                [monthly.index.year, monthly.index.month],
                names=["year", "period"],
            )
            return monthly

        monthly = Q.groupby([Q.index.year, Q.index.month]).sum()
        monthly.index = pd.MultiIndex.from_tuples(
            monthly.index, names=["year", "period"]
        )
        return monthly

    @staticmethod
    def _aggregate_to_weekly(
        Q: pd.DataFrame, input_freq: Optional[str]
    ) -> pd.DataFrame:
        """
        Aggregate input flows to a (iso_year, iso_week) MultiIndex DataFrame.

        ISO week 53 is dropped so every retained year has exactly 52 periods.

        Parameters
        ----------
        Q : pd.DataFrame
            Observed flows, daily or W-SUN frequency.
        input_freq : str or None
            Detected input frequency; ``'W-SUN'`` skips the resample step.

        Returns
        -------
        pd.DataFrame
            Weekly flows indexed by a ``(iso_year, iso_week)`` MultiIndex
            with level names ``('year', 'period')``. ISO week 53 rows are
            removed.
        """
        if input_freq == _WEEKLY_FREQ:
            weekly = Q.copy()
        else:
            weekly = Q.resample(_WEEKLY_FREQ).sum()

        iso = weekly.index.isocalendar()
        mask = iso["week"] <= 52
        weekly = weekly.loc[mask]
        iso = iso.loc[mask]

        weekly.index = pd.MultiIndex.from_arrays(
            [iso["year"].astype(int).values, iso["week"].astype(int).values],
            names=["year", "period"],
        )
        return weekly

    def fit(self, Q_obs=None, *, sites=None, timestep=None, **kwargs):
        """
        Fit Kirsch generator to preprocessed data.

        Parameters
        ----------
        Q_obs : pd.DataFrame, optional
            If provided, calls preprocessing automatically.
        sites : list, optional
            Sites to use (passed to preprocessing if Q_obs provided).
        timestep : {'monthly', 'weekly', None}, optional
            Target output resolution; forwarded to preprocessing.
        **kwargs
            Additional fitting parameters.
        """
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites, timestep=timestep)
        self.validate_preprocessing()

        n_per_year = self.n_periods_per_year

        # Per-period statistics by direct groupby on the (year, period) MultiIndex.
        grouped = self.Qm.groupby(level="period")
        self.mean_period = grouped.mean()
        self.std_period = grouped.std()

        # Keep only years that contain all n_per_year periods so the reshape is safe.
        period_counts = self.Qm.groupby(level="year").size()
        complete_years = period_counts[period_counts == n_per_year].index
        year_mask = self.Qm.index.get_level_values("year").isin(complete_years)
        Qm_complete = self.Qm.loc[year_mask].sort_index()

        valid_years = Qm_complete.index.get_level_values("year").unique().to_numpy()
        n_years = len(valid_years)

        Qm_arr = Qm_complete.values.reshape(n_years, n_per_year, self.n_sites)
        mean_arr = self.mean_period.values  # (n_per_year, n_sites)
        std_arr = self.std_period.values
        self.Z_h = (Qm_arr - mean_arr) / std_arr
        self.historic_years = valid_years
        self.n_historic_years = n_years

        if self.generate_using_log_flow:
            self._fit_normal_score_transforms()
            self.Y = self._apply_normal_score_transform(self.Z_h)
        else:
            self.Y = self.Z_h.copy()

        half_year = n_per_year // 2
        self.Y_prime = np.zeros_like(self.Y[:-1])
        self.Y_prime[:, :half_year, :] = self.Y[:-1, half_year:, :]
        self.Y_prime[:, half_year:, :] = self.Y[1:, :half_year, :]

        for s in range(self.n_sites):
            y_s = self.Y[:, :, s]  # (n_years, n_per_year)
            y_prime_s = self.Y_prime[:, :, s]

            corr_s = np.corrcoef(y_s.T)
            corr_prime_s = np.corrcoef(y_prime_s.T)

            self.U_site[s] = self._repair_and_cholesky(corr_s)
            self.U_prime_site[s] = self._repair_and_cholesky(corr_prime_s)

        self.update_state(fitted=True)
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(
            "Fitting complete: %d years, %d sites, %d periods/year",
            self.n_historic_years,
            self.n_sites,
            n_per_year,
        )

    def _compute_fitted_params(self) -> FittedParams:
        """
        Extract and package fitted parameters from Kirsch generator.

        Returns
        -------
        FittedParams
            Dataclass containing all fitted parameters.
        """
        n_per_year = self.n_periods_per_year
        # mean and std per (site, period): n_sites * n_per_year * 2
        # Cholesky factors (two matrices) per site: n_sites * n_per_year * n_per_year * 2
        n_params = (self.n_sites * n_per_year * 2) + (
            self.n_sites * n_per_year * n_per_year * 2
        )

        training_period = (str(self.Q.index[0].date()), str(self.Q.index[-1].date()))

        corr_matrices = {"U_site": self.U_site, "U_prime_site": self.U_prime_site}

        return FittedParams(
            means_=self.mean_period,
            stds_=self.std_period,
            correlations_=corr_matrices,
            distributions_=None,
            n_parameters_=n_params,
            sample_size_=self.n_historic_years * n_per_year,
            n_sites_=self.n_sites,
            training_period_=training_period,
        )

    def _repair_and_cholesky(self, corr):
        """
        Given a correlation matrix, repair to PSD if needed and return its
        upper-triangular Cholesky factor.

        Parameters
        ----------
        corr : np.ndarray
            Correlation matrix to be repaired and decomposed.

        Returns
        -------
        np.ndarray
            Upper-triangular Cholesky factor of the (possibly repaired) input.
        """
        try:
            return np.linalg.cholesky(corr).T
        except np.linalg.LinAlgError:
            self.logger.warning(
                "Matrix not positive definite, repairing... This may cause correlation inflation."
            )
            repaired_corr = repair_correlation_matrix(
                corr, method=self.matrix_repair_method
            )
            return np.linalg.cholesky(repaired_corr).T

    def _fit_normal_score_transforms(self):
        """
        Compute empirical CDF mappings for normal score transform per period per site.

        For each (period, site), stores the sorted standardized residuals and their
        corresponding normal scores (via Hazen plotting position). These are used
        during generation to map Cholesky-mixed values back to the correct
        per-period distribution, preventing bias from the exp() back-transform.
        """
        n_years = self.Z_h.shape[0]
        self._nst_sorted = {}
        self._nst_nscores = {}

        pp = (np.arange(1, n_years + 1) - 0.5) / n_years
        nscores = norm.ppf(pp)

        for m in range(self.n_periods_per_year):
            for s in range(self.n_sites):
                sorted_vals = np.sort(self.Z_h[:, m, s])
                self._nst_sorted[(m, s)] = sorted_vals
                self._nst_nscores[(m, s)] = nscores

    def _apply_normal_score_transform(self, Z):
        """
        Forward transform: standardized residuals -> normal scores.

        Parameters
        ----------
        Z : np.ndarray
            Standardized residuals with shape (n_years, n_periods_per_year, n_sites).

        Returns
        -------
        np.ndarray
            Normal-scored residuals with same shape.
        """
        Z_norm = np.zeros_like(Z)
        for m in range(self.n_periods_per_year):
            for s in range(self.n_sites):
                Z_norm[:, m, s] = np.interp(
                    Z[:, m, s], self._nst_sorted[(m, s)], self._nst_nscores[(m, s)]
                )
        return Z_norm

    def _apply_inverse_normal_score_transform(self, ZC):
        """
        Inverse transform: normal scores -> original standardized space per target period.

        Uses linear extrapolation at tails so synthetic values beyond the
        observed range are not clamped.

        Parameters
        ----------
        ZC : np.ndarray
            Normal-scored combined tensor with shape
            (n_years, n_periods_per_year, n_sites).

        Returns
        -------
        np.ndarray
            Standardized residuals in original space with same shape.
        """
        ZC_orig = np.zeros_like(ZC)
        for m in range(self.n_periods_per_year):
            for s in range(self.n_sites):
                ns = self._nst_nscores[(m, s)]
                sv = self._nst_sorted[(m, s)]
                slope_lo = (sv[1] - sv[0]) / (ns[1] - ns[0]) if ns[1] != ns[0] else 1.0
                slope_hi = (
                    (sv[-1] - sv[-2]) / (ns[-1] - ns[-2]) if ns[-1] != ns[-2] else 1.0
                )
                ns_ext = np.concatenate([[-10.0], ns, [10.0]])
                sv_ext = np.concatenate(
                    [
                        [sv[0] + slope_lo * (-10.0 - ns[0])],
                        sv,
                        [sv[-1] + slope_hi * (10.0 - ns[-1])],
                    ]
                )
                ZC_orig[:, m, s] = np.interp(ZC[:, m, s], ns_ext, sv_ext)
        return ZC_orig

    def _get_bootstrap_indices(self, n_years, max_idx=None, rng=None):
        """
        Return 'M', a matrix of bootstrap indices for the synthetic time series.

        Parameters
        ----------
        n_years : int
            Number of years for which to generate bootstrap indices.
        max_idx : int, optional
            Maximum index for the historic years. If None, uses the number of historic years.
        rng : np.random.Generator, optional
            Random number generator instance. If None, creates a new default generator.

        Returns
        -------
        np.ndarray
            A matrix of shape (n_years, n_periods_per_year) containing bootstrap indices.
        """
        if rng is None:
            rng = np.random.default_rng()
        max_idx = self.n_historic_years if max_idx is None else max_idx
        return rng.choice(
            max_idx, size=(n_years, self.n_periods_per_year), replace=True
        )

    def _create_bootstrap_tensor(self, M):
        """
        Create the bootstrap tensor of standardized flows from index matrix M.

        Parameters
        ----------
        M : np.ndarray
            Bootstrap indices with shape (n_years, n_periods_per_year), values in
            [0, self.Y.shape[0]).

        Returns
        -------
        np.ndarray
            Bootstrap tensor with shape (n_years, n_periods_per_year, n_sites).
        """
        source = self.Y
        n_years, n_periods = M.shape
        max_idx = source.shape[0]
        output = np.zeros((n_years, n_periods, self.n_sites))
        for i in range(n_years):
            for m in range(n_periods):
                h_idx = M[i, m]
                if h_idx >= max_idx:
                    h_idx = max_idx - 1
                output[i, m] = source[h_idx, m]
        return output

    def _apply_cholesky_and_combine(self, X, X_prime, n_years):
        """
        Apply Cholesky decomposition and combine Z with Z_prime tensors.

        This is the core pipeline: apply Cholesky to decorrelated residuals,
        then combine across years to preserve the cross-year boundary.

        Parameters
        ----------
        X : np.ndarray
            Bootstrap standardized residuals with shape
            (n_years+1, n_periods_per_year, n_sites).
        X_prime : np.ndarray
            Bootstrap standardized residuals from Y_prime with shape
            (n_years+1, n_periods_per_year, n_sites).
        n_years : int
            Number of years for the output (before buffering).

        Returns
        -------
        np.ndarray
            Combined and Cholesky-transformed flows with shape
            (n_years, n_periods_per_year, n_sites).
        """
        Z = np.zeros_like(X)
        Z_prime = np.zeros_like(X_prime)

        for s in range(self.n_sites):
            Z[:, :, s] = X[:, :, s] @ self.U_site[s]
            Z_prime[:, :, s] = X_prime[:, :, s] @ self.U_prime_site[s]

        ZC = self._combine_Z_and_Z_prime(Z, Z_prime)
        return ZC

    def _combine_Z_and_Z_prime(self, Z, Z_prime):
        """
        Combine Z and Z_prime into a single tensor, to preserve intra-year correlations.

        Parameters
        ----------
        Z : np.ndarray
            Standardized flows with shape (n_years, n_periods_per_year, n_sites).
        Z_prime : np.ndarray
            Standardized flows with shape (n_years, n_periods_per_year, n_sites).

        Returns
        -------
        np.ndarray
            Combined standardized flows with shape
            (n_years-1, n_periods_per_year, n_sites).
        """
        n_years = Z.shape[0] - 1
        half_year = self.n_periods_per_year // 2
        ZC = np.zeros((n_years, self.n_periods_per_year, self.n_sites))
        ZC[:, :half_year, :] = Z_prime[:n_years, half_year:, :]
        ZC[:, half_year:, :] = Z[1 : n_years + 1, half_year:, :]
        return ZC

    def _destandardize_flows(self, Z_combined):
        """
        Reapply per-period mean and standard deviation to standardized flows.

        Parameters
        ----------
        Z_combined : np.ndarray
            Standardized flows with shape (n_years, n_periods_per_year, n_sites).

        Returns
        -------
        np.ndarray
            Destandardized flows with shape (n_years, n_periods_per_year, n_sites).
        """
        Q_syn = np.zeros_like(Z_combined)
        for m in range(self.n_periods_per_year):
            Q_syn[:, m, :] = (
                Z_combined[:, m, :] * self.std_period.iloc[m].values
                + self.mean_period.iloc[m].values
            )
        return Q_syn

    def _reshape_output(self, Q_syn):
        return Q_syn.reshape(-1, self.n_sites)

    def _derive_X_prime(self, X):
        """
        Deterministically derive X_prime from X per Kirsch et al. (2013), p. 6.

        X_prime is the half-year shift of X: row i of X_prime combines the
        second half of year i with the first half of year i+1, mirroring how
        Y_prime is constructed from Y at fit time. There is exactly one
        bootstrap in the Kirsch algorithm; X_prime is not independently
        resampled.

        Parameters
        ----------
        X : np.ndarray
            Bootstrap residual tensor with shape
            (n_years+1, n_periods_per_year, n_sites).

        Returns
        -------
        np.ndarray
            X_prime of shape (n_years+1, n_periods_per_year, n_sites). The
            first n_years rows are the paper-correct shifted content; the
            last row repeats the penultimate row to satisfy the shape expected
            by ``_apply_cholesky_and_combine``.
        """
        n_years_plus_1, n_periods, n_sites = X.shape
        if n_periods != self.n_periods_per_year:
            raise ValueError(
                f"X has {n_periods} periods, expected {self.n_periods_per_year}"
            )
        n_years = n_years_plus_1 - 1
        half_year = self.n_periods_per_year // 2
        X_prime_temp = np.zeros((n_years, self.n_periods_per_year, n_sites))
        X_prime_temp[:, :half_year, :] = X[:-1, half_year:, :]
        X_prime_temp[:, half_year:, :] = X[1:, :half_year, :]
        X_prime = np.zeros((n_years + 1, self.n_periods_per_year, n_sites))
        X_prime[:n_years] = X_prime_temp
        X_prime[n_years] = X_prime_temp[-1]
        return X_prime

    def _pipeline_from_X(self, X, n_years, *, as_array=True, synthetic_index=None):
        """
        Run the paper-correct post-bootstrap pipeline on a pre-built X tensor.

        Shared by ``generate_single_series``, ``generate_from_indices``, and
        ``generate_from_residuals``. Derives X_prime (via ``_derive_X_prime``),
        applies the per-site Cholesky combination, the inverse normal-score
        transform when ``generate_using_log_flow`` is True, destandardizes,
        exponentiates, and reshapes.

        Parameters
        ----------
        X : np.ndarray
            Bootstrap residual tensor, shape
            (n_years+1, n_periods_per_year, n_sites).
        n_years : int
            Number of years for the synthetic output.
        as_array : bool, default=True
            If True, returns numpy array; if False, returns pandas DataFrame.
        synthetic_index : pd.DatetimeIndex, optional
            Custom DatetimeIndex for the output. If None, a default index
            is generated.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Synthetic per-period flows with shape
            (n_years * n_periods_per_year, n_sites).
        """
        X_prime = self._derive_X_prime(X)
        ZC = self._apply_cholesky_and_combine(X, X_prime, n_years)
        if self.generate_using_log_flow:
            ZC = self._apply_inverse_normal_score_transform(ZC)
        Q_syn = self._destandardize_flows(ZC)
        if self.generate_using_log_flow:
            Q_syn = np.exp(Q_syn)
        Q_flat = self._reshape_output(Q_syn)
        if as_array:
            return Q_flat
        if synthetic_index is None:
            synthetic_index = self._get_synthetic_index(n_years)
        return pd.DataFrame(Q_flat, columns=self._sites, index=synthetic_index)

    def generate_from_indices(
        self, indices, n_years=None, as_array=True, synthetic_index=None
    ):
        """
        Generate synthetic flows by directly specifying historical year indices.

        This method allows external code (e.g., MOEA-FIND) to inject decision
        variables (year indices) instead of random sampling. Runs the full
        post-bootstrap pipeline: Cholesky, normal-score inversion, re-seasonalization.

        Parameters
        ----------
        indices : np.ndarray
            Array of historical year indices to resample. Shape
            ``(n_years+1, n_periods_per_year)`` where each entry is in
            ``[0, n_historic_years)``. The extra year allows cross-year
            boundary handling. Can be floats (will be cast to int).
        n_years : int, optional
            Number of years for the synthetic output. If None, inferred from
            indices.shape[0] - 1.
        as_array : bool, default=True
            If True, returns numpy array; if False, returns pandas DataFrame.
        synthetic_index : pd.DatetimeIndex, optional
            Custom DatetimeIndex for the output. If None, a default index is generated.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Synthetic flows with shape ``(n_years * n_periods_per_year, n_sites)``
            if as_array=True, otherwise a pandas DataFrame.

        Notes
        -----
        This method assumes the generator has been fitted. Indices are treated as
        indices into the historic years array (self.historic_years or [0, 1, ..., n-1]).
        """
        self.validate_fit()

        indices = np.asarray(indices, dtype=int)

        if n_years is None:
            n_years = indices.shape[0] - 1

        if indices.shape != (n_years + 1, self.n_periods_per_year):
            raise ValueError(
                f"indices must have shape ({n_years + 1}, {self.n_periods_per_year}), "
                f"got {indices.shape}"
            )

        indices = np.clip(indices, 0, self.Y.shape[0] - 1)
        X = self._create_bootstrap_tensor(indices)
        return self._pipeline_from_X(
            X, n_years, as_array=as_array, synthetic_index=synthetic_index
        )

    def generate_from_residuals(self, residuals, as_array=True, synthetic_index=None):
        """
        Generate synthetic flows from pre-computed standardized residuals.

        This method allows external code (e.g., MOEA-FIND) to inject decision
        variables (standardized residuals) directly, bypassing the bootstrap
        resampling step. Runs steps 4-8 of the Kirsch pipeline:
        normal-score transform, Cholesky, inverse normal-score, cross-year
        combination, and re-seasonalization.

        Parameters
        ----------
        residuals : np.ndarray
            Array of standardized residuals with shape
            ``(n_years, n_periods_per_year, n_sites)``. Each residual should
            be approximately N(0,1) or representable as such within
            period-specific empirical distributions.
        as_array : bool, default=True
            If True, returns numpy array; if False, returns pandas DataFrame.
        synthetic_index : pd.DatetimeIndex, optional
            Custom DatetimeIndex for the output. If None, a default index is generated.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Synthetic flows with shape ``(n_years * n_periods_per_year, n_sites)``
            if as_array=True, otherwise a pandas DataFrame.

        Notes
        -----
        This method assumes the generator has been fitted. Residuals are assumed
        to be standardized residuals; they will be normal-score transformed,
        processed through Cholesky factors, and combined to preserve the
        cross-year boundary.
        """
        self.validate_fit()

        residuals = np.asarray(residuals)

        if residuals.ndim != 3 or residuals.shape[1:] != (
            self.n_periods_per_year,
            self.n_sites,
        ):
            raise ValueError(
                f"residuals must have shape (n_years, {self.n_periods_per_year}, "
                f"{self.n_sites}), got {residuals.shape}"
            )

        n_years = residuals.shape[0]

        X = np.zeros((n_years + 1, self.n_periods_per_year, self.n_sites))
        X[:n_years] = residuals
        X[n_years] = residuals[-1]

        return self._pipeline_from_X(
            X, n_years, as_array=as_array, synthetic_index=synthetic_index
        )

    def generate_single_series(
        self, n_years, M=None, as_array=True, synthetic_index=None, rng=None
    ):
        """
        Generate a single synthetic time series.

        Parameters
        ----------
        n_years : int
            Number of years for the synthetic time series.
        M : np.ndarray, optional
            Bootstrap indices for the synthetic time series. If None, random indices will be generated.
        as_array : bool
            If True, returns a numpy array; if False, returns a pandas DataFrame.
        synthetic_index : pd.DatetimeIndex, optional
            Custom index for the synthetic time series. If None, a default index will be generated.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Synthetic time series data.
        """
        if rng is None:
            rng = np.random.default_rng()

        n_years_buffered = n_years + 1

        if M is None:
            M = self._get_bootstrap_indices(
                n_years_buffered, max_idx=self.Y.shape[0], rng=rng
            )
        else:
            M = np.asarray(M)
            if M.shape != (n_years_buffered, self.n_periods_per_year):
                raise ValueError(
                    f"M must have shape ({n_years_buffered}, {self.n_periods_per_year})"
                )

        X = self._create_bootstrap_tensor(M)
        return self._pipeline_from_X(
            X, n_years, as_array=as_array, synthetic_index=synthetic_index
        )

    def generate(
        self, n_realizations=1, n_years=None, n_timesteps=None, seed=None, **kwargs
    ):
        """
        Generate an ensemble of synthetic monthly or weekly flows.

        Parameters
        ----------
        n_realizations : int, default=1
            Number of synthetic time series to generate.
        n_years : int, optional
            Number of years for each synthetic time series. If None, uses the
            number of historic years.
        n_timesteps : int, optional
            Not used (Kirsch generates whole years).
        seed : int, optional
            Random seed for reproducibility.
        **kwargs
            Additional generation parameters.

        Returns
        -------
        Ensemble
            Ensemble object containing all generated realizations.
        """
        self.validate_fit()

        rng = np.random.default_rng(seed)

        if n_years is None:
            n_years = self.n_historic_years

        realization_dict = {}
        for i in range(n_realizations):
            df = self.generate_single_series(n_years, as_array=False, rng=rng)
            realization_dict[i] = df

        metadata = EnsembleMetadata(
            generator_class=self.__class__.__name__,
            n_realizations=n_realizations,
            n_sites=self.n_sites,
            time_resolution=self.output_frequency,
            time_period=(
                str(realization_dict[0].index[0].date()),
                str(realization_dict[0].index[-1].date()),
            ),
        )

        ensemble = Ensemble(realization_dict, metadata=metadata)

        self.logger.info(
            "Generated %d realizations of %d years each",
            n_realizations,
            n_years,
        )

        return ensemble
