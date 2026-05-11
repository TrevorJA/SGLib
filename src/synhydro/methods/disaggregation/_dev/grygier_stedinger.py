"""
Grygier-Stedinger condensed disaggregation for temporal downscaling.

NOT EXPORTED FROM THE PUBLIC API.

TODO: The current implementation does not match Grygier and Stedinger (1988).
What is here is effectively Valencia-Schaake with a Kalman-style additive
residual projection D = S_e * 1 / (1^T S_e 1) applied in transformed space.
Specifically missing from the published method:
  - The condensed regression structure (per-month regressions with B_t * X_{y,t-1}
    lag-1 terms and optional cumulative-sum C_t * sum(W_u X_{y,u}) term, eqs. 9-11).
  - The four real-space conservation-adjustment schemes the paper actually
    compares (Proportional, ABS, SD, Stedinger-Vogel Exponential); the
    paper recommends the proportional scheme. The current covariance-based
    correction is not from the paper.
  - Inter-annual serial correlation via lag-1 cross-month coupling and
    previous-year terms.
Conservation only holds in transformed (log) space; after inverse transform
the 1^T X_syn = Y_syn guarantee is broken.

This module is retained on disk for a future rewrite. It is not imported
by synhydro.__init__ and is excluded from the documentation site.

References
----------
Grygier, J.C., and Stedinger, J.R. (1988). Condensed disaggregation procedures
and conservation corrections for stochastic hydrology. Water Resources Research,
24(10), 1574-1584. https://doi.org/10.1029/WR024i010p01574
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Any, List, Union

from synhydro.core.base import Disaggregator, DisaggregatorParams, FittedParams
from synhydro.core.ensemble import Ensemble, EnsembleMetadata


logger = logging.getLogger(__name__)


class GrygierStedingerDisaggregator(Disaggregator):
    """
    Grygier-Stedinger condensed disaggregation for annual to monthly flows.

    Implements the condensed parameterization and conservation correction
    matrix approach described in Grygier and Stedinger (1988). The method
    fits a reduced set of parameters (sub-period means, standard deviations,
    lag-0 correlations with aggregate, and lag-1 serial correlations) and
    applies a rigorous conservation correction that preserves the conditional
    covariance structure while ensuring exact summation.

    References
    ----------
    Grygier, J.C., and Stedinger, J.R. (1988). Condensed disaggregation
    procedures and conservation corrections for stochastic hydrology.
    Water Resources Research, 24(10), 1574-1584.
    """

    def __init__(
        self,
        *,
        n_subperiods: int = 12,
        transform: str = "log",
        name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the Grygier-Stedinger Disaggregator.

        Parameters
        ----------
        n_subperiods : int, default=12
            Number of sub-periods per aggregate period (typically 12 for
            annual to monthly). Must equal the number of columns if aggregating.
        transform : str, default='log'
            Transformation to apply before fitting: 'log', 'wilson_hilferty',
            or 'none'. Log transformation is standard for flow data.
        name : str, optional
            Name identifier for this disaggregator instance.
        debug : bool, default=False
            Enable debug logging.
        """
        super().__init__(name=name, debug=debug)

        # Store algorithm-specific parameters
        self.n_subperiods = n_subperiods
        self.transform = transform
        if transform not in ("log", "wilson_hilferty", "none"):
            raise ValueError(
                f"transform must be 'log', 'wilson_hilferty', or 'none', got '{transform}'"
            )

        # Update init_params
        self.init_params.algorithm_params = {
            "method": "Grygier-Stedinger Condensed Disaggregation",
            "n_subperiods": n_subperiods,
            "transform": transform,
        }

        # Fitted parameters (computed during fit())
        self.mu_X_ = None  # sub-period means (m,)
        self.sigma_X_ = None  # sub-period stds (m,)
        self.mu_Y_ = None  # aggregate mean (scalar)
        self.sigma_Y_ = None  # aggregate std (scalar)
        self.A_ = None  # regression coefficients (m,)
        self.C_ = None  # Cholesky factor of conditional covariance (m, m)
        self.D_ = None  # conservation correction matrix (m, 1)
        self.transformation_params_ = {}  # Parameters for inverse transformation

    @property
    def input_frequency(self) -> str:
        """Grygier-Stedinger disaggregator expects annual input."""
        return "YS"  # Year Start

    @property
    def output_frequency(self) -> str:
        """Grygier-Stedinger disaggregator produces monthly output."""
        return "MS"  # Monthly Start

    def preprocessing(
        self,
        Q_obs: Union[pd.Series, pd.DataFrame],
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Preprocess observed monthly flow data.

        Validates input data and prepares for aggregation. Expects monthly
        flow data (finer resolution); will aggregate to annual internally.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed historical flow data at monthly resolution (finer temporal
            resolution, i.e., input/output of disaggregation). Must have
            DatetimeIndex with monthly frequency (MS - month start).
        sites : list of str, optional
            Sites to use. If None, uses all columns.
        **kwargs : Any
            Additional preprocessing parameters (currently unused).
        """
        # Validate and store observed data
        Q_monthly = self._store_obs_data(Q_obs, sites)

        # Store validated data
        self.Q_monthly_ = Q_monthly

        # Detect single-site vs multisite
        self.is_multisite = (
            isinstance(self.Q_monthly_, pd.DataFrame) and self.Q_monthly_.shape[1] > 1
        )

        if self.is_multisite:
            # Create aggregate (index gauge) as sum of all sites
            self.Q_aggregate_ = self.Q_monthly_.sum(axis=1)
        else:
            # Convert to Series if single column DataFrame
            if isinstance(self.Q_monthly_, pd.DataFrame):
                self.Q_monthly_ = self.Q_monthly_.iloc[:, 0]
            self._sites = [self.Q_monthly_.name if self.Q_monthly_.name else "site"]
            self.Q_aggregate_ = self.Q_monthly_

        # Aggregate monthly to annual (sum of 12 months)
        self.Q_annual_ = self._aggregate_to_annual()

        # Get complete years for fitting
        self.n_years_fit_ = len(self.Q_annual_)

        if self.n_years_fit_ == 0:
            raise ValueError(
                "No complete years found. Grygier-Stedinger requires monthly data with at least one complete annual cycle."
            )

        # Update state
        self.update_state(preprocessed=True)
        self.logger.info(
            f"Preprocessing complete: {self.n_sites} sites, {self.n_years_fit_} years, "
            f"{len(self.Q_monthly_)} monthly observations"
        )

    def fit(
        self,
        Q_obs: Optional[Union[pd.Series, pd.DataFrame]] = None,
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the Grygier-Stedinger Disaggregator to the data.

        Computes sub-period statistics, condensed parameters, regression
        coefficients, conditional covariance, and conservation correction
        matrix. Applies transformation (log or Wilson-Hilferty) if specified.

        If ``Q_obs`` is provided, ``preprocessing()`` is called automatically.
        If omitted, a prior call to ``preprocessing()`` is required.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            Observed data. If provided, runs preprocessing automatically.
        sites : list of str, optional
            Sites to use (only when Q_obs is provided).
        **kwargs : Any
            Additional fitting parameters (currently unused).
        """
        # Auto-call preprocessing if Q_obs is provided
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)
        # Validate preprocessing
        self.validate_preprocessing()

        # Apply transformation if specified
        Q_monthly_transformed = self._apply_transformation(self.Q_monthly_)

        # Organize monthly sub-periods into annual vectors
        X_array = self._organize_subperiods(
            Q_monthly_transformed
        )  # (n_years, m, n_sites) or (n_years, m)

        # Compute annual aggregates from the organized subperiods
        # This guarantees alignment (only complete years) and correct shape
        if self.is_multisite:
            Y_array = X_array.sum(axis=(1, 2))  # (n_years,)
        else:
            Y_array = X_array.sum(axis=1)  # (n_years,)

        # Compute sub-period statistics
        self._compute_statistics(X_array, Y_array)

        # Compute regression coefficients A and conditional covariance S_e
        self._compute_regression_and_covariance()

        # Compute conservation correction matrix D
        self._compute_conservation_correction()

        # Cholesky decomposition of conditional covariance
        self._compute_cholesky()

        # Update state
        self.update_state(fitted=True)

        # Compute and store fitted parameters
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(
            f"Fitting complete: {self.n_subperiods} sub-periods, "
            f"transform={self.transform}, sites={self.n_sites}"
        )

    def _aggregate_to_annual(self) -> pd.Series:
        """
        Aggregate monthly flows to annual.

        Returns
        -------
        pd.Series
            Annual aggregates (sum of monthly flows).
        """
        if self.is_multisite:
            monthly_index = self.Q_aggregate_
        else:
            monthly_index = self.Q_monthly_

        # Group by year and sum
        annual = monthly_index.groupby(monthly_index.index.year).sum()
        annual.index = pd.to_datetime(annual.index, format="%Y")
        return annual

    def _apply_transformation(
        self, Q: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply log or Wilson-Hilferty transformation.

        Parameters
        ----------
        Q : pd.Series or pd.DataFrame
            Flow data to transform.

        Returns
        -------
        pd.Series or pd.DataFrame
            Transformed data.
        """
        if self.transform == "none":
            return Q.copy()

        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        Q_safe = Q.clip(lower=epsilon)

        if self.transform == "log":
            return np.log(Q_safe)
        elif self.transform == "wilson_hilferty":
            # Wilson-Hilferty (cube-root) transformation
            return np.cbrt(Q_safe)

    def _inverse_transformation(
        self, Q_transformed: Union[np.ndarray, pd.Series]
    ) -> Union[np.ndarray, pd.Series]:
        """
        Inverse transformation (exp or cube for transformed data).

        Parameters
        ----------
        Q_transformed : np.ndarray or pd.Series
            Transformed data to inverse transform.

        Returns
        -------
        np.ndarray or pd.Series
            Original-scale data.
        """
        if self.transform == "none":
            return Q_transformed

        if self.transform == "log":
            return np.exp(Q_transformed)
        elif self.transform == "wilson_hilferty":
            return np.power(Q_transformed, 3)

    def _organize_subperiods(
        self, Q_monthly_transformed: Union[pd.Series, pd.DataFrame]
    ) -> np.ndarray:
        """
        Organize monthly flows into annual vectors.

        Groups 12 consecutive months into annual vectors. Returns array of
        shape (n_years, 12) for single-site or (n_years, 12, n_sites) for
        multisite.

        Parameters
        ----------
        Q_monthly_transformed : pd.Series or pd.DataFrame
            Transformed monthly flows.

        Returns
        -------
        np.ndarray
            Sub-periods organized by year.
        """
        # Group by year
        grouped = Q_monthly_transformed.groupby(Q_monthly_transformed.index.year)

        if self.is_multisite:
            # Multisite: (n_years, 12, n_sites)
            X_list = []
            for year, group_df in grouped:
                if len(group_df) == 12:
                    X_list.append(group_df.values)
            X_array = np.array(X_list)
        else:
            # Single-site: (n_years, 12)
            X_list = []
            for year, group_series in grouped:
                if len(group_series) == 12:
                    X_list.append(group_series.values)
            X_array = np.array(X_list)

        return X_array

    def _compute_statistics(self, X_array: np.ndarray, Y_array: np.ndarray) -> None:
        """
        Compute sub-period and aggregate statistics.

        Computes means and standard deviations for sub-periods and aggregate,
        as well as covariance matrices.

        Parameters
        ----------
        X_array : np.ndarray
            Sub-period array of shape (n_years, m) for single-site or
            (n_years, m, n_sites) for multisite.
        Y_array : np.ndarray
            Aggregate array of shape (n_years,).
        """
        if self.is_multisite:
            # Single-site: average across years to get means for each month
            self.mu_X_ = X_array.mean(axis=0)  # (m, n_sites)
            self.sigma_X_ = X_array.std(axis=0, ddof=1)  # (m, n_sites)

            # Flatten for covariance computation: (n_years, m*n_sites)
            X_flat = X_array.reshape(X_array.shape[0], -1)
            self.S_XX_ = np.cov(X_flat.T)  # (m*n_sites, m*n_sites)

            # Covariance between sub-periods and aggregate
            self.S_XY_ = np.cov(X_flat.T, Y_array)[:-1, -1]  # (m*n_sites,)
        else:
            # Single-site: means and stds for each month
            self.mu_X_ = X_array.mean(axis=0)  # (m,)
            self.sigma_X_ = X_array.std(axis=0, ddof=1)  # (m,)

            # Covariance matrix between sub-periods
            self.S_XX_ = np.cov(X_array.T)  # (m, m)

            # Covariance between sub-periods and aggregate
            self.S_XY_ = np.cov(X_array.T, Y_array)[:-1, -1]  # (m,)

        # Aggregate statistics
        self.mu_Y_ = Y_array.mean()
        self.sigma_Y_ = Y_array.std(ddof=1)

    def _compute_regression_and_covariance(self) -> None:
        """
        Compute regression coefficients A and conditional covariance S_e.

        Following Valencia-Schaake approach:
        A = S_XY / sigma_Y^2
        S_e = S_XX - A * S_XY^T
        """
        if self.is_multisite:
            m = self.n_subperiods
            n_sites = self.n_sites

            # Regression coefficients (m*n_sites,)
            self.A_ = self.S_XY_ / (self.sigma_Y_**2)

            # Conditional covariance (m*n_sites, m*n_sites)
            S_XY_outer = np.outer(self.S_XY_, self.S_XY_)
            self.S_e_ = self.S_XX_ - S_XY_outer / (self.sigma_Y_**2)

            # Ensure symmetry and positive definiteness
            self.S_e_ = (self.S_e_ + self.S_e_.T) / 2
            self.S_e_ = np.maximum(self.S_e_, np.eye(m * n_sites) * 1e-10)
        else:
            # Single-site
            self.A_ = self.S_XY_ / (self.sigma_Y_**2)

            # Conditional covariance (m, m)
            S_XY_outer = np.outer(self.S_XY_, self.S_XY_)
            self.S_e_ = self.S_XX_ - S_XY_outer / (self.sigma_Y_**2)

            # Ensure symmetry and positive definiteness
            self.S_e_ = (self.S_e_ + self.S_e_.T) / 2
            self.S_e_ = np.maximum(self.S_e_, np.eye(self.n_subperiods) * 1e-10)

    def _compute_conservation_correction(self) -> None:
        """
        Compute conservation correction matrix D.

        The correction matrix D is derived from the conditional covariance
        and ensures that the sum of disaggregated flows exactly equals the
        aggregate while preserving the conditional covariance structure:

        D = S_e * 1_m * (1_m^T * S_e * 1_m)^{-1}

        where 1_m is a vector of ones. This is the key innovation of
        Grygier-Stedinger over Valencia-Schaake.
        """
        if self.is_multisite:
            m = self.n_subperiods
            n_sites = self.n_sites
            size = m * n_sites

            # Vector of ones
            ones = np.ones(size)

            # S_e * ones
            S_e_ones = self.S_e_ @ ones

            # ones^T * S_e * ones (scalar)
            denominator = ones @ S_e_ones

            # Avoid division by zero
            if abs(denominator) < 1e-10:
                self.logger.warning(
                    "denominator in conservation correction is near zero"
                )
                denominator = 1e-10

            # D = S_e * ones / denominator
            self.D_ = S_e_ones / denominator
        else:
            m = self.n_subperiods

            # Vector of ones
            ones = np.ones(m)

            # S_e * ones
            S_e_ones = self.S_e_ @ ones

            # ones^T * S_e * ones (scalar)
            denominator = ones @ S_e_ones

            # Avoid division by zero
            if abs(denominator) < 1e-10:
                self.logger.warning(
                    "denominator in conservation correction is near zero"
                )
                denominator = 1e-10

            # D = S_e * ones / denominator
            self.D_ = S_e_ones / denominator

    def _compute_cholesky(self) -> None:
        """
        Cholesky decomposition of conditional covariance.

        Factorizes S_e = C * C^T for efficient sampling.

        Raises
        ------
        np.linalg.LinAlgError
            If S_e is not positive definite.
        """
        try:
            self.C_ = np.linalg.cholesky(self.S_e_)
        except np.linalg.LinAlgError:
            # Add small diagonal jitter if Cholesky fails
            self.logger.warning("Cholesky decomposition failed, adding diagonal jitter")
            size = self.S_e_.shape[0]
            self.S_e_ = self.S_e_ + np.eye(size) * 1e-8
            self.C_ = np.linalg.cholesky(self.S_e_)

    def disaggregate(
        self, ensemble: Ensemble, seed: Optional[int] = None, **kwargs: Any
    ) -> Ensemble:
        """
        Disaggregate annual ensemble to monthly flows using Grygier-Stedinger method.

        For each synthetic annual value, computes conditional mean, generates
        uncorrected sub-periods from multivariate normal, applies conservation
        correction, inverse transforms, and enforces non-negativity.

        Parameters
        ----------
        ensemble : Ensemble
            Annual streamflow ensemble to disaggregate.
            Must have frequency 'AS' (annual start).
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : Any
            Additional disaggregation parameters (currently unused).

        Returns
        -------
        Ensemble
            Disaggregated monthly streamflow ensemble with frequency 'MS'.
        """
        # Validate fit
        self.validate_fit()

        # Validate input ensemble
        self.validate_input_ensemble(ensemble)

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Disaggregate each realization
        monthly_realization_dict = {}

        for realization_id, annual_df in ensemble.data_by_realization.items():
            # Disaggregate this realization
            monthly_df = self._disaggregate_single_realization(annual_df, rng=rng)
            monthly_realization_dict[realization_id] = monthly_df

        # Create metadata for monthly ensemble
        metadata = EnsembleMetadata(
            generator_class=ensemble.metadata.generator_class,
            generator_params=ensemble.metadata.generator_params,
            n_realizations=len(monthly_realization_dict),
            n_sites=len(self._sites),
            time_resolution=self.output_frequency,
            time_period=(
                str(monthly_realization_dict[0].index[0].date()),
                str(monthly_realization_dict[0].index[-1].date()),
            ),
        )

        # Create and return monthly ensemble
        monthly_ensemble = Ensemble(monthly_realization_dict, metadata=metadata)

        self.logger.info(
            f"Disaggregated {len(monthly_realization_dict)} realizations "
            f"from annual to monthly (Grygier-Stedinger)"
        )

        return monthly_ensemble

    def _disaggregate_single_realization(
        self, Y_syn_annual: pd.DataFrame, *, rng: np.random.Generator
    ) -> pd.DataFrame:
        """
        Disaggregate a single realization from annual to monthly.

        Parameters
        ----------
        Y_syn_annual : pd.DataFrame
            Annual synthetic flows for a single realization.
            Columns are site names, index is years.

        Returns
        -------
        pd.DataFrame
            Monthly disaggregated flows with DatetimeIndex (month start frequency).
        """
        # Transform annual values
        Y_syn_transformed = self._apply_transformation(Y_syn_annual.sum(axis=1))

        n_years = len(Y_syn_transformed)
        m = self.n_subperiods

        # Create monthly output index
        start_year = (
            Y_syn_annual.index[0].year
            if hasattr(Y_syn_annual.index[0], "year")
            else Y_syn_annual.index[0]
        )
        start_date = pd.Timestamp(year=start_year, month=1, day=1)
        end_date = pd.Timestamp(year=start_year + n_years - 1, month=12, day=31)
        monthly_index = pd.date_range(start=start_date, end=end_date, freq="MS")

        if self.is_multisite:
            X_syn_monthly = np.zeros((n_years, m, self.n_sites))
        else:
            X_syn_monthly = np.zeros((n_years, m))

        # Disaggregate each year
        for y in range(n_years):
            Y_y = Y_syn_transformed.iloc[y]

            if self.is_multisite:
                # Conditional mean for each site
                mu_X_y = self.mu_X_ + self.A_.reshape(m, self.n_sites) * (
                    Y_y - self.mu_Y_
                )

                # Generate uncorrected sub-periods
                Z = rng.standard_normal(m * self.n_sites)
                X_raw = mu_X_y.flatten() + self.C_ @ Z

                # Apply conservation correction
                delta = Y_y - X_raw.reshape(m, self.n_sites).sum()
                X_corrected = X_raw + self.D_ * delta

                # Reshape and inverse transform
                X_y = X_corrected.reshape(m, self.n_sites)
                X_y_original = self._inverse_transformation(X_y)

                # Enforce non-negativity
                X_y_final = self._enforce_non_negativity_multisite(X_y_original)
                X_syn_monthly[y, :, :] = X_y_final
            else:
                # Conditional mean
                mu_X_y = self.mu_X_ + self.A_ * (Y_y - self.mu_Y_)

                # Generate uncorrected sub-periods
                Z = rng.standard_normal(m)
                X_raw = mu_X_y + self.C_ @ Z

                # Apply conservation correction
                delta = Y_y - X_raw.sum()
                X_corrected = X_raw + self.D_ * delta

                # Inverse transform
                X_y_original = self._inverse_transformation(X_corrected)

                # Enforce non-negativity
                X_y_final = self._enforce_non_negativity_singlesite(X_y_original)
                X_syn_monthly[y, :] = X_y_final

        # Format as DataFrame
        if self.is_multisite:
            # Shape: (n_years * 12, n_sites)
            X_syn_flat = X_syn_monthly.reshape(-1, self.n_sites)
            monthly_df = pd.DataFrame(
                X_syn_flat, index=monthly_index, columns=self._sites
            )
        else:
            # Shape: (n_years * 12,)
            X_syn_flat = X_syn_monthly.flatten()
            monthly_df = pd.DataFrame(
                X_syn_flat, index=monthly_index, columns=[self._sites[0]]
            )

        return monthly_df

    def _enforce_non_negativity_singlesite(self, X: np.ndarray) -> np.ndarray:
        """
        Enforce non-negativity for single-site flows.

        If any month is negative, set to zero and redistribute the deficit
        proportionally across remaining months.

        Parameters
        ----------
        X : np.ndarray
            Sub-period array of shape (m,).

        Returns
        -------
        np.ndarray
            Non-negative sub-period array.
        """
        X = X.copy()
        X = np.maximum(X, 0)  # Clip to zero

        # Redistribute negative values
        deficit = 0
        for i in range(len(X)):
            if X[i] < 0:
                deficit += -X[i]
                X[i] = 0

        if deficit > 0:
            positive_mask = X > 0
            if positive_mask.sum() > 0:
                X[positive_mask] += deficit / positive_mask.sum()

        return X

    def _enforce_non_negativity_multisite(self, X: np.ndarray) -> np.ndarray:
        """
        Enforce non-negativity for multisite flows.

        If any flow is negative, set to zero and redistribute the deficit
        proportionally across remaining positive values.

        Parameters
        ----------
        X : np.ndarray
            Sub-period array of shape (m, n_sites).

        Returns
        -------
        np.ndarray
            Non-negative sub-period array.
        """
        X = X.copy()
        X = np.maximum(X, 0)  # Clip to zero

        # Redistribute negative values
        for s in range(X.shape[1]):
            deficit = 0
            for i in range(X.shape[0]):
                if X[i, s] < 0:
                    deficit += -X[i, s]
                    X[i, s] = 0

            if deficit > 0:
                positive_mask = X[:, s] > 0
                if positive_mask.sum() > 0:
                    X[positive_mask, s] += deficit / positive_mask.sum()

        return X

    def _compute_fitted_params(self) -> FittedParams:
        """
        Extract and package fitted parameters.

        Returns
        -------
        FittedParams
            Dataclass containing all fitted parameters.
        """
        # Count parameters: m means + m stds + m lags + m conditional variances + m conservation weights
        n_params = self.n_subperiods * 4 + self.n_subperiods

        # Get training period
        training_period = (
            str(self.Q_monthly_.index[0].date()),
            str(self.Q_monthly_.index[-1].date()),
        )

        # Package fitted model info
        fitted_models_info = {
            "method": "Grygier-Stedinger Condensed Disaggregation",
            "n_subperiods": self.n_subperiods,
            "transform": self.transform,
            "A_shape": tuple(self.A_.shape) if hasattr(self.A_, "shape") else None,
            "S_e_shape": tuple(self.S_e_.shape),
            "C_shape": tuple(self.C_.shape),
            "D_shape": tuple(self.D_.shape),
        }

        return FittedParams(
            means_=pd.Series(
                self.mu_X_, index=[f"month_{i+1}" for i in range(self.n_subperiods)]
            ),
            stds_=pd.Series(
                self.sigma_X_, index=[f"month_{i+1}" for i in range(self.n_subperiods)]
            ),
            correlations_={"S_e": self.S_e_, "A": self.A_, "D": self.D_},
            fitted_models_=fitted_models_info,
            n_parameters_=n_params,
            sample_size_=len(self.Q_monthly_),
            n_sites_=self.n_sites,
            training_period_=training_period,
        )
