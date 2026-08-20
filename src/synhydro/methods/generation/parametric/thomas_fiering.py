"""
Thomas-Fiering generator for synthetic monthly streamflow generation.

Implements the Thomas-Fiering method (1962) with Stedinger-Taylor (1982)
normalization for generating synthetic streamflow sequences.
"""

import logging
from typing import Optional, Union, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from synhydro.core.base import Generator, FittedParams, make_output_index
from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.core.statistics import compute_monthly_statistics
from synhydro.transformations import SteddingerTransform

logger = logging.getLogger(__name__)


class ThomasFieringGenerator(Generator):
    """
    Thomas-Fiering autoregressive model for monthly streamflow generation.

    Generates synthetic monthly streamflows using a lag-1 autoregressive model
    with Stedinger-Taylor normalization. Preserves monthly means, standard
    deviations, and lag-1 serial correlations.

    Note: Thomas-Fiering is a univariate method (single site only).

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.methods.generate.parametric.thomas_fiering import ThomasFieringGenerator
    >>> Q_monthly = pd.read_csv('monthly_flows.csv', index_col=0, parse_dates=True)
    >>> tf = ThomasFieringGenerator()
    >>> tf.fit(Q_monthly.iloc[:, 0])
    >>> ensemble = tf.generate(n_years=10, n_realizations=5)

    References
    ----------
    Thomas, H.A., and Fiering, M.B. (1962). Mathematical synthesis of streamflow
    sequences for the analysis of river basins by simulation.

    Stedinger, J.R., and Taylor, M.R. (1982). Synthetic streamflow generation:
    1. Model verification and validation. Water Resources Research, 18(4), 909-918.
    """

    supports_multisite = False
    supported_frequencies = ("MS",)

    def __init__(self, *, name: Optional[str] = None, debug: bool = False, **kwargs):
        """
        Initialize the ThomasFieringGenerator.

        Parameters
        ----------
        name : str, optional
            Name for this generator instance.
        debug : bool, default=False
            Enable debug logging.
        **kwargs : dict, optional
            Additional parameters (currently unused).
        """
        super().__init__(name=name, debug=debug)

        # Initialize Stedinger transform
        self.stedinger_transform = SteddingerTransform(by_month=True)

        # Store initialization parameters
        self.init_params.algorithm_params = {
            "method": "Thomas-Fiering AR(1)",
            "distribution": "Normal (after Stedinger transformation)",
        }
        self.init_params.transformation_params = {
            "transformation": "SteddingerTransform",
            "by_month": True,
        }

    @property
    def output_frequency(self) -> str:
        """Thomas-Fiering generator produces monthly output."""
        return "MS"  # Month Start

    def preprocessing(self, Q_obs, *, sites: Optional[list] = None, **kwargs) -> None:
        """
        Preprocess observed data for Thomas-Fiering generation.

        Validates input, resamples to monthly if needed, and applies
        Stedinger-Taylor normalization.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Streamflow data with DatetimeIndex. Must be single site.
        sites : list, optional
            Not used (Thomas-Fiering is univariate).
        **kwargs : dict, optional
            Additional parameters (currently unused).
        """
        Q = self._store_obs_data(Q_obs, sites)

        # Resample to monthly if needed. Use inferred_freq so input without an
        # explicit DatetimeIndex.freq (the common case for loaded CSVs) is still
        # detected.
        detected_freq = Q.index.freq or pd.infer_freq(Q.index)
        detected_freq = str(detected_freq) if detected_freq is not None else None
        if detected_freq not in ("MS", "M", "<MonthBegin>", "<MonthEnd>"):
            self.logger.info(
                "Resampling from detected freq %r to monthly", detected_freq
            )
            Q = Q.resample("MS").sum()

        # Store monthly data
        self.Q_obs_monthly = Q.iloc[:, 0]  # Convert to Series for Thomas-Fiering

        # Apply Stedinger-Taylor normalization
        self.Q_norm = self.stedinger_transform.fit_transform(self.Q_obs_monthly)

        # Update state
        self.update_state(preprocessed=True)
        self.logger.info(f"Preprocessing complete: {len(self.Q_obs_monthly)} months")

    def fit(self, Q_obs=None, *, sites=None, **kwargs) -> None:
        """
        Estimate Thomas-Fiering model parameters from normalized flows.

        Calculates monthly means, standard deviations, and lag-1 serial
        correlations from normalized flows.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            If provided, calls preprocessing automatically.
        sites : list, optional
            Sites to use (passed to preprocessing if Q_obs provided).
        **kwargs : dict, optional
            Additional parameters (currently unused).
        """
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)
        self.validate_preprocessing()

        # Compute monthly statistics using centralized function
        monthly_stats = compute_monthly_statistics(self.Q_norm)
        self.mu_monthly = monthly_stats["mean"]
        self.sigma_monthly = monthly_stats["std"]

        # Compute monthly lag-1 correlation
        self.rho_monthly = self._compute_lag1_correlation(self.Q_norm)

        # Update state
        self.update_state(fitted=True)

        # Compute and store fitted parameters
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(f"Fitting complete: {len(self.Q_obs_monthly)} months")

    def _compute_lag1_correlation(self, data: pd.Series) -> pd.Series:
        """
        Compute lag-1 serial correlation between consecutive months.

        Returns the "arriving" correlation: ``rho[m]`` is the Pearson
        correlation between month ``m-1`` and month ``m`` (with the
        December-to-January transition wrapping across the year boundary),
        matching the convention used in the algorithm formulation.

        Parameters
        ----------
        data : pd.Series
            Monthly flow time series data.

        Returns
        -------
        pd.Series
            Series with months (1-12) as index, values are lag-1 correlations.
        """
        monthly_corr = self.mu_monthly.copy()

        for month in range(1, 13):
            prev_month = 12 if month == 1 else month - 1

            # Get paired consecutive month values: prev_month -> month
            first_values = []
            second_values = []

            for i in range(len(data) - 1):
                if (
                    data.index[i].month == prev_month
                    and data.index[i + 1].month == month
                ):
                    first_values.append(data.iloc[i])
                    second_values.append(data.iloc[i + 1])

            # Filter out NaN and Inf values
            if len(first_values) > 1:
                # Convert to numpy arrays for easier filtering
                first_arr = np.array(first_values)
                second_arr = np.array(second_values)

                # Keep only finite values
                valid_mask = np.isfinite(first_arr) & np.isfinite(second_arr)
                first_clean = first_arr[valid_mask]
                second_clean = second_arr[valid_mask]

                # Compute Pearson correlation if enough valid data
                if (
                    len(first_clean) > 1
                    and np.std(first_clean) > 0
                    and np.std(second_clean) > 0
                ):
                    try:
                        corr, _ = pearsonr(first_clean, second_clean)
                        # Replace NaN with 0
                        monthly_corr[month] = corr if np.isfinite(corr) else 0.0
                    except (ValueError, RuntimeWarning):
                        monthly_corr[month] = 0.0
                else:
                    monthly_corr[month] = 0.0  # Default to 0 if insufficient valid data
            else:
                monthly_corr[month] = 0.0  # Default to 0 if insufficient data

        return monthly_corr

    def _compute_fitted_params(self) -> FittedParams:
        """
        Extract and package fitted parameters.

        Returns
        -------
        FittedParams
            Dataclass containing all fitted parameters from Thomas-Fiering model.
        """
        # Count parameters: 12 months x (1 mean + 1 std + 1 correlation)
        n_params = 12 * 3

        # Get training period
        training_period = (
            str(self.Q_obs_monthly.index[0].date()),
            str(self.Q_obs_monthly.index[-1].date()),
        )

        # Package transformation parameters
        transform_params = {
            "stedinger_transform": {
                "tau_monthly": self.stedinger_transform.params_.get("tau"),
                "by_month": True,
            }
        }

        return FittedParams(
            means_=self.mu_monthly,
            stds_=self.sigma_monthly,
            correlations_=self.rho_monthly,
            distributions_={
                "type": "normal",
                "assumption": "AR(1) with normal innovations after Stedinger transformation",
            },
            transformations_=transform_params,
            n_parameters_=n_params,
            sample_size_=len(self.Q_obs_monthly),
            n_sites_=1,  # Thomas-Fiering is univariate
            training_period_=training_period,
        )

    def _generate(
        self, n_years: int, *, rng: np.random.Generator, **kwargs
    ) -> pd.DataFrame:
        """
        Generate a single realization of synthetic flows (internal method).

        Parameters
        ----------
        n_years : int
            Number of years to generate.
        rng : np.random.Generator
            Random number generator instance.
        **kwargs : dict, optional
            Additional parameters (currently unused).

        Returns
        -------
        pd.DataFrame
            Single realization of synthetic monthly flows.
        """

        # Generate synthetic sequences
        self.x_syn = np.zeros((n_years * 12))
        for i in range(n_years):
            for m in range(12):
                prev_month = m if m > 0 else 12
                month = m + 1

                if (i == 0) and (m == 0):
                    self.x_syn[0] = (
                        self.mu_monthly[month]
                        + rng.normal(0, 1) * self.sigma_monthly[month]
                    )

                else:

                    e_rand = rng.normal(0, 1)

                    self.x_syn[i * 12 + m] = (
                        self.mu_monthly[month]
                        + self.rho_monthly[month]
                        * (self.sigma_monthly[month] / self.sigma_monthly[prev_month])
                        * (self.x_syn[i * 12 + m - 1] - self.mu_monthly[prev_month])
                        + np.sqrt(1 - self.rho_monthly[month] ** 2)
                        * self.sigma_monthly[month]
                        * e_rand
                    )

        # Convert to DataFrame
        syn_start_year = self.Q_obs_monthly.index[0].year
        syn_start_date = f"{syn_start_year}-01-01"
        x_syn_df = pd.DataFrame(
            self.x_syn,
            index=make_output_index(syn_start_date, len(self.x_syn), "MS"),
        )

        # Replace negative values in normalized space
        x_syn_df[x_syn_df < 0] = self.Q_norm.min()

        # Inverse transform to original space
        Q_syn = self.stedinger_transform.inverse_transform(x_syn_df)

        # Handle negative and NaN values
        Q_syn[Q_syn < 0] = self.Q_obs_monthly.min()
        Q_syn = Q_syn.fillna(self.Q_obs_monthly.min())

        return Q_syn

    def generate(
        self,
        n_years: Optional[int] = None,
        n_realizations: int = 1,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Ensemble:
        """
        Generate synthetic monthly streamflows.

        Parameters
        ----------
        n_years : int, optional
            Number of years to generate per realization.
            If None, uses the length of historic data.
        n_realizations : int, default=1
            Number of synthetic realizations to generate.
        n_timesteps : int, optional
            Number of monthly timesteps to generate. If provided, overrides n_years.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : dict, optional
            Additional parameters (currently unused).

        Returns
        -------
        Ensemble
            Ensemble object containing all realizations.

        Raises
        ------
        ValueError
            If neither n_years nor n_timesteps is provided.
        """
        # Validate fit
        self.validate_fit()

        # Create random number generator
        rng = np.random.default_rng(seed)

        # Determine number of years
        if n_timesteps is not None:
            n_years = int(np.ceil(n_timesteps / 12))
        elif n_years is None:
            n_years = len(self.Q_obs_monthly) // 12  # Convert months to years

        if n_years <= 0:
            raise ValueError(f"n_years must be positive, got {n_years}")

        # Generate realizations
        realizations = {}
        for i in range(n_realizations):
            Q_syn = self._generate(n_years, rng=rng)

            # Extract values as Series
            if isinstance(Q_syn, pd.DataFrame):
                Q_syn = Q_syn.iloc[:, 0]

            # If n_timesteps specified, truncate to exact length
            if n_timesteps is not None and len(Q_syn) > n_timesteps:
                Q_syn = Q_syn.iloc[:n_timesteps]

            # Convert Series to DataFrame for Ensemble
            realizations[i] = Q_syn.to_frame(name=self._sites[0])

        self.logger.info(
            f"Generated {n_realizations} realizations of {n_years} years each"
        )

        first = realizations[0]
        metadata = EnsembleMetadata(
            generator_class=self.__class__.__name__,
            n_realizations=n_realizations,
            n_sites=1,
            time_resolution=self.output_frequency,
            time_period=(str(first.index[0].date()), str(first.index[-1].date())),
        )
        return Ensemble(realizations, metadata=metadata)
