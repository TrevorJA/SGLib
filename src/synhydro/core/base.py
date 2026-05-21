"""
Improved Base Generator Class for SynHydro

This module provides an abstract base class for all synthetic generation methods.
"""

from abc import ABC, abstractmethod
import logging
import warnings
from typing import Union, Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict


@dataclass
class GeneratorState:
    """Track generator preprocessing and fitting state."""

    is_preprocessed: bool = False
    is_fitted: bool = False
    fit_timestamp: Optional[str] = None


@dataclass
class GeneratorParams:
    """
    Store initialization/configuration parameters for generators.

    These are user-specified settings that control algorithm behavior,
    not learned from data.
    """

    # Common parameters across all generators
    random_seed: Optional[int] = None
    verbose: bool = False
    debug: bool = False

    # Flexible storage for generator-specific parameters
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    transformation_params: Dict[str, Any] = field(default_factory=dict)
    computational_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary."""
        result = {
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "debug": self.debug,
        }
        result.update(self.algorithm_params)
        result.update(self.transformation_params)
        result.update(self.computational_params)
        return result

    def __repr__(self) -> str:
        """Readable string representation."""
        lines = ["GeneratorParams:"]
        if self.random_seed is not None:
            lines.append(f"  random_seed: {self.random_seed}")
        if self.verbose:
            lines.append(f"  verbose: {self.verbose}")
        if self.debug:
            lines.append(f"  debug: {self.debug}")

        if self.algorithm_params:
            lines.append("  Algorithm parameters:")
            for key, val in self.algorithm_params.items():
                lines.append(f"    {key}: {val}")

        if self.transformation_params:
            lines.append("  Transformation parameters:")
            for key, val in self.transformation_params.items():
                lines.append(f"    {key}: {val}")

        if self.computational_params:
            lines.append("  Computational parameters:")
            for key, val in self.computational_params.items():
                lines.append(f"    {key}: {val}")

        return "\n".join(lines)


@dataclass
class FittedParams:
    """
    Store parameters learned from data during fit().

    Following scikit-learn convention, parameter names end with underscore.
    """

    # Statistical moments (common across many generators)
    means_: Optional[Union[pd.Series, pd.DataFrame]] = None
    stds_: Optional[Union[pd.Series, pd.DataFrame]] = None
    correlations_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None

    # Distribution parameters (for parametric methods)
    distributions_: Optional[Dict[str, Any]] = None

    # Transformation parameters (if transformations were fitted)
    transformations_: Optional[Dict[str, Any]] = None

    # Other fitted objects (flexible storage)
    fitted_models_: Optional[Dict[str, Any]] = None

    # Metadata about fitting
    n_parameters_: int = 0  # Total number of fitted parameters
    sample_size_: int = 0  # Training data size (timesteps)
    n_sites_: int = 0  # Number of sites
    training_period_: Optional[Tuple[str, str]] = None  # (start, end) dates

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling numpy/pandas types."""
        result = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                result[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        """Readable string representation."""
        lines = ["FittedParams:"]
        lines.append(f"  n_parameters: {self.n_parameters_}")
        lines.append(f"  sample_size: {self.sample_size_}")
        lines.append(f"  n_sites: {self.n_sites_}")

        if self.training_period_ is not None:
            lines.append(
                f"  training_period: {self.training_period_[0]} to {self.training_period_[1]}"
            )

        if self.means_ is not None:
            lines.append(f"  means: fitted ({self._describe_shape(self.means_)})")
        if self.stds_ is not None:
            lines.append(f"  stds: fitted ({self._describe_shape(self.stds_)})")
        if self.correlations_ is not None:
            if isinstance(self.correlations_, dict):
                lines.append(
                    f"  correlations: {len(self.correlations_)} matrices fitted"
                )
            else:
                lines.append(f"  correlations: fitted ({self.correlations_.shape})")

        if self.distributions_:
            lines.append(f"  distributions: {list(self.distributions_.keys())}")

        if self.transformations_:
            lines.append(f"  transformations: {list(self.transformations_.keys())}")

        return "\n".join(lines)

    @staticmethod
    def _describe_shape(obj):
        """Describe shape of pandas or numpy object."""
        if isinstance(obj, pd.DataFrame):
            return f"{obj.shape[0]}x{obj.shape[1]} DataFrame"
        elif isinstance(obj, pd.Series):
            return f"Series of length {len(obj)}"
        elif isinstance(obj, np.ndarray):
            return f"array {obj.shape}"
        return str(type(obj))


class Generator(ABC):
    """
    Abstract base class for all synthetic generation methods.

    All generator implementations should inherit from this class.
    Follows the scikit-learn pattern: ``__init__`` configures the algorithm,
    ``fit(Q_obs)`` learns from data, ``generate()`` produces synthetic flows.

    Class Attributes
    ----------------
    supports_multisite : bool
        Whether this generator supports multiple sites. Default False.
    supported_frequencies : tuple of str
        Pandas frequency strings this generator accepts (e.g., ``('MS',)``).
    """

    supports_multisite: bool = False
    supported_frequencies: tuple = ("MS",)

    @abstractmethod
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the generator with algorithm configuration.

        Subclasses add algorithm-specific keyword-only parameters before
        ``name`` and ``debug``. Data is not passed here — use ``fit(Q_obs)``
        or ``preprocessing(Q_obs)`` instead.

        Parameters
        ----------
        name : str, optional
            Name identifier for this generator instance.
        debug : bool, default False
            Enable debug logging.
        """
        self.name = name or self.__class__.__name__
        self.debug = debug

        self.state = GeneratorState()

        # Initialize parameter storage
        self.init_params = GeneratorParams(debug=debug)
        self.fitted_params_ = None  # Set during _compute_fitted_params()

        # Setup logging
        self._setup_logging(debug)

    def _setup_logging(self, debug: bool) -> None:
        """Setup logging infrastructure."""
        self.logger = logging.getLogger(f"synhydro.{self.name}")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def validate_input_data(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Validate and standardize input data format.

        Checks type, DatetimeIndex, NaN content, negative values, data
        frequency, and minimum record length.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Input time series data

        Returns
        -------
        pd.DataFrame
            Validated and standardized data

        Raises
        ------
        ValueError
            If data format is invalid
        TypeError
            If data type is unsupported
        """
        # Type checking
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f"Input data must be pandas Series or DataFrame, got {type(data)}"
            )

        # Convert Series to DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=data.name or "flow")

        # Empty data
        if len(data) == 0:
            raise ValueError("Input data is empty.")

        # Index validation
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(
                    f"Data index must be a DatetimeIndex, got {type(data.index)}: {e}"
                )

        # NaN check
        nan_counts = data.isna().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            pct = total_nans / data.size * 100
            nan_sites = nan_counts[nan_counts > 0]
            self.logger.warning(
                "Input data contains %d NaN values (%.1f%%). "
                "Sites with NaNs: %s. NaN rows will propagate through "
                "fitting and may cause errors or poor results.",
                total_nans,
                pct,
                dict(nan_sites),
            )

        # Negative value check
        numeric_data = data.select_dtypes(include="number")
        if (numeric_data < 0).any().any():
            neg_sites = [
                col for col in numeric_data.columns if (numeric_data[col] < 0).any()
            ]
            self.logger.warning(
                "Input data contains negative values in sites: %s. "
                "Generators using log transforms will fail on negative values.",
                neg_sites,
            )

        # Frequency detection and validation
        freq = pd.infer_freq(data.index)
        if freq is None and data.index.freq is not None:
            freq = data.index.freq.freqstr
        if freq is not None:
            # Normalize common frequency aliases
            freq_str = str(freq)
            freq_map = {
                "D": "D",
                "MS": "MS",
                "M": "MS",
                "ME": "MS",
                "YS": "YS",
                "YE": "YS",
                "AS": "YS",
                "A": "YS",
                "AS-JAN": "YS",
                "<MonthBegin>": "MS",
                "<MonthEnd>": "MS",
                "<YearBegin: month=1>": "YS",
                "<YearEnd: month=12>": "YS",
                "YS-JAN": "YS",
                "YE-DEC": "YS",
                "<Day>": "D",
                "W": "W-SUN",
                "W-SUN": "W-SUN",
                "W-MON": "W-SUN",
                "W-TUE": "W-SUN",
                "W-WED": "W-SUN",
                "W-THU": "W-SUN",
                "W-FRI": "W-SUN",
                "W-SAT": "W-SUN",
            }
            normalized = freq_map.get(freq_str, freq_str)
            if (
                hasattr(self, "supported_frequencies")
                and normalized not in self.supported_frequencies
            ):
                self.logger.warning(
                    "Detected frequency '%s' is not in supported frequencies %s "
                    "for %s. The generator may resample internally or raise an "
                    "error during preprocessing.",
                    normalized,
                    self.supported_frequencies,
                    self.__class__.__name__,
                )

        # Minimum record length
        n = len(data)
        if n < 24:
            self.logger.warning(
                "Input data has only %d timesteps. Most generators need at "
                "least 2-3 years of data for reliable parameter estimation. "
                "Results may be unreliable.",
                n,
            )

        self.logger.info(
            "Validated data: %d timesteps, %d sites, period %s to %s",
            data.shape[0],
            data.shape[1],
            data.index[0],
            data.index[-1],
        )

        return data

    def validate_preprocessing(self) -> None:
        """
        Check if preprocessing has been completed.

        Raises
        ------
        ValueError
            If preprocessing() has not been run.
        """
        if not self.state.is_preprocessed:
            raise ValueError(
                f"{self.name} must run preprocessing() before fit() or generate()"
            )

    def validate_fit(self) -> None:
        """
        Check if generator has been fitted.

        Raises
        ------
        ValueError
            If fit() has not been run.
        """
        if not self.state.is_fitted:
            raise ValueError(f"{self.name} must run fit() before generate()")

    def update_state(
        self, preprocessed: Optional[bool] = None, fitted: Optional[bool] = None
    ) -> None:
        """
        Update generator state flags.

        Parameters
        ----------
        preprocessed : bool, optional
            Set preprocessing state.
        fitted : bool, optional
            Set fitted state.
        """
        if preprocessed is not None:
            self.state.is_preprocessed = preprocessed
            self.logger.debug(f"State updated: is_preprocessed={preprocessed}")

        if fitted is not None:
            self.state.is_fitted = fitted
            if fitted:
                self.state.fit_timestamp = datetime.now().isoformat()
            self.logger.debug(f"State updated: is_fitted={fitted}")

    @property
    def is_fitted(self) -> bool:
        """Check if generator is fitted."""
        return self.state.is_fitted

    @property
    def is_preprocessed(self) -> bool:
        """Check if preprocessing is complete."""
        return self.state.is_preprocessed

    @property
    def n_sites(self) -> int:
        """
        Number of sites in the generator.

        Returns
        -------
        int
            Number of sites.

        Raises
        ------
        ValueError
            If preprocessing not yet run.
        """
        if not hasattr(self, "_sites"):
            raise ValueError("Run preprocessing() first to access n_sites")
        return len(self._sites)

    @property
    def sites(self) -> List[str]:
        """
        List of site names.

        Returns
        -------
        List[str]
            Site identifiers.

        Raises
        ------
        ValueError
            If preprocessing not yet run.
        """
        if not hasattr(self, "_sites"):
            raise ValueError("Run preprocessing() first to access sites")
        return self._sites

    @property
    @abstractmethod
    def output_frequency(self) -> str:
        """
        Temporal frequency of generated output.

        Returns
        -------
        str
            Pandas frequency string (e.g., 'MS' for monthly, 'D' for daily).
        """
        pass

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get initialization parameters (scikit-learn style).

        Returns only constructor/configuration parameters, not fitted values.
        Following scikit-learn convention for compatibility.

        Parameters
        ----------
        deep : bool, default=True
            If True, return deep copy of parameters.

        Returns
        -------
        Dict[str, Any]
            Dictionary of initialization parameters.
        """
        return self.init_params.to_dict()

    def get_fitted_params(self) -> Dict[str, Any]:
        """
        Get parameters learned from data during fit().

        Returns
        -------
        Dict[str, Any]
            Dictionary of fitted parameters (all keys end with underscore).

        Raises
        ------
        ValueError
            If generator has not been fitted yet.
        """
        if self.fitted_params_ is None:
            raise ValueError(f"{self.name} has not been fitted yet. Run fit() first.")
        return self.fitted_params_.to_dict()

    @abstractmethod
    def _compute_fitted_params(self) -> FittedParams:
        """
        Extract and package fitted parameters after fit().

        Must be implemented by each generator subclass to specify
        what parameters were learned during fitting.

        Returns
        -------
        FittedParams
            Dataclass containing all fitted parameters.
        """
        pass

    def summary(self, show_fitted: bool = True) -> str:
        """
        Generate comprehensive summary of generator configuration and fit.

        Parameters
        ----------
        show_fitted : bool, default=True
            Whether to include fitted parameters in summary.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"{self.name} Summary".center(80))
        lines.append("=" * 80)
        lines.append("")

        # Model Information
        lines.append("Model Information")
        lines.append("-" * 80)
        lines.append(f"Generator Type:          {self.__class__.__name__}")
        lines.append(
            f"Status:                  {'Fitted' if self.is_fitted else 'Not Fitted'}"
        )

        if self.is_fitted and self.state.fit_timestamp:
            lines.append(f"Fitted:                  {self.state.fit_timestamp}")

        if hasattr(self, "_sites") and self._sites:
            lines.append(f"Number of Sites:         {len(self._sites)}")
            if len(self._sites) <= 5:
                lines.append(f"Sites:                   {self._sites}")
            else:
                lines.append(
                    f"Sites:                   {self._sites[:3]} ... ({len(self._sites)} total)"
                )

        lines.append("")

        # Initialization Parameters
        lines.append("Initialization Parameters")
        lines.append("-" * 80)
        params_str = str(self.init_params)
        for line in params_str.split("\n")[1:]:  # Skip first line "GeneratorParams:"
            lines.append(line)
        lines.append("")

        # Fitted Parameters
        if show_fitted and self.is_fitted and self.fitted_params_ is not None:
            lines.append("Fitted Parameters")
            lines.append("-" * 80)
            fitted_str = str(self.fitted_params_)
            for line in fitted_str.split("\n")[1:]:  # Skip first line "FittedParams:"
                lines.append(line)
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Scikit-learn style repr showing algorithm parameters."""
        params = self.init_params.algorithm_params
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"

    def get_state_info(self) -> Dict[str, Any]:
        """
        Get complete state information including params and metadata.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all generator state, parameters, and metadata.
        """
        info = {
            "name": self.name,
            "class": self.__class__.__name__,
            "is_preprocessed": self.state.is_preprocessed,
            "is_fitted": self.state.is_fitted,
            "fit_timestamp": self.state.fit_timestamp,
            "init_params": self.init_params.to_dict(),
        }

        if self.is_preprocessed:
            if hasattr(self, "_sites"):
                info["n_sites"] = len(self._sites)
                info["sites"] = self._sites

        if self.is_fitted and self.fitted_params_:
            info["fitted_params"] = self.fitted_params_.to_dict()

        return info

    def _create_output_index(
        self,
        n_timesteps: int,
        freq: str = "MS",
        start_date: Optional[pd.Timestamp] = None,
    ) -> pd.DatetimeIndex:
        """
        Create DatetimeIndex for generated synthetic data.

        Parameters
        ----------
        n_timesteps : int
            Number of timesteps to generate.
        freq : str, default='MS'
            Pandas frequency string ('MS' for month start, 'D' for daily).
        start_date : pd.Timestamp, optional
            Start date for index. If None, continues from last observed date.

        Returns
        -------
        pd.DatetimeIndex
            Index for synthetic data.
        """
        if start_date is None and hasattr(self, "_Q_obs"):
            # Start after last observed date
            start_date = self._Q_obs.index[-1] + pd.Timedelta(days=1)
        elif start_date is None:
            # Default to arbitrary start date
            start_date = pd.Timestamp("2000-01-01")

        return pd.date_range(start=start_date, periods=n_timesteps, freq=freq)

    def _format_output(
        self, data: np.ndarray, dates: pd.DatetimeIndex, n_realizations: int = 1
    ) -> pd.DataFrame:
        """
        Format output array as DataFrame with proper structure.

        Parameters
        ----------
        data : np.ndarray
            Generated flow data. Shape: (n_timesteps * n_realizations, n_sites)
            or (n_timesteps, n_sites) if n_realizations=1.
        dates : pd.DatetimeIndex
            DatetimeIndex for the data.
        n_realizations : int, default=1
            Number of realizations in the data.

        Returns
        -------
        pd.DataFrame
            Formatted output with DatetimeIndex (single realization) or
            MultiIndex [realization, date] (multiple realizations).
        """
        if n_realizations == 1:
            # Single realization: simple DatetimeIndex
            return pd.DataFrame(data, index=dates, columns=self.sites)
        else:
            # Multiple realizations: MultiIndex
            idx = pd.MultiIndex.from_product(
                [range(n_realizations), dates], names=["realization", "date"]
            )
            return pd.DataFrame(
                data.reshape(-1, len(self.sites)), index=idx, columns=self.sites
            )

    def save(self, filepath: str) -> None:
        """
        Save fitted generator to file using pickle.

        Parameters
        ----------
        filepath : str
            Path to save the generator.

        Raises
        ------
        ValueError
            If generator is not fitted.
        """
        import pickle

        if not self.is_fitted:
            raise ValueError("Cannot save unfitted generator. Run fit() first.")

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        self.logger.info(f"Generator saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "Generator":
        """
        Load fitted generator from file.

        Parameters
        ----------
        filepath : str
            Path to saved generator file.

        Returns
        -------
        Generator
            Loaded generator instance.
        """
        import pickle

        with open(filepath, "rb") as f:
            generator = pickle.load(f)

        generator.logger.info(f"Generator loaded from {filepath}")
        return generator

    def _store_obs_data(
        self,
        Q_obs: Union[pd.Series, pd.DataFrame],
        sites: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Validate, store, and filter observed data. Called at the start of
        ``preprocessing()`` in every subclass.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed historical flow data.
        sites : list of str, optional
            Sites to keep. If None, uses all columns.

        Returns
        -------
        pd.DataFrame
            Validated and filtered DataFrame ready for preprocessing.
        """
        self._Q_obs_raw = Q_obs
        Q = self.validate_input_data(Q_obs)
        if sites is not None:
            missing = set(sites) - set(Q.columns)
            if missing:
                raise ValueError(f"Sites not found in data: {missing}")
            Q = Q[sites]
        if not self.supports_multisite and Q.shape[1] > 1:
            raise ValueError(
                f"{self.__class__.__name__} is univariate and does not support "
                f"multi-site data ({Q.shape[1]} columns provided). "
                f"Pass a single-column DataFrame, or use a multi-site generator "
                f"such as KirschGenerator, MatalasGenerator, or "
                f"MultiSiteHMMGenerator."
            )
        self._sites = Q.columns.tolist()
        return Q

    @abstractmethod
    def preprocessing(
        self,
        Q_obs: Union[pd.Series, pd.DataFrame],
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Preprocess and validate observed flow data.

        Implementations should:
        1. Call ``_store_obs_data(Q_obs, sites)`` to validate and store data
        2. Perform generator-specific data preparation
        3. Call ``update_state(preprocessed=True)`` at end

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed historical flow data.
        sites : list of str, optional
            Sites to use. If None, uses all columns.
        **kwargs : Any
            Additional preprocessing parameters.
        """
        pass

    @abstractmethod
    def fit(
        self,
        Q_obs: Optional[Union[pd.Series, pd.DataFrame]] = None,
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the generator to observed flow data.

        If ``Q_obs`` is provided, ``preprocessing()`` is called automatically.
        If omitted, a prior call to ``preprocessing()`` is required.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            Observed data. If provided, runs preprocessing automatically.
        sites : list of str, optional
            Sites to use (only when Q_obs is provided).
        **kwargs : Any
            Additional fitting parameters.
        """
        pass

    @abstractmethod
    def generate(
        self,
        n_realizations: int = 1,
        n_years: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Ensemble":
        """
        Generate synthetic streamflow realizations.

        Implementations should:
        1. Call validate_fit() at start
        2. Set random seed if provided
        3. Generate synthetic flows
        4. Return Ensemble object containing all realizations

        Parameters
        ----------
        n_realizations : int, default=1
            Number of synthetic realizations to generate.
        n_years : int, optional
            Number of years to generate (alternative to n_timesteps).
        n_timesteps : int, optional
            Number of timesteps to generate explicitly.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        Ensemble
            Generated synthetic flows as an Ensemble object.
        """
        pass


@dataclass
class DisaggregatorState:
    """Track disaggregator preprocessing and fitting state."""

    is_preprocessed: bool = False
    is_fitted: bool = False
    fit_timestamp: Optional[str] = None


@dataclass
class DisaggregatorParams:
    """
    Store initialization/configuration parameters for disaggregators.

    These are user-specified settings that control algorithm behavior,
    not learned from data.
    """

    # Common parameters across all disaggregators
    random_seed: Optional[int] = None
    verbose: bool = False
    debug: bool = False

    # Flexible storage for disaggregator-specific parameters
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    computational_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary."""
        result = {
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "debug": self.debug,
        }
        result.update(self.algorithm_params)
        result.update(self.computational_params)
        return result

    def __repr__(self) -> str:
        """Readable string representation."""
        lines = ["DisaggregatorParams:"]
        if self.random_seed is not None:
            lines.append(f"  random_seed: {self.random_seed}")
        if self.verbose:
            lines.append(f"  verbose: {self.verbose}")
        if self.debug:
            lines.append(f"  debug: {self.debug}")

        if self.algorithm_params:
            lines.append("  Algorithm parameters:")
            for key, val in self.algorithm_params.items():
                lines.append(f"    {key}: {val}")

        if self.computational_params:
            lines.append("  Computational parameters:")
            for key, val in self.computational_params.items():
                lines.append(f"    {key}: {val}")

        return "\n".join(lines)


class Disaggregator(ABC):
    """
    Abstract base class for all temporal disaggregation methods.

    Disaggregators transform synthetic flows from one temporal resolution
    to a finer resolution (e.g., monthly to daily).

    All disaggregator implementations should inherit from this class.
    """

    @abstractmethod
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the disaggregator with algorithm configuration.

        Subclasses add algorithm-specific keyword-only parameters before
        ``name`` and ``debug``. Data is not passed here — use ``fit(Q_obs)``
        or ``preprocessing(Q_obs)`` instead.

        Parameters
        ----------
        name : str, optional
            Name identifier for this disaggregator instance.
        debug : bool, default False
            Enable debug logging.
        """
        self.name = name or self.__class__.__name__
        self.debug = debug

        self.state = DisaggregatorState()

        # Initialize parameter storage
        self.init_params = DisaggregatorParams(debug=debug)
        self.fitted_params_ = None  # Set during _compute_fitted_params()

        # Setup logging
        self._setup_logging(debug)

    def _setup_logging(self, debug: bool) -> None:
        """Setup logging infrastructure."""
        self.logger = logging.getLogger(f"synhydro.{self.name}")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def validate_input_data(self, data: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Validate and standardize input data format.

        Checks type, DatetimeIndex, NaN content, and negative values.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Input time series data

        Returns
        -------
        pd.DataFrame
            Validated and standardized data

        Raises
        ------
        ValueError
            If data format is invalid
        TypeError
            If data type is unsupported
        """
        # Type checking
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f"Input data must be pandas Series or DataFrame, got {type(data)}"
            )

        # Convert Series to DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=data.name or "flow")

        # Empty data
        if len(data) == 0:
            raise ValueError("Input data is empty.")

        # Index validation
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(
                    f"Data index must be a DatetimeIndex, got {type(data.index)}: {e}"
                )

        # NaN check
        nan_counts = data.isna().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            pct = total_nans / data.size * 100
            nan_sites = nan_counts[nan_counts > 0]
            self.logger.warning(
                "Input data contains %d NaN values (%.1f%%). "
                "Sites with NaNs: %s. NaN rows will propagate through "
                "fitting and may cause errors or poor results.",
                total_nans,
                pct,
                dict(nan_sites),
            )

        # Negative value check
        numeric_data = data.select_dtypes(include="number")
        if (numeric_data < 0).any().any():
            neg_sites = [
                col for col in numeric_data.columns if (numeric_data[col] < 0).any()
            ]
            self.logger.warning(
                "Input data contains negative values in sites: %s. "
                "Disaggregators using log transforms will fail on negative values.",
                neg_sites,
            )

        self.logger.info(
            "Validated data: %d timesteps, %d sites, period %s to %s",
            data.shape[0],
            data.shape[1],
            data.index[0],
            data.index[-1],
        )

        return data

    def validate_preprocessing(self) -> None:
        """
        Check if preprocessing has been completed.

        Raises
        ------
        ValueError
            If preprocessing() has not been run.
        """
        if not self.state.is_preprocessed:
            raise ValueError(
                f"{self.name} must run preprocessing() before fit() or disaggregate()"
            )

    def validate_fit(self) -> None:
        """
        Check if disaggregator has been fitted.

        Raises
        ------
        ValueError
            If fit() has not been run.
        """
        if not self.state.is_fitted:
            raise ValueError(f"{self.name} must run fit() before disaggregate()")

    def update_state(
        self, preprocessed: Optional[bool] = None, fitted: Optional[bool] = None
    ) -> None:
        """
        Update disaggregator state flags.

        Parameters
        ----------
        preprocessed : bool, optional
            Set preprocessing state.
        fitted : bool, optional
            Set fitted state.
        """
        if preprocessed is not None:
            self.state.is_preprocessed = preprocessed
            self.logger.debug(f"State updated: is_preprocessed={preprocessed}")

        if fitted is not None:
            self.state.is_fitted = fitted
            if fitted:
                self.state.fit_timestamp = datetime.now().isoformat()
            self.logger.debug(f"State updated: is_fitted={fitted}")

    @property
    def is_fitted(self) -> bool:
        """Check if disaggregator is fitted."""
        return self.state.is_fitted

    @property
    def is_preprocessed(self) -> bool:
        """Check if preprocessing is complete."""
        return self.state.is_preprocessed

    @property
    def n_sites(self) -> int:
        """
        Number of sites in the disaggregator.

        Returns
        -------
        int
            Number of sites.

        Raises
        ------
        ValueError
            If preprocessing not yet run.
        """
        if not hasattr(self, "_sites"):
            raise ValueError("Run preprocessing() first to access n_sites")
        return len(self._sites)

    @property
    def sites(self) -> List[str]:
        """
        List of site names.

        Returns
        -------
        List[str]
            Site identifiers.

        Raises
        ------
        ValueError
            If preprocessing not yet run.
        """
        if not hasattr(self, "_sites"):
            raise ValueError("Run preprocessing() first to access sites")
        return self._sites

    @property
    @abstractmethod
    def input_frequency(self) -> str:
        """
        Expected temporal frequency of input ensemble.

        Returns
        -------
        str
            Pandas frequency string (e.g., 'MS' for monthly, 'W' for weekly).
        """
        pass

    @property
    @abstractmethod
    def output_frequency(self) -> str:
        """
        Temporal frequency of disaggregated output.

        Returns
        -------
        str
            Pandas frequency string (e.g., 'D' for daily, 'H' for hourly).
        """
        pass

    def validate_input_ensemble(self, ensemble: "Ensemble") -> None:
        """
        Validate that input ensemble is compatible with disaggregator.

        Checks temporal frequency and site consistency.

        Parameters
        ----------
        ensemble : Ensemble
            Input ensemble to validate

        Raises
        ------
        ValueError
            If ensemble is incompatible with disaggregator
        """
        from .ensemble import Ensemble

        # Type check
        if not isinstance(ensemble, Ensemble):
            raise TypeError(f"Input must be an Ensemble object, got {type(ensemble)}")

        # Check temporal frequency
        if ensemble.frequency != self.input_frequency:
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

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get initialization parameters (scikit-learn style).

        Returns only constructor/configuration parameters, not fitted values.

        Parameters
        ----------
        deep : bool, default=True
            If True, return deep copy of parameters.

        Returns
        -------
        Dict[str, Any]
            Dictionary of initialization parameters.
        """
        return self.init_params.to_dict()

    def get_fitted_params(self) -> Dict[str, Any]:
        """
        Get parameters learned from data during fit().

        Returns
        -------
        Dict[str, Any]
            Dictionary of fitted parameters (all keys end with underscore).

        Raises
        ------
        ValueError
            If disaggregator has not been fitted yet.
        """
        if self.fitted_params_ is None:
            raise ValueError(f"{self.name} has not been fitted yet. Run fit() first.")
        return self.fitted_params_.to_dict()

    @abstractmethod
    def _compute_fitted_params(self) -> FittedParams:
        """
        Extract and package fitted parameters after fit().

        Must be implemented by each disaggregator subclass to specify
        what parameters were learned during fitting.

        Returns
        -------
        FittedParams
            Dataclass containing all fitted parameters.
        """
        pass

    def summary(self, show_fitted: bool = True) -> str:
        """
        Generate comprehensive summary of disaggregator configuration and fit.

        Parameters
        ----------
        show_fitted : bool, default=True
            Whether to include fitted parameters in summary.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"{self.name} Summary".center(80))
        lines.append("=" * 80)
        lines.append("")

        # Model Information
        lines.append("Model Information")
        lines.append("-" * 80)
        lines.append(f"Disaggregator Type:      {self.__class__.__name__}")
        lines.append(
            f"Status:                  {'Fitted' if self.is_fitted else 'Not Fitted'}"
        )
        lines.append(f"Input Frequency:         {self.input_frequency}")
        lines.append(f"Output Frequency:        {self.output_frequency}")

        if self.is_fitted and self.state.fit_timestamp:
            lines.append(f"Fitted:                  {self.state.fit_timestamp}")

        if hasattr(self, "_sites") and self._sites:
            lines.append(f"Number of Sites:         {len(self._sites)}")
            if len(self._sites) <= 5:
                lines.append(f"Sites:                   {self._sites}")
            else:
                lines.append(
                    f"Sites:                   {self._sites[:3]} ... ({len(self._sites)} total)"
                )

        lines.append("")

        # Initialization Parameters
        lines.append("Initialization Parameters")
        lines.append("-" * 80)
        params_str = str(self.init_params)
        for line in params_str.split("\n")[1:]:  # Skip first line
            lines.append(line)
        lines.append("")

        # Fitted Parameters
        if show_fitted and self.is_fitted and self.fitted_params_ is not None:
            lines.append("Fitted Parameters")
            lines.append("-" * 80)
            fitted_str = str(self.fitted_params_)
            for line in fitted_str.split("\n")[1:]:  # Skip first line
                lines.append(line)
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Scikit-learn style repr showing algorithm parameters."""
        params = self.init_params.algorithm_params
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"

    def save(self, filepath: str) -> None:
        """
        Save fitted disaggregator to file using pickle.

        Parameters
        ----------
        filepath : str
            Path to save the disaggregator.

        Raises
        ------
        ValueError
            If disaggregator is not fitted.
        """
        import pickle

        if not self.is_fitted:
            raise ValueError("Cannot save unfitted disaggregator. Run fit() first.")

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        self.logger.info(f"Disaggregator saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "Disaggregator":
        """
        Load fitted disaggregator from file.

        Parameters
        ----------
        filepath : str
            Path to saved disaggregator file.

        Returns
        -------
        Disaggregator
            Loaded disaggregator instance.
        """
        import pickle

        with open(filepath, "rb") as f:
            disaggregator = pickle.load(f)

        disaggregator.logger.info(f"Disaggregator loaded from {filepath}")
        return disaggregator

    def _store_obs_data(
        self,
        Q_obs: Union[pd.Series, pd.DataFrame],
        sites: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Validate, store, and filter observed data.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed historical flow data.
        sites : list of str, optional
            Sites to keep. If None, uses all columns.

        Returns
        -------
        pd.DataFrame
            Validated and filtered DataFrame.
        """
        self._Q_obs_raw = Q_obs
        Q = self.validate_input_data(Q_obs)
        if sites is not None:
            missing = set(sites) - set(Q.columns)
            if missing:
                raise ValueError(f"Sites not found in data: {missing}")
            Q = Q[sites]
        self._sites = Q.columns.tolist()
        return Q

    @abstractmethod
    def preprocessing(
        self,
        Q_obs: Union[pd.Series, pd.DataFrame],
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Preprocess and validate observed flow data.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed historical flow data at the output resolution.
        sites : list of str, optional
            Sites to use. If None, uses all columns.
        **kwargs : Any
            Additional preprocessing parameters.
        """
        pass

    @abstractmethod
    def fit(
        self,
        Q_obs: Optional[Union[pd.Series, pd.DataFrame]] = None,
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the disaggregator to observed flow data.

        If ``Q_obs`` is provided, ``preprocessing()`` is called automatically.
        If omitted, a prior call to ``preprocessing()`` is required.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            Observed data. If provided, runs preprocessing automatically.
        sites : list of str, optional
            Sites to use (only when Q_obs is provided).
        **kwargs : Any
            Additional fitting parameters.
        """
        pass

    @abstractmethod
    def disaggregate(self, ensemble: "Ensemble", **kwargs: Any) -> "Ensemble":
        """
        Disaggregate synthetic flows from coarser to finer temporal resolution.

        Implementations should:
        1. Call validate_fit() at start
        2. Call validate_input_ensemble() to check compatibility
        3. Disaggregate each realization in the ensemble
        4. Return new Ensemble with finer temporal resolution

        Parameters
        ----------
        ensemble : Ensemble
            Input ensemble at coarser temporal resolution
        **kwargs : Any
            Additional disaggregation parameters.

        Returns
        -------
        Ensemble
            Disaggregated ensemble at finer temporal resolution
        """
        pass
