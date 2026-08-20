"""
Phase Randomization Generator for synthetic streamflow generation.

Implements the Brunner et al. (2019) phase randomization method for generating
synthetic streamflow time series that preserve both short- and long-range
temporal dependence using Fourier transform phase randomization.

References
----------
Brunner, M.I., Bardossy, A., and Furrer, R. (2019). Technical note: Stochastic
simulation of streamflow time series using phase randomization. Hydrology and
Earth System Sciences, 23, 3175-3187. https://doi.org/10.5194/hess-23-3175-2019
"""

import logging
import warnings
from typing import Optional, Dict, Any, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import norm as _norm

from synhydro.core.base import Generator, FittedParams, make_output_index
from synhydro.core.ensemble import Ensemble

logger = logging.getLogger(__name__)


class PhaseRandomizationGenerator(Generator):
    """
    Phase randomization generator for synthetic streamflow using Brunner et al. (2019).

    Generates synthetic daily streamflow time series using Fourier transform
    phase randomization combined with the four-parameter kappa distribution.
    The method preserves both short- and long-range temporal dependence by
    conserving the power spectrum while randomizing phases.

    Attributes
    ----------
    par_day_ : dict
        Fitted kappa distribution parameters for each day of year (1-365).
        Each entry contains {'xi', 'alfa', 'k', 'h'}.
    modulus_ : np.ndarray
        Amplitude spectrum (modulus of FFT) from fitted data.
    phases_ : np.ndarray
        Phase spectrum from fitted data.
    norm_ : np.ndarray
        Normalized/deseasonalized data after normal score transform.

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.methods.generation.hybrid import PhaseRandomizationGenerator
    >>> Q_daily = pd.read_csv('daily_flows.csv', index_col=0, parse_dates=True)
    >>> gen = PhaseRandomizationGenerator(marginal='kappa')
    >>> gen.preprocessing(Q_daily)
    >>> gen.fit()
    >>> ensemble = gen.generate(n_realizations=100, seed=42)

    Notes
    -----
    - Requires at least 2 years (730 days) of daily data
    - February 29 observations are removed to ensure consistent 365-day years
    - The method generates series of the same length as the observed data
    """

    supports_multisite: bool = False
    supported_frequencies: tuple = ("D",)

    def __init__(
        self,
        *,
        marginal: str = "kappa",
        win_h_length: int = 15,
        name: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ):
        """Initialize the PhaseRandomizationGenerator.

        Parameters
        ----------
        marginal : str, default='kappa'
            Marginal distribution type for back-transformation:
            - 'kappa': Four-parameter kappa distribution (default, allows extrapolation)
            - 'empirical': Empirical distribution (no extrapolation beyond observed)
        win_h_length : int, default=15
            Half-window length for daily distribution fitting. Values within
            +-win_h_length days are used, giving a total window of 2*win_h_length+1 days.
        name : str, optional
            Name identifier for this generator instance.
        debug : bool, default=False
            Enable debug logging.
        **kwargs : dict, optional
            Additional parameters (currently unused).
        """
        super().__init__(name=name, debug=debug)

        # Validate marginal distribution type
        if marginal not in ("kappa", "empirical"):
            raise ValueError(
                f"marginal must be 'kappa' or 'empirical', got '{marginal}'"
            )

        self.marginal = marginal
        self.win_h_length = win_h_length

        # Store initialization parameters
        self.init_params.algorithm_params = {
            "marginal": marginal,
            "win_h_length": win_h_length,
        }

        # Initialize fitted parameter storage
        self.par_day_ = {}  # Kappa parameters per day of year
        self.norm_ = None  # Normalized data
        self.modulus_ = None  # FFT modulus
        self.phases_ = None  # FFT phases

    @property
    def output_frequency(self) -> str:
        """Phase randomization generates daily output."""
        return "D"

    def preprocessing(self, Q_obs, *, sites=None, **kwargs) -> None:
        """
        Preprocess observed data for phase randomization generation.

        Validates input data, removes leap days, and creates day-of-year index.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed daily streamflow data with DatetimeIndex.
        sites : list, optional
            Sites to keep. If None, uses all columns.
        **kwargs : dict
            Additional preprocessing parameters (currently unused).

        Raises
        ------
        ValueError
            If data has fewer than 730 days or has missing days.
        """
        # Validate, filter, and store observed data
        Q = self._store_obs_data(Q_obs, sites=sites)

        # Convert to Series for processing
        Q_series = Q.iloc[:, 0]

        # Remove February 29 (leap days)
        leap_mask = (Q_series.index.month == 2) & (Q_series.index.day == 29)
        n_leap_removed = leap_mask.sum()
        Q_series = Q_series[~leap_mask]

        if n_leap_removed > 0:
            self.logger.info(f"Removed {n_leap_removed} leap day observations")

        # Validate minimum data requirement (at least 2 years)
        if len(Q_series) < 730:
            raise ValueError(
                f"At least 730 days (2 years) of data required, got {len(Q_series)}"
            )

        # Trim trailing incomplete year so the record is a multiple of 365.
        if len(Q_series) % 365 != 0:
            n_trim = len(Q_series) % 365
            n_before = len(Q_series)
            Q_series = Q_series.iloc[:-n_trim]
            self.logger.warning(
                "Record length (%d days) is not a multiple of 365 after removing "
                "leap days. Trimmed %d trailing days to %d days (%d complete years).",
                n_before,
                n_trim,
                len(Q_series),
                len(Q_series) // 365,
            )

        # Create day-of-year index (1-365)
        # Using a consistent mapping that handles the Feb 29 removal
        self.day_index_ = self._create_day_index(Q_series.index)

        # Store preprocessed data
        self.Q_obs_ = Q_series.values
        self.Q_obs_index_ = Q_series.index
        self.n_years_ = len(Q_series) // 365

        # Update state
        self.update_state(preprocessed=True)
        self.logger.info(
            f"Preprocessing complete: {len(Q_series)} days ({self.n_years_} years)"
        )

    def _create_day_index(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Create day-of-year index (1-365) accounting for leap year removal.

        Parameters
        ----------
        index : pd.DatetimeIndex
            DateTime index of the data (leap days already removed).

        Returns
        -------
        np.ndarray
            Array of day-of-year values (1-365).
        """
        # Get day of year, but adjust for dates after Feb 28 in leap years
        # Since Feb 29 is removed, we need consistent 1-365 mapping
        day_of_year = index.dayofyear.values.copy()

        # For leap years, days after Feb 29 have dayofyear > 59 (Feb 28)
        # but since Feb 29 is removed, we need to subtract 1 for those days
        is_leap_year = index.is_leap_year
        after_feb28 = day_of_year > 59

        # Adjust: in leap years, days after Feb 28 should be shifted down by 1
        # because Feb 29 was removed
        day_of_year[is_leap_year & after_feb28] -= 1

        return day_of_year

    def fit(self, Q_obs=None, *, sites=None, **kwargs) -> None:
        """
        Fit the phase randomization model to observed data.

        This method:
        1. Fits kappa distribution parameters for each day of year (if marginal='kappa')
        2. Applies normal score transform per day of year
        3. Computes FFT and extracts modulus/phases

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            If provided, calls preprocessing automatically.
        sites : list, optional
            Sites to keep. Passed to preprocessing if Q_obs is provided.
        **kwargs : dict
            Additional fitting parameters (currently unused).
        """
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)
        self.validate_preprocessing()

        # Step 1: Fit daily kappa distributions (if using kappa marginal)
        if self.marginal == "kappa":
            self._fit_kappa_distributions()

        # Step 2: Normal score transform
        self._apply_normal_score_transform()

        # Step 3: Fourier transform
        self._compute_fft()

        # Update state
        self.update_state(fitted=True)

        # Compute fitted params
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(
            f"Fitting complete: {len(self.Q_obs_)} observations, "
            f"marginal={self.marginal}"
        )

    def _fit_kappa_distributions(self) -> None:
        """Fit kappa distribution for each day of year using L-moments."""
        self.logger.debug("Fitting kappa distributions for each day of year...")

        for d in range(1, 366):
            # Get window days (circular indexing)
            window_days = self._get_window_days(d)

            # Get values for window days across all years
            mask = np.isin(self.day_index_, window_days)
            data_window = self.Q_obs_[mask]

            # Compute L-moments
            try:
                lmom = self._compute_lmoments(data_window)

                # Fit kappa parameters
                kappa_params = self._fit_kappa_params(lmom)

                if kappa_params is not None:
                    self.par_day_[d] = kappa_params
                else:
                    # Use previous day's parameters if fitting failed
                    if d > 1 and d - 1 in self.par_day_:
                        self.par_day_[d] = self.par_day_[d - 1].copy()
                        self.logger.debug(f"Day {d}: Using previous day's parameters")
                    else:
                        self.par_day_[d] = None
                        self.logger.warning(f"Day {d}: Kappa fitting failed")

            except Exception as e:
                # Use previous day's parameters if fitting failed
                if d > 1 and d - 1 in self.par_day_:
                    self.par_day_[d] = self.par_day_[d - 1].copy()
                    self.logger.debug(
                        f"Day {d}: Using previous day's parameters due to: {e}"
                    )
                else:
                    self.par_day_[d] = None
                    self.logger.warning(f"Day {d}: Kappa fitting failed: {e}")

        # Handle any remaining None values by propagating from subsequent day
        for d in range(365, 0, -1):
            if self.par_day_.get(d) is None and d < 365:
                if self.par_day_.get(d + 1) is not None:
                    self.par_day_[d] = self.par_day_[d + 1].copy()

    def _get_window_days(self, day: int) -> list:
        """
        Get the window of days around a target day (circular indexing).

        Parameters
        ----------
        day : int
            Target day of year (1-365).

        Returns
        -------
        list
            List of day-of-year values in the window.
        """
        window_days = []
        for w in range(1, self.win_h_length + 1):
            # Days before
            before = (day - w - 1) % 365 + 1
            # Days after
            after = (day + w - 1) % 365 + 1
            window_days.extend([before, after])

        # Add the target day itself
        window_days.append(day)

        return list(set(window_days))

    def _compute_lmoments(self, x: np.ndarray) -> Dict[str, float]:
        """
        Compute L-moments from data.

        Implements the algorithm from PRSim/R/Lmoments.R.

        Parameters
        ----------
        x : np.ndarray
            Sample data.

        Returns
        -------
        dict
            Dictionary with keys 'l1', 'l2', 'lcv', 'lca' (tau3), 'lkur' (tau4).
        """
        x_sorted = np.sort(x)
        n = len(x_sorted)

        if n < 4:
            raise ValueError("Need at least 4 observations for L-moments")

        # Probability weighted moments
        pp = np.arange(n)
        nn = n - 1

        p1 = pp / nn
        p2 = p1 * (pp - 1) / np.maximum(nn - 1, 1)
        p3 = p2 * (pp - 2) / np.maximum(nn - 2, 1)

        b0 = np.mean(x_sorted)
        b1 = np.mean(p1 * x_sorted)
        b2 = np.mean(p2 * x_sorted)
        b3 = np.mean(p3 * x_sorted)

        l1 = b0
        l2 = 2 * b1 - b0

        if b0 == 0:
            lcv = 0
        else:
            lcv = 2 * b1 / b0 - 1

        denom = 2 * b1 - b0
        if denom == 0:
            lca = 0
            lkur = 0
        else:
            lca = 2 * (3 * b2 - b0) / denom - 3  # tau3 (L-skewness)
            lkur = 5 * (2 * (2 * b3 - 3 * b2) + b0) / denom + 6  # tau4 (L-kurtosis)

        return {"l1": l1, "l2": l2, "lcv": lcv, "lca": lca, "lkur": lkur}

    def _fit_kappa_params(self, lmom: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Fit kappa distribution parameters from L-moments.

        Implements the algorithm from PRSim/R/par.kappa.R.

        Parameters
        ----------
        lmom : dict
            L-moments dictionary with keys 'l1', 'l2', 'lca' (tau3), 'lkur' (tau4).

        Returns
        -------
        dict or None
            Dictionary with keys 'xi', 'alfa', 'k', 'h' or None if fitting fails.
        """
        lambda1 = lmom["l1"]
        lambda2 = lmom["l2"]
        tau3 = lmom["lca"]
        tau4 = lmom["lkur"]

        def compute_theoretical_tau(k, h):
            """Compute theoretical tau3 and tau4 from k and h."""
            if h == 0:
                # GEV case
                if k == 0:
                    k = 1e-100
                tau3_th = 2 * (1 - 3 ** (-k)) / (1 - 2 ** (-k)) - 3
                tau4_th = (
                    5 * (1 - 4 ** (-k)) - 10 * (1 - 3 ** (-k)) + 6 * (1 - 2 ** (-k))
                ) / (1 - 2 ** (-k))
            else:
                g = np.zeros(4)
                for r in range(1, 5):
                    try:
                        if h > 0:
                            g[r - 1] = (r * gamma(1 + k) * gamma(r / h)) / (
                                h ** (1 + k) * gamma(1 + k + r / h)
                            )
                        else:
                            g[r - 1] = (r * gamma(1 + k) * gamma(-k - r / h)) / (
                                (-h) ** (1 + k) * gamma(1 - r / h)
                            )
                    except (ValueError, ZeroDivisionError):
                        return None, None

                if g[0] - g[1] == 0:
                    return None, None

                tau3_th = (-g[0] + 3 * g[1] - 2 * g[2]) / (g[0] - g[1])
                tau4_th = -(-g[0] + 6 * g[1] - 10 * g[2] + 5 * g[3]) / (g[0] - g[1])

            return tau3_th, tau4_th

        def objective(kh):
            k, h = kh
            # Check validity constraints
            if (k < -1 and h >= 0) or (h < 0 and (k <= -1 or k >= -1 / h)):
                return 1e10

            tau3_th, tau4_th = compute_theoretical_tau(k, h)
            if tau3_th is None:
                return 1e10

            return (tau3 - tau3_th) ** 2 + (tau4 - tau4_th) ** 2

        # Optimize to find k and h
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    [1.0, 1.0],
                    method="Nelder-Mead",
                    options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-6},
                )

            if result.fun > 0.1:  # Poor fit
                return None

            k, h = result.x

            # Compute xi and alfa
            xi, alfa = self._compute_xi_alfa(lambda1, lambda2, k, h)

            if alfa is None or alfa <= 0:
                return None

            return {"xi": xi, "alfa": alfa, "k": k, "h": h}

        except Exception:
            return None

    def _compute_xi_alfa(
        self, lambda1: float, lambda2: float, k: float, h: float
    ) -> tuple:
        """
        Compute xi (location) and alfa (scale) parameters.

        Parameters
        ----------
        lambda1 : float
            First L-moment (mean).
        lambda2 : float
            Second L-moment (L-scale).
        k : float
            Shape parameter k.
        h : float
            Shape parameter h.

        Returns
        -------
        tuple
            (xi, alfa) or (None, None) if computation fails.
        """
        try:
            if h == 0:
                # GEV case
                if k == 0:
                    k = 1e-100
                alfa = (lambda2 * k) / ((1 - 2 ** (-k)) * gamma(1 + k))
                xi = lambda1 - alfa * (1 - gamma(1 + k)) / k
            else:
                g = np.zeros(2)
                for r in range(1, 3):
                    if h > 0:
                        g[r - 1] = (r * gamma(1 + k) * gamma(r / h)) / (
                            h ** (1 + k) * gamma(1 + k + r / h)
                        )
                    else:
                        g[r - 1] = (r * gamma(1 + k) * gamma(-k - r / h)) / (
                            (-h) ** (1 + k) * gamma(1 - r / h)
                        )

                if g[0] - g[1] == 0:
                    return None, None

                alfa = (lambda2 * k) / (g[0] - g[1])
                xi = lambda1 - alfa * (1 - g[0]) / k

            return xi, alfa

        except Exception:
            return None, None

    def _apply_normal_score_transform(self) -> None:
        """Apply normal score transform per day of year."""
        self.logger.debug("Applying normal score transform...")

        self.norm_ = np.zeros(len(self.Q_obs_))

        for d in range(1, 366):
            mask = self.day_index_ == d
            day_values = self.Q_obs_[mask]
            n = len(day_values)

            if n == 0:
                continue

            # Rank the values (1-based ranking)
            ranks = np.argsort(np.argsort(day_values)) + 1

            # Map ranks to deterministic Van der Waerden normal scores
            self.norm_[mask] = _norm.ppf(ranks / (n + 1.0))

    def _compute_fft(self) -> None:
        """Compute FFT and extract modulus and phases."""
        self.logger.debug("Computing FFT...")

        self.ft_ = np.fft.fft(self.norm_)
        self.modulus_ = np.abs(self.ft_)
        self.phases_ = np.angle(self.ft_)

        n = len(self.ft_)
        # First half: indices 1 to floor(n/2)
        self.first_part_ = np.arange(1, n // 2 + 1)
        # Second half: mirror indices
        self.second_part_ = n - self.first_part_

    def _compute_fitted_params(self) -> FittedParams:
        """Extract and package fitted parameters."""
        n_params = 0
        if self.marginal == "kappa":
            # 365 days x 4 parameters
            n_params = 365 * 4

        training_period = (
            str(self.Q_obs_index_[0].date()),
            str(self.Q_obs_index_[-1].date()),
        )

        return FittedParams(
            distributions_={
                "type": self.marginal,
                "kappa_params": self.par_day_ if self.marginal == "kappa" else None,
            },
            transformations_={"normal_score": True, "win_h_length": self.win_h_length},
            fitted_models_={
                "modulus_shape": (
                    self.modulus_.shape if self.modulus_ is not None else None
                )
            },
            n_parameters_=n_params,
            sample_size_=len(self.Q_obs_),
            n_sites_=1,
            training_period_=training_period,
        )

    def generate(
        self,
        n_realizations: int = 1,
        n_years: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Ensemble:
        """
        Generate synthetic streamflow realizations using phase randomization.

        Parameters
        ----------
        n_realizations : int, default=1
            Number of synthetic realizations to generate.
        n_years : int, optional
            Target length of each realization in years (365-day years, no leap
            days).  When provided, independent phase-randomized chunks are
            concatenated until the target length is reached, then trimmed.
            When None the output length equals the observed record length.
        n_timesteps : int, optional
            Not used.  Length is controlled via n_years.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional generation parameters (currently unused).

        Returns
        -------
        Ensemble
            Generated synthetic flows as an Ensemble object.
        """
        self.validate_fit()

        rng = np.random.default_rng(seed)

        n_obs = len(self.Q_obs_)
        if n_years is not None:
            n_target = int(n_years) * 365
        else:
            n_target = n_obs

        n_chunks = max(1, int(np.ceil(n_target / n_obs)))
        self.logger.info(
            "Generating %d realizations of %d days (%d chunks of %d each)",
            n_realizations,
            n_target,
            n_chunks,
            n_obs,
        )

        out_index = self._build_noleap_index(n_target)
        realizations = {}

        for r in range(n_realizations):
            segments = []
            for _ in range(n_chunks):
                ts_chunk = self._phase_randomize(rng=rng)
                segments.append(self._back_transform(ts_chunk, rng=rng))
            simulated = np.concatenate(segments)[:n_target]

            realizations[r] = pd.DataFrame(
                simulated, index=out_index, columns=self._sites
            )

        self.logger.info("Generated %d realizations", n_realizations)
        return Ensemble(realizations)

    @staticmethod
    def _build_noleap_index(n_days: int, start_year: int = 2000) -> pd.DatetimeIndex:
        """Build a no-leap daily DatetimeIndex of length n_days.

        Parameters
        ----------
        n_days : int
            Number of days required.
        start_year : int
            Calendar year for the first day of the index.

        Returns
        -------
        pd.DatetimeIndex
            Daily index with Feb 29 rows removed, of exactly n_days length.
        """
        # Generate enough calendar days to cover n_days after removing leap days.
        # Worst case: ~0.25 % of days are Feb 29, so pad by 1 % plus a fixed buffer.
        n_calendar = n_days + n_days // 365 + 10
        all_days = make_output_index(f"{start_year}-01-01", n_calendar, "D")
        noleap = all_days[~((all_days.month == 2) & (all_days.day == 29))]
        return noleap[:n_days]

    def _phase_randomize(self, rng=None) -> np.ndarray:
        """
        Generate phase-randomized time series.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator instance. If None, creates a new default generator.

        Returns
        -------
        np.ndarray
            Phase-randomized time series in normal domain.
        """
        if rng is None:
            rng = np.random.default_rng()
        n = len(self.ft_)

        # Generate random phases for first half
        random_phases = rng.uniform(-np.pi, np.pi, len(self.first_part_))

        # Construct new complex spectrum
        ft_new = np.zeros(n, dtype=complex)

        # Keep first element (DC component/mean) unchanged
        ft_new[0] = self.ft_[0]

        # First half: modulus with random phases
        ft_new[self.first_part_] = self.modulus_[self.first_part_] * np.exp(
            1j * random_phases
        )

        # Second half: conjugate symmetry for real output
        ft_new[self.second_part_] = np.conj(ft_new[self.first_part_])

        # Handle Nyquist frequency for even-length signals
        if n % 2 == 0:
            ft_new[n // 2] = self.modulus_[n // 2]  # Real-valued

        # Inverse FFT
        ts_new = np.real(np.fft.ifft(ft_new))

        return ts_new

    def _back_transform(self, ts_new: np.ndarray, rng=None) -> np.ndarray:
        """
        Back-transform from normal to original distribution.

        Parameters
        ----------
        ts_new : np.ndarray
            Phase-randomized series in normal domain.
        rng : np.random.Generator, optional
            Random number generator instance. If None, creates a new default generator.

        Returns
        -------
        np.ndarray
            Simulated streamflow in original units.
        """
        if rng is None:
            rng = np.random.default_rng()
        simulated = np.zeros(len(ts_new))

        for d in range(1, 366):
            mask = self.day_index_ == d
            day_values_new = ts_new[mask]
            n = len(day_values_new)

            if n == 0:
                continue

            if self.marginal == "kappa" and self.par_day_.get(d) is not None:
                # Generate kappa sample
                kappa_sample = self._rand_kappa(n, rng=rng, **self.par_day_[d])

                # Rank-based mapping
                new_ranks = np.argsort(np.argsort(day_values_new))
                sorted_kappa = np.sort(kappa_sample)
                simulated[mask] = sorted_kappa[new_ranks]

            else:
                # Use empirical distribution
                day_obs = self.Q_obs_[mask]
                new_ranks = np.argsort(np.argsort(day_values_new))
                sorted_obs = np.sort(day_obs)
                simulated[mask] = sorted_obs[new_ranks]

            # Handle negative values
            day_mask_global = np.where(mask)[0]
            neg_indices = day_mask_global[simulated[mask] < 0]

            if len(neg_indices) > 0:
                min_obs = self.Q_obs_[mask].min()
                if min_obs > 0:
                    rep_value = rng.uniform(0, min_obs, len(neg_indices))
                else:
                    rep_value = np.zeros(len(neg_indices))
                simulated[neg_indices] = rep_value

        return simulated

    def _rand_kappa(
        self, n: int, xi: float, alfa: float, k: float, h: float, rng=None
    ) -> np.ndarray:
        """
        Generate random samples from kappa distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        xi : float
            Location parameter.
        alfa : float
            Scale parameter.
        k : float
            Shape parameter k.
        h : float
            Shape parameter h.
        rng : np.random.Generator, optional
            Random number generator instance. If None, creates a new default generator.

        Returns
        -------
        np.ndarray
            Random samples from kappa distribution.
        """
        if rng is None:
            rng = np.random.default_rng()
        F = rng.uniform(1e-10, 1 - 1e-10, n)
        return self._invF_kappa(F, xi, alfa, k, h)

    def _invF_kappa(
        self, F: np.ndarray, xi: float, alfa: float, k: float, h: float
    ) -> np.ndarray:
        """
        Inverse CDF (quantile function) for kappa distribution.

        Parameters
        ----------
        F : np.ndarray
            Probability values (0 < F < 1).
        xi : float
            Location parameter.
        alfa : float
            Scale parameter.
        k : float
            Shape parameter k.
        h : float
            Shape parameter h.

        Returns
        -------
        np.ndarray
            Quantile values.
        """
        # Handle edge cases
        if k == 0:
            k = 1e-100

        if h == 0:
            # GEV case
            x = xi + alfa * (1 - (-np.log(F)) ** k) / k
        else:
            x = xi + (alfa / k) * (1 - ((1 - F**h) / h) ** k)

        return x
