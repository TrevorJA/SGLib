"""
Multi-Site Hidden Markov Model Generator (Gold et al. 2024)

Implements a Gaussian HMM (one multivariate Gaussian emission per state,
i.e. n_mix = 1) for generating synthetic multi-site streamflow that preserves
both temporal dependencies (via hidden states) and spatial correlations (via
multivariate emissions with full covariance matrices).

Only the annual multi-site HMM of Gold et al. (2024) is implemented. The
paper's subsequent KNN spatial and annual-to-monthly disaggregation step is
out of scope for this generator; use NowakDisaggregator for temporal
disaggregation of the annual output.

Based on the methodology from:
Gold, D.F., Gupta, R.S., and Reed, P.M. (2024). Exploring the spatially
compounding multi-sectoral drought vulnerabilities in Colorado's West Slope
river basins. Earth's Future. https://doi.org/10.1029/2024EF004841

References
----------
Gold, D.F., Gupta, R.S., and Reed, P.M. (2024). Exploring the spatially
compounding multi-sectoral drought vulnerabilities in Colorado's West Slope
river basins. Earth's Future. https://doi.org/10.1029/2024EF004841
"""

import logging
import warnings
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from hmmlearn import hmm

from synhydro.core.base import Generator, FittedParams, make_output_index
from synhydro.core.ensemble import Ensemble, EnsembleMetadata

logger = logging.getLogger(__name__)


class MultiSiteHMMGenerator(Generator):
    """
    Multi-site Hidden Markov Model generator for synthetic streamflow.

    Generates synthetic streamflow using a Gaussian HMM (a single multivariate
    Gaussian emission per state) that models temporal dependencies through hidden states and spatial correlations
    through multivariate Gaussian emissions with state-specific covariance matrices.

    The method is particularly suited for capturing drought dynamics across
    multiple sites/basins simultaneously.

    Parameters
    ----------
    n_states : int, default=2
        Number of hidden states. Default is 2 (dry/wet states).
    offset : float, default=1.0
        Small value added before log transformation to handle zeros.
        Recommended: 1.0 for flows in standard units.
    max_iterations : int, default=1000
        Maximum EM iterations per HMM fit. If EM has not converged when this
        limit is reached a ``UserWarning`` is emitted.
    n_init : int, default=1
        Number of random EM restarts. Each restart fits the HMM from a
        different random initialization and the fit with the highest
        log-likelihood is retained. The default of 1 preserves the original
        single-fit behaviour; values of 5-10 are recommended in practice
        because EM frequently converges to local optima.
    covariance_type : str, default='full'
        Type of covariance matrix:
        - 'full': Full covariance matrix per state (captures all correlations)
        - 'diag': Diagonal covariance per state (independent sites)
        - 'spherical': Single variance per state for all dimensions
        - 'tied': One full covariance matrix shared by all states
    name : str, optional
        Name identifier for this generator instance.
    debug : bool, default=False
        Enable debug logging.

    Attributes
    ----------
    means_ : np.ndarray
        State means for each site. Shape: (n_states, n_sites).
    covariances_ : np.ndarray
        Covariance matrices for each state. Shape: (n_states, n_sites, n_sites).
    transition_matrix_ : np.ndarray
        State transition probability matrix. Shape: (n_states, n_states).
    stationary_distribution_ : np.ndarray
        Stationary distribution of states. Shape: (n_states,).
    Q_log_ : np.ndarray
        Log-transformed observed flows used for fitting.
    log_likelihood_ : float
        Log-likelihood of the retained (best) fit.
    log_likelihoods_ : list of float
        Log-likelihood of every restart, in restart order. Failed restarts
        are recorded as ``nan``.
    converged_ : bool
        Whether EM converged for the retained fit within ``max_iterations``.

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.methods.generation.parametric import MultiSiteHMMGenerator
    >>>
    >>> # Load multi-site annual flows
    >>> Q_annual = pd.read_csv('annual_flows.csv', index_col=0, parse_dates=True)
    >>>
    >>> # Initialize generator
    >>> gen = MultiSiteHMMGenerator(n_states=2, n_init=10)
    >>> gen.preprocessing(Q_annual)
    >>> gen.fit(random_state=42)
    >>>
    >>> # Generate 100 realizations of 50 years each
    >>> ensemble = gen.generate(n_realizations=100, n_years=50, seed=42)

    Notes
    -----
    - Annual timestep data only (``supported_frequencies = ("YS",)``)
    - Log transformation ensures positive emissions
    - Full covariance preserves spatial correlations between sites
    - State ordering: states sorted by mean (low mean = dry state)
    - EM can converge to local optima; use ``n_init > 1`` and inspect
      ``log_likelihoods_`` to check that restarts agree
    """

    supports_multisite = True
    supported_frequencies = ("YS",)

    _COVARIANCE_TYPES = ("full", "diag", "spherical", "tied")

    # Restarts whose log-likelihoods differ by more than this many nats are
    # treated as distinct local optima.
    _LOGLIK_SPREAD_TOL = 1.0

    def __init__(
        self,
        *,
        n_states: int = 2,
        offset: float = 1.0,
        max_iterations: int = 1000,
        n_init: int = 1,
        covariance_type: str = "full",
        name: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ):
        """Initialize the MultiSiteHMMGenerator."""
        super().__init__(name=name, debug=debug)

        # Validate parameters
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")

        if offset <= 0:
            raise ValueError(f"offset must be positive, got {offset}")

        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

        if n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}")

        if covariance_type not in self._COVARIANCE_TYPES:
            raise ValueError(
                f"covariance_type must be one of {self._COVARIANCE_TYPES}, "
                f"got '{covariance_type}'"
            )

        self.n_states = n_states
        self.offset = offset
        self.max_iterations = max_iterations
        self.n_init = n_init
        self.covariance_type = covariance_type

        # Store initialization parameters
        self.init_params.algorithm_params = {
            "method": "Multi-Site Hidden Markov Model (Gold et al. 2024)",
            "n_states": n_states,
            "offset": offset,
            "max_iterations": max_iterations,
            "n_init": n_init,
            "covariance_type": covariance_type,
        }

        # Initialize fitted parameter storage
        self.means_ = None
        self.covariances_ = None
        self.transition_matrix_ = None
        self.stationary_distribution_ = None
        self.Q_log_ = None
        self.log_likelihood_ = None
        self.log_likelihoods_ = None
        self.converged_ = None
        self._hmm_model = None

    @property
    def output_frequency(self) -> str:
        """
        Output frequency matches input frequency.

        Typically used for annual data ('YS'), but flexible. Anchored
        annual aliases inferred by pandas (e.g. 'YS-JAN', 'AS-JAN') are
        normalized to 'YS' and monthly aliases to 'MS' so that the value
        matches the canonical ``input_frequency`` of disaggregators.
        """
        if hasattr(self, "_Q_obs") and self._Q_obs is not None:
            # Infer from preprocessed data
            freq = pd.infer_freq(self._Q_obs.index)
            if freq is None:
                return "YS"
            if freq[0] in ("Y", "A"):
                return "YS"
            if freq[0] == "M":
                return "MS"
            return freq
        return "YS"  # Default to annual start

    def preprocessing(
        self, Q_obs, *, sites: Optional[List[str]] = None, **kwargs
    ) -> None:
        """
        Preprocess observed data for HMM fitting.

        Applies offset and log transformation to handle zeros and ensure
        positive values for fitting.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed streamflow data with DatetimeIndex.
        sites : List[str], optional
            Subset of sites to use. If None, uses all columns.
        **kwargs : dict
            Additional preprocessing parameters (currently unused).

        Raises
        ------
        ValueError
            If data has fewer than 2 sites for multi-site modeling.
        """
        Q = self._store_obs_data(Q_obs, sites=sites)

        # Validate minimum sites for multi-site HMM
        if len(self._sites) < 2:
            self.logger.warning(
                "Multi-site HMM with only 1 site. Consider using univariate HMM."
            )

        # Store original observed data
        self._Q_obs = Q.copy()

        # Apply offset and log transformation
        Q_adj = Q + self.offset
        with np.errstate(invalid="ignore"):
            self.Q_log_ = np.log(Q_adj).values

        # Check for invalid values
        if not np.all(np.isfinite(self.Q_log_)):
            raise ValueError(
                "Log-transformed data contains non-finite values. "
                "Check for negative flows or adjust offset parameter."
            )

        self.logger.info(
            f"Preprocessing complete: {len(Q)} observations, "
            f"{len(self._sites)} sites, offset={self.offset}"
        )

        # Update state
        self.update_state(preprocessed=True)

    def fit(self, Q_obs=None, *, sites=None, **kwargs) -> None:
        """
        Fit the multi-site HMM to observed data.

        Estimates transition probabilities, state-specific means, and
        covariance matrices with the Baum-Welch (EM) algorithm via hmmlearn's
        ``GaussianHMM``.

        Runs ``n_init`` EM restarts from different random initializations and
        retains the fit with the highest log-likelihood. Restart seeds are
        derived deterministically from ``random_state`` so that the same
        ``random_state`` always yields the same fit.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            Observed streamflow data. If provided, preprocessing is called
            automatically.
        sites : list of str, optional
            Sites to use (only when Q_obs is provided).
        **kwargs : dict
            Additional fitting parameters. May include ``random_state`` (int
            or None) for reproducible fitting.

        Warns
        -----
        UserWarning
            If EM did not converge within ``max_iterations`` for the retained
            fit, or if the restarts converged to different local optima
            (log-likelihood spread greater than 1 nat).

        Raises
        ------
        RuntimeError
            If every restart fails.

        Notes
        -----
        States are automatically ordered by mean (ascending), so state 0
        represents the dry state and higher-numbered states represent
        progressively wetter states.
        """
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)
        self.validate_preprocessing()

        random_state = kwargs.pop("random_state", None)

        self.logger.debug(
            f"Fitting GaussianHMM with {self.n_states} states, "
            f"covariance_type='{self.covariance_type}', n_init={self.n_init}"
        )

        # Derive one hmmlearn seed per restart from the user-supplied seed
        seed_rng = np.random.default_rng(random_state)
        restart_seeds = seed_rng.integers(0, 2**31 - 1, size=self.n_init)

        best_model = None
        best_score = -np.inf
        log_likelihoods = []

        for i, restart_seed in enumerate(restart_seeds):
            model = hmm.GaussianHMM(
                n_components=self.n_states,
                n_iter=self.max_iterations,
                covariance_type=self.covariance_type,
                random_state=int(restart_seed),
            )
            try:
                model.fit(self.Q_log_)
                score = float(model.score(self.Q_log_))
            except (ValueError, np.linalg.LinAlgError) as e:
                self.logger.debug(f"HMM restart {i} failed: {e}")
                log_likelihoods.append(np.nan)
                continue

            log_likelihoods.append(score)
            if score > best_score:
                best_score = score
                best_model = model

        if best_model is None:
            raise RuntimeError(
                "All HMM restarts failed. "
                "Try reducing n_states or increasing the record length."
            )

        self._hmm_model = best_model
        self.log_likelihood_ = best_score
        self.log_likelihoods_ = log_likelihoods
        self.converged_ = self._em_converged(best_model.monitor_)

        if not self.converged_:
            warnings.warn(
                f"HMM EM did not converge within max_iterations="
                f"{self.max_iterations} (final log-likelihood {best_score:.3f}). "
                "Consider increasing max_iterations.",
                UserWarning,
                stacklevel=2,
            )

        finite_ll = [ll for ll in log_likelihoods if np.isfinite(ll)]
        if len(finite_ll) > 1:
            spread = max(finite_ll) - min(finite_ll)
            if spread > self._LOGLIK_SPREAD_TOL:
                warnings.warn(
                    f"HMM EM restarts converged to different local optima "
                    f"(log-likelihood range {min(finite_ll):.3f} to "
                    f"{max(finite_ll):.3f}). The best fit was retained; "
                    f"consider increasing n_init (currently {self.n_init}).",
                    UserWarning,
                    stacklevel=2,
                )

        # Extract parameters
        means = np.array(best_model.means_)  # Shape: (n_states, n_sites)
        transition_matrix = np.array(best_model.transmat_)
        covariances = self._expand_covariances(best_model.covars_)

        # Order states by mean of first site (dry to wet)
        mean_order = np.argsort(means[:, 0])

        self.means_ = means[mean_order]
        self.covariances_ = covariances[mean_order]
        self.transition_matrix_ = transition_matrix[mean_order, :][:, mean_order]

        # Compute stationary distribution
        self.stationary_distribution_ = self._compute_stationary_distribution()

        self.logger.info(
            f"Fitting complete: {self.n_states} states, "
            f"transition matrix:\n{self.transition_matrix_}"
        )

        # Compute fitted params
        self.fitted_params_ = self._compute_fitted_params()

        # Update state
        self.update_state(fitted=True)

    @staticmethod
    def _em_converged(monitor) -> bool:
        """
        Whether EM converged by the log-likelihood tolerance criterion.

        hmmlearn's ``ConvergenceMonitor.converged`` also returns True when
        the iteration budget is exhausted, so it cannot distinguish genuine
        convergence from hitting ``n_iter``. This checks the tolerance
        criterion only.
        """
        history = list(monitor.history)
        return len(history) >= 2 and (history[-1] - history[-2]) < monitor.tol

    def _expand_covariances(self, covars_raw) -> np.ndarray:
        """
        Expand hmmlearn covariances to full (n_states, n_sites, n_sites) arrays.

        hmmlearn's ``GaussianHMM.covars_`` property returns full matrices for
        every covariance type, but the compact shapes are handled explicitly
        as a safeguard against version differences.
        """
        n_sites = len(self._sites)
        covars_raw = np.asarray(covars_raw)

        if covars_raw.shape == (self.n_states, n_sites, n_sites):
            return covars_raw.copy()
        if self.covariance_type == "diag":  # (n_states, n_sites)
            return np.array([np.diag(c) for c in covars_raw])
        if self.covariance_type == "spherical":  # (n_states,)
            return np.array([c * np.eye(n_sites) for c in covars_raw])
        if self.covariance_type == "tied":  # (n_sites, n_sites)
            return np.array([covars_raw.copy() for _ in range(self.n_states)])
        raise ValueError(
            f"Unexpected covariance shape {covars_raw.shape} for "
            f"covariance_type='{self.covariance_type}'"
        )

    def _count_covariance_parameters(self) -> int:
        """Number of free covariance parameters for the chosen covariance_type."""
        S = len(self._sites)
        K = self.n_states
        if self.covariance_type == "full":
            return K * S * (S + 1) // 2
        if self.covariance_type == "diag":
            return K * S
        if self.covariance_type == "spherical":
            return K
        return S * (S + 1) // 2  # tied: one full matrix shared by all states

    def _compute_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution from transition matrix.

        Returns
        -------
        np.ndarray
            Stationary probabilities for each state.
        """
        # Find eigenvector corresponding to eigenvalue = 1
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix_.T)

        # Find index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvals - 1.0))

        # Extract and normalize eigenvector
        pi = np.real(eigenvecs[:, idx])
        pi = pi / pi.sum()

        return pi

    def generate(
        self,
        n_realizations: int = 1,
        n_years: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Ensemble:
        """
        Generate synthetic streamflow realizations.

        Parameters
        ----------
        n_realizations : int, default=1
            Number of synthetic realizations to generate.
        n_years : int, optional
            Number of years to generate. If provided with annual data,
            this equals n_timesteps.
        n_timesteps : int, optional
            Number of timesteps to generate explicitly. Takes precedence
            over n_years if both provided.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional generation parameters (currently unused).

        Returns
        -------
        Ensemble
            Generated synthetic flows as an Ensemble object.

        Raises
        ------
        ValueError
            If neither n_years nor n_timesteps is provided.
        """
        self.validate_fit()

        # Determine number of timesteps
        if n_timesteps is not None:
            n_steps = n_timesteps
        elif n_years is not None:
            n_steps = n_years
        else:
            raise ValueError("Must provide either n_years or n_timesteps")

        # Create random number generator
        rng = np.random.default_rng(seed)

        self.logger.debug(
            f"Generating {n_realizations} realizations of {n_steps} timesteps"
        )

        realizations = {}

        for r in range(n_realizations):
            # Generate state trajectory
            states = self._generate_state_trajectory(n_steps, rng=rng)

            # Generate emissions for each timestep
            Q_log_syn = np.zeros((n_steps, len(self._sites)))
            for t, state in enumerate(states):
                Q_log_syn[t, :] = rng.multivariate_normal(
                    self.means_[state], self.covariances_[state]
                )

            # Back-transform from log space
            Q_syn = np.exp(Q_log_syn) - self.offset

            # Ensure non-negative flows
            Q_syn = np.maximum(Q_syn, 0.0)

            # Create DataFrame with appropriate index
            start_date = self._Q_obs.index[0]
            dates = make_output_index(start_date, n_steps, self.output_frequency)

            realizations[r] = pd.DataFrame(Q_syn, index=dates, columns=self._sites)

        self.logger.info(f"Generated {n_realizations} realizations")

        first = realizations[0]
        metadata = EnsembleMetadata(
            generator_class=self.__class__.__name__,
            n_realizations=n_realizations,
            n_sites=len(self._sites),
            time_resolution=self.output_frequency,
            time_period=(str(first.index[0].date()), str(first.index[-1].date())),
        )
        return Ensemble(realizations, metadata=metadata)

    def _generate_state_trajectory(
        self, n_timesteps: int, *, rng: np.random.Generator
    ) -> List[int]:
        """
        Generate hidden state trajectory.

        Parameters
        ----------
        n_timesteps : int
            Number of timesteps in trajectory.
        rng : np.random.Generator
            Random number generator instance.

        Returns
        -------
        List[int]
            Sequence of hidden states.
        """
        # Sample initial state from stationary distribution
        state = rng.choice(self.n_states, p=self.stationary_distribution_)

        states = [state]

        # Generate remaining states using transition matrix
        for _ in range(1, n_timesteps):
            state = rng.choice(self.n_states, p=self.transition_matrix_[state, :])
            states.append(state)

        return states

    def _compute_fitted_params(self) -> FittedParams:
        """Extract and package fitted parameters."""
        # Count free parameters: means + covariances + transition rows
        n_params = (
            self.n_states * len(self._sites)
            + self._count_covariance_parameters()
            + self.n_states * (self.n_states - 1)
        )

        training_period = (
            str(self._Q_obs.index[0].date()),
            str(self._Q_obs.index[-1].date()),
        )

        return FittedParams(
            means_=self.means_,
            correlations_={
                "covariance_matrices": self.covariances_,
                "transition_matrix": self.transition_matrix_,
                "stationary_distribution": self.stationary_distribution_,
            },
            distributions_={
                "type": "Multivariate Gaussian per state",
                "n_states": self.n_states,
                "covariance_type": self.covariance_type,
            },
            transformations_={"log_transform": True, "offset": self.offset},
            n_parameters_=n_params,
            sample_size_=len(self._Q_obs),
            n_sites_=len(self._sites),
            training_period_=training_period,
        )
