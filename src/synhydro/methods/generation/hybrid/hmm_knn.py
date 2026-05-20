"""
HMM-KNN Generator

Generates synthetic annual multisite streamflow by combining Hidden Markov Model
state sequencing with K-Nearest Neighbor bootstrapped resampling. The HMM learns
K hidden hydrologic regimes from log-annual flows and drives a sequence of regime
transitions. Within each regime-transition category, a KNN search conditioned on
the previous year's log-flows selects a historical analog year, and the full
multisite annual flow vector for that year is resampled directly.

This implementation is not attributable to a single primary publication. It
combines two independent developments:

* The non-parametric category (NPC) framework of Prairie et al. (2008), which
  conditions a KNN bootstrap on a discrete regime-transition category. Prairie
  (2008) used a non-homogeneous Markov chain (NHM) with kernel-smoothed
  transition probabilities, not a hidden Markov model.

* The hidden-Markov regime-state representation introduced for hydrologic
  applications by Akintug and Rasmussen (2005) and applied at multi-site
  annual scale by Gold et al. (2024).

The HMM substitutes for Prairie's NHM as the upper-layer regime sequencer; the
lower-layer KNN bootstrap follows Prairie (2008) directly, with the rank-based
kernel weights of Lall and Sharma (1996).

References
----------
Prairie, J., Rajagopalan, B., Lall, U., and Fulp, T. (2008). A stochastic
nonparametric approach for streamflow generation combining observational and
paleoreconstructed data. Water Resources Research, 44, W06423.
https://doi.org/10.1029/2007WR006684

Lall, U., and Sharma, A. (1996). A nearest neighbor bootstrap for resampling
hydrologic time series. Water Resources Research, 32(3), 679-693.
https://doi.org/10.1029/95WR02966

Akintug, B., and Rasmussen, P.F. (2005). A Markov switching model for annual
hydrologic time series. Water Resources Research, 41(9).
https://doi.org/10.1029/2004WR003605

Gold, D.F., Reed, P.M., and Gupta, R.S. (2024). Exploring the spatially
compounding multi-sectoral drought vulnerabilities in Colorado's West Slope
river basins. Earth's Future. https://doi.org/10.1029/2024EF004841
"""

import logging
import warnings
from math import ceil
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from hmmlearn import hmm

from synhydro.core.base import FittedParams, Generator
from synhydro.core.ensemble import Ensemble

logger = logging.getLogger(__name__)


class HMMKNNGenerator(Generator):
    """
    HMM-KNN generator for synthetic annual multisite streamflow.

    Combines a Gaussian Hidden Markov Model for regime sequencing with
    K-Nearest Neighbor bootstrapping for within-regime resampling. Regime
    transitions are governed by a learned Markov transition matrix. For each
    synthetic year the generator identifies the regime-transition category
    (previous state, current state), searches the historical record for analog
    years within that category using normalized log-flow distances, and resamples
    the full multisite flow vector from one of the K nearest analogs.

    Parameters
    ----------
    n_states : int, default=2
        Number of hidden hydrologic regimes. State 0 is the driest regime.
    delta : float, default=1.0
        Additive offset applied before log transformation to handle near-zero
        flows. Must be positive.
    covariance_type : str, default='full'
        Covariance structure for the Gaussian HMM emissions. One of 'full',
        'diag', or 'spherical'. 'full' preserves all inter-site correlations
        within each state.
    n_init : int, default=10
        Number of random initializations for HMM fitting. The fit with the
        highest log-likelihood is retained.
    name : str, optional
        Name identifier for this generator instance.
    debug : bool, default=False
        Enable debug-level logging.

    Attributes
    ----------
    transition_matrix_ : np.ndarray of shape (n_states, n_states)
        Learned HMM transition probability matrix.
    stationary_distribution_ : np.ndarray of shape (n_states,)
        Stationary distribution of the Markov chain.
    state_sequence_ : np.ndarray of shape (N,)
        Viterbi state assignment for each year of the historical record.
    log_std_ : np.ndarray of shape (n_sites,)
        Per-site standard deviation of historical log-flows, used to normalize
        distances in KNN search.
    Q_log_ : np.ndarray of shape (N, n_sites)
        Log-transformed historical flows.

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.methods.generation.hybrid import HMMKNNGenerator
    >>>
    >>> Q_annual = pd.read_csv('annual_flows.csv', index_col=0, parse_dates=True)
    >>>
    >>> gen = HMMKNNGenerator(n_states=2)
    >>> gen.fit(Q_annual)
    >>>
    >>> ensemble = gen.generate(n_realizations=100, n_years=50, seed=42)
    """

    supports_multisite = True
    supported_frequencies = ("YS",)

    def __init__(
        self,
        *,
        n_states: int = 2,
        delta: float = 1.0,
        covariance_type: str = "full",
        n_init: int = 10,
        name: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the HMMKNNGenerator."""
        super().__init__(name=name, debug=debug)

        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        if covariance_type not in ("full", "diag", "spherical"):
            raise ValueError(
                f"covariance_type must be 'full', 'diag', or 'spherical', "
                f"got '{covariance_type}'"
            )
        if n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}")

        self.n_states = n_states
        self.delta = delta
        self.covariance_type = covariance_type
        self.n_init = n_init

        self.init_params.algorithm_params = {
            "method": "HMM-KNN (HMM upper layer; Prairie et al. 2008 KNN bootstrap)",
            "n_states": n_states,
            "delta": delta,
            "covariance_type": covariance_type,
            "n_init": n_init,
        }

        # Fitted attributes set during fit()
        self.transition_matrix_ = None
        self.stationary_distribution_ = None
        self.state_sequence_ = None
        self.log_std_ = None
        self.Q_log_ = None
        self._Q_obs = None
        self._hmm_model = None
        self._n_sites = None

        # KNN pool data structures built during fit()
        # _state_pools[s] = array of historical year indices assigned to state s
        # _category_pools[(s_prev, s_curr)] = array of historical year indices
        #   for years where (state[i-1], state[i]) == (s_prev, s_curr), i >= 1
        self._state_pools: Dict[int, np.ndarray] = {}
        self._category_pools: Dict[tuple, np.ndarray] = {}

    @property
    def output_frequency(self) -> str:
        """
        Temporal frequency of generated output.

        Returns
        -------
        str
            Always 'YS' (annual start).
        """
        return "YS"

    def preprocessing(
        self,
        Q_obs: Union[pd.Series, pd.DataFrame],
        *,
        sites: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Preprocess observed annual flow data for HMM-KNN fitting.

        Applies the log transform Y = log(Q + delta) and stores the result.
        The preprocessed data are used by fit().

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed annual streamflow with DatetimeIndex.
        sites : list of str, optional
            Subset of sites to use. If None, all columns are used.
        **kwargs : dict
            Additional preprocessing parameters (unused).

        Raises
        ------
        ValueError
            If log-transformed data contain non-finite values.
        """
        Q = self._store_obs_data(Q_obs, sites=sites)
        self._Q_obs = Q.copy()
        self._n_sites = len(self._sites)

        Q_adj = Q.values + self.delta
        with np.errstate(invalid="ignore"):
            self.Q_log_ = np.log(Q_adj)

        if not np.all(np.isfinite(self.Q_log_)):
            raise ValueError(
                "Log-transformed data contain non-finite values. "
                "Check for negative flows or increase the delta parameter."
            )

        self.logger.info(
            "Preprocessing complete: %d years, %d sites, delta=%.2f",
            len(Q),
            self._n_sites,
            self.delta,
        )
        self.update_state(preprocessed=True)

    def fit(
        self,
        Q_obs: Optional[Union[pd.Series, pd.DataFrame]] = None,
        *,
        sites: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Fit the HMM-KNN model to observed annual flow data.

        Runs n_init random HMM initializations and retains the fit with the
        highest log-likelihood. Decodes the historical state sequence via the
        Viterbi algorithm, reorders states by ascending mean log-flow at the
        first site (state 0 = driest), and builds KNN pool index structures.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            Observed annual streamflow. If provided, preprocessing() is called
            automatically. If None, preprocessing() must have been called first.
        sites : list of str, optional
            Sites to use (only applied when Q_obs is provided).
        **kwargs : dict
            Additional fitting parameters (unused).
        """
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)
        self.validate_preprocessing()

        self.logger.debug(
            "Fitting GaussianHMM: n_states=%d, covariance_type='%s', n_init=%d",
            self.n_states,
            self.covariance_type,
            self.n_init,
        )

        best_model = None
        best_score = -np.inf

        for i in range(self.n_init):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=self.covariance_type,
                    n_iter=1000,
                    random_state=i,
                )
                try:
                    model.fit(self.Q_log_)
                    score = model.score(self.Q_log_)
                    if score > best_score:
                        best_score = score
                        best_model = model
                except Exception:
                    self.logger.debug("HMM init %d failed, skipping.", i)

        if best_model is None:
            raise RuntimeError(
                "All HMM initializations failed. "
                "Try reducing n_states or increasing the record length."
            )

        self._hmm_model = best_model

        # Extract and reorder states by ascending mean of first site
        means_raw = best_model.means_  # (n_states, n_sites)
        trans_raw = best_model.transmat_  # (n_states, n_states)
        order = np.argsort(means_raw[:, 0])
        inv_order = np.argsort(order)

        self.transition_matrix_ = trans_raw[order, :][:, order]

        # Decode Viterbi state sequence and remap to reordered indices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, raw_states = best_model.decode(self.Q_log_, algorithm="viterbi")
        self.state_sequence_ = inv_order[raw_states]

        # Compute stationary distribution
        self.stationary_distribution_ = self._compute_stationary_distribution()

        # Normalize log-flows by historical std for KNN distances
        self.log_std_ = self.Q_log_.std(axis=0)
        self.log_std_ = np.where(self.log_std_ == 0, 1.0, self.log_std_)

        # Build KNN pool index structures
        self._build_pools()

        self.fitted_params_ = self._compute_fitted_params()
        self.update_state(fitted=True)

        self.logger.info(
            "Fitting complete: %d states, best log-likelihood=%.4f, "
            "transition matrix:\n%s",
            self.n_states,
            best_score,
            self.transition_matrix_,
        )

    def _compute_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution of the Markov chain.

        Returns
        -------
        np.ndarray of shape (n_states,)
            Normalized stationary probabilities.
        """
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix_.T)
        idx = np.argmin(np.abs(eigenvals - 1.0))
        pi = np.real(eigenvecs[:, idx])
        pi = pi / pi.sum()
        return pi

    def _build_pools(self) -> None:
        """
        Build state and category pool index structures for KNN resampling.

        _state_pools[s] contains indices into the historical record for all
        years assigned to state s.

        _category_pools[(s_prev, s_curr)] contains indices i (i >= 1) into
        the historical record where (state_sequence_[i-1], state_sequence_[i])
        equals (s_prev, s_curr). The index i is the year whose flow is returned
        as the synthetic value; year i-1 provides the feature vector for distance
        computation.
        """
        N = len(self.state_sequence_)

        # State pools: every year by its state assignment
        for s in range(self.n_states):
            self._state_pools[s] = np.where(self.state_sequence_ == s)[0]

        # Category pools: years i in 1..N-1 keyed by (state[i-1], state[i])
        for i in range(1, N):
            s_prev = int(self.state_sequence_[i - 1])
            s_curr = int(self.state_sequence_[i])
            key = (s_prev, s_curr)
            if key not in self._category_pools:
                self._category_pools[key] = []
            self._category_pools[key].append(i)

        # Convert lists to arrays
        for key in self._category_pools:
            self._category_pools[key] = np.array(self._category_pools[key], dtype=int)

        self.logger.debug(
            "Built %d state pools and %d category pools.",
            len(self._state_pools),
            len(self._category_pools),
        )

    @staticmethod
    def _lall_sharma_weights(k: int) -> np.ndarray:
        """
        Compute Lall-Sharma kernel weights for k neighbors.

        Parameters
        ----------
        k : int
            Number of neighbors (k >= 1).

        Returns
        -------
        np.ndarray of shape (k,)
            Probability weights summing to 1, with rank 1 (nearest) receiving
            the highest weight.
        """
        ranks = np.arange(1, k + 1, dtype=float)
        raw = 1.0 / ranks
        return raw / raw.sum()

    def generate(
        self,
        n_realizations: int = 1,
        n_years: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Ensemble:
        """
        Generate synthetic annual streamflow realizations.

        Parameters
        ----------
        n_realizations : int, default=1
            Number of independent synthetic sequences to generate.
        n_years : int, optional
            Number of years per realization. If None and n_timesteps is also
            None, defaults to the length of the historical record.
        n_timesteps : int, optional
            Explicit number of timesteps. Takes precedence over n_years.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional generation parameters (unused).

        Returns
        -------
        Ensemble
            Generated synthetic flows.

        Raises
        ------
        ValueError
            If the generator has not been fitted.
        """
        self.validate_fit()

        if n_timesteps is not None:
            n_steps = n_timesteps
        elif n_years is not None:
            n_steps = n_years
        else:
            n_steps = len(self._Q_obs)

        rng = np.random.default_rng(seed)

        self.logger.debug(
            "Generating %d realizations of %d years", n_realizations, n_steps
        )

        realizations = {}
        for r in range(n_realizations):
            realizations[r] = self._generate_single_realization(n_steps, rng=rng)

        self.logger.info("Generated %d realizations.", n_realizations)
        return Ensemble(realizations)

    def _generate_single_realization(
        self, n_timesteps: int, rng: Optional[np.random.Generator] = None
    ) -> pd.DataFrame:
        """
        Generate a single synthetic annual flow realization.

        Parameters
        ----------
        n_timesteps : int
            Number of annual timesteps to generate.
        rng : np.random.Generator, optional
            Random number generator. If None, a default generator is created.

        Returns
        -------
        pd.DataFrame
            Synthetic annual flows with DatetimeIndex and site columns.
            Shape (n_timesteps, n_sites).
        """
        if rng is None:
            rng = np.random.default_rng()

        Q_obs_values = self._Q_obs.values  # (N, n_sites)
        N = len(Q_obs_values)

        Q_syn = np.zeros((n_timesteps, self._n_sites))
        states = np.empty(n_timesteps, dtype=int)

        # -- Year t=0: draw initial state, sample uniformly from state pool --
        s0 = int(rng.choice(self.n_states, p=self.stationary_distribution_))
        states[0] = s0
        pool0 = self._state_pools[s0]
        if len(pool0) == 0:
            # Fallback: sample from any year (should be extremely rare)
            j0 = int(rng.integers(0, N))
        else:
            j0 = int(rng.choice(pool0))
        Q_syn[0] = Q_obs_values[j0]

        # -- Years t=1..n_timesteps-1 --
        for t in range(1, n_timesteps):
            # Draw next state
            s_prev = int(states[t - 1])
            s_curr = int(
                rng.choice(self.n_states, p=self.transition_matrix_[s_prev, :])
            )
            states[t] = s_curr

            # Identify pool for category (s_prev, s_curr)
            cat_key = (s_prev, s_curr)
            if (
                cat_key in self._category_pools
                and len(self._category_pools[cat_key]) > 0
            ):
                pool = self._category_pools[cat_key]
            else:
                # Fallback: use state-only pool for s_curr
                self.logger.debug(
                    "Category (%d, %d) has no historical members; "
                    "falling back to state %d pool.",
                    s_prev,
                    s_curr,
                    s_curr,
                )
                pool = self._state_pools[s_curr]
                if len(pool) == 0:
                    # Ultimate fallback: sample uniformly from all years
                    j = int(rng.integers(0, N))
                    Q_syn[t] = Q_obs_values[j]
                    continue

            n_pool = len(pool)
            k = max(1, ceil(np.sqrt(n_pool)))

            # Feature vectors: log-flow of the year BEFORE each pool member
            # pool[i] is a historical year index; pool[i]-1 is its predecessor
            # For pool members at year 0 (no predecessor), we skip distance
            # conditioning and sample uniformly -- but since category pools
            # only include indices >= 1, this is not an issue.
            prev_indices = pool - 1  # indices of predecessor years
            # Clip to valid range (should already be >= 0 by construction)
            prev_indices = np.clip(prev_indices, 0, N - 1)

            features_hist = self.Q_log_[prev_indices]  # (n_pool, n_sites)

            # Normalized Euclidean distance from previous synthetic log-flow
            y_prev_syn = np.log(np.maximum(Q_syn[t - 1], 0.0) + self.delta)
            diffs = (y_prev_syn - features_hist) / self.log_std_
            distances = np.sqrt((diffs**2).sum(axis=1))  # (n_pool,)

            # Select k nearest and apply Lall-Sharma weights
            if k >= n_pool:
                # All pool members are neighbors
                sorted_idx = np.argsort(distances)
                k_actual = n_pool
            else:
                sorted_idx = np.argpartition(distances, k)[:k]
                sorted_idx = sorted_idx[np.argsort(distances[sorted_idx])]
                k_actual = k

            weights = self._lall_sharma_weights(k_actual)
            chosen_local = int(rng.choice(k_actual, p=weights))
            j_star = pool[sorted_idx[chosen_local]]

            Q_syn[t] = Q_obs_values[j_star]

        # Enforce non-negativity
        Q_syn = np.maximum(Q_syn, 0.0)

        start_date = self._Q_obs.index[0]
        dates = pd.date_range(start=start_date, periods=n_timesteps, freq="YS")

        return pd.DataFrame(Q_syn, index=dates, columns=self._sites)

    def _compute_fitted_params(self) -> FittedParams:
        """
        Package fitted parameters into a FittedParams dataclass.

        Returns
        -------
        FittedParams
            Dataclass containing transition matrix, stationary distribution,
            state sequence, and metadata.
        """
        training_period = (
            str(self._Q_obs.index[0].date()),
            str(self._Q_obs.index[-1].date()),
        )

        # Parameter count: transition matrix (off-diagonal) + means + state sequence
        n_params = self.n_states * (self.n_states - 1) + self.n_states * self._n_sites

        return FittedParams(
            correlations_={
                "transition_matrix": self.transition_matrix_,
                "stationary_distribution": self.stationary_distribution_,
            },
            distributions_={
                "type": "Nonparametric KNN resampling conditioned on HMM state categories",
                "n_states": self.n_states,
                "covariance_type": self.covariance_type,
            },
            transformations_={"log_transform": True, "delta": self.delta},
            fitted_models_={"hmm": self._hmm_model},
            n_parameters_=n_params,
            sample_size_=len(self._Q_obs),
            n_sites_=self._n_sites,
            training_period_=training_period,
        )
