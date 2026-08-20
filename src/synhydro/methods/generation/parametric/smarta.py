"""
SMARTA: Symmetric Moving Average (neaRly) To Anything.

Multisite stationary generator for annual streamflow with arbitrary marginal
distributions and any-range autocorrelation structure (SRD or LRD).

Reference:
    Tsoukalas, I., Makropoulos, C., & Koutsoyiannis, D. (2018). Simulation of
    stochastic processes exhibiting any-range dependence and arbitrary marginal
    distributions. Water Resources Research, 54(11), 9484-9513.
    https://doi.org/10.1029/2017WR022462
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, norm

from synhydro.core.base import FittedParams, Generator, make_output_index
from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.core.nataf import (
    cas_acf,
    fit_cas,
    hurst_acf,
    nataf_inverse,
    sma_weights_fft,
)
from synhydro.core.statistics import repair_correlation_matrix

logger = logging.getLogger(__name__)


def _bic(n: int, neg_loglik: float, k: int) -> float:
    """Bayesian Information Criterion."""
    return 2.0 * neg_loglik + k * np.log(n)


class SMARTAGenerator(Generator):
    """Symmetric Moving Average (neaRly) To Anything generator.

    Generates multisite stationary synthetic timeseries at annual resolution
    with arbitrary marginal distributions and any-range autocorrelation
    structure via the SMA model with Nataf ICDF mapping.

    Parameters
    ----------
    acf_model : str
        Autocorrelation model: ``"cas"`` (default), ``"hurst"``, or ``"custom"``.
    sma_order : int
        SMA truncation order q (default 512, should be power of 2).
    nataf_method : str
        Nataf evaluation method: ``"GH"`` (default), ``"MC"``, or ``"Int"``.
    nataf_n_eval : int
        Number of support points for Nataf polynomial fitting (default 9).
    nataf_poly_deg : int
        Polynomial degree for Nataf approximation (default 8).
    nataf_gh_nodes : int
        Gauss-Hermite quadrature nodes (default 21).
    marginal_method : str
        Marginal fitting method: ``"parametric"`` (default, gamma/lognorm BIC).
    matrix_repair_method : str
        Method for repairing non-PD matrices: ``"spectral"`` (default),
        ``"nearest"``, or ``"hypersphere"``.
    name : str, optional
        Generator name.
    debug : bool
        Enable debug logging (default False).
    """

    supports_multisite = True
    supported_frequencies = ("YS", "YE", "AS", "A-DEC", "MS")

    def __init__(
        self,
        *,
        acf_model: str = "cas",
        sma_order: int = 512,
        nataf_method: str = "GH",
        nataf_n_eval: int = 9,
        nataf_poly_deg: int = 8,
        nataf_gh_nodes: int = 21,
        marginal_method: str = "parametric",
        matrix_repair_method: str = "spectral",
        name: Optional[str] = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, debug=debug)
        self.acf_model = acf_model
        self.sma_order = sma_order
        self.nataf_method = nataf_method
        self.nataf_n_eval = nataf_n_eval
        self.nataf_poly_deg = nataf_poly_deg
        self.nataf_gh_nodes = nataf_gh_nodes
        self.marginal_method = marginal_method
        self.matrix_repair_method = matrix_repair_method

        # Fitted state
        self._Q_annual: Optional[pd.DataFrame] = None
        self._marginal_params: Dict[int, Dict[str, Any]] = {}
        self._icdfs: List = []
        self._target_acf: List[np.ndarray] = []
        self._equiv_acf: List[np.ndarray] = []
        self._sma_weights: List[np.ndarray] = []
        self._B_tilde: Optional[np.ndarray] = None
        self._cas_params: Dict[int, Tuple[float, float]] = {}

    @property
    def output_frequency(self) -> str:
        """Annual frequency."""
        return "YS"

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocessing(
        self,
        Q_obs: Optional[Union[pd.Series, pd.DataFrame]] = None,
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Validate and prepare annual data.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            Observed streamflow. If not provided, uses data from constructor.
        sites : list of str, optional
            Subset of site names to use.
        """
        Q = self._store_obs_data(Q_obs, sites)
        self._n_sites = len(self._sites)

        # Resample to annual if finer resolution
        inferred = pd.infer_freq(Q.index[: min(30, len(Q))])
        if (
            inferred is not None
            and not inferred.startswith("Y")
            and not inferred.startswith("A")
        ):
            self.logger.info("Resampling from %s to annual (sum)", inferred)
            Q = Q.resample("YS").sum()

        Q = Q.clip(lower=1e-6)
        self._Q_annual = Q

        self.logger.info(
            "Preprocessing complete: %d years, %d sites",
            len(Q),
            self._n_sites,
        )
        self.update_state(preprocessed=True)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        Q_obs: Optional[Union[pd.Series, pd.DataFrame]] = None,
        *,
        sites: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Fit the SMARTA model to observed data.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            If provided, calls preprocessing first.
        sites : list of str, optional
            Subset of site names.
        """
        if Q_obs is not None or not self.is_preprocessed:
            self.preprocessing(Q_obs, sites=sites)
        self.validate_preprocessing()

        Q = self._Q_annual
        n_years = len(Q)
        n_sites = self._n_sites
        q = self.sma_order

        if n_years < 20:
            self.logger.warning(
                "Short record (%d years). CAS parameter estimation may be unreliable.",
                n_years,
            )

        if n_sites * q > 10000:
            self.logger.warning(
                "Large problem size (n_sites=%d, sma_order=%d). "
                "Fitting may be slow. Consider reducing sma_order or using nataf_method='GH'.",
                n_sites,
                q,
            )

        # Step 1: Fit marginals per site
        self.logger.info("Step 1: Fitting marginal distributions...")
        self._marginal_params = {}
        self._icdfs = []
        for s_idx, site in enumerate(self._sites):
            vals = Q[site].values
            params, icdf = self._fit_marginal(vals, s_idx)
            self._marginal_params[s_idx] = params
            self._icdfs.append(icdf)

        # Step 2: Compute target ACF per site
        self.logger.info("Step 2: Computing target autocorrelation structures...")
        self._target_acf = []
        for s_idx, site in enumerate(self._sites):
            acf = self._compute_target_acf(Q[site].values, s_idx)
            self._target_acf.append(acf)

        # Step 3: Nataf inversion for autocorrelations
        self.logger.info("Step 3: Computing equivalent autocorrelations via Nataf...")
        nataf_kw = {}
        if self.nataf_method == "GH":
            nataf_kw["nodes"] = self.nataf_gh_nodes

        self._equiv_acf = []
        for s_idx in range(n_sites):
            target = self._target_acf[s_idx][1:]  # exclude lag 0
            equiv, _ = nataf_inverse(
                target,
                self._icdfs[s_idx],
                self._icdfs[s_idx],
                method=self.nataf_method,
                n_eval=self.nataf_n_eval,
                poly_deg=self.nataf_poly_deg,
                **nataf_kw,
            )
            self._equiv_acf.append(np.concatenate([[1.0], equiv]))

        # Step 4: SMA weights via FFT
        self.logger.info("Step 4: Computing SMA weights via FFT...")
        self._sma_weights = []
        for s_idx in range(n_sites):
            weights = sma_weights_fft(self._equiv_acf[s_idx])
            self._sma_weights.append(weights)

        # Step 5: Nataf inversion for cross-correlations + Cholesky
        if n_sites > 1:
            self.logger.info("Step 5: Computing equivalent cross-correlations...")
            # Empirical lag-0 cross-correlations
            data = Q.values  # (n_years, n_sites)
            corr_target = np.corrcoef(data.T)

            # Equivalent cross-correlations
            C_tilde = np.eye(n_sites)
            for i in range(n_sites):
                for j in range(i + 1, n_sites):
                    equiv_ij, _ = nataf_inverse(
                        np.atleast_1d(corr_target[i, j]),
                        self._icdfs[i],
                        self._icdfs[j],
                        method=self.nataf_method,
                        n_eval=self.nataf_n_eval,
                        poly_deg=self.nataf_poly_deg,
                        **nataf_kw,
                    )
                    C_tilde[i, j] = equiv_ij[0]
                    C_tilde[j, i] = equiv_ij[0]

            # Build G_tilde = C_tilde / SSumA
            SSumA = np.zeros((n_sites, n_sites))
            for i in range(n_sites):
                for j in range(n_sites):
                    SSumA[i, j] = np.sum(self._sma_weights[i] * self._sma_weights[j])

            G_tilde = C_tilde / SSumA

            # Cholesky decomposition
            try:
                self._B_tilde = np.linalg.cholesky(G_tilde)
            except np.linalg.LinAlgError:
                self.logger.warning(
                    "G_tilde matrix is not positive-definite. Attempting repair..."
                )
                G_repaired = self._repair_innovation_covariance(G_tilde)
                try:
                    self._B_tilde = np.linalg.cholesky(G_repaired)
                except np.linalg.LinAlgError:
                    raise ValueError(
                        "Innovation cross-correlation matrix G_tilde is not "
                        "positive-definite even after repair. The target "
                        "cross-correlations may be incompatible with the "
                        "marginal distributions and autocorrelation structures."
                    )
        else:
            self._B_tilde = np.array([[1.0]])

        self.logger.info("SMARTA model fitting complete.")
        self.update_state(fitted=True)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        n_realizations: int = 1,
        n_years: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Ensemble:
        """Generate synthetic annual timeseries.

        Parameters
        ----------
        n_realizations : int
            Number of realizations to generate (default 1).
        n_years : int, optional
            Number of years per realization. Defaults to observed length.
        n_timesteps : int, optional
            Alias for n_years at annual resolution.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Ensemble
            Generated synthetic data.
        """
        self.validate_fit()
        rng = np.random.default_rng(seed)

        if n_timesteps is not None:
            n_years = n_timesteps
        elif n_years is None:
            n_years = len(self._Q_annual)

        realizations: Dict[int, pd.DataFrame] = {}
        for r in range(n_realizations):
            df = self._generate_one(n_years, rng=rng)
            realizations[r] = df

        metadata = EnsembleMetadata(
            generator_class=self.name or self.__class__.__name__,
            n_realizations=n_realizations,
            n_sites=self._n_sites,
            description=f"SMARTA(q={self.sma_order}, acf={self.acf_model})",
        )
        return Ensemble(realizations, metadata=metadata)

    def _generate_one(self, n_years: int, *, rng: np.random.Generator) -> pd.DataFrame:
        """Generate a single realization.

        Following SimSMARTA.R from anySim.
        """
        q = self.sma_order
        n_sites = self._n_sites
        total_len = 2 * q + 1 + n_years

        # Draw i.i.d. standard normals
        V = rng.standard_normal((total_len, n_sites))

        # Cross-correlate innovations
        W = V @ self._B_tilde.T

        # SMA convolution per site
        Z = np.empty((n_years, n_sites))
        for s_idx in range(n_sites):
            aj = self._sma_weights[s_idx]
            w_site = W[:, s_idx]
            for t in range(n_years):
                end_idx = 2 * q + 1 + t
                segment = w_site[t:end_idx]
                Z[t, s_idx] = np.dot(aj, segment)

        # Map to target domain
        U = norm.cdf(Z)
        X = np.empty_like(Z)
        for s_idx in range(n_sites):
            u_clipped = np.clip(U[:, s_idx], 1e-10, 1.0 - 1e-10)
            X[:, s_idx] = self._icdfs[s_idx](u_clipped)

        # Build DataFrame
        dates = make_output_index(
            f"{int(self._Q_annual.index[0].year)}-01-01", n_years, "YS"
        )
        return pd.DataFrame(X, index=dates, columns=self._sites)

    def _repair_innovation_covariance(self, G: np.ndarray) -> np.ndarray:
        """Repair a non-PD innovation covariance while preserving its diagonal.

        ``G_tilde`` (Tsoukalas et al., 2018b, Eq. 27) is a covariance matrix
        whose diagonal ``1 / sum(a_j^2)`` fixes the variance of the SMA output
        ``Z``. Applying :func:`repair_correlation_matrix` directly would
        rescale that diagonal to one. Instead the matrix is split as
        ``G = D^(1/2) R D^(1/2)``, only the correlation part ``R`` is repaired
        with ``matrix_repair_method``, and the original diagonal ``D`` is
        restored, so ``diag(B B^T) == diag(G)`` exactly.

        Parameters
        ----------
        G : np.ndarray
            Symmetric innovation covariance matrix that failed Cholesky.

        Returns
        -------
        np.ndarray
            Positive-definite matrix with the same diagonal as ``G``.
        """
        d = np.sqrt(np.diag(G))
        R = G / np.outer(d, d)
        R_repaired = repair_correlation_matrix(R, method=self.matrix_repair_method)
        return R_repaired * np.outer(d, d)

    # ------------------------------------------------------------------
    # Fitted params
    # ------------------------------------------------------------------

    def _compute_fitted_params(self) -> FittedParams:
        """Compute fitted parameters summary."""
        training_period = (
            str(self._Q_annual.index[0].date()),
            str(self._Q_annual.index[-1].date()),
        )
        n = self._n_sites
        # Parameters: marginal (2 per site) + CAS (2 per site) + cross-corr (n*(n-1)/2)
        n_params = n * 4 + n * (n - 1) // 2

        return FittedParams(
            distributions_=self._marginal_params,
            transformations_={
                "acf_model": self.acf_model,
                "sma_order": self.sma_order,
                "cas_params": self._cas_params,
            },
            n_parameters_=n_params,
            sample_size_=len(self._Q_annual),
            n_sites_=self._n_sites,
            training_period_=training_period,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_marginal(
        self, values: np.ndarray, site_idx: int
    ) -> Tuple[Dict[str, Any], Any]:
        """Fit a marginal distribution to site data via BIC selection.

        Fits gamma and log-normal, selects the better model by BIC.

        Returns
        -------
        params : dict
            Distribution parameters.
        icdf : callable
            Quantile function (ICDF) for the fitted distribution.
        """
        n = len(values)
        vals = values[values > 0]

        best_bic = np.inf
        best_params = None
        best_icdf = None

        # Gamma fit
        try:
            shape, loc, scale = gamma_dist.fit(vals, floc=0)
            nll = -np.sum(gamma_dist.logpdf(vals, a=shape, loc=0, scale=scale))
            bic_val = _bic(n, nll, 2)
            if bic_val < best_bic:
                best_bic = bic_val
                best_params = {
                    "dist": "gamma",
                    "shape": shape,
                    "loc": 0.0,
                    "scale": scale,
                    "bic": bic_val,
                }
                best_icdf = lambda p, s=shape, sc=scale: gamma_dist.ppf(
                    p, a=s, loc=0, scale=sc
                )
        except Exception:
            self.logger.debug("Gamma fit failed for site %d", site_idx)

        # Log-normal fit
        try:
            s, loc, scale = lognorm.fit(vals, floc=0)
            nll = -np.sum(lognorm.logpdf(vals, s=s, loc=0, scale=scale))
            bic_val = _bic(n, nll, 2)
            if bic_val < best_bic:
                best_bic = bic_val
                best_params = {
                    "dist": "lognorm",
                    "s": s,
                    "loc": 0.0,
                    "scale": scale,
                    "bic": bic_val,
                }
                best_icdf = lambda p, s_=s, sc=scale: lognorm.ppf(
                    p, s=s_, loc=0, scale=sc
                )
        except Exception:
            self.logger.debug("Log-normal fit failed for site %d", site_idx)

        if best_params is None:
            raise ValueError(
                f"Could not fit any distribution to site {site_idx}. "
                "Data may be degenerate."
            )

        self.logger.debug(
            "Site %d: best marginal = %s (BIC=%.1f)",
            site_idx,
            best_params["dist"],
            best_bic,
        )
        return best_params, best_icdf

    def _compute_target_acf(self, values: np.ndarray, site_idx: int) -> np.ndarray:
        """Compute or specify the target autocorrelation function for a site.

        Returns
        -------
        np.ndarray
            ACF of length ``sma_order + 1``.
        """
        q = self.sma_order

        if self.acf_model == "cas":
            # Compute empirical ACF
            emp_acf = self._empirical_acf(values, max_lag=min(q, len(values) // 3))
            kappa, beta = fit_cas(emp_acf)
            self._cas_params[site_idx] = (kappa, beta)
            self.logger.debug(
                "Site %d CAS fit: kappa=%.4f, beta=%.4f", site_idx, kappa, beta
            )
            return cas_acf(kappa, beta, q)

        elif self.acf_model == "hurst":
            # Estimate Hurst coefficient from empirical ACF
            emp_acf = self._empirical_acf(values, max_lag=min(q, len(values) // 3))
            kappa, beta = fit_cas(emp_acf)
            if beta > 1:
                H = 1.0 - 1.0 / (2.0 * beta)
            else:
                H = 0.6
                self.logger.warning(
                    "Site %d: CAS fit gave beta=%.4f <= 1, which does not map "
                    "to a valid Hurst coefficient via H = 1 - 1/(2*beta). "
                    "Falling back to the default H=0.6. Consider "
                    "acf_model='cas' for this site.",
                    site_idx,
                    beta,
                )
            self._cas_params[site_idx] = (H, 0.0)
            self.logger.debug("Site %d Hurst fit: H=%.4f", site_idx, H)
            return hurst_acf(H, q)

        elif self.acf_model == "custom":
            raise NotImplementedError(
                "Custom ACF vectors are not yet supported. "
                "Pass acf_model='cas' or 'hurst'."
            )
        else:
            raise ValueError(f"Unknown acf_model: '{self.acf_model}'")

    @staticmethod
    def _empirical_acf(values: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute empirical autocorrelation function."""
        n = len(values)
        mean = np.mean(values)
        var = np.var(values)
        if var < 1e-15:
            return np.ones(max_lag + 1)

        acf = np.ones(max_lag + 1)
        centered = values - mean
        for lag in range(1, max_lag + 1):
            acf[lag] = np.sum(centered[: n - lag] * centered[lag:]) / (n * var)

        return acf
