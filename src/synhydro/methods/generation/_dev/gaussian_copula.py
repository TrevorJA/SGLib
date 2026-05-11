"""
Gaussian / Student-t Copula Generator for multi-site monthly streamflow.

NOT EXPORTED FROM THE PUBLIC API. NOT BUILT INTO THE DOCUMENTATION SITE.

Status: needs realignment to a primary publication before public release. The
original module attribution was Chen et al. (2015), but a careful re-read of
that paper showed it describes a fundamentally different method (Archimedean
cascade with Pearson Type III marginals and Kendall-tau / partial-tau
estimation). What this code actually implements -- per-(month, site)
parametric or empirical marginal -> PIT to normal -> PAR(1) on normal scores
-> Gaussian or Student-t copula correlation matrix on the PAR residuals -> PAR
recursion + inverse marginal -- is not described in Chen (2015) or in any
single primary publication. It is closest in structure to a Pereira et al.
(2017)-style framework with the vine replaced by an elliptical copula, but
Pereira's distinguishing contributions (the periodic R-vine and the spatial
T1 reparametrization) are not part of this generator.

This module is retained on disk for a future rewrite. It is excluded from
``synhydro/__init__.py``, ``synhydro/methods/generation/__init__.py``,
``synhydro/methods/generation/parametric/__init__.py``,
``docs/algorithms/index.md``, ``docs/api/generators.md``, ``mkdocs.yml`` nav,
``README.md``, ``CHANGELOG.md``, and the docs-site build (``mkdocs.yml``
``exclude_docs``). The existing test file
``tests/test_gaussian_copula_generator.py`` still imports the class via its
full module path and continues to pass.

================================================================================
TODO: refactor decisions for a future session
================================================================================

The earlier verification review and the comparison with the vine copula
generator identified the following issues:

  (1) Misattribution. Chen et al. (2015) is cited as primary but does not
      describe this method. The strings "Gaussian", "Spearman", and
      "Pearson correlation" do not appear in Chen (2015). Chen (2015) builds
      a trivariate Archimedean (Gumbel/Frank/Clayton) cascade with Pearson
      Type III marginals fit by L-moments, conditional copulas, and Kendall
      tau / partial Kendall tau parameter estimation. None of that is here.

  (2) Citation typo. Chen et al. (2015) volume/pages were "526, 360-381" in
      the original docstring; the correct citation is *Journal of Hydrology*
      528, 369-384.

  (3) Code duplication with VineCopulaGenerator. Roughly 520 of the 764
      lines in this file are byte-equivalent duplicates of the corresponding
      sections in vine_copula.py: ``preprocessing``, ``_fit_parametric_marginals``,
      ``_fit_empirical_marginals``, ``_pit_to_normal``, ``_fit_par_residuals``,
      ``output_frequency``, ``generate``, the inverse-marginal block of
      ``_generate_one``, and ``_bic`` / ``_LOG2PI``. Only the dependence step
      (single S x S Gaussian/t correlation matrix vs. R-vine of bivariate
      pair-copulas) genuinely differs.

  (4) Asymmetric back-transform vs. vine. The Gaussian path uses
      ``t_dist.cdf`` for the t-copula back-transform; the vine path always
      uses ``norm.cdf`` regardless of the per-pair families. Document the
      difference if both classes are kept.

  (5) Empirical-CDF tail extrapolation. ``u`` is clipped to
      ``[1e-6, 1 - 1e-6]`` before ``norm.ppf`` and ``[1e-8, 1 - 1e-8]`` before
      parametric ``ppf``; no explicit tail-extrapolation strategy. Document
      this as a limitation.

REFACTOR OPTIONS FOR A FUTURE SESSION

  Option A: Delete this module entirely.
    Mathematically, what this generator does is equivalent to
    ``VineCopulaGenerator(family_set=['gaussian'])`` (or
    ``family_set=['student_t']``) when the vine copula is realigned to
    Pereira (2017). Users who want elliptical-copula behavior can simply
    restrict the vine's family set. This is the cleanest design.

  Option B: Refactor to align with Bardossy & Pegram (2009).
    Bardossy & Pegram (2009) describe a multisite copula model with a hidden
    multisite AR(1) Gaussian process, asymmetric V-transform, and quantile
    back-transform. Streamflow adaptation drops their dry-day pole. This
    would produce a multisite generator with explicit asymmetric upper-tail
    dependence, a real value-add over a generic Gaussian copula. Cite
    Bardossy & Pegram (2009).

  Option C: Refactor to align with Lee & Salas (2011).
    Lee & Salas (2011) describe a single-site bivariate copula AR(1) with
    user-selectable family (Clayton / Frank / Gumbel / Gaussian). This is a
    simpler, single-site generator suitable for sites where heteroscedastic
    tail dependence (Gumbel for upper-tail persistence) matters.

  See the conversation handoff in vine_copula.py for the broader
  publication-aligned three-generator design (BardossyPegramGenerator +
  LeeSalasGenerator + a Pereira-aligned VineCopulaGenerator).

REFERENCES (kept for the future rewrite, with citations corrected)

References
----------
Genest, C., and Favre, A.-C. (2007). Everything you always wanted to know
    about copula modeling but were afraid to ask. Journal of Hydrologic
    Engineering, 12(4), 347-368.
Chen, L., Singh, V.P., Guo, S., Zhou, J., and Zhang, J. (2015). Copula-based
    method for multisite monthly and daily streamflow simulation. Journal of
    Hydrology, 528, 369-384.
Pereira, G.A.A., Veiga, A., Erhardt, T., and Czado, C. (2017). A periodic
    spatial vine copula model for multi-site streamflow simulation. Electric
    Power Systems Research, 152, 9-17.
Bardossy, A., and Pegram, G.G.S. (2009). Copula based multisite model for
    daily precipitation simulation. Hydrology and Earth System Sciences,
    13(12), 2299-2314. https://doi.org/10.5194/hess-13-2299-2009
Lee, T., and Salas, J.D. (2011). Copula-based stochastic simulation of
    hydrological data applied to Nile River flows. Hydrology Research, 42(4),
    318-330. https://doi.org/10.2166/nh.2011.085
Tootoonchi, F. et al. (2022). Copulas for hydroclimatic analysis: A
    practice-oriented overview. WIREs Water, 9(2), e1579.
"""

import logging
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import norm, gamma as gamma_dist, lognorm, t as t_dist

from synhydro.core.base import Generator, FittedParams
from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.core.statistics import (
    repair_correlation_matrix,
    NormalScoreTransform,
)

logger = logging.getLogger(__name__)

# BIC helper
_LOG2PI = np.log(2.0 * np.pi)


def _bic(n: int, neg_loglik: float, k: int) -> float:
    """Bayesian Information Criterion: BIC = 2*NLL + k*ln(n)."""
    return 2.0 * neg_loglik + k * np.log(n)


class GaussianCopulaGenerator(Generator):
    """Multi-site monthly generator based on elliptical copulas.

    Fits per-(month, site) marginal distributions and an n_sites x n_sites
    copula correlation matrix per calendar month.  Supports Gaussian copula
    (zero tail dependence) and Student-t copula (symmetric tail dependence).

    Parameters
    ----------
    copula_type : str, default="gaussian"
        ``"gaussian"`` or ``"t"``.  The t-copula adds a degrees-of-freedom
        parameter that controls symmetric tail dependence.
    marginal_method : str, default="parametric"
        ``"parametric"`` fits gamma and log-normal per (month, site) and
        selects the winner by BIC.  ``"empirical"`` uses the Hazen plotting-
        position CDF via :class:`NormalScoreTransform`.
    log_transform : bool, default=False
        Apply ``log(Q + offset)`` before fitting.  Usually unnecessary
        with parametric marginals but may help empirical marginals.
    offset : float, default=1.0
        Additive offset for the log transform.
    matrix_repair_method : str, default="spectral"
        Method passed to :func:`repair_correlation_matrix`.
    name : str, optional
        Instance name.
    debug : bool, default=False
        Enable debug logging.

    References
    ----------
    Genest & Favre (2007), Chen et al. (2015), Tootoonchi et al. (2022).
    """

    supports_multisite = True
    supported_frequencies = ("MS",)

    _VALID_COPULAS = ("gaussian", "t")
    _VALID_MARGINALS = ("parametric", "empirical")

    def __init__(
        self,
        *,
        copula_type: str = "gaussian",
        marginal_method: str = "parametric",
        log_transform: bool = False,
        offset: float = 1.0,
        matrix_repair_method: str = "spectral",
        name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        super().__init__(name=name, debug=debug)

        if copula_type not in self._VALID_COPULAS:
            raise ValueError(
                f"copula_type must be one of {self._VALID_COPULAS}, "
                f"got '{copula_type}'"
            )
        if marginal_method not in self._VALID_MARGINALS:
            raise ValueError(
                f"marginal_method must be one of {self._VALID_MARGINALS}, "
                f"got '{marginal_method}'"
            )

        self.copula_type = copula_type
        self.marginal_method = marginal_method
        self.log_transform = log_transform
        self.offset = offset
        self.matrix_repair_method = matrix_repair_method

        self.init_params.algorithm_params = {
            "method": f"Copula ({copula_type})",
            "copula_type": copula_type,
            "marginal_method": marginal_method,
            "log_transform": log_transform,
            "offset": offset,
            "matrix_repair_method": matrix_repair_method,
        }

        # Fitted attributes (populated by fit)
        self._marginal_params: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._nst: Optional[NormalScoreTransform] = None
        self._monthly_correlations: Dict[int, np.ndarray] = {}
        self._monthly_cholesky: Dict[int, np.ndarray] = {}
        self._df: Optional[float] = None  # t-copula degrees of freedom
        self._Q_monthly: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def output_frequency(self) -> str:
        return "MS"

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocessing(
        self,
        Q_obs,
        *,
        sites: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Validate input and prepare monthly flow data.

        Parameters
        ----------
        Q_obs : pd.DataFrame or pd.Series
            Monthly streamflow with DatetimeIndex.
        sites : list of str, optional
            Subset of sites to use.
        """
        Q = self._store_obs_data(Q_obs, sites)
        self._n_sites = len(self._sites)

        # Resample daily/weekly to monthly if needed
        inferred = pd.infer_freq(Q.index[: min(30, len(Q))])
        if inferred is not None and (
            inferred.startswith("D") or inferred.startswith("W")
        ):
            self.logger.info("Resampling from %s to monthly (sum)", inferred)
            Q = Q.resample("MS").sum()

        if self._n_sites == 1:
            self.logger.warning(
                "Only 1 site provided. The copula correlation matrix is "
                "trivially [1]. Consider a univariate generator instead."
            )

        # Optional log transform
        if self.log_transform:
            Q = np.log(Q + self.offset)

        self._Q_monthly = Q.copy()
        self.update_state(preprocessed=True)
        self.logger.info(
            "Preprocessing complete: %d months, %d sites",
            len(Q),
            self._n_sites,
        )

    # ------------------------------------------------------------------
    # Marginal fitting helpers
    # ------------------------------------------------------------------

    def _fit_parametric_marginals(self) -> None:
        """Fit gamma and log-normal per (month, site); select by BIC."""
        Q = self._Q_monthly
        self._marginal_params = {}

        for m in range(1, 13):
            mask = Q.index.month == m
            month_data = Q.loc[mask]
            n = len(month_data)

            for s_idx, site in enumerate(self._sites):
                vals = month_data[site].values
                vals = vals[np.isfinite(vals)]
                # Ensure strictly positive for gamma/lognorm fitting
                vals_pos = np.maximum(vals, 1e-6)

                best_bic = np.inf
                best_params: Dict[str, Any] = {}

                # Candidate 1: Gamma
                try:
                    ga, gloc, gscale = gamma_dist.fit(vals_pos, floc=0)
                    nll_gamma = -np.sum(
                        gamma_dist.logpdf(vals_pos, ga, loc=0, scale=gscale)
                    )
                    bic_gamma = _bic(len(vals_pos), nll_gamma, k=2)
                    if bic_gamma < best_bic:
                        best_bic = bic_gamma
                        best_params = {
                            "dist": "gamma",
                            "shape": float(ga),
                            "loc": 0.0,
                            "scale": float(gscale),
                            "bic": float(bic_gamma),
                        }
                except Exception:
                    self.logger.debug("Gamma fit failed for month=%d, site=%s", m, site)

                # Candidate 2: Log-normal
                try:
                    ls, lloc, lscale = lognorm.fit(vals_pos, floc=0)
                    nll_ln = -np.sum(lognorm.logpdf(vals_pos, ls, loc=0, scale=lscale))
                    bic_ln = _bic(len(vals_pos), nll_ln, k=2)
                    if bic_ln < best_bic:
                        best_bic = bic_ln
                        best_params = {
                            "dist": "lognorm",
                            "shape": float(ls),
                            "loc": 0.0,
                            "scale": float(lscale),
                            "bic": float(bic_ln),
                        }
                except Exception:
                    self.logger.debug(
                        "Log-normal fit failed for month=%d, site=%s", m, site
                    )

                if not best_params:
                    self.logger.warning(
                        "All parametric fits failed for month=%d, site=%s; "
                        "falling back to empirical normal scores.",
                        m,
                        site,
                    )
                    best_params = {"dist": "empirical_fallback"}

                self._marginal_params[(m, s_idx)] = best_params

    def _fit_empirical_marginals(self) -> None:
        """Fit NormalScoreTransform per (month, site)."""
        Q = self._Q_monthly
        values_by_group: Dict[tuple, np.ndarray] = {}

        for m in range(1, 13):
            mask = Q.index.month == m
            month_data = Q.loc[mask]
            for s_idx, site in enumerate(self._sites):
                vals = month_data[site].values
                vals = vals[np.isfinite(vals)]
                values_by_group[(m, s_idx)] = vals

        self._nst = NormalScoreTransform()
        self._nst.fit(values_by_group)

    def _pit_to_normal(self) -> Dict[int, np.ndarray]:
        """Apply PIT and normal score transform; return normal scores by month.

        Returns
        -------
        Dict[int, np.ndarray]
            Mapping from month (1-12) to array of shape (n_years, n_sites).
        """
        Q = self._Q_monthly
        scores_by_month: Dict[int, np.ndarray] = {}

        for m in range(1, 13):
            mask = Q.index.month == m
            month_data = Q.loc[mask].values  # (n_years, n_sites)
            n_years = month_data.shape[0]
            z = np.zeros_like(month_data)

            for s_idx in range(self._n_sites):
                vals = month_data[:, s_idx]

                if self.marginal_method == "parametric":
                    params = self._marginal_params.get((m, s_idx), {})
                    dist_name = params.get("dist", "empirical_fallback")

                    if dist_name == "gamma":
                        u = gamma_dist.cdf(
                            np.maximum(vals, 1e-6),
                            params["shape"],
                            loc=params["loc"],
                            scale=params["scale"],
                        )
                    elif dist_name == "lognorm":
                        u = lognorm.cdf(
                            np.maximum(vals, 1e-6),
                            params["shape"],
                            loc=params["loc"],
                            scale=params["scale"],
                        )
                    else:
                        # Fallback to empirical
                        if self._nst is None:
                            self._fit_empirical_marginals()
                        z[:, s_idx] = self._nst.transform(vals, (m, s_idx))
                        continue

                    # Clip to avoid +/- inf from norm.ppf
                    u = np.clip(u, 1e-6, 1.0 - 1e-6)
                    z[:, s_idx] = norm.ppf(u)

                else:
                    # Empirical path uses NormalScoreTransform directly
                    z[:, s_idx] = self._nst.transform(vals, (m, s_idx))

            scores_by_month[m] = z

        return scores_by_month

    # ------------------------------------------------------------------
    # PAR(1) temporal model (Pereira et al. 2017)
    # ------------------------------------------------------------------

    def _fit_par_residuals(
        self, scores_by_month: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Fit PAR(1) per (month, site) and return residuals.

        For each month transition m -> m+1, estimates the lag-1
        autocorrelation rho_m(s) from the normal scores.  The PAR(1)
        residuals are approximately i.i.d. and capture only spatial
        dependence (Pereira et al. 2017, Section 3.1).

        Parameters
        ----------
        scores_by_month : Dict[int, np.ndarray]
            Normal scores per month, shape (n_years, n_sites).

        Returns
        -------
        Dict[int, np.ndarray]
            PAR residuals per month, shape (n_years, n_sites).
            One fewer year than input for months that reference the
            previous month's data.
        """
        self._par_rho = {}  # (month, site_idx) -> float

        # Compute lag-1 autocorrelation for each month transition
        for m in range(1, 13):
            m_prev = 12 if m == 1 else m - 1
            z_curr = scores_by_month[m]  # (n_years, n_sites)
            z_prev = scores_by_month[m_prev]  # (n_years, n_sites)

            # Align: for Jan(m=1), pair with Dec of previous year
            if m == 1:
                z_prev = z_prev[:-1]  # Dec years 0..T-2
                z_curr = z_curr[1:]  # Jan years 1..T-1
                n_min = min(len(z_prev), len(z_curr))
                z_prev = z_prev[:n_min]
                z_curr = z_curr[:n_min]
            else:
                # Same year alignment
                n_min = min(len(z_curr), len(z_prev))
                z_curr = z_curr[:n_min]
                z_prev = z_prev[:n_min]

            for s in range(self._n_sites):
                if len(z_curr) > 2:
                    std_c = np.std(z_curr[:, s])
                    std_p = np.std(z_prev[:, s])
                    if std_c > 1e-10 and std_p > 1e-10:
                        rho = float(np.corrcoef(z_prev[:, s], z_curr[:, s])[0, 1])
                        rho = np.clip(rho, -0.99, 0.99)
                    else:
                        rho = 0.0
                else:
                    rho = 0.0
                self._par_rho[(m, s)] = rho

        # Compute PAR residuals: e[t] = (z[t] - rho * z[t-1]) / sqrt(1 - rho^2)
        residuals_by_month: Dict[int, np.ndarray] = {}
        for m in range(1, 13):
            m_prev = 12 if m == 1 else m - 1
            z_curr = scores_by_month[m]
            z_prev = scores_by_month[m_prev]

            if m == 1:
                z_prev = z_prev[:-1]
                z_curr = z_curr[1:]
                n_min = min(len(z_prev), len(z_curr))
                z_prev = z_prev[:n_min]
                z_curr = z_curr[:n_min]
            else:
                n_min = min(len(z_curr), len(z_prev))
                z_curr = z_curr[:n_min]
                z_prev = z_prev[:n_min]

            e = np.zeros_like(z_curr)
            for s in range(self._n_sites):
                rho = self._par_rho[(m, s)]
                scale = np.sqrt(max(1.0 - rho**2, 1e-6))
                e[:, s] = (z_curr[:, s] - rho * z_prev[:, s]) / scale

            residuals_by_month[m] = e

        return residuals_by_month

    # ------------------------------------------------------------------
    # t-copula df estimation
    # ------------------------------------------------------------------

    def _estimate_t_df(self, scores_by_month: Dict[int, np.ndarray]) -> float:
        """Estimate t-copula degrees of freedom via profile likelihood.

        Grid search over df in [2, 50].  For each df, compute the
        multivariate-t log-likelihood across all months and pick the
        maximum.

        Parameters
        ----------
        scores_by_month : Dict[int, np.ndarray]
            Normal scores per month, shape (n_years, n_sites).

        Returns
        -------
        float
            Estimated degrees of freedom.
        """
        from scipy.special import gammaln

        df_grid = np.concatenate(
            [
                np.arange(2, 10, 1),
                np.arange(10, 52, 2),
            ]
        )
        best_ll = -np.inf
        best_df = 5.0
        p = self._n_sites

        for df in df_grid:
            total_ll = 0.0
            for m in range(1, 13):
                z = scores_by_month[m]  # (n_years, p)
                R = self._monthly_correlations[m]
                try:
                    L = np.linalg.cholesky(R)
                except np.linalg.LinAlgError:
                    continue

                log_det_R = 2.0 * np.sum(np.log(np.diag(L)))
                n_years = z.shape[0]

                for i in range(n_years):
                    x = z[i]
                    # Mahalanobis distance
                    v = np.linalg.solve(L, x)
                    quad = float(np.dot(v, v))

                    # Multivariate t log-density (unnormalized by constant)
                    ll_i = (
                        gammaln((df + p) / 2.0)
                        - gammaln(df / 2.0)
                        - 0.5 * p * np.log(df * np.pi)
                        - 0.5 * log_det_R
                        - ((df + p) / 2.0) * np.log(1.0 + quad / df)
                    )
                    total_ll += ll_i

            if total_ll > best_ll:
                best_ll = total_ll
                best_df = float(df)

        self.logger.info("Estimated t-copula df = %.1f", best_df)
        return best_df

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, Q_obs=None, *, sites=None, **kwargs) -> None:
        """Fit marginal distributions and copula correlation structure.

        Parameters
        ----------
        Q_obs : pd.DataFrame or pd.Series, optional
            If provided, calls preprocessing automatically.
        sites : list of str, optional
            Sites to use.
        """
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)
        self.validate_preprocessing()

        # Step 1: Fit marginals
        if self.marginal_method == "parametric":
            self._fit_parametric_marginals()
        else:
            self._fit_empirical_marginals()

        # Step 2: PIT + normal score transform
        scores_by_month = self._pit_to_normal()

        # Step 3: Fit PAR(1) temporal model (Pereira et al. 2017)
        # Removes temporal dependence; residuals capture spatial structure only
        residuals_by_month = self._fit_par_residuals(scores_by_month)

        # Step 4: Correlation matrices on PAR residuals per month
        self._monthly_correlations = {}
        self._monthly_cholesky = {}

        for m in range(1, 13):
            e = residuals_by_month[m]  # (n_years-1, n_sites)

            if self._n_sites == 1:
                corr = np.array([[1.0]])
            else:
                corr = np.corrcoef(e, rowvar=False)
                corr = repair_correlation_matrix(corr, method=self.matrix_repair_method)

            self._monthly_correlations[m] = corr

            try:
                L = np.linalg.cholesky(corr)
            except np.linalg.LinAlgError:
                corr = repair_correlation_matrix(corr, method="spectral")
                L = np.linalg.cholesky(corr)
                self._monthly_correlations[m] = corr

            self._monthly_cholesky[m] = L

        # Step 5: t-copula df estimation (on residuals)
        if self.copula_type == "t":
            self._df = self._estimate_t_df(residuals_by_month)

        self.update_state(fitted=True)
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(
            "Fit complete: copula=%s, marginals=%s, %d sites",
            self.copula_type,
            self.marginal_method,
            self._n_sites,
        )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        n_realizations: int = 1,
        n_years: Optional[int] = None,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Ensemble:
        """Generate synthetic monthly streamflow realizations.

        Parameters
        ----------
        n_realizations : int, default=1
            Number of independent realizations.
        n_years : int, optional
            Years per realization.  Defaults to length of the historic record.
        n_timesteps : int, optional
            Total monthly timesteps; overrides *n_years* if provided.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Ensemble
            Synthetic flow realizations.
        """
        self.validate_fit()

        rng = np.random.default_rng(seed)

        if n_timesteps is not None:
            total_months = n_timesteps
            n_years_gen = int(np.ceil(n_timesteps / 12))
        elif n_years is not None:
            total_months = n_years * 12
            n_years_gen = n_years
        else:
            total_months = len(self._Q_monthly)
            n_years_gen = total_months // 12

        start_date = self._Q_monthly.index[0]

        realizations: Dict[int, pd.DataFrame] = {}
        for r in range(n_realizations):
            Q_syn = self._generate_one(n_years_gen, total_months, rng=rng)
            dates = pd.date_range(start=start_date, periods=len(Q_syn), freq="MS")
            realizations[r] = pd.DataFrame(Q_syn, index=dates, columns=self._sites)

        metadata = EnsembleMetadata(
            generator_class=self.name or self.__class__.__name__,
            n_realizations=n_realizations,
            n_sites=self._n_sites,
            description=(
                f"Copula ({self.copula_type}) with " f"{self.marginal_method} marginals"
            ),
        )

        self.logger.info(
            "Generated %d realizations of %d months x %d sites",
            n_realizations,
            total_months,
            self._n_sites,
        )
        return Ensemble(realizations, metadata=metadata)

    def _generate_one(
        self,
        n_years: int,
        total_months: int,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate a single realization.

        Returns
        -------
        np.ndarray
            Shape (total_months, n_sites).
        """
        n_sites = self._n_sites
        n_months = n_years * 12
        Q = np.zeros((n_months, n_sites))

        # Track normal scores for PAR(1) temporal recursion
        z_prev = np.zeros(n_sites)  # z[t-1] in normal-score space

        for t in range(n_months):
            m = (t % 12) + 1  # calendar month 1-12
            L = self._monthly_cholesky[m]

            # Draw independent samples (copula innovations)
            if self.copula_type == "t":
                eps = t_dist.rvs(df=self._df, size=n_sites, random_state=rng)
            else:
                eps = rng.standard_normal(n_sites)

            # Impose spatial correlation on residuals via Cholesky
            e_spatial = L @ eps

            # PAR(1) temporal recursion (Pereira et al. 2017):
            # z[t] = rho_m * z[t-1] + sqrt(1 - rho_m^2) * e[t]
            z = np.zeros(n_sites)
            for s_idx in range(n_sites):
                rho = self._par_rho.get((m, s_idx), 0.0)
                scale = np.sqrt(max(1.0 - rho**2, 1e-6))
                z[s_idx] = rho * z_prev[s_idx] + scale * e_spatial[s_idx]

            z_prev = z.copy()

            # Map from normal-score space to original marginal space
            if self.copula_type == "t":
                u = t_dist.cdf(z, df=self._df)
            else:
                u = norm.cdf(z)

            # Inverse marginal CDF
            for s_idx in range(n_sites):
                if self.marginal_method == "parametric":
                    params = self._marginal_params.get((m, s_idx), {})
                    dist_name = params.get("dist", "empirical_fallback")

                    if dist_name == "gamma":
                        Q[t, s_idx] = gamma_dist.ppf(
                            np.clip(u[s_idx], 1e-8, 1 - 1e-8),
                            params["shape"],
                            loc=params["loc"],
                            scale=params["scale"],
                        )
                    elif dist_name == "lognorm":
                        Q[t, s_idx] = lognorm.ppf(
                            np.clip(u[s_idx], 1e-8, 1 - 1e-8),
                            params["shape"],
                            loc=params["loc"],
                            scale=params["scale"],
                        )
                    else:
                        Q[t, s_idx] = self._nst.inverse_transform(
                            np.array([z[s_idx]]), (m, s_idx)
                        )[0]
                else:
                    Q[t, s_idx] = self._nst.inverse_transform(
                        np.array([z[s_idx]]), (m, s_idx)
                    )[0]

        # Undo log transform if applied
        if self.log_transform:
            Q = np.exp(Q) - self.offset

        # Enforce non-negativity
        Q = np.maximum(Q, 0.0)

        return Q[:total_months]

    # ------------------------------------------------------------------
    # Fitted params
    # ------------------------------------------------------------------

    def _compute_fitted_params(self) -> FittedParams:
        n = self._n_sites
        # Parameters per month: n marginal (2 each) + n(n-1)/2 corr + n PAR rho
        n_marginal = 12 * n * 2
        n_corr = 12 * n * (n - 1) // 2
        n_par = 12 * n  # PAR(1) rho per month per site
        n_df = 1 if self.copula_type == "t" else 0
        n_params = n_marginal + n_corr + n_par + n_df

        training_period = (
            str(self._Q_monthly.index[0].date()),
            str(self._Q_monthly.index[-1].date()),
        )

        return FittedParams(
            means_=None,
            stds_=None,
            correlations_={m: self._monthly_correlations[m] for m in range(1, 13)},
            distributions_={
                "copula_type": self.copula_type,
                "marginal_method": self.marginal_method,
                "marginal_params": {
                    str(k): v for k, v in self._marginal_params.items()
                },
                "df": self._df,
            },
            transformations_={
                "log_transform": self.log_transform,
                "offset": self.offset,
            },
            n_parameters_=n_params,
            sample_size_=len(self._Q_monthly),
            n_sites_=self._n_sites,
            training_period_=training_period,
        )
