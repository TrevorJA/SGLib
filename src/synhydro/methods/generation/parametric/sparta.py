"""
SPARTA: Stochastic Periodic AutoRegressive To Anything.

Multisite cyclostationary generator for monthly streamflow with per-season
parametric marginal distributions and PAR(1)-N auxiliary Gaussian model.

The Tsoukalas et al. (2018) framework supports any continuous marginal
distribution. This implementation restricts marginal selection to gamma and
lognormal families, chosen per (month, site) by BIC. Extending to additional
families (e.g., kappa, Weibull, GEV) is a planned enhancement.

Reference:
    Tsoukalas, I., Efstratiadis, A., & Makropoulos, C. (2018). Stochastic
    periodic autoregressive to anything (SPARTA): Modeling and simulation of
    cyclostationary processes with arbitrary marginal distributions. Water
    Resources Research, 54(1), 161-185.
    https://doi.org/10.1002/2017WR021394
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, norm

from synhydro.core.base import FittedParams, Generator, make_output_index
from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.core.nataf import nataf_inverse
from synhydro.core.statistics import (
    repair_correlation_matrix,
    repair_covariance_matrix,
)

logger = logging.getLogger(__name__)

# Flows at or below this floor are treated as zero. Preprocessing clips the
# stored monthly record to this value; marginal fitting excludes such values.
_FLOW_FLOOR = 1e-6


def _bic(n: int, neg_loglik: float, k: int) -> float:
    """Bayesian Information Criterion."""
    return 2.0 * neg_loglik + k * np.log(n)


class SPARTAGenerator(Generator):
    """Stochastic Periodic AutoRegressive To Anything generator.

    Generates multisite cyclostationary synthetic timeseries at monthly
    resolution with per-month marginal distributions and PAR(1)-N auxiliary
    Gaussian model with Nataf ICDF mapping.

    Parameters
    ----------
    nataf_method : str
        Nataf evaluation method: ``"GH"`` (default), ``"MC"``, or ``"Int"``.
    nataf_n_eval : int
        Number of support points for Nataf polynomial fitting (default 9).
    nataf_poly_deg : int
        Polynomial degree for Nataf approximation (default 8, identical
        to SMARTA and to ``nataf_inverse``).
    nataf_gh_nodes : int
        Gauss-Hermite quadrature nodes (default 21).
    marginal_method : str
        Marginal fitting: ``"parametric"`` (default, gamma/lognorm BIC).
    matrix_repair_method : str
        Method for repairing non-positive-definite innovation covariances
        (default ``"spectral"``, i.e. eigenvalue clipping). The diagonal of
        ``G_s`` is preserved for every method.
    name : str, optional
        Generator name.
    debug : bool
        Enable debug logging (default False).
    """

    supports_multisite = True
    supported_frequencies = ("MS",)

    def __init__(
        self,
        *,
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
        self.nataf_method = nataf_method
        self.nataf_n_eval = nataf_n_eval
        self.nataf_poly_deg = nataf_poly_deg
        self.nataf_gh_nodes = nataf_gh_nodes
        self.marginal_method = marginal_method
        self.matrix_repair_method = matrix_repair_method

        # Fitted state
        self._Q_monthly: Optional[pd.DataFrame] = None
        self._marginal_params: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._icdfs: Dict[Tuple[int, int], Any] = {}
        # Per-site equivalent lag-1 season-to-season autocorrelations (n_sites, 12)
        self._equiv_auto: Optional[np.ndarray] = None
        # Per-season equivalent cross-correlation matrices
        self._equiv_cross: Dict[int, np.ndarray] = {}
        # PAR(1) model parameters per season
        self._A_s: List[np.ndarray] = []
        self._B_s: List[np.ndarray] = []
        # Univariate shortcut
        self._rmod: Optional[np.ndarray] = None

    @property
    def output_frequency(self) -> str:
        """Monthly frequency."""
        return "MS"

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
        """Validate and prepare monthly data.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            Observed monthly streamflow.
        sites : list of str, optional
            Subset of site names.
        """
        Q = self._store_obs_data(Q_obs, sites)
        self._n_sites = len(self._sites)

        # Resample to monthly if needed
        inferred = pd.infer_freq(Q.index[: min(30, len(Q))])
        if inferred is not None and (
            inferred.startswith("D") or inferred.startswith("W")
        ):
            self.logger.info("Resampling from %s to monthly (sum)", inferred)
            Q = Q.resample("MS").sum()

        # Clip to a small positive floor so the stored record is strictly
        # positive; zeros (values at or below the floor) are excluded from
        # marginal fitting in ``_fit_marginal``.
        Q = Q.clip(lower=_FLOW_FLOOR)
        self._Q_monthly = Q

        self.logger.info(
            "Preprocessing complete: %d months, %d sites",
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
        """Fit the SPARTA model to observed monthly data.

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

        Q = self._Q_monthly
        n_sites = self._n_sites

        nataf_kw = {}
        if self.nataf_method == "GH":
            nataf_kw["nodes"] = self.nataf_gh_nodes

        # Step 1: Fit marginals per (month, site)
        self.logger.info("Step 1: Fitting marginal distributions per (month, site)...")
        self._marginal_params = {}
        self._icdfs = {}
        for m in range(1, 13):
            mask = Q.index.month == m
            for s_idx, site in enumerate(self._sites):
                vals = Q.loc[mask, site].values
                params, icdf = self._fit_marginal(vals, m, s_idx)
                self._marginal_params[(m, s_idx)] = params
                self._icdfs[(m, s_idx)] = icdf

        # Step 2: Compute empirical lag-1 season-to-season correlations
        # on a (n_years, 12) calendar matrix (column 0 = January) so the
        # month pairs are correct regardless of the record's start month.
        self.logger.info("Step 2: Computing season-to-season correlations...")
        year_matrix = self._calendar_year_matrix(Q)
        target_auto = np.zeros((n_sites, 12))
        for s_idx, site in enumerate(self._sites):
            target_auto[s_idx] = self._season_to_season_corr(year_matrix[site])

        # Step 3: Compute lag-0 cross-correlations per season (multisite)
        target_cross: Dict[int, np.ndarray] = {}
        if n_sites > 1:
            self.logger.info("Step 3: Computing per-season cross-correlations...")
            for m in range(1, 13):
                mask = Q.index.month == m
                month_data = Q.loc[mask].values  # (n_years, n_sites)
                target_cross[m] = np.corrcoef(month_data.T)
        else:
            for m in range(1, 13):
                target_cross[m] = np.array([[1.0]])

        # Step 4: Nataf inversion for autocorrelations
        self.logger.info("Step 4: Computing equivalent autocorrelations via Nataf...")
        self._equiv_auto = np.zeros((n_sites, 12))
        for s_idx in range(n_sites):
            for m_idx in range(12):
                m = m_idx + 1  # 1-based month
                m_prev = 12 if m == 1 else m - 1
                target_r = target_auto[s_idx, m_idx]

                equiv_r, _ = nataf_inverse(
                    np.atleast_1d(target_r),
                    self._icdfs[(m, s_idx)],
                    self._icdfs[(m_prev, s_idx)],
                    method=self.nataf_method,
                    n_eval=self.nataf_n_eval,
                    poly_deg=self.nataf_poly_deg,
                    **nataf_kw,
                )
                self._equiv_auto[s_idx, m_idx] = equiv_r[0]

        # Step 5: Nataf inversion for cross-correlations (multisite)
        self._equiv_cross = {}
        if n_sites > 1:
            self.logger.info(
                "Step 5: Computing equivalent cross-correlations via Nataf..."
            )
            for m in range(1, 13):
                C_tilde = np.eye(n_sites)
                for i in range(n_sites):
                    for j in range(i + 1, n_sites):
                        equiv_ij, _ = nataf_inverse(
                            np.atleast_1d(target_cross[m][i, j]),
                            self._icdfs[(m, i)],
                            self._icdfs[(m, j)],
                            method=self.nataf_method,
                            n_eval=self.nataf_n_eval,
                            poly_deg=self.nataf_poly_deg,
                            **nataf_kw,
                        )
                        C_tilde[i, j] = equiv_ij[0]
                        C_tilde[j, i] = equiv_ij[0]
                self._equiv_cross[m] = C_tilde
        else:
            for m in range(1, 13):
                self._equiv_cross[m] = np.array([[1.0]])

        # Step 6: Build PAR(1)-N model per season
        self.logger.info("Step 6: Building PAR(1)-N model per season...")
        self._A_s = []
        self._B_s = []

        if n_sites == 1:
            # Univariate shortcut: rmod_s = sqrt(1 - r_s^2)
            r = self._equiv_auto[0, :]  # (12,)
            self._rmod = np.sqrt(1.0 - r**2)
            # Fill A_s, B_s for consistency
            for m_idx in range(12):
                self._A_s.append(np.array([[r[m_idx]]]))
                self._B_s.append(np.array([[self._rmod[m_idx]]]))
        else:
            for m_idx in range(12):
                m = m_idx + 1
                m_prev = 12 if m == 1 else m - 1

                A_s = np.diag(self._equiv_auto[:, m_idx])
                C_s = self._equiv_cross[m]
                C_prev = self._equiv_cross[m_prev]

                G_s = C_s - A_s @ C_prev @ A_s.T
                G_s = (G_s + G_s.T) / 2.0  # symmetrize

                try:
                    B_s = np.linalg.cholesky(G_s)
                except np.linalg.LinAlgError:
                    self.logger.warning(
                        "G_s for month %d is not positive-definite "
                        "(min eigenvalue %.2e). Repairing...",
                        m,
                        np.linalg.eigvalsh(G_s).min(),
                    )
                    G_repaired = self._repair_innovation_covariance(G_s)
                    try:
                        B_s = np.linalg.cholesky(G_repaired)
                    except np.linalg.LinAlgError:
                        raise ValueError(
                            f"Innovation covariance for month {m} is not "
                            "positive-definite even after repair."
                        )

                self._A_s.append(A_s)
                self._B_s.append(B_s)

        self.logger.info("SPARTA model fitting complete.")
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
        """Generate synthetic monthly timeseries.

        Parameters
        ----------
        n_realizations : int
            Number of realizations (default 1).
        n_years : int, optional
            Number of years. Defaults to observed length.
        n_timesteps : int, optional
            Total months. Overrides n_years.
        seed : int, optional
            Random seed.

        Returns
        -------
        Ensemble
            Generated synthetic data.
        """
        self.validate_fit()
        rng = np.random.default_rng(seed)

        if n_timesteps is not None:
            n_years = int(np.ceil(n_timesteps / 12))
        elif n_years is None:
            n_years = len(self._Q_monthly) // 12

        realizations: Dict[int, pd.DataFrame] = {}
        for r in range(n_realizations):
            df = self._generate_one(n_years, rng=rng)
            if n_timesteps is not None:
                df = df.iloc[:n_timesteps]
            realizations[r] = df

        metadata = EnsembleMetadata(
            generator_class=self.name or self.__class__.__name__,
            n_realizations=n_realizations,
            n_sites=self._n_sites,
            description="SPARTA PAR(1)-N with Nataf ICDF mapping",
        )
        return Ensemble(realizations, metadata=metadata)

    def _generate_one(self, n_years: int, *, rng: np.random.Generator) -> pd.DataFrame:
        """Generate a single realization.

        Following SimSPARTA.R from anySim.
        """
        n_sites = self._n_sites

        if n_sites == 1:
            return self._generate_one_univariate(n_years, rng=rng)
        else:
            return self._generate_one_multivariate(n_years, rng=rng)

    def _generate_one_univariate(
        self, n_years: int, *, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Univariate generation matching SimSPARTA.R exactly."""
        r = self._equiv_auto[0, :]  # (12,)
        rmod = self._rmod  # (12,)

        W = rng.standard_normal((n_years, 12))
        Y = np.empty((n_years, 12))

        # PAR(1) recursion
        for i in range(n_years):
            for j in range(12):
                if j > 0:
                    Y[i, j] = Y[i, j - 1] * r[j] + rmod[j] * W[i, j]
                elif i == 0 and j == 0:
                    Y[i, j] = W[i, j]
                else:  # j == 0, i > 0
                    Y[i, j] = Y[i - 1, 11] * r[j] + rmod[j] * W[i, j]

        # Map to target domain
        U = norm.cdf(Y)
        X = np.empty_like(Y)
        for j in range(12):
            m = j + 1
            u_col = np.clip(U[:, j], 1e-10, 1.0 - 1e-10)
            X[:, j] = self._icdfs[(m, 0)](u_col)

        # Reshape to long format; position 0 is January
        X_long = X.reshape(-1)
        dates = make_output_index(self._output_start_date(), n_years * 12, "MS")
        return pd.DataFrame(X_long, index=dates, columns=[self._sites[0]])

    def _generate_one_multivariate(
        self, n_years: int, *, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Multivariate generation using PAR(1)-N matrices."""
        n_sites = self._n_sites
        total_steps = n_years * 12

        W = rng.standard_normal((total_steps, n_sites))
        Z = np.empty((total_steps, n_sites))

        # PAR(1) recursion: z_t = A_s @ z_{t-1} + B_s @ w_t
        for t in range(total_steps):
            m_idx = t % 12  # season index 0-11
            A = self._A_s[m_idx]
            B = self._B_s[m_idx]

            if t == 0:
                Z[t] = B @ W[t]
            else:
                Z[t] = A @ Z[t - 1] + B @ W[t]

        # Map to target domain
        U = norm.cdf(Z)
        X = np.empty_like(Z)
        for t in range(total_steps):
            m = (t % 12) + 1  # 1-based month
            for s_idx in range(n_sites):
                u_val = np.clip(U[t, s_idx], 1e-10, 1.0 - 1e-10)
                X[t, s_idx] = self._icdfs[(m, s_idx)](np.atleast_1d(u_val))[0]

        # Position 0 is January
        dates = make_output_index(self._output_start_date(), total_steps, "MS")
        return pd.DataFrame(X, index=dates, columns=list(self._sites))

    # ------------------------------------------------------------------
    # Fitted params
    # ------------------------------------------------------------------

    def _compute_fitted_params(self) -> FittedParams:
        """Compute fitted parameters summary."""
        training_period = (
            str(self._Q_monthly.index[0].date()),
            str(self._Q_monthly.index[-1].date()),
        )
        n = self._n_sites
        # 12*n marginal params (2 each) + 12*n autocorrelation + 12*n*(n-1)/2 cross-corr
        n_params = 12 * n * 2 + 12 * n + 12 * n * (n - 1) // 2

        return FittedParams(
            distributions_=self._marginal_params,
            transformations_={"model": "PAR(1)-N"},
            n_parameters_=n_params,
            sample_size_=len(self._Q_monthly),
            n_sites_=self._n_sites,
            training_period_=training_period,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_marginal(
        self, values: np.ndarray, month: int, site_idx: int
    ) -> Tuple[Dict[str, Any], Any]:
        """Fit a marginal distribution via BIC selection (gamma vs lognorm).

        Values at or below ``_FLOW_FLOOR`` (true zeros, and zeros clipped
        to the floor in preprocessing) are excluded from the fit so that
        they do not distort the gamma/lognormal MLE. The BIC sample size
        counts all values, which does not affect the selection because
        both candidates have two parameters. Zero-inflated months are
        not explicitly modeled: the fitted marginal is strictly positive,
        so the generator never produces zeros.
        """
        n = len(values)
        vals = values[values > _FLOW_FLOOR]
        if len(vals) < 5:
            # Fallback for sparse data
            mean_val = np.mean(values) if np.mean(values) > 0 else 1.0
            params = {"dist": "gamma", "shape": 1.0, "loc": 0.0, "scale": mean_val}
            icdf = lambda p, sc=mean_val: gamma_dist.ppf(p, a=1.0, loc=0, scale=sc)
            return params, icdf

        best_bic = np.inf
        best_params = None
        best_icdf = None

        # Gamma
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
                }
                best_icdf = lambda p, s=shape, sc=scale: gamma_dist.ppf(
                    p, a=s, loc=0, scale=sc
                )
        except Exception:
            pass

        # Log-normal
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
                }
                best_icdf = lambda p, s_=s, sc=scale: lognorm.ppf(
                    p, s=s_, loc=0, scale=sc
                )
        except Exception:
            pass

        if best_params is None:
            raise ValueError(
                f"Could not fit distribution for month {month}, site {site_idx}."
            )

        return best_params, best_icdf

    def _output_start_date(self) -> pd.Timestamp:
        """January 1 of the first observed year (generation starts in January)."""
        return pd.Timestamp(year=self._Q_monthly.index[0].year, month=1, day=1)

    def _calendar_year_matrix(self, Q: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Arrange monthly data as (n_years, 12) matrices with column 0 = January.

        Only complete calendar years are kept; partial years at either end
        of the record are trimmed with a warning.

        Parameters
        ----------
        Q : pd.DataFrame
            Monthly data with a DatetimeIndex.

        Returns
        -------
        dict of str to np.ndarray
            Site name to ``(n_years, 12)`` array of complete calendar years.
        """
        years = Q.index.year
        months = Q.index.month
        counts = pd.Series(1, index=Q.index).groupby(years).sum()
        complete = counts.index[counts == 12]
        n_partial = len(counts) - len(complete)
        if n_partial > 0:
            self.logger.warning(
                "Trimming %d partial calendar year(s) (%s) before computing "
                "season-to-season correlations; record spans %s to %s",
                n_partial,
                ", ".join(str(y) for y in counts.index[counts != 12]),
                Q.index[0].date(),
                Q.index[-1].date(),
            )
        if len(complete) < 2:
            raise ValueError(
                "SPARTA requires at least 2 complete calendar years "
                f"(January to December); found {len(complete)}."
            )
        keep = np.isin(years, complete)
        Q_full = Q.loc[keep]
        order = np.lexsort((months[keep], years[keep]))
        n_years = len(complete)
        return {
            site: Q_full[site].values[order].reshape(n_years, 12)
            for site in Q.columns
        }

    def _repair_innovation_covariance(self, G_s: np.ndarray) -> np.ndarray:
        """Make the innovation covariance ``G_s`` positive definite.

        ``G_s = C_s - A_s C_{s-1} A_s^T`` is a covariance whose diagonal
        ``1 - r_s^2`` sets the innovation variance, so it must not be
        rescaled to unit diagonal. With ``matrix_repair_method="spectral"``
        the eigenvalues are clipped directly; other methods repair the
        correlation form of ``G_s`` and restore its original diagonal.
        """
        if self.matrix_repair_method == "spectral":
            return repair_covariance_matrix(G_s)
        d = np.sqrt(np.diag(G_s))
        R = repair_correlation_matrix(
            G_s / np.outer(d, d), method=self.matrix_repair_method
        )
        return R * np.outer(d, d)

    @staticmethod
    def _season_to_season_corr(data: np.ndarray) -> np.ndarray:
        """Compute lag-1 season-to-season correlations.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_years, 12) of monthly values for one site.

        Returns
        -------
        np.ndarray
            Length-12 vector of lag-1 correlations.
            Index 0 = cor(month 1, month 12 of previous year).
            Index j = cor(month j+1, month j) for j > 0.

        References
        ----------
        s2scor.R in anySim.
        """
        n_years = data.shape[0]
        r = np.zeros(12)

        # Month 1 vs month 12 of previous year
        if n_years > 1:
            col_jan = data[1:, 0]  # Jan of years 2..N
            col_dec = data[:-1, 11]  # Dec of years 1..N-1
            r[0] = np.corrcoef(col_jan, col_dec)[0, 1]

        # Within-year adjacent months
        for j in range(1, 12):
            r[j] = np.corrcoef(data[:, j], data[:, j - 1])[0, 1]

        return r
