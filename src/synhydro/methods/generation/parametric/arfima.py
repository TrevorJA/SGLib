"""
ARFIMA (Autoregressive Fractionally Integrated Moving Average) generator for synthetic streamflow.

Implements the ARFIMA(p,d,q) model for generating synthetic hydrologic timeseries with
long-range dependence (LRD), preserving the Hurst phenomenon. Primary reference:
Hosking, J.R.M. (1984). Modeling persistence in hydrological time series using fractional
differencing. Water Resources Research, 20(12), 1898-1908.
"""

import logging
from typing import Optional, Union, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.linalg import solve_toeplitz
from scipy.optimize import minimize
from scipy.signal import fftconvolve, lfilter

from synhydro.core.base import (
    Generator,
    FittedParams,
    GeneratorParams,
    make_output_index,
)
from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.transformations import SteddingerTransform, StandardScaler

logger = logging.getLogger(__name__)


class ARFIMAGenerator(Generator):
    """
    Autoregressive Fractionally Integrated Moving Average (ARFIMA) generator for synthetic monthly/annual streamflow generation.

    Generates synthetic streamflows using an ARFIMA model that captures long-range
    dependence through fractional differencing parameter d in (-0.5, 0.5). The model
    preserves Hurst exponent, seasonal patterns (if monthly), and autocorrelation
    structure.

    By default d and the ARMA(p,q) coefficients are estimated jointly by the
    approximate maximum likelihood method of Hosking (1984, Sec. 4.2): the
    series is fractionally differenced with backcast presample values
    (Eq. 9, M = 30) and the conditional sum of squares of the ARMA
    innovations is minimised over (d, phi, theta).

    The Hurst exponent H relates to the fractional differencing parameter via H = d + 0.5,
    providing direct parameterization of long-memory behavior.

    Preprocessing applies a shifted-lognormal transformation (Stedinger and
    Taylor, 1982) followed by per-period z-score standardization. The Gaussian
    ARFIMA process is fit in this transformed space. On back-transform,
    Q = tau + exp(Y) is strictly positive by construction, so no hard-clipping
    of synthetic flows is required (Hosking, 1984; Montanari et al., 1997).

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.methods.generation.parametric.arfima import ARFIMAGenerator
    >>> Q_monthly = pd.read_csv('monthly_flows.csv', index_col=0, parse_dates=True)
    >>> arfima = ARFIMAGenerator()
    >>> arfima.preprocessing(Q_monthly.iloc[:, 0])
    >>> arfima.fit()
    >>> ensemble = arfima.generate(n_years=50, n_realizations=100)

    References
    ----------
    Hosking, J.R.M. (1984). Modeling persistence in hydrological time series using
    fractional differencing. Water Resources Research, 20(12), 1898-1908.
    https://doi.org/10.1029/WR020i012p01898
    """

    supports_multisite: bool = False
    supported_frequencies: tuple = ("MS", "YS")

    def __init__(
        self,
        *,
        p: int = 1,
        q: int = 0,
        d_method: str = "mle",
        truncation_lag: int = 100,
        auto_order: bool = False,
        order_criterion: str = "aic",
        backcast_length: int = 30,
        d_bounds: Tuple[float, float] = (-0.49, 0.49),
        name: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the ARFIMAGenerator.

        Parameters
        ----------
        p : int, default=1
            AR order for the short-memory ARMA(p,q) component.
        q : int, default=0
            MA order for the short-memory ARMA(p,q) component.
        d_method : str, default='mle'
            Estimation method.  'mle' estimates d and the ARMA(p,q)
            coefficients jointly by Hosking's (1984, Sec. 4.2) approximate
            maximum likelihood (conditional sum of squares of the ARMA
            innovations of the fractionally differenced series, with
            backcast presample values).  The remaining options are
            two-stage procedures that first estimate d from the series
            alone and then fit the ARMA part to the fractionally
            differenced residual: 'whittle' (profile Whittle likelihood of
            the ARFIMA(0,d,0) spectrum), 'gph' (Geweke-Porter-Hudak) or
            'rs' (R/S analysis).  The two-stage estimates of d are
            contaminated by short-memory structure when p + q > 0.
        truncation_lag : int, default=100
            Truncation lag K for the inverse fractional differencing filter
            used in generation (and for the fit-side differencing of the
            two-stage methods).  The truncated inverse filter caps the
            simulated variance at sum_{k<=K} psi_k^2 (about 97% of the
            exact value at d = 0.3, but only 55% at d = 0.45 for K = 100);
            raise K (e.g. 1000 or more) when the estimated d exceeds
            about 0.4.
        auto_order : bool, default=False
            If True, select (p, q) by an information-criterion grid search
            over p in {0, 1, 2} and q in {0, 1, 2}.  Overrides user-supplied
            p and q values.
        order_criterion : str, default='aic'
            Information criterion for ``auto_order``: 'aic' (Hosking 1984,
            Sec. 5.1, with the delta_d term) or 'bic' (consistent for
            ARFIMA order selection, Huang et al. 2022).
        backcast_length : int, default=30
            Number M of presample values backcast with an AR(M) model
            before fractional differencing (Hosking 1984, Eq. 9 and
            Table 1).  Used by ``d_method='mle'`` only.  Set to 0 for
            Hosking's Eq. 8 (presample values equal to the mean).  M is
            reduced to n // 4 for short records.
        d_bounds : tuple of float, default=(-0.49, 0.49)
            Search interval for d in the joint estimator.  Hosking's model
            is stationary and invertible for -0.5 < d < 0.5; use
            (0.01, 0.49) to restrict the fit to persistent processes.
        name : str, optional
            Name identifier for this generator instance.
        debug : bool, default=False
            Enable debug logging.
        **kwargs : dict, optional
            Additional parameters (stored in init_params).
        """
        super().__init__(name=name, debug=debug)

        self.p = p
        self.q = q
        self.d_method = d_method
        self.truncation_lag = truncation_lag
        self.auto_order = auto_order
        self.order_criterion = order_criterion
        self.backcast_length = backcast_length
        self.d_bounds = (float(d_bounds[0]), float(d_bounds[1]))

        if d_method not in ("mle", "whittle", "gph", "rs"):
            raise ValueError(f"Unknown d_method: {d_method}")
        if order_criterion not in ("aic", "bic"):
            raise ValueError(f"Unknown order_criterion: {order_criterion}")
        if not (-0.5 <= self.d_bounds[0] < self.d_bounds[1] <= 0.5):
            raise ValueError(
                f"d_bounds must satisfy -0.5 <= lo < hi <= 0.5, got {d_bounds}"
            )

        # Store initialization parameters
        self.init_params.algorithm_params = {
            "p": p,
            "q": q,
            "d_method": d_method,
            "truncation_lag": truncation_lag,
            "auto_order": auto_order,
            "order_criterion": order_criterion,
            "backcast_length": backcast_length,
            "d_bounds": self.d_bounds,
        }
        self.init_params.transformation_params = {
            "transformation": "SteddingerTransform + StandardScaler",
            "by_month": "auto (True if monthly input, False if annual)",
            "reference": "Stedinger & Taylor (1982) WRR 18(4):909-918",
        }

    @property
    def output_frequency(self) -> str:
        """Return output frequency based on input data."""
        if hasattr(self, "_output_freq"):
            return self._output_freq
        return "MS"  # Default to monthly

    def preprocessing(self, Q_obs, *, sites=None, **kwargs) -> None:
        """
        Preprocess observed data for ARFIMA generation.

        Validates input, ensures univariate data, applies a shifted-lognormal
        transformation (Stedinger and Taylor, 1982) followed by per-month
        z-score standardization to produce a stationary, approximately Gaussian
        residual series suitable for ARFIMA fitting.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed historical flow data.
        sites : list, optional
            Sites to keep. If None, uses all columns.
        **kwargs : dict, optional
            Additional preprocessing parameters.

        Raises
        ------
        ValueError
            If data has insufficient length or multiple sites.

        Notes
        -----
        The two-stage transformation guarantees strictly positive synthetic
        flows on back-transform (Q = tau + exp(Y) with tau >= 0), removing
        the need for hard-clipping. See Hosking (1984), Montanari et al.
        (1997), and Stedinger and Taylor (1982).
        """
        Q = self._store_obs_data(Q_obs, sites=sites)

        if len(Q) < 30:
            raise ValueError(f"ARFIMA requires at least 30 timesteps, got {len(Q)}")

        # Detect frequency from index or infer from median spacing
        freq = getattr(Q.index, "freq", None)
        if freq is None and len(Q) > 1:
            median_days = (Q.index[1:] - Q.index[:-1]).median().days
            if 25 <= median_days <= 35:
                freq = "MS"
            elif 350 <= median_days <= 380:
                freq = "YS"

        freq_str = getattr(freq, "freqstr", str(freq)) if freq is not None else None
        if freq_str is not None and freq_str in ("MS", "M", "ME"):
            self._output_freq = "MS"
            self._is_monthly = True
        elif freq_str is not None and any(
            freq_str.startswith(p) for p in ("AS", "YS", "Y", "A")
        ):
            self._output_freq = "YS"
            self._is_monthly = False
        else:
            self._output_freq = "MS"
            self._is_monthly = len(Q) > 24  # assume monthly if enough data

        self.Q_obs = Q.iloc[:, 0]  # Convert to Series

        # Two-stage transformation: shifted-lognormal then per-period z-score.
        # by_month=True for monthly input (per-month tau, mean, std); by_month=False
        # for annual input (single global tau, mean, std).
        by_month = self._is_monthly
        self.log_transform = SteddingerTransform(by_month=by_month)
        self.scaler = StandardScaler(by_month=by_month)

        self.Q_log = self.log_transform.fit_transform(self.Q_obs)
        self.Q_norm = self.scaler.fit_transform(self.Q_log)

        self.update_state(preprocessed=True)
        self.logger.info(
            f"Preprocessing complete: {len(self.Q_obs)} timesteps, "
            f"frequency={'monthly' if self._is_monthly else 'annual'}"
        )

    def fit(self, Q_obs=None, *, sites=None, **kwargs) -> None:
        """
        Estimate ARFIMA model parameters from preprocessed data.

        With ``d_method='mle'`` (default) d and the ARMA(p,q) coefficients
        are estimated jointly by Hosking's (1984, Sec. 4.2) approximate
        maximum likelihood; see ``_fit_joint_mle``.  With the two-stage
        methods the sequence is:

        1. Estimate d from the series alone ('whittle', 'gph' or 'rs')
        2. Apply truncated fractional differencing (Hosking Eq. 8)
        3. Fit ARMA(p,q) to the differenced series (Yule-Walker / CSS)

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            If provided, calls preprocessing automatically.
        sites : list, optional
            Sites to keep. Passed to preprocessing if Q_obs is provided.
        **kwargs : dict, optional
            Additional fitting parameters.

        Raises
        ------
        ValueError
            If fitting fails (e.g., ARMA estimation error).
        """
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)
        self.validate_preprocessing()

        if self.d_method == "mle":
            self._fit_joint_mle()
        else:
            self._fit_two_stage()

        # Innovation variance from one-step-ahead prediction errors
        W_vals = self.W.values
        residuals = self._compute_css_residuals(W_vals, self.phi, self.theta)
        burn_in = max(self.p, self.q, 1)
        self.sigma_eps_sq = float(np.var(residuals[burn_in:]))

        self.update_state(fitted=True)
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(f"Fitting complete: sigma_eps^2 = {self.sigma_eps_sq:.4f}")

    # ------------------------------------------------------------------
    # Joint approximate maximum likelihood (Hosking 1984, Sec. 4.2)
    # ------------------------------------------------------------------

    def _fit_joint_mle(self) -> None:
        """
        Estimate (d, phi, theta) jointly by approximate maximum likelihood.

        Implements Hosking (1984, Sec. 4.2, Eqs. 6-9).  For a candidate
        parameter vector the standardized series X_t (mean removed) is
        fractionally differenced with the operator of Eq. 9, in which the
        M presample values X_{-M}, ..., X_{-1} are backcast with an AR(M)
        model and earlier values are set to the mean (zero); the
        differenced series W_t is then filtered through the ARMA(p,q)
        recursion to obtain the innovations e_t.  The likelihood of the
        ARMA model (Eq. 6) is replaced by its conditional-sum-of-squares
        approximation, so the objective minimised over (d, phi, theta) is
        sum_{t>p} e_t^2.  Optimisation uses L-BFGS-B from several starting
        values (the profile Whittle estimate of d and a small grid); the
        best local optimum is kept.

        When ``auto_order`` is set, the joint fit is repeated for every
        (p, q) in {0, 1, 2}^2 and the order minimising the information
        criterion of ``order_criterion`` is selected.

        Sets ``d``, ``phi``, ``theta``, ``pi_coeffs``, ``W`` and, with
        ``auto_order``, ``p`` and ``q``.
        """
        x = self.Q_norm.values.astype(float)
        x = x - np.mean(x)
        n = len(x)

        presample = self._backcast_presample(x)
        x_ext = np.concatenate([presample, x])

        if self.auto_order:
            self.p, self.q = self._select_order_joint(x_ext, n)
            self.logger.info(
                "%s selected ARFIMA(%d,d,%d)",
                self.order_criterion.upper(),
                self.p,
                self.q,
            )
            d, phi, theta, _ = self._joint_fit_cache[(self.p, self.q)]
        else:
            d, phi, theta, _ = self._fit_joint_css(x_ext, n, self.p, self.q)

        self.d = float(d)
        self.phi = np.asarray(phi, dtype=float)
        self.theta = np.asarray(theta, dtype=float)
        self.pi_coeffs = self._compute_fractional_diff_coefficients(self.d)
        W = self._fractional_difference_backcast(x_ext, self.d, n)
        self.W = pd.Series(W, index=self.Q_norm.index)

        self.logger.info(
            "Joint approximate ML: d = %.4f (H = %.4f), phi = %s, theta = %s",
            self.d,
            self.d + 0.5,
            self.phi,
            self.theta,
        )

    def _backcast_presample(self, x: np.ndarray) -> np.ndarray:
        """
        Backcast M presample values with an AR(M) model (Hosking Eq. 9).

        An AR(M) model is fitted to the centred series by Yule-Walker and
        the time-reversed series is forecast M steps ahead (a stationary
        Gaussian process is time-reversible; Box and Jenkins 1976,
        p. 199).  M is ``backcast_length`` reduced to n // 4 for short
        records.

        Parameters
        ----------
        x : np.ndarray
            Centred series.

        Returns
        -------
        np.ndarray
            Backcast values [X_{-M}, ..., X_{-1}] (empty when M = 0).
        """
        n = len(x)
        M = int(min(self.backcast_length, n // 4))
        if M <= 0:
            return np.zeros(0)

        acov = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(M + 1)])
        if acov[0] <= 0:
            return np.zeros(M)
        try:
            a = solve_toeplitz(acov[:M], acov[1 : M + 1])
        except np.linalg.LinAlgError:
            self.logger.warning(
                "Singular Yule-Walker system in backcasting; using mean presample"
            )
            return np.zeros(M)

        # Forecast the reversed series: rev[0] = x_{n-1}, ..., rev[n-1] = x_0,
        # rev[n] = x_{-1}, rev[n+1] = x_{-2}, ...
        rev = np.concatenate([x[::-1], np.zeros(M)])
        for t in range(n, n + M):
            rev[t] = np.dot(a, rev[t - 1 : t - M - 1 : -1])
        return rev[n:][::-1]

    @staticmethod
    def _fractional_difference_backcast(
        x_ext: np.ndarray, d: float, n: int
    ) -> np.ndarray:
        """
        Apply the fractional differencing operator of Hosking (1984) Eq. 9.

        W_t = sum_{j=0}^{t+M-1} pi_j X_{t-j} for t = 1..n, where the
        extended series holds the M backcast presample values followed by
        the n observations.  The full-length filter is used (no truncation
        at ``truncation_lag``).

        Parameters
        ----------
        x_ext : np.ndarray
            Presample values followed by the centred observations.
        d : float
            Fractional differencing parameter.
        n : int
            Number of observations (last n entries of ``x_ext``).

        Returns
        -------
        np.ndarray
            Fractionally differenced series of length n.
        """
        L = len(x_ext)
        k = np.arange(1, L)
        pi = np.concatenate([[1.0], np.cumprod((k - 1 - d) / k)])
        w = fftconvolve(x_ext, pi)[:L]
        return w[L - n :]

    @staticmethod
    def _arma_is_admissible(phi: np.ndarray, theta: np.ndarray) -> bool:
        """
        Check that the AR and MA polynomials have roots outside the unit circle.

        Parameters
        ----------
        phi : np.ndarray
            AR coefficients (may be empty).
        theta : np.ndarray
            MA coefficients (may be empty).

        Returns
        -------
        bool
            True when the ARMA part is stationary and invertible.
        """
        for coeffs, sign in ((phi, -1.0), (theta, 1.0)):
            if len(coeffs) == 0:
                continue
            poly = np.concatenate([[1.0], sign * np.asarray(coeffs, dtype=float)])
            roots = np.roots(poly[::-1])
            if len(roots) and np.any(np.abs(roots) <= 1.0 + 1e-6):
                return False
        return True

    @staticmethod
    def _arma_innovations(
        w: np.ndarray, phi: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        One-step ARMA(p,q) innovations with zero presample values.

        Equivalent to ``_compute_css_residuals`` but vectorised through
        ``scipy.signal.lfilter``: phi(B) W_t = theta(B) e_t.

        Parameters
        ----------
        w : np.ndarray
            Fractionally differenced series.
        phi : np.ndarray
            AR coefficients (may be empty).
        theta : np.ndarray
            MA coefficients (may be empty).

        Returns
        -------
        np.ndarray
            Innovations of the same length as ``w``.
        """
        b = np.concatenate([[1.0], -np.asarray(phi, dtype=float)])
        a = np.concatenate([[1.0], np.asarray(theta, dtype=float)])
        return lfilter(b, a, w)

    def _fit_joint_css(
        self, x_ext: np.ndarray, n: int, p: int, q: int
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Minimise the innovation sum of squares over (d, phi, theta).

        Parameters
        ----------
        x_ext : np.ndarray
            Presample values followed by the centred observations.
        n : int
            Number of observations.
        p : int
            AR order.
        q : int
            MA order.

        Returns
        -------
        d : float
            Estimated fractional differencing parameter.
        phi : np.ndarray
            Estimated AR coefficients.
        theta : np.ndarray
            Estimated MA coefficients.
        ssr : float
            Minimised sum of squared innovations over t > p.
        """
        lo, hi = self.d_bounds
        burn = p

        def objective(params: np.ndarray) -> float:
            d = params[0]
            phi = params[1 : 1 + p]
            theta = params[1 + p :]
            if not self._arma_is_admissible(phi, theta):
                return 1e300
            w = self._fractional_difference_backcast(x_ext, d, n)
            e = self._arma_innovations(w, phi, theta)
            ssr = float(np.sum(e[burn:] ** 2))
            return ssr if np.isfinite(ssr) else 1e300

        # Starting values: profile Whittle estimate of d (two-stage
        # estimator) plus a small grid, ARMA coefficients at zero.
        margin = 0.01
        d_whittle = float(np.clip(self._whittle_estimator(), lo + margin, hi - margin))
        d_starts = [d_whittle] + [
            float(np.clip(v, lo + margin, hi - margin)) for v in (0.1, 0.3)
        ]
        bounds = [(lo, hi)] + [(-0.99, 0.99)] * (p + q)

        best = None
        for d0 in d_starts:
            x0 = np.concatenate([[d0], np.zeros(p + q)])
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500},
            )
            if not result.success:
                self.logger.debug(
                    "Joint CSS start d0=%.3f did not converge: %s",
                    d0,
                    result.message,
                )
            if best is None or result.fun < best.fun:
                best = result

        d_hat = float(best.x[0])
        phi_hat = np.array(best.x[1 : 1 + p], dtype=float)
        theta_hat = np.array(best.x[1 + p :], dtype=float)
        self.logger.debug(
            "Joint CSS ARFIMA(%d,d,%d): d=%.4f phi=%s theta=%s ssr=%.4f",
            p,
            q,
            d_hat,
            phi_hat,
            theta_hat,
            best.fun,
        )
        return d_hat, phi_hat, theta_hat, float(best.fun)

    def _information_criterion(self, sigma2: float, n_eff: int, k: int) -> float:
        """
        Information criterion for order selection, up to a constant.

        With Gaussian innovations -2 log L_max = n_eff log(sigma2) + const,
        so Hosking's (1984, Sec. 5.1) AIC = -2 log L_max + 2 log L_0 +
        2 (p + q + delta_d) reduces, up to terms common to all candidate
        models, to n_eff log(sigma2) + 2 k with k = p + q + delta_d.  BIC
        replaces the penalty 2 k by k log(n_eff).

        Parameters
        ----------
        sigma2 : float
            Innovation variance estimate (sum of squares / n_eff).
        n_eff : int
            Number of residuals entering the sum of squares.
        k : int
            Number of estimated parameters (p + q + delta_d).

        Returns
        -------
        float
            Criterion value (smaller is better).
        """
        if self.order_criterion == "bic":
            return float(n_eff * np.log(sigma2) + k * np.log(n_eff))
        return float(n_eff * np.log(sigma2) + 2.0 * k)

    def _select_order_joint(self, x_ext: np.ndarray, n: int) -> Tuple[int, int]:
        """
        Select (p, q) by information criterion using the joint estimator.

        Every (p, q) in {0, 1, 2}^2 is fitted by ``_fit_joint_css`` and
        the innovation variance is evaluated on the common sample t >= 2
        so that all candidates are compared on the same residuals.  d is
        estimated in every candidate, so delta_d = 1 throughout.  Fits are
        cached in ``_joint_fit_cache`` keyed by (p, q).

        Parameters
        ----------
        x_ext : np.ndarray
            Presample values followed by the centred observations.
        n : int
            Number of observations.

        Returns
        -------
        best_p : int
            Selected AR order.
        best_q : int
            Selected MA order.
        """
        max_order = 2
        common_burn = max_order
        n_eff = n - common_burn
        self._joint_fit_cache: Dict[Tuple[int, int], Tuple] = {}

        best_val = np.inf
        best_p, best_q = self.p, self.q
        for p_cand in range(max_order + 1):
            for q_cand in range(max_order + 1):
                try:
                    d, phi, theta, _ = self._fit_joint_css(x_ext, n, p_cand, q_cand)
                except Exception as exc:
                    self.logger.debug(
                        "Order search: ARFIMA(%d,d,%d) failed: %s",
                        p_cand,
                        q_cand,
                        exc,
                    )
                    continue
                w = self._fractional_difference_backcast(x_ext, d, n)
                e = self._arma_innovations(w, phi, theta)
                sigma2 = float(np.sum(e[common_burn:] ** 2) / n_eff)
                if sigma2 <= 0:
                    continue
                val = self._information_criterion(sigma2, n_eff, p_cand + q_cand + 1)
                self._joint_fit_cache[(p_cand, q_cand)] = (d, phi, theta, val)
                self.logger.debug(
                    "%s(%d,%d) = %.2f  (d=%.4f, sigma2=%.4f)",
                    self.order_criterion.upper(),
                    p_cand,
                    q_cand,
                    val,
                    d,
                    sigma2,
                )
                if val < best_val:
                    best_val = val
                    best_p, best_q = p_cand, q_cand

        if (best_p, best_q) not in self._joint_fit_cache:
            raise ValueError("Joint ARFIMA order selection failed for every (p, q)")
        return best_p, best_q

    # ------------------------------------------------------------------
    # Two-stage estimators (d first, then ARMA on the differenced series)
    # ------------------------------------------------------------------

    def _fit_two_stage(self) -> None:
        """
        Two-stage estimation: d from the series alone, then ARMA(p,q).

        Used for ``d_method`` in {'whittle', 'gph', 'rs'}.  The estimate of
        d ignores the ARMA part and is therefore contaminated by
        short-memory structure (Hosking 1984, Table 4); see the algorithm
        documentation.

        Sets ``d``, ``pi_coeffs``, ``W``, ``phi``, ``theta`` and, with
        ``auto_order``, ``p`` and ``q``.
        """
        self.logger.info(f"Estimating d using {self.d_method} method...")
        self.d = self._estimate_d()
        self.logger.info(
            f"Estimated d = {self.d:.4f}, Hurst exponent H = {self.d + 0.5:.4f}"
        )

        # Compute fractional differencing coefficients
        self.pi_coeffs = self._compute_fractional_diff_coefficients(self.d)

        # Apply fractional differencing
        self.W = self._apply_fractional_differencing()

        if self.auto_order:
            self.p, self.q = self._select_order_two_stage(self.W)
            self.logger.info(
                "%s selected ARMA(%d,%d) for short-memory component",
                self.order_criterion.upper(),
                self.p,
                self.q,
            )

        # Fit ARMA(p,q) to differenced series
        if self.q > 0:
            self.phi, self.theta = self._fit_arma_css(self.W, self.p, self.q)
            self.logger.info(
                "Fitted ARMA(%d,%d) via CSS: phi=%s, theta=%s",
                self.p,
                self.q,
                self.phi,
                self.theta,
            )
        elif self.p > 0:
            self.phi = self._fit_ar(self.W, self.p)
            self.theta = np.array([])
            self.logger.info(f"Fitted AR({self.p}) coefficients: {self.phi}")
        else:
            self.phi = np.array([])
            self.theta = np.array([])

    def _estimate_d(self) -> float:
        """
        Estimate fractional differencing parameter d (two-stage methods).

        Returns
        -------
        float
            Estimated d in (0, 0.5).

        Raises
        ------
        ValueError
            If estimation fails.
        """
        if self.d_method == "whittle":
            return self._whittle_estimator()
        elif self.d_method == "rs":
            return self._rs_estimator()
        elif self.d_method == "gph":
            return self._gph_estimator()
        else:
            raise ValueError(f"Unknown d_method: {self.d_method}")

    def _whittle_estimator(self) -> float:
        """
        Estimate d via the profile Whittle likelihood in the frequency domain.

        Writes the spectral density as f(w; d) = c * g(w; d) with
        g(w; d) = [2(1 - cos w)]^(-d) and profiles out the scale c
        analytically (c_hat = mean_j I(w_j) / g(w_j; d)), leaving

            L(d) = m * log( mean_j I(w_j) / g(w_j; d) ) + sum_j log g(w_j; d)

        where I(w_j) is the periodogram at the m = floor(n/2) - 1 Fourier
        frequencies (Fox and Taqqu, 1986; Beran, 1994, Sec. 5.5).  This
        form is invariant to the scale of the input series.  The
        non-profiled form sum_j [log g + I/g] with c fixed at 1 is biased
        upward by roughly +0.04 to +0.06 on unit-variance data.

        Only the ARFIMA(0,d,0) spectrum is used; the ARMA(p,q) part is
        fitted afterwards to the fractionally differenced series (see
        ``fit``).

        Returns
        -------
        float
            Estimated d.
        """
        data = self.Q_norm.values

        # Periodogram
        n = len(data)
        dft = np.fft.fft(data)
        I = (np.abs(dft) ** 2) / (2 * np.pi * n)

        # Frequencies (exclude 0 and Nyquist)
        freqs = np.fft.fftfreq(n)
        idx = np.arange(1, n // 2)
        I = I[idx]
        w = 2 * np.pi * freqs[idx]

        # Objective function: Whittle likelihood
        def whittle_likelihood(d_test):
            if d_test <= 0 or d_test >= 0.5:
                return 1e10
            # Spectral density of ARFIMA(0,d,0):
            #   f(w) ~ |1 - e^{-iw}|^{-2d}
            # Since |1 - e^{-iw}|^2 = 2(1 - cos(w)):
            #   f(w) ~ [2(1 - cos(w))]^{-d}
            # Ref: Hosking (1981) eq. 2.3; Beran (1994) Ch. 5
            g_w = (2.0 * (1.0 - np.cos(w))) ** (-d_test)

            # Avoid log(0)
            g_w = np.maximum(g_w, 1e-10)

            # Profile Whittle likelihood (Fox & Taqqu 1986): the scale
            # factor c of f = c * g is concentrated out analytically,
            # c_hat = mean(I / g), which makes the estimate scale-invariant.
            m = len(I)
            likelihood = m * np.log(np.mean(I / g_w)) + np.sum(np.log(g_w))
            return likelihood

        # Optimize
        result = minimize(
            whittle_likelihood, x0=0.3, bounds=[(0.01, 0.49)], method="L-BFGS-B"
        )

        d_hat = float(result.x[0])
        self.logger.debug(
            f"Whittle optimization result: d={d_hat:.4f}, loss={result.fun:.4f}"
        )
        return d_hat

    def _rs_estimator(self) -> float:
        """
        Estimate d via R/S (rescaled range) analysis for Hurst exponent.

        Computes Hurst exponent H, then d = H - 0.5.

        Returns
        -------
        float
            Estimated d.
        """
        data = self.Q_norm.values
        H = self._compute_hurst_exponent(data)
        d = H - 0.5
        d = np.clip(d, 0.01, 0.49)
        self.logger.debug(f"R/S estimator: H={H:.4f}, d={d:.4f}")
        return float(d)

    def _gph_estimator(self) -> float:
        """
        Estimate d via GPH (Geweke-Porter-Hudak) log-periodogram regression.

        Uses the low-frequency region of the periodogram.

        Returns
        -------
        float
            Estimated d.
        """
        data = self.Q_norm.values
        n = len(data)

        # Periodogram
        dft = np.fft.fft(data)
        I = (np.abs(dft) ** 2) / (2 * np.pi * n)

        # Use low frequencies
        m = int(np.sqrt(n))  # Number of low frequencies
        freqs = np.fft.fftfreq(n)
        idx = np.arange(1, m + 1)

        I_freqs = I[idx]
        w_freqs = 2 * np.pi * freqs[idx]

        # GPH regression (Geweke & Porter-Hudak 1983):
        # log I(w_j) = c - d * log|1 - e^{-iw_j}|^2 + u_j
        # Since |1-e^{-iw}|^2 = 2(1-cos(w)):
        #   log I(w_j) = c - d * log(2(1 - cos(w_j))) + u_j
        # Regressing on log(2(1-cos(w))), slope = -d.
        x = np.log(2.0 * (1.0 - np.cos(w_freqs)))
        y = np.log(I_freqs)

        # Remove NaN/Inf
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        # Linear regression: slope = -d
        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            d_hat = -coeffs[0]
            d_hat = np.clip(d_hat, 0.01, 0.49)
        else:
            d_hat = 0.3

        self.logger.debug(f"GPH estimator: d={d_hat:.4f}")
        return float(d_hat)

    def _compute_hurst_exponent(
        self, data: np.ndarray, lags: Optional[int] = None
    ) -> float:
        """
        Compute Hurst exponent via R/S analysis.

        This is a crude estimator (range of the global cumulative sum over a
        narrow block-size range) intended only for comparison against the
        Whittle and GPH estimates, not for production use.

        Parameters
        ----------
        data : np.ndarray
            Time series data.
        lags : int, optional
            Number of lags to analyze. Default is int(sqrt(len(data))).

        Returns
        -------
        float
            Estimated Hurst exponent H.
        """
        if lags is None:
            lags = int(np.sqrt(len(data)))

        tau = []
        for k in range(10, min(lags, len(data) // 2)):
            # Mean-centered cumulative sum
            y = np.cumsum(data - np.mean(data))

            # Reshape into chunks of size k
            n_chunks = len(y) // k
            if n_chunks == 0:
                break

            y_reshaped = y[: n_chunks * k].reshape(n_chunks, k)

            # Range for each chunk
            R = np.max(y_reshaped, axis=1) - np.min(y_reshaped, axis=1)

            # Standard deviation for each chunk
            S = np.std(data[: n_chunks * k].reshape(n_chunks, k), axis=1)

            # Avoid division by zero
            S[S == 0] = 1

            # R/S statistic
            rs = np.mean(R / S)
            tau.append((k, rs))

        # Linear regression: log(R/S) = H * log(k) + const
        if len(tau) > 1:
            lags_log = np.log([t[0] for t in tau])
            rs_log = np.log([t[1] for t in tau])

            coeffs = np.polyfit(lags_log, rs_log, 1)
            H = coeffs[0]
        else:
            H = 0.5

        return float(np.clip(H, 0.1, 1.0))

    def _compute_fractional_diff_coefficients(self, d: float) -> np.ndarray:
        """
        Compute fractional differencing coefficients pi_k.

        pi_0 = 1
        pi_k = pi_{k-1} * (k - 1 - d) / k, for k >= 1

        Parameters
        ----------
        d : float
            Fractional differencing parameter.

        Returns
        -------
        np.ndarray
            Array of coefficients [pi_0, pi_1, ..., pi_K].
        """
        K = self.truncation_lag
        pi = np.zeros(K + 1)
        pi[0] = 1.0

        for k in range(1, K + 1):
            pi[k] = pi[k - 1] * (k - 1 - d) / k

        return pi

    def _apply_fractional_differencing(self) -> pd.Series:
        """
        Apply fractional differencing to obtain differenced series.

        W_t = sum_{k=0}^{K} pi_k * X_{t-k}

        Returns
        -------
        pd.Series
            Fractionally differenced series.
        """
        X = self.Q_norm.values
        K = self.truncation_lag
        W = np.zeros(len(X))

        for t in range(len(X)):
            for k in range(min(t + 1, K + 1)):
                W[t] += self.pi_coeffs[k] * X[t - k]

        W_series = pd.Series(W, index=self.Q_norm.index)
        return W_series

    @staticmethod
    def _compute_css_residuals(
        W: np.ndarray, phi: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute one-step-ahead prediction residuals for ARMA(p,q).

        Parameters
        ----------
        W : np.ndarray
            Fractionally differenced series.
        phi : np.ndarray
            AR coefficients (may be empty).
        theta : np.ndarray
            MA coefficients (may be empty).

        Returns
        -------
        np.ndarray
            Residuals (innovations) of the same length as *W*.
        """
        n = len(W)
        p = len(phi)
        q = len(theta)
        eps = np.zeros(n)
        for t in range(n):
            ar_part = 0.0
            for k in range(min(p, t)):
                ar_part += phi[k] * W[t - 1 - k]
            ma_part = 0.0
            for j in range(min(q, t)):
                ma_part += theta[j] * eps[t - 1 - j]
            eps[t] = W[t] - ar_part - ma_part
        return eps

    def _fit_arma_css(
        self, data: pd.Series, p: int, q: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit ARMA(p,q) via Conditional Sum of Squares (CSS).

        CSS minimizes the sum of squared one-step-ahead prediction errors.
        Asymptotically equivalent to MLE (Chung & Baillie 1993).

        Parameters
        ----------
        data : pd.Series
            Fractionally differenced series (centred).
        p : int
            AR order.
        q : int
            MA order.

        Returns
        -------
        phi : np.ndarray
            AR coefficients [phi_1, ..., phi_p].
        theta : np.ndarray
            MA coefficients [theta_1, ..., theta_q].
        """
        W = data.values.copy()
        W = W - np.mean(W)
        n = len(W)

        # Initial guess: Yule-Walker for AR, zeros for MA
        if p > 0:
            phi0 = self._fit_ar(data, p)
        else:
            phi0 = np.array([])
        theta0 = np.zeros(q)
        x0 = np.concatenate([phi0, theta0])

        def css_objective(params: np.ndarray) -> float:
            phi_c = params[:p]
            theta_c = params[p:]
            if not self._arma_is_admissible(phi_c, theta_c):
                return 1e300
            eps = self._compute_css_residuals(W, phi_c, theta_c)
            burn = max(p, q, 1)
            ssr = float(np.sum(eps[burn:] ** 2))
            return ssr if np.isfinite(ssr) else 1e300

        bounds = [(-0.99, 0.99)] * (p + q)
        result = minimize(
            css_objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )

        if not result.success:
            self.logger.warning(
                "CSS optimisation did not converge (%s). "
                "Falling back to AR-only (Yule-Walker).",
                result.message,
            )
            return phi0 if p > 0 else np.array([]), np.zeros(q)

        phi_fit = result.x[:p]
        theta_fit = result.x[p:]
        return phi_fit, theta_fit

    def _select_order_two_stage(self, data: pd.Series) -> Tuple[int, int]:
        """Select ARMA(p,q) order for the two-stage estimators.

        Searches p in {0, 1, 2} and q in {0, 1, 2} on the fractionally
        differenced series and applies ``order_criterion`` (AIC per
        Hosking 1984 Sec. 5.1, or BIC, which is consistent for ARFIMA
        order selection, Huang et al. 2022).

        Parameters
        ----------
        data : pd.Series
            Fractionally differenced series.

        Returns
        -------
        best_p : int
            Selected AR order.
        best_q : int
            Selected MA order.
        """
        W = data.values.copy()
        W = W - np.mean(W)
        n = len(W)

        best_val = np.inf
        best_p, best_q = self.p, self.q

        for p_cand in range(3):
            for q_cand in range(3):
                try:
                    if q_cand > 0:
                        phi_c, theta_c = self._fit_arma_css(data, p_cand, q_cand)
                    elif p_cand > 0:
                        phi_c = self._fit_ar(data, p_cand)
                        theta_c = np.array([])
                    else:
                        phi_c = np.array([])
                        theta_c = np.array([])

                    eps = self._compute_css_residuals(W, phi_c, theta_c)
                    burn = max(p_cand, q_cand, 1)
                    n_eff = n - burn
                    sigma2 = float(np.var(eps[burn:]))
                    if sigma2 <= 0:
                        continue
                    val = self._information_criterion(
                        sigma2, n_eff, p_cand + q_cand + 1
                    )

                    self.logger.debug(
                        "%s(%d,%d) = %.2f  (sigma2=%.4f)",
                        self.order_criterion.upper(),
                        p_cand,
                        q_cand,
                        val,
                        sigma2,
                    )

                    if val < best_val:
                        best_val = val
                        best_p, best_q = p_cand, q_cand
                except Exception as exc:
                    self.logger.debug(
                        "Order search: ARMA(%d,%d) failed: %s",
                        p_cand,
                        q_cand,
                        exc,
                    )

        return best_p, best_q

    def _fit_ar(self, data: pd.Series, p: int) -> np.ndarray:
        """
        Fit AR(p) model using Yule-Walker equations.

        Parameters
        ----------
        data : pd.Series
            Time series to fit.
        p : int
            AR order.

        Returns
        -------
        np.ndarray
            AR coefficients [phi_1, phi_2, ..., phi_p].
        """
        data = data.values
        data = data - np.mean(data)  # Center

        # Autocovariance
        acov = np.array(
            [
                np.mean(data[:-k] * data[k:]) if k > 0 else np.var(data)
                for k in range(p + 1)
            ]
        )

        # Yule-Walker system
        R = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                R[i, j] = acov[abs(i - j)]

        r = acov[1 : p + 1]

        try:
            phi = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            self.logger.warning("Singular matrix in Yule-Walker; using least-squares")
            phi = np.linalg.lstsq(R, r, rcond=None)[0]

        return phi

    def _compute_fitted_params(self) -> FittedParams:
        """
        Extract and package fitted parameters.

        Returns
        -------
        FittedParams
            Dataclass containing all fitted ARFIMA parameters.
        """
        n_params = 1 + self.p + self.q + 1  # d, AR, MA, sigma_eps^2
        if self._is_monthly:
            n_params += 36  # 12 tau + 12 means + 12 stds
        else:
            n_params += 3  # 1 tau + 1 mean + 1 std

        training_period = (
            str(self.Q_obs.index[0].date()),
            str(self.Q_obs.index[-1].date()),
        )

        fitted_models = {
            "d": self.d,
            "phi": self.phi.tolist() if len(self.phi) > 0 else None,
            "theta": self.theta.tolist() if len(self.theta) > 0 else None,
            "sigma_eps_sq": float(self.sigma_eps_sq),
            "pi_coefficients": self.pi_coeffs.tolist(),
            "truncation_lag": self.truncation_lag,
        }

        tau = self.log_transform.params_["tau"]
        scaler_mean = self.scaler.params_["mean"]
        scaler_std = self.scaler.params_["std"]
        fitted_models["transformation"] = {
            "stedinger_tau": tau.to_dict() if hasattr(tau, "to_dict") else float(tau),
            "log_mean": (
                scaler_mean.to_dict()
                if hasattr(scaler_mean, "to_dict")
                else float(scaler_mean)
            ),
            "log_std": (
                scaler_std.to_dict()
                if hasattr(scaler_std, "to_dict")
                else float(scaler_std)
            ),
        }

        return FittedParams(
            means_=None,
            stds_=None,
            correlations_=None,
            distributions_={
                "type": "normal_with_fractional_differencing",
                "assumption": f"ARFIMA({self.p},{self.d:.4f},{self.q}) with Gaussian innovations",
            },
            fitted_models_=fitted_models,
            n_parameters_=n_params,
            sample_size_=len(self.Q_obs),
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
        Generate synthetic streamflow realizations.

        Sequence:
        1. Generate white noise innovations
        2. Apply AR recursion to obtain ARMA differenced series W_t
        3. Invert fractional differencing via MA convolution (FIR filter) to recover X_t
        4. Un-standardize and inverse Stedinger transform to original streamflow units
        5. Return as Ensemble

        Parameters
        ----------
        n_realizations : int, default=1
            Number of synthetic realizations to generate.
        n_years : int, optional
            Number of years to generate. If None, uses length of training data.
        n_timesteps : int, optional
            Number of timesteps to generate. Overrides n_years if provided.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : dict, optional
            Additional parameters (unused).

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

        rng = np.random.default_rng(seed)

        # Determine number of timesteps
        if n_timesteps is not None:
            n_timesteps_final = n_timesteps
        elif n_years is not None:
            if self._is_monthly:
                n_timesteps_final = n_years * 12
            else:
                n_timesteps_final = n_years
        else:
            n_timesteps_final = len(self.Q_obs)

        if n_timesteps_final <= 0:
            raise ValueError(f"n_timesteps must be positive, got {n_timesteps_final}")

        # Generate realizations
        realizations = {}
        for i in range(n_realizations):
            Q_syn = self._generate_single(n_timesteps_final, rng=rng)
            realizations[i] = Q_syn.to_frame(name=self._sites[0])

        self.logger.info(
            f"Generated {n_realizations} realizations of {n_timesteps_final} timesteps each"
        )

        # Create metadata
        metadata = EnsembleMetadata(
            generator_class=self.__class__.__name__,
            generator_params=self.get_params(),
            n_realizations=n_realizations,
            n_sites=1,
            time_resolution="monthly" if self._is_monthly else "annual",
            description=f"ARFIMA({self.p},{self.d:.4f},{self.q}) with d_method={self.d_method}",
        )

        return Ensemble(realizations, metadata=metadata)

    def _generate_single(
        self, n_timesteps: int, *, rng: np.random.Generator
    ) -> pd.Series:
        """
        Generate a single realization of synthetic flows.

        Parameters
        ----------
        n_timesteps : int
            Number of timesteps to generate.
        rng : np.random.Generator
            Random number generator instance.

        Returns
        -------
        pd.Series
            Single realization of synthetic flows.
        """
        # Burn-in: the ARMA recursion starts from W_0 = eps_0 and the FIR
        # inversion below only reaches its full (truncated) variance once
        # t >= K.  Simulate extra steps and discard them so the returned
        # series starts in the (truncated) stationary regime.  The extra
        # draws come from the same rng stream, so the seed contract is
        # unchanged.
        burn_in = self._burn_in_length()
        n_total = n_timesteps + burn_in

        # Generate ARMA innovations
        eps = rng.normal(0, np.sqrt(self.sigma_eps_sq), n_total)

        # Apply ARMA(p,q) recursion to get differenced series
        W = np.zeros(n_total)
        for t in range(n_total):
            W[t] = eps[t]
            for k in range(min(t, self.p)):
                W[t] += self.phi[k] * W[t - 1 - k]
            for j in range(min(t, self.q)):
                W[t] += self.theta[j] * eps[t - 1 - j]

        # Invert fractional differencing via MA convolution (Hosking 1984):
        # X_t = sum_{k=0}^{K} psi_k * W_{t-k}
        # This is a FIR filter (convolution), NOT an AR recursion.  The
        # truncation at K caps the variance at sum_k psi_k^2 (97% of the
        # exact value at d = 0.3 with K = 100, but only 55% at d = 0.45).
        psi = self._compute_inverse_fractional_diff_coefficients(self.d)
        X_full = np.zeros(n_total)

        for t in range(n_total):
            for k in range(min(t + 1, len(psi))):
                X_full[t] += psi[k] * W[t - k]

        X = X_full[burn_in:]

        # Create index first so the per-month transforms can recover calendar info
        if self._is_monthly:
            start_date = self.Q_obs.index[-1] + pd.DateOffset(months=1)
            index = make_output_index(start_date, n_timesteps, "MS")
        else:
            start_date = self.Q_obs.index[-1] + pd.DateOffset(years=1)
            index = make_output_index(start_date, n_timesteps, "YS")

        # Reverse the two-stage transformation: un-standardize in log space,
        # then exponentiate and add the Stedinger lower bound. The shifted
        # lognormal Q = tau + exp(Y) is strictly positive by construction.
        # The Series name must match the fit-time site name so the per-column
        # tau lookup inside SteddingerTransform aligns correctly.
        Y_norm = pd.Series(X, index=index, name=self._sites[0])
        Y_log = self.scaler.inverse_transform(Y_norm)
        Q_synth = self.log_transform.inverse_transform(Y_log)

        return Q_synth

    def _burn_in_length(self) -> int:
        """
        Number of leading simulated steps discarded in ``_generate_single``.

        Covers the full FIR filter length ``truncation_lag`` plus a margin
        for the ARMA(p,q) recursion transient, which decays geometrically.

        Returns
        -------
        int
            Burn-in length in timesteps (at least ``truncation_lag``).
        """
        arma_margin = max(50, 10 * max(self.p, self.q))
        return int(self.truncation_lag + arma_margin)

    def _compute_inverse_fractional_diff_coefficients(self, d: float) -> np.ndarray:
        """
        Compute inverse fractional differencing coefficients psi_k.

        psi_0 = 1
        psi_k = psi_{k-1} * (k - 1 + d) / k, for k >= 1

        Parameters
        ----------
        d : float
            Fractional differencing parameter.

        Returns
        -------
        np.ndarray
            Array of inverse coefficients [psi_0, psi_1, ..., psi_K].
        """
        K = self.truncation_lag
        psi = np.zeros(K + 1)
        psi[0] = 1.0

        for k in range(1, K + 1):
            psi[k] = psi[k - 1] * (k - 1 + d) / k

        return psi
