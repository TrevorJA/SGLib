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
from scipy.optimize import minimize
from scipy.signal import correlate

from synhydro.core.base import Generator, FittedParams, GeneratorParams
from synhydro.core.ensemble import Ensemble, EnsembleMetadata

logger = logging.getLogger(__name__)


class ARFIMAGenerator(Generator):
    """
    Autoregressive Fractionally Integrated Moving Average (ARFIMA) generator for synthetic monthly/annual streamflow generation.

    Generates synthetic streamflows using an ARFIMA model that captures long-range
    dependence through fractional differencing parameter d in (0, 0.5). The model
    preserves Hurst exponent, seasonal patterns (if monthly), and autocorrelation
    structure.

    The Hurst exponent H relates to the fractional differencing parameter via H = d + 0.5,
    providing direct parameterization of long-memory behavior.

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
        d_method: str = "whittle",
        truncation_lag: int = 100,
        deseasonalize: bool = True,
        auto_order: bool = False,
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
        d_method : str, default='whittle'
            Method for estimating d: 'whittle' (frequency domain MLE),
            'gph' (Geweke-Porter-Hudak), or 'rs' (R/S analysis).
        truncation_lag : int, default=100
            Truncation lag K for fractional differencing coefficients.
        deseasonalize : bool, default=True
            Remove seasonal component (monthly means/stds) before fitting.
            Set False for annual data.
        auto_order : bool, default=False
            If True, select (p, q) via BIC grid search over
            p in {0, 1, 2} and q in {0, 1, 2}.  Overrides user-supplied
            p and q values.  Uses BIC which is proven consistent for
            ARFIMA (Huang et al. 2022, Annals of Statistics).
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
        self.deseasonalize = deseasonalize
        self.auto_order = auto_order

        # Store initialization parameters
        self.init_params.algorithm_params = {
            "p": p,
            "q": q,
            "d_method": d_method,
            "truncation_lag": truncation_lag,
            "deseasonalize": deseasonalize,
            "auto_order": auto_order,
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

        Validates input, ensures univariate data, optionally deseasonalizes
        for monthly data, and checks stationarity.

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

        # Deseasonalize if monthly
        if self._is_monthly and self.deseasonalize:
            self._deseasonalize_data()
        else:
            self.Q_norm = self.Q_obs.copy()
            self.seasonal_params = None

        self.update_state(preprocessed=True)
        self.logger.info(
            f"Preprocessing complete: {len(self.Q_obs)} timesteps, "
            f"frequency={'monthly' if self._is_monthly else 'annual'}"
        )

    def _deseasonalize_data(self) -> None:
        """
        Remove monthly seasonality from monthly data.

        Computes monthly means and stds, then standardizes the data
        to create a stationary residual series.
        """
        monthly_means = self.Q_obs.groupby(self.Q_obs.index.month).mean()
        monthly_stds = self.Q_obs.groupby(self.Q_obs.index.month).std()

        # Avoid division by zero
        monthly_stds = monthly_stds.replace(0, 1)

        # Standardize by month
        months = self.Q_obs.index.month
        self.Q_norm = (self.Q_obs - monthly_means[months].values) / monthly_stds[
            months
        ].values
        self.Q_norm.index = self.Q_obs.index

        self.seasonal_params = {"means": monthly_means, "stds": monthly_stds}

    def fit(self, Q_obs=None, *, sites=None, **kwargs) -> None:
        """
        Estimate ARFIMA model parameters from preprocessed data.

        Sequence:
        1. Estimate fractional differencing parameter d using specified method
        2. Apply fractional differencing to obtain differenced series
        3. Fit ARMA(p,q) to differenced series using Yule-Walker equations
        4. Store all fitted parameters

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

        # Estimate d
        self.logger.info(f"Estimating d using {self.d_method} method...")
        self.d = self._estimate_d()
        self.logger.info(
            f"Estimated d = {self.d:.4f}, Hurst exponent H = {self.d + 0.5:.4f}"
        )

        # Compute fractional differencing coefficients
        self.pi_coeffs = self._compute_fractional_diff_coefficients(self.d)

        # Apply fractional differencing
        self.W = self._apply_fractional_differencing()

        # BIC-based automatic order selection (Huang et al. 2022)
        if self.auto_order:
            self.p, self.q = self._select_order_bic(self.W)
            self.logger.info(
                "BIC selected ARMA(%d,%d) for short-memory component",
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

        # Innovation variance from one-step-ahead prediction errors
        W_vals = self.W.values
        residuals = self._compute_css_residuals(W_vals, self.phi, self.theta)
        burn_in = max(self.p, self.q, 1)
        self.sigma_eps_sq = float(np.var(residuals[burn_in:]))

        self.update_state(fitted=True)
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(f"Fitting complete: sigma_eps^2 = {self.sigma_eps_sq:.4f}")

    def _estimate_d(self) -> float:
        """
        Estimate fractional differencing parameter d.

        Uses Whittle estimator (frequency-domain MLE) by default.

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
        Estimate d via Whittle likelihood in frequency domain.

        Minimizes the Whittle likelihood:
        L(d) = sum_j [ log f(w_j; d) + I(w_j) / f(w_j; d) ]

        where I(w_j) is the periodogram and f(w_j; d) is the spectral
        density of the ARFIMA(p,d,q) process.

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
            #   f(w) ∝ |1 - e^{-iw}|^{-2d}
            # Since |1 - e^{-iw}|^2 = 2(1 - cos(w)):
            #   f(w) ∝ [2(1 - cos(w))]^{-d}
            # Ref: Hosking (1981) eq. 2.3; Beran (1994) Ch. 5
            g_w = (2.0 * (1.0 - np.cos(w))) ** (-d_test)

            # Avoid log(0)
            g_w = np.maximum(g_w, 1e-10)

            # Profile Whittle likelihood (Fox & Taqqu 1986)
            likelihood = np.sum(np.log(g_w) + I / g_w)
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
            eps = self._compute_css_residuals(W, phi_c, theta_c)
            burn = max(p, q, 1)
            return float(np.sum(eps[burn:] ** 2))

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

    def _select_order_bic(self, data: pd.Series) -> Tuple[int, int]:
        """Select ARMA(p,q) order by BIC grid search.

        Searches p in {0, 1, 2} and q in {0, 1, 2}.  BIC is proven
        consistent for ARFIMA order selection (Huang et al. 2022,
        Annals of Statistics 50(3)).

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

        best_bic = np.inf
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
                    k = p_cand + q_cand
                    bic_val = n_eff * np.log(sigma2) + k * np.log(n_eff)

                    self.logger.debug(
                        "BIC(%d,%d) = %.2f  (sigma2=%.4f)",
                        p_cand,
                        q_cand,
                        bic_val,
                        sigma2,
                    )

                    if bic_val < best_bic:
                        best_bic = bic_val
                        best_p, best_q = p_cand, q_cand
                except Exception as exc:
                    self.logger.debug(
                        "BIC search: ARMA(%d,%d) failed: %s",
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
        if self.seasonal_params:
            n_params += 24  # 12 means + 12 stds

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

        if self.seasonal_params:
            fitted_models["seasonal"] = {
                "means": self.seasonal_params["means"].to_dict(),
                "stds": self.seasonal_params["stds"].to_dict(),
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
        4. Re-seasonalize if monthly
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
        # Generate ARMA innovations
        eps = rng.normal(0, np.sqrt(self.sigma_eps_sq), n_timesteps)

        # Apply ARMA(p,q) recursion to get differenced series
        W = np.zeros(n_timesteps)
        for t in range(n_timesteps):
            W[t] = eps[t]
            for k in range(min(t, self.p)):
                W[t] += self.phi[k] * W[t - 1 - k]
            for j in range(min(t, self.q)):
                W[t] += self.theta[j] * eps[t - 1 - j]

        # Invert fractional differencing via MA convolution (Hosking 1984):
        # X_t = sum_{k=0}^{K} psi_k * W_{t-k}
        # This is a FIR filter (convolution), NOT an AR recursion.
        psi = self._compute_inverse_fractional_diff_coefficients(self.d)
        X = np.zeros(n_timesteps)

        for t in range(n_timesteps):
            for k in range(min(t + 1, len(psi))):
                X[t] += psi[k] * W[t - k]

        # Create index first so we know the start month
        if self._is_monthly:
            start_date = self.Q_obs.index[-1] + pd.DateOffset(months=1)
            index = pd.date_range(start=start_date, periods=n_timesteps, freq="MS")
        else:
            start_date = self.Q_obs.index[-1] + pd.DateOffset(years=1)
            index = pd.date_range(start=start_date, periods=n_timesteps, freq="YS")

        # Re-seasonalize if monthly
        if self._is_monthly and self.seasonal_params:
            X = self._re_seasonalize(X, start_month=index[0].month)

        # Enforce non-negativity
        X = np.maximum(X, 0)

        return pd.Series(X, index=index)

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

    def _re_seasonalize(self, X: np.ndarray, start_month: int) -> np.ndarray:
        """
        Re-apply seasonal component (multiply by monthly stds and add means).

        Parameters
        ----------
        X : np.ndarray
            Deseasonalized synthetic flows.
        start_month : int
            Calendar month (1-12) of the first timestep.

        Returns
        -------
        np.ndarray
            Re-seasonalized flows.
        """
        if not self.seasonal_params:
            return X

        means = self.seasonal_params["means"].values
        stds = self.seasonal_params["stds"].values

        # Build month indices aligned to the actual start month
        months = np.array([(start_month - 1 + i) % 12 for i in range(len(X))])

        X_reseasonal = X * stds[months] + means[months]

        return X_reseasonal
