"""
Nataf joint distribution model utilities.

Provides forward and inverse Nataf transformations for mapping between
Gaussian-domain and actual-domain correlation coefficients, as well as
autocorrelation structures (CAS, Hurst) and SMA weight computation.

Reference implementation: anySim R package (Tsoukalas et al., 2020).
"""

import logging
import warnings
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.polynomial import hermite_e as hermite
from scipy import integrate
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Type alias for inverse CDF (quantile) functions: p -> x
ICDF = Callable[[np.ndarray], np.ndarray]

# Open-interval bounds for probabilities fed to ICDFs in the Int method
_U_MIN = np.nextafter(0.0, 1.0)
_U_MAX = np.nextafter(1.0, 0.0)


# ---------------------------------------------------------------------------
# Forward Nataf: Gaussian-domain rho -> actual-domain rho
# ---------------------------------------------------------------------------


def nataf_forward_gh(
    rho_z: Union[float, np.ndarray],
    icdf_x: ICDF,
    icdf_y: ICDF,
    nodes: int = 21,
    prune: float = 0.0,
) -> np.ndarray:
    """Evaluate the Nataf integral via Gauss-Hermite quadrature.

    Given equivalent (Gaussian-domain) correlations ``rho_z``, compute the
    resulting actual-domain correlations after mapping through the ICDFs.

    Parameters
    ----------
    rho_z : float or array-like
        Equivalent correlation coefficient(s) in [-1, 1].
    icdf_x : callable
        Quantile function (ICDF) for variable x. Signature: p -> x.
    icdf_y : callable
        Quantile function (ICDF) for variable y.
    nodes : int
        Number of Gauss-Hermite quadrature nodes per dimension (default 21).
    prune : float
        Fraction of low-weight quadrature points to discard, in [0, 1).

    Returns
    -------
    np.ndarray
        Actual-domain correlation(s), same shape as ``rho_z``.

    References
    ----------
    NatafGH.R in anySim R package.
    """
    rho_z = np.atleast_1d(np.asarray(rho_z, dtype=float))

    # Physicist's Hermite quadrature nodes and weights
    # numpy hermegauss gives probabilist's; we need physicist's convention
    # matching R's pracma::gaussHermite
    gh_x, gh_w = np.polynomial.hermite.hermgauss(nodes)

    # Scale: nodes * sqrt(2), weights / sqrt(pi) -- matches R code
    gh_x = gh_x * np.sqrt(2.0)
    gh_w = gh_w / np.sqrt(np.pi)

    # Build 2D grid
    idx_i, idx_j = np.meshgrid(np.arange(nodes), np.arange(nodes), indexing="ij")
    pts_raw = np.column_stack([gh_x[idx_i.ravel()], gh_x[idx_j.ravel()]])
    wts = gh_w[idx_i.ravel()] * gh_w[idx_j.ravel()]

    # Prune low-weight points
    if prune > 0:
        threshold = np.quantile(wts, prune)
        mask = wts > threshold
        pts_raw = pts_raw[mask]
        wts = wts[mask]

    rho_x = np.empty_like(rho_z)

    for t in range(len(rho_z)):
        rz = rho_z[t]

        if rz == 0.0:
            rho_x[t] = 0.0
            continue

        # Rotate/scale points by correlation matrix eigendecomposition
        sigma = np.array([[1.0, rz], [rz, 1.0]])
        eigvals, eigvecs = np.linalg.eigh(sigma)
        eigvals = np.maximum(eigvals, 0.0)  # guard numerical negatives
        rot = eigvecs * np.sqrt(eigvals)
        pts = pts_raw @ rot.T  # (n_pts, 2)

        # Map to uniform via standard normal CDF
        # Match anySim: only replace u == 1 (here u >= 1 - eps) with 0.999.
        # The 0.999 clamp is inherited verbatim from anySim
        # (ifelse(u == 1, 0.999, u)); it only affects the outermost
        # Gauss-Hermite nodes, whose weights are negligible.
        # A symmetric tiny floor guards ICDFs undefined at exactly 0.
        u = norm.cdf(pts)
        u = np.where(u >= 1.0 - 1e-15, 0.999, u)
        u = np.maximum(u, 1e-300)

        # Evaluate ICDFs
        x1 = icdf_x(u[:, 0])
        x2 = icdf_y(u[:, 1])

        # Weighted correlation
        rho_x[t] = _weighted_correlation(x1, x2, wts)

    return rho_x


def nataf_forward_mc(
    rho_z: Union[float, np.ndarray],
    icdf_x: ICDF,
    icdf_y: ICDF,
    n_samples: int = 100_000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Evaluate the Nataf integral via Monte Carlo simulation.

    Parameters
    ----------
    rho_z : float or array-like
        Equivalent correlation coefficient(s) in [-1, 1].
    icdf_x : callable
        Quantile function for variable x.
    icdf_y : callable
        Quantile function for variable y.
    n_samples : int
        Number of Monte Carlo samples (default 100,000).
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Actual-domain correlation(s).

    References
    ----------
    NatafMC.R in anySim R package.
    """
    rho_z = np.atleast_1d(np.asarray(rho_z, dtype=float))
    if rng is None:
        rng = np.random.default_rng()

    z1 = rng.standard_normal(n_samples)
    z2_indep = rng.standard_normal(n_samples)

    rho_x = np.empty_like(rho_z)

    for t in range(len(rho_z)):
        rz = rho_z[t]

        if rz == 0.0:
            rho_x[t] = 0.0
            continue

        # Correlate
        z2 = rz * z1 + np.sqrt(1.0 - rz**2) * z2_indep

        # Map through CDF then ICDFs
        u1 = norm.cdf(z1)
        u2 = norm.cdf(z2)

        x1 = icdf_x(u1)
        x2 = icdf_y(u2)

        rho_x[t] = np.corrcoef(x1, x2)[0, 1]

    return rho_x


def nataf_forward_int(
    rho_z: Union[float, np.ndarray],
    icdf_x: ICDF,
    icdf_y: ICDF,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
) -> np.ndarray:
    """Evaluate the Nataf integral via 2D numerical integration.

    Parameters
    ----------
    rho_z : float or array-like
        Equivalent correlation coefficient(s) in [-1, 1].
    icdf_x : callable
        Quantile function for variable x.
    icdf_y : callable
        Quantile function for variable y.
    epsabs, epsrel : float
        Absolute and relative tolerances passed to ``scipy.integrate.dblquad``.

    Returns
    -------
    np.ndarray
        Actual-domain correlation(s).

    Raises
    ------
    RuntimeError
        If the integration does not yield a finite value on any of the
        progressively narrowed domains [-lim, lim]^2, lim = 7, 6, ..., 3.
        Raising (rather than returning NaN) prevents ``nataf_inverse`` from
        silently fitting its polynomial through NaN support points.

    References
    ----------
    NatafInt.R in anySim R package.
    """
    rho_z = np.atleast_1d(np.asarray(rho_z, dtype=float))

    # Compute distribution statistics (mean, variance) via integration of ICDF
    mu_x, var_x = _distribution_stats_from_icdf(icdf_x)
    mu_y, var_y = _distribution_stats_from_icdf(icdf_y)
    sigma_x = np.sqrt(var_x)
    sigma_y = np.sqrt(var_y)

    q1 = -(mu_x * mu_y) / (sigma_x * sigma_y)
    q2 = 1.0 / (2.0 * np.pi * sigma_x * sigma_y)

    rho_x = np.empty_like(rho_z)

    for t in range(len(rho_z)):
        rz = rho_z[t]

        if rz == 0.0:
            rho_x[t] = 0.0
            continue

        lim = 7.0

        def integrand(u2, u1):
            z2_corr = rz * u1 + np.sqrt(1.0 - rz**2) * u2
            # The rotated coordinate can reach |z| ~ lim * sqrt(2), where
            # norm.cdf rounds to exactly 0 or 1 in double precision and the
            # ICDF returns +/-inf. Keep u strictly inside (0, 1); the
            # Gaussian weight there is < 1e-20 so the clip is harmless.
            u1_p = np.clip(norm.cdf(np.atleast_1d(u1)), _U_MIN, _U_MAX)
            u2_p = np.clip(norm.cdf(np.atleast_1d(z2_corr)), _U_MIN, _U_MAX)
            val = icdf_x(u1_p)[0] * icdf_y(u2_p)[0] * np.exp(-0.5 * (u1**2 + u2**2))
            return val

        # Adaptive integration over [-lim, lim]^2; on a genuine integration
        # failure (non-finite value or numerical error in the integrand)
        # retry with a narrower domain. Programming errors propagate.
        result = np.nan
        while lim >= 3.0:
            try:
                val, _ = integrate.dblquad(
                    integrand, -lim, lim, -lim, lim, epsabs=epsabs, epsrel=epsrel
                )
            except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError) as exc:
                logger.debug(
                    "Nataf integration at rho_z=%.4f, lim=%.1f failed: %s", rz, lim, exc
                )
            else:
                if np.isfinite(val):
                    result = val
                    break
            lim -= 1.0

        if not np.isfinite(result):
            raise RuntimeError(
                f"Nataf numerical integration failed for rho_z={rz:.4f} on all "
                "integration domains (lim 7 down to 3). Check that the ICDFs are "
                "finite on (0, 1), or use method='GH' or 'MC'."
            )
        rho_x[t] = q1 + q2 * result

    return rho_x


# ---------------------------------------------------------------------------
# Inverse Nataf: target rho -> equivalent rho
# ---------------------------------------------------------------------------


def nataf_inverse(
    target_rho: Union[float, np.ndarray],
    icdf_x: ICDF,
    icdf_y: ICDF,
    method: str = "GH",
    n_eval: int = 9,
    poly_deg: int = 8,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert the Nataf relationship to find equivalent correlations.

    Given target actual-domain correlations, find the Gaussian-domain
    correlations that produce them after ICDF mapping.

    Parameters
    ----------
    target_rho : float or array-like
        Target correlation(s) in the actual domain.
    icdf_x : callable
        Quantile function for variable x.
    icdf_y : callable
        Quantile function for variable y.
    method : str
        Forward Nataf method: ``"GH"`` (default), ``"MC"``, or ``"Int"``.
    n_eval : int
        Number of evaluation points for forward mapping (default 9).
    poly_deg : int
        Degree of polynomial for approximating the relationship (default 8).
    **kwargs
        Additional arguments passed to the forward Nataf method.

    Returns
    -------
    rho_equiv : np.ndarray
        Equivalent Gaussian-domain correlations.
    df_nataf : np.ndarray
        Shape (n_eval, 2) array of (rz, rx) support points used for fitting.

    References
    ----------
    NatafInvD.R in anySim R package.
    """
    target_rho = np.atleast_1d(np.asarray(target_rho, dtype=float))

    # Short-circuit: if every target is exactly zero, the grid would
    # collapse (rmin = rmax = 0) and np.polyfit would fail. By Lemma 2,
    # rho_x = 0 iff rho_z = 0, so the answer is identically zero.
    if np.all(target_rho == 0.0):
        rz_grid = np.zeros(n_eval)
        return np.zeros_like(target_rho), np.column_stack([rz_grid, rz_grid])

    # Determine evaluation range based on sign of target correlations
    rmin = -1.0 if np.any(target_rho < 0) else 0.0
    rmax = 1.0 if np.any(target_rho > 0) else 0.0

    # Evaluation grid in Gaussian domain
    rz_grid = np.linspace(rmin, rmax, n_eval)

    # Forward Nataf at support points
    forward_fn = _get_forward_method(method)
    rx_grid = forward_fn(rz_grid, icdf_x, icdf_y, **kwargs)

    # Ensure rho=0 maps to 0
    zero_idx = np.where(rz_grid == 0.0)[0]
    if len(zero_idx) > 0:
        rx_grid[zero_idx] = 0.0

    df_nataf = np.column_stack([rz_grid, rx_grid])

    # Fit polynomial through (rz, rx) pairs
    coeffs = np.polyfit(rz_grid, rx_grid, poly_deg)

    # Dense evaluation of the polynomial for inversion
    rz_fine = np.arange(rmin, rmax + 0.0001, 0.0002)
    rx_fine = np.polyval(coeffs, rz_fine)

    # Invert: given target rx, find rz via interpolation
    # rx_fine should be monotonically increasing (Lemma 1)
    rho_equiv = np.interp(target_rho, rx_fine, rz_fine)

    # Warn when targets fall outside the attainable (Frechet-Hoeffding) range;
    # np.interp clamps them to the boundary rz values (-1 or 1).
    rx_lo, rx_hi = float(rx_fine.min()), float(rx_fine.max())
    below = target_rho < rx_lo
    above = target_rho > rx_hi
    if np.any(below):
        logger.warning(
            "%d target correlation(s) (min %.4f) are below the attainable "
            "Frechet-Hoeffding lower bound %.4f for these marginals; "
            "clipped to the bound (rho_equiv = %.4f).",
            int(np.sum(below)), float(target_rho[below].min()), rx_lo, float(rmin),
        )
    if np.any(above):
        logger.warning(
            "%d target correlation(s) (max %.4f) exceed the attainable upper "
            "bound %.4f for these marginals; clipped to the bound.",
            int(np.sum(above)), float(target_rho[above].max()), rx_hi,
        )

    # Clip to valid range
    rho_equiv = np.clip(rho_equiv, -1.0, 1.0)

    return rho_equiv, df_nataf


# ---------------------------------------------------------------------------
# Autocorrelation structures
# ---------------------------------------------------------------------------


def cas_acf(kappa: float, beta: float, max_lag: int) -> np.ndarray:
    """Compute the Cauchy-type autocorrelation structure (CAS).

    Parameters
    ----------
    kappa : float
        Rate parameter (kappa > 0).
    beta : float
        Shape parameter (beta >= 0). beta=0 gives exponential (SRD),
        beta>1 gives long-range dependence (LRD).
    max_lag : int
        Maximum lag.

    Returns
    -------
    np.ndarray
        ACF values of length ``max_lag + 1``, starting with rho(0) = 1.

    References
    ----------
    csCAS.R in anySim; Eq. 6 in Tsoukalas et al. (2018).
    """
    tau = np.arange(1, max_lag + 1, dtype=float)

    if beta == 0.0:
        acf_vals = np.exp(-kappa * tau)
    else:
        acf_vals = (1.0 + kappa * beta * tau) ** (-1.0 / beta)

    return np.concatenate([[1.0], acf_vals])


def hurst_acf(H: float, max_lag: int) -> np.ndarray:
    """Compute the fractional Gaussian noise (fGn) autocorrelation.

    Parameters
    ----------
    H : float
        Hurst coefficient in (0, 1). H=0.5 is white noise.
    max_lag : int
        Maximum lag.

    Returns
    -------
    np.ndarray
        ACF values of length ``max_lag + 1``.

    References
    ----------
    csHurst.R in anySim; Eq. 7 in Tsoukalas et al. (2018).
    """
    tau = np.arange(1, max_lag + 1, dtype=float)
    two_h = 2.0 * H
    acf_vals = 0.5 * (
        np.abs(tau + 1) ** two_h + np.abs(tau - 1) ** two_h - 2.0 * np.abs(tau) ** two_h
    )
    return np.concatenate([[1.0], acf_vals])


def fit_cas(
    empirical_acf: np.ndarray,
    max_lag: Optional[int] = None,
) -> Tuple[float, float]:
    """Fit CAS parameters to an empirical autocorrelation function.

    Parameters
    ----------
    empirical_acf : np.ndarray
        Empirical ACF starting from lag 0. Must include rho(0) = 1.
    max_lag : int, optional
        Maximum lag to use for fitting. Defaults to len(empirical_acf) - 1.

    Returns
    -------
    kappa : float
        Fitted rate parameter.
    beta : float
        Fitted shape parameter.
    """
    if max_lag is None:
        max_lag = len(empirical_acf) - 1

    target = empirical_acf[1 : max_lag + 1]
    lags = np.arange(1, max_lag + 1, dtype=float)

    def objective(params):
        kappa, beta = params
        if beta == 0.0:
            pred = np.exp(-kappa * lags)
        else:
            pred = (1.0 + kappa * beta * lags) ** (-1.0 / beta)
        return np.mean((target - pred) ** 2)

    result = minimize(
        objective,
        x0=[1.0, 0.5],
        bounds=[(1e-6, 100.0), (0.0, 10.0)],
        method="L-BFGS-B",
    )

    return float(result.x[0]), float(result.x[1])


# ---------------------------------------------------------------------------
# SMA weight computation
# ---------------------------------------------------------------------------


def sma_weights_fft(equivalent_acf: np.ndarray) -> np.ndarray:
    """Compute SMA model weights via FFT of the equivalent ACF.

    Parameters
    ----------
    equivalent_acf : np.ndarray
        Equivalent autocorrelation values starting from lag 0, of length
        ``q + 1`` where q is the SMA order. Must satisfy acf[0] = 1.

    Returns
    -------
    np.ndarray
        SMA weight vector of length ``2q + 1`` (symmetric).

    References
    ----------
    EstSMARTA.R in anySim (FFT section); Koutsoyiannis (2000).
    """
    q = len(equivalent_acf) - 1

    # Build full symmetric ACF: [rho_q, ..., rho_1, rho_0, rho_1, ..., rho_q]
    g0 = equivalent_acf[0]
    g_right = equivalent_acf[1:]
    g_left = g_right[::-1]
    co = np.concatenate([g_left, [g0], g_right])

    # DFT, sqrt of absolute power spectrum, inverse DFT
    co_hat = np.fft.fft(co)
    ft = np.real(np.fft.ifft(np.sqrt(np.abs(co_hat))))

    # Rearrange to symmetric form: [a_q,...,a_1, a_0, a_1,...,a_q]
    # The IFFT result has a_0 at index 0, a_1 at index 1, ..., a_q at index q,
    # then the mirrored part. We need [a_q,...,a_1, a_0, a_1,...,a_q].
    a_0 = ft[0]
    a_right = ft[1 : q + 1]
    a_left = a_right[::-1]
    weights = np.concatenate([a_left, [a_0], a_right])

    # Verify unit variance constraint
    sum_sq = np.sum(weights**2)
    if not np.isclose(sum_sq, 1.0, atol=0.05):
        logger.warning(
            "SMA weights sum of squares = %.4f (expected 1.0). "
            "ACF may not be valid positive-definite.",
            sum_sq,
        )

    return weights


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _weighted_correlation(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted Pearson correlation coefficient.

    Parameters
    ----------
    x, y : np.ndarray
        Data vectors.
    w : np.ndarray
        Weights (non-negative).

    Returns
    -------
    float
        Weighted correlation.
    """
    w_sum = np.sum(w)
    mean_x = np.sum(w * x) / w_sum
    mean_y = np.sum(w * y) / w_sum
    dx = x - mean_x
    dy = y - mean_y
    cov_xy = np.sum(w * dx * dy) / w_sum
    var_x = np.sum(w * dx**2) / w_sum
    var_y = np.sum(w * dy**2) / w_sum
    denom = np.sqrt(var_x * var_y)
    if denom < 1e-15:
        return 0.0
    return cov_xy / denom


def _distribution_stats_from_icdf(
    icdf: ICDF,
    lb: float = 0.0,
    ub: float = 1.0,
) -> Tuple[float, float]:
    """Compute mean and variance of a distribution from its ICDF.

    Integrates the quantile function over [0, 1] to obtain the mean,
    and the squared quantile function for the second moment.

    Parameters
    ----------
    icdf : callable
        Quantile function (ICDF).
    lb : float
        Lower bound of integration (default 0).
    ub : float
        Upper bound of integration (default 1).

    Returns
    -------
    mean : float
        Distribution mean.
    variance : float
        Distribution variance.

    References
    ----------
    DistrStats2.R in anySim.
    """
    mean, _ = integrate.quad(lambda p: icdf(np.atleast_1d(p))[0], lb, ub)
    moment2, _ = integrate.quad(lambda p: icdf(np.atleast_1d(p))[0] ** 2, lb, ub)
    variance = moment2 - mean**2
    return mean, variance


def _get_forward_method(method: str) -> Callable:
    """Return the forward Nataf function for the given method string."""
    methods = {
        "GH": nataf_forward_gh,
        "MC": nataf_forward_mc,
        "Int": nataf_forward_int,
    }
    if method not in methods:
        raise ValueError(
            f"Unknown Nataf method '{method}'. Choose from {list(methods.keys())}."
        )
    return methods[method]
