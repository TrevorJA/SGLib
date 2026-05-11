"""
Statistical Utilities for SynHydro

Specialized statistical functions for flow data analysis, correlation computations,
distribution fitting, long-range dependence estimation, and spectral analysis.

For standardization/normalization, use synhydro.transformations.StandardScaler or
synhydro.transformations.DeseasonalizeTransform instead.
"""

from typing import Optional, Union, Tuple, Dict, List, Literal
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import cholesky
from scipy import signal
import logging

logger = logging.getLogger(__name__)


### Function to compute autocorrelation ###
def compute_autocorrelation(
    data: pd.DataFrame, max_lag: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute autocorrelation for each column in DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input flow data with DatetimeIndex.
    max_lag : int, optional
        Maximum lag to compute. If None, defaults to len(data) - 1.

    Returns
    -------
    pd.DataFrame
        DataFrame of autocorrelations with lags as index and columns as sites.
    """
    n = len(data)
    if max_lag is None:
        max_lag = n - 1

    acf_dict = {}
    for col in data.columns:
        series = data[col].dropna()
        mean = series.mean()
        var = series.var()
        acf_values = []
        for lag in range(max_lag + 1):
            if lag >= len(series):
                acf_values.append(np.nan)
                continue
            cov = (
                ((series[:-lag] - mean) * (series[lag:] - mean)).sum()
                if lag > 0
                else ((series - mean) ** 2).sum()
            )
            acf_values.append(cov / (len(series) - lag) / var)
        acf_dict[col] = acf_values

    acf_df = pd.DataFrame(acf_dict, index=np.arange(max_lag + 1))
    acf_df.index.name = "lag"
    return acf_df


def compute_spatial_correlation(
    data: pd.DataFrame, method: Literal["pearson", "spearman", "kendall"] = "pearson"
) -> pd.DataFrame:
    """
    Compute spatial correlation matrix between columns in DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input flow data with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Correlation matrix with sites as both index and columns.
    """
    corr_matrix = data.corr(method=method)

    return corr_matrix


def compute_monthly_statistics(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute statistics by month.

    Parameters
    ----------
    data : pd.DataFrame
        Flow data with DatetimeIndex.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys: 'mean', 'std', 'min', 'max', 'skew', 'cv'
        Each value is DataFrame with months (1-12) as index, sites as columns.
    """
    monthly = data.groupby(data.index.month)

    return {
        "mean": monthly.mean(),
        "std": monthly.std(),
        "min": monthly.min(),
        "max": monthly.max(),
        "skew": monthly.skew(),
        "cv": monthly.std() / monthly.mean(),  # Coefficient of variation
    }


def repair_correlation_matrix(
    corr: NDArray[np.float64],
    method: Literal["spectral", "hypersphere", "nearest"] = "spectral",
) -> NDArray[np.float64]:
    """
    Repair non-positive definite correlation matrix.

    Parameters
    ----------
    corr : NDArray
        Correlation matrix to repair.
    method : {'spectral', 'hypersphere', 'nearest'}, default='spectral'
        Method for repairing matrix.

    Returns
    -------
    NDArray
        Repaired positive definite correlation matrix.

    Notes
    -----
    - 'spectral': Eigenvalue truncation (set negative eigenvalues to small positive)
    - 'hypersphere': Project onto space of valid correlation matrices
    - 'nearest': Find nearest positive definite matrix (Higham algorithm)
    """
    if method == "spectral":
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr)

        # Truncate negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 1e-8)

        # Reconstruct matrix
        repaired = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Rescale to unit diagonal
        d = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(d, d)

        return repaired

    elif method == "hypersphere":
        # Project correlations onto valid range [-1, 1]
        repaired = corr.copy()
        np.fill_diagonal(repaired, 1.0)
        repaired = np.clip(repaired, -1.0, 1.0)

        # Ensure symmetry
        repaired = (repaired + repaired.T) / 2

        # Check if positive definite, if not use spectral method
        try:
            cholesky(repaired, lower=True)
            return repaired
        except np.linalg.LinAlgError:
            logger.warning("Hypersphere method failed, falling back to spectral")
            return repair_correlation_matrix(corr, method="spectral")

    elif method == "nearest":
        # Higham's algorithm for nearest correlation matrix
        return _nearest_correlation_matrix(corr)

    else:
        raise ValueError(f"Unknown method: {method}")


def _nearest_correlation_matrix(
    A: NDArray[np.float64], max_iter: int = 100, tol: float = 1e-7
) -> NDArray[np.float64]:
    """
    Find nearest correlation matrix using Higham's algorithm.

    Parameters
    ----------
    A : NDArray
        Input matrix.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    NDArray
        Nearest correlation matrix.
    """
    n = A.shape[0]
    Y = A.copy()
    dS = np.zeros_like(A)

    for _ in range(max_iter):
        R = Y - dS

        # Project onto positive semidefinite cone
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Project onto unit diagonal
        dS = X - R
        np.fill_diagonal(X, 1.0)

        # Check convergence
        if np.linalg.norm(Y - X, ord="fro") < tol:
            return X

        Y = X

    logger.warning(
        f"Nearest correlation matrix did not converge in {max_iter} iterations"
    )
    return Y


# =============================================================================
# Long-Range Dependence Estimation
# =============================================================================


def compute_hurst_exponent(
    data: Union[pd.Series, np.ndarray],
    method: Literal["rs", "dfa"] = "rs",
    min_window: int = 10,
    max_window: Optional[int] = None,
) -> Dict[str, float]:
    """
    Estimate the Hurst exponent of a timeseries.

    The Hurst exponent H quantifies long-range dependence:
    H = 0.5 indicates no long-range dependence (random walk).
    H > 0.5 indicates persistent (positively autocorrelated) behavior.
    H < 0.5 indicates anti-persistent behavior.

    For hydrologic timeseries, H is related to the ARFIMA fractional
    differencing parameter d by H = d + 0.5 (Hosking 1984).

    Parameters
    ----------
    data : pd.Series or np.ndarray
        Input timeseries (1-D).
    method : {'rs', 'dfa'}, default='rs'
        Estimation method.
        'rs': Rescaled range (R/S) analysis (Hurst 1951).
        'dfa': Detrended fluctuation analysis (Peng et al. 1994).
    min_window : int, default=10
        Minimum window size for analysis.
    max_window : int, optional
        Maximum window size. If None, defaults to len(data) // 4.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        'H': estimated Hurst exponent,
        'c': intercept of log-log regression,
        'r_squared': R-squared of the log-log fit.

    References
    ----------
    Hurst, H.E. (1951). Long-term storage capacity of reservoirs.
    Transactions of the American Society of Civil Engineers, 116, 770-799.

    Koutsoyiannis, D. (2002). The Hurst phenomenon and fractional Gaussian
    noise made easy. Hydrological Sciences Journal, 47(4), 573-595.
    """
    if isinstance(data, pd.Series):
        x = data.dropna().values
    else:
        x = np.asarray(data, dtype=np.float64)
        x = x[~np.isnan(x)]

    n = len(x)
    if n < 2 * min_window:
        raise ValueError(
            f"Timeseries too short ({n}) for Hurst estimation with "
            f"min_window={min_window}. Need at least {2 * min_window} values."
        )

    if max_window is None:
        max_window = n // 4

    if method == "rs":
        return _hurst_rs(x, min_window, max_window)
    elif method == "dfa":
        return _hurst_dfa(x, min_window, max_window)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rs' or 'dfa'.")


def _hurst_rs(
    x: np.ndarray,
    min_window: int,
    max_window: int,
) -> Dict[str, float]:
    """
    Rescaled range (R/S) analysis for Hurst exponent estimation.

    For each window size w, computes the mean R/S statistic across
    non-overlapping blocks, then estimates H from the log-log slope.
    """
    n = len(x)

    # Generate window sizes (logarithmically spaced)
    window_sizes = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(min(max_window, n // 2)),
            num=20,
        ).astype(int)
    )
    window_sizes = window_sizes[window_sizes >= min_window]

    log_n = []
    log_rs = []

    for w in window_sizes:
        n_blocks = n // w
        if n_blocks < 1:
            continue

        rs_values = []
        for i in range(n_blocks):
            block = x[i * w : (i + 1) * w]
            mean_block = block.mean()
            cumdev = np.cumsum(block - mean_block)
            R = cumdev.max() - cumdev.min()
            S = block.std(ddof=1)
            if S > 0:
                rs_values.append(R / S)

        if len(rs_values) > 0:
            log_n.append(np.log(w))
            log_rs.append(np.log(np.mean(rs_values)))

    log_n = np.array(log_n)
    log_rs = np.array(log_rs)

    if len(log_n) < 3:
        logger.warning("Too few window sizes for reliable Hurst estimation.")
        return {"H": np.nan, "c": np.nan, "r_squared": np.nan}

    # Linear regression in log-log space
    slope, intercept = np.polyfit(log_n, log_rs, 1)

    # R-squared
    ss_res = np.sum((log_rs - (slope * log_n + intercept)) ** 2)
    ss_tot = np.sum((log_rs - log_rs.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"H": float(slope), "c": float(intercept), "r_squared": float(r_squared)}


def _hurst_dfa(
    x: np.ndarray,
    min_window: int,
    max_window: int,
) -> Dict[str, float]:
    """
    Detrended fluctuation analysis for Hurst exponent estimation.

    Computes the RMS of detrended cumulative sums across window sizes,
    then estimates H from the log-log slope of the fluctuation function.
    """
    n = len(x)
    y = np.cumsum(x - x.mean())

    window_sizes = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(min(max_window, n // 2)),
            num=20,
        ).astype(int)
    )
    window_sizes = window_sizes[window_sizes >= min_window]

    log_n = []
    log_f = []

    for w in window_sizes:
        n_blocks = n // w
        if n_blocks < 1:
            continue

        fluctuations = []
        for i in range(n_blocks):
            segment = y[i * w : (i + 1) * w]
            t = np.arange(w)
            # Linear detrending
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            fluctuations.append(np.sqrt(np.mean((segment - trend) ** 2)))

        if len(fluctuations) > 0:
            log_n.append(np.log(w))
            log_f.append(np.log(np.mean(fluctuations)))

    log_n = np.array(log_n)
    log_f = np.array(log_f)

    if len(log_n) < 3:
        logger.warning("Too few window sizes for reliable DFA estimation.")
        return {"H": np.nan, "c": np.nan, "r_squared": np.nan}

    slope, intercept = np.polyfit(log_n, log_f, 1)
    ss_res = np.sum((log_f - (slope * log_n + intercept)) ** 2)
    ss_tot = np.sum((log_f - log_f.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"H": float(slope), "c": float(intercept), "r_squared": float(r_squared)}


# =============================================================================
# Spectral Analysis
# =============================================================================


def compute_power_spectral_density(
    data: Union[pd.Series, pd.DataFrame],
    method: Literal["welch", "periodogram"] = "welch",
    nperseg: Optional[int] = None,
    detrend: str = "linear",
) -> Tuple[np.ndarray, Union[np.ndarray, pd.DataFrame]]:
    """
    Compute the power spectral density (PSD) of flow timeseries.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input flow timeseries. If DataFrame, PSD is computed per column.
    method : {'welch', 'periodogram'}, default='welch'
        PSD estimation method. Welch's method provides smoother estimates.
    nperseg : int, optional
        Segment length for Welch's method. If None, defaults to
        min(256, len(data)).
    detrend : str, default='linear'
        Detrending method applied to each segment: 'linear', 'constant',
        or False.

    Returns
    -------
    frequencies : np.ndarray
        Array of sample frequencies.
    psd : np.ndarray or pd.DataFrame
        Power spectral density estimate. If input is DataFrame,
        returns DataFrame with same column names.
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    if isinstance(data, pd.Series):
        x = data.dropna().values
        is_series = True
    elif isinstance(data, pd.DataFrame):
        is_series = False
    else:
        raise TypeError(
            f"Expected pd.Series, pd.DataFrame, or np.ndarray, got {type(data)}"
        )

    if is_series:
        if nperseg is None:
            nperseg = min(256, len(x))
        if method == "welch":
            freqs, psd = signal.welch(x, fs=1.0, nperseg=nperseg, detrend=detrend)
        else:
            freqs, psd = signal.periodogram(x, fs=1.0, detrend=detrend)
        return freqs, psd
    else:
        psd_dict = {}
        freqs = None
        for col in data.columns:
            x = data[col].dropna().values
            seg = nperseg if nperseg is not None else min(256, len(x))
            if method == "welch":
                f, p = signal.welch(x, fs=1.0, nperseg=seg, detrend=detrend)
            else:
                f, p = signal.periodogram(x, fs=1.0, detrend=detrend)
            if freqs is None:
                freqs = f
            psd_dict[col] = p
        return freqs, pd.DataFrame(psd_dict, index=freqs)


def compare_spectral_properties(
    observed: Union[pd.Series, pd.DataFrame],
    synthetic: Union[pd.Series, pd.DataFrame],
    method: Literal["welch", "periodogram"] = "welch",
    nperseg: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compare spectral properties between observed and synthetic timeseries.

    Parameters
    ----------
    observed : pd.Series or pd.DataFrame
        Observed flow timeseries.
    synthetic : pd.Series or pd.DataFrame
        Synthetic flow timeseries.
    method : {'welch', 'periodogram'}, default='welch'
        PSD estimation method.
    nperseg : int, optional
        Segment length for Welch's method.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        'spectral_rmse': RMSE of log-PSD between observed and synthetic.
        'spectral_correlation': Pearson correlation of log-PSD.
        'low_freq_ratio': Ratio of low-frequency power (synthetic/observed).
        'high_freq_ratio': Ratio of high-frequency power (synthetic/observed).
    """
    # Compute PSDs
    if isinstance(observed, pd.DataFrame):
        observed = observed.iloc[:, 0]
    if isinstance(synthetic, pd.DataFrame):
        synthetic = synthetic.iloc[:, 0]

    freq_obs, psd_obs = compute_power_spectral_density(
        observed, method=method, nperseg=nperseg
    )
    freq_syn, psd_syn = compute_power_spectral_density(
        synthetic, method=method, nperseg=nperseg
    )

    # Interpolate to common frequency grid if needed
    if len(freq_obs) != len(freq_syn):
        common_freqs = np.linspace(
            max(freq_obs[1], freq_syn[1]),
            min(freq_obs[-1], freq_syn[-1]),
            num=min(len(freq_obs), len(freq_syn)),
        )
        psd_obs = np.interp(common_freqs, freq_obs, psd_obs)
        psd_syn = np.interp(common_freqs, freq_syn, psd_syn)
        freq_obs = common_freqs

    # Avoid log(0)
    eps = 1e-20
    log_psd_obs = np.log10(psd_obs + eps)
    log_psd_syn = np.log10(psd_syn + eps)

    # RMSE in log space
    spectral_rmse = float(np.sqrt(np.mean((log_psd_obs - log_psd_syn) ** 2)))

    # Correlation in log space
    if np.std(log_psd_obs) > 0 and np.std(log_psd_syn) > 0:
        spectral_corr = float(np.corrcoef(log_psd_obs, log_psd_syn)[0, 1])
    else:
        spectral_corr = 0.0

    # Low vs high frequency power ratio
    mid_idx = len(freq_obs) // 2
    low_obs = np.sum(psd_obs[:mid_idx])
    low_syn = np.sum(psd_syn[:mid_idx])
    high_obs = np.sum(psd_obs[mid_idx:])
    high_syn = np.sum(psd_syn[mid_idx:])

    low_freq_ratio = float(low_syn / low_obs) if low_obs > 0 else np.nan
    high_freq_ratio = float(high_syn / high_obs) if high_obs > 0 else np.nan

    return {
        "spectral_rmse": spectral_rmse,
        "spectral_correlation": spectral_corr,
        "low_freq_ratio": low_freq_ratio,
        "high_freq_ratio": high_freq_ratio,
    }


# ============================================================================
# Extreme Value Analysis
# ============================================================================


def fit_gev(
    annual_maxima: Union[pd.Series, np.ndarray],
    method: Literal["mle", "lmom"] = "lmom",
) -> Dict[str, float]:
    """
    Fit a Generalized Extreme Value (GEV) distribution to annual maxima.

    The GEV distribution unifies the Gumbel (Type I), Frechet (Type II),
    and Weibull (Type III) extreme value distributions through the shape
    parameter.

    Parameters
    ----------
    annual_maxima : pd.Series or np.ndarray
        Annual maximum flow values.
    method : {'mle', 'lmom'}, default='lmom'
        Fitting method. 'mle' uses scipy maximum likelihood estimation.
        'lmom' uses L-moments (preferred for small samples in hydrology).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        'shape' (xi): shape parameter (xi < 0 = bounded upper tail).
        'loc' (mu): location parameter.
        'scale' (sigma): scale parameter.
        'aic': Akaike Information Criterion (MLE only).
    """
    from scipy.stats import genextreme

    if isinstance(annual_maxima, pd.Series):
        x = annual_maxima.dropna().values
    else:
        x = np.asarray(annual_maxima)
        x = x[~np.isnan(x)]

    if len(x) < 5:
        raise ValueError("Need at least 5 annual maxima for GEV fitting.")

    if method == "mle":
        shape, loc, scale = genextreme.fit(x)
        n = len(x)
        log_lik = np.sum(genextreme.logpdf(x, shape, loc=loc, scale=scale))
        aic = 2 * 3 - 2 * log_lik
        return {
            "shape": float(-shape),  # scipy uses negated shape convention
            "loc": float(loc),
            "scale": float(scale),
            "aic": float(aic),
        }
    elif method == "lmom":
        l1, l2, t3 = _compute_lmoments(x)
        # Hosking (1997) rational approximation for GEV shape from L-skewness
        # k is the GEV shape parameter (xi in some notations)
        c = 2.0 / (3.0 + t3) - np.log(2) / np.log(3)
        k = 7.8590 * c + 2.9554 * c**2
        if abs(k) > 1e-6:
            gamma_val = _gamma_func(1 + k)
            scale = float(l2 * k / (gamma_val * (1 - 2 ** (-k))))
            loc = float(l1 - scale * (gamma_val - 1) / k)
        else:
            # Gumbel limit (k -> 0)
            scale = float(l2 / np.log(2))
            loc = float(l1 - scale * 0.5772)
        return {
            "shape": float(k),
            "loc": loc,
            "scale": scale,
        }
    else:
        raise ValueError(f"Unknown method: {method}")


def fit_lp3(
    annual_maxima: Union[pd.Series, np.ndarray],
    method: Literal["mle", "mom"] = "mom",
) -> Dict[str, float]:
    """
    Fit a Log-Pearson Type III (LP3) distribution to annual maxima.

    LP3 is the standard flood frequency distribution recommended by the
    U.S. Interagency Advisory Committee on Water Data (Bulletin 17C).

    Parameters
    ----------
    annual_maxima : pd.Series or np.ndarray
        Annual maximum flow values. Must be positive.
    method : {'mle', 'mom'}, default='mom'
        Fitting method. 'mom' uses method of moments on log-transformed data
        (standard Bulletin 17C approach).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        'mean_log': mean of log-transformed data.
        'std_log': standard deviation of log-transformed data.
        'skew_log': skewness of log-transformed data.
        'alpha', 'beta', 'gamma': Pearson III parameters on log scale.
    """
    if isinstance(annual_maxima, pd.Series):
        x = annual_maxima.dropna().values
    else:
        x = np.asarray(annual_maxima)
        x = x[~np.isnan(x)]

    if np.any(x <= 0):
        raise ValueError("LP3 requires strictly positive annual maxima.")

    if len(x) < 5:
        raise ValueError("Need at least 5 annual maxima for LP3 fitting.")

    log_x = np.log10(x)
    n = len(log_x)

    mean_log = float(np.mean(log_x))
    std_log = float(np.std(log_x, ddof=1))
    skew_log = (
        float(n * np.sum((log_x - mean_log) ** 3) / ((n - 1) * (n - 2) * std_log**3))
        if std_log > 1e-10
        else 0.0
    )

    if abs(skew_log) > 1e-6:
        alpha = 4.0 / (skew_log**2)
        beta = std_log * abs(skew_log) / 2.0
        if skew_log < 0:
            beta = -beta
        gamma = mean_log - 2.0 * std_log / skew_log
    else:
        alpha = np.inf
        beta = std_log
        gamma = mean_log

    return {
        "mean_log": mean_log,
        "std_log": std_log,
        "skew_log": skew_log,
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
    }


def flood_frequency_quantiles(
    annual_maxima: Union[pd.Series, np.ndarray],
    return_periods: Optional[List] = None,
    distribution: Literal["gev", "lp3"] = "gev",
    **fit_kwargs,
) -> pd.DataFrame:
    """
    Compute flood frequency quantiles for specified return periods.

    Parameters
    ----------
    annual_maxima : pd.Series or np.ndarray
        Annual maximum flow values.
    return_periods : list of float, optional
        Return periods in years. Default: [2, 5, 10, 25, 50, 100, 200, 500].
    distribution : {'gev', 'lp3'}, default='gev'
        Distribution to use.
    **fit_kwargs
        Additional arguments passed to fit_gev or fit_lp3.

    Returns
    -------
    pd.DataFrame
        Columns: return_period, exceedance_prob, quantile.
    """
    from scipy.stats import genextreme, pearson3

    if return_periods is None:
        return_periods = [2, 5, 10, 25, 50, 100, 200, 500]

    if isinstance(annual_maxima, pd.Series):
        x = annual_maxima.dropna().values
    else:
        x = np.asarray(annual_maxima)
        x = x[~np.isnan(x)]

    exceedance_probs = [1.0 / T for T in return_periods]

    if distribution == "gev":
        params = fit_gev(x, **fit_kwargs)
        # scipy uses negated shape
        quantiles = [
            float(
                genextreme.isf(
                    p, -params["shape"], loc=params["loc"], scale=params["scale"]
                )
            )
            for p in exceedance_probs
        ]
    elif distribution == "lp3":
        params = fit_lp3(x, **fit_kwargs)
        # Use Pearson III on log scale, then back-transform
        if abs(params["skew_log"]) > 1e-6:
            a = params["alpha"]
            b = params["beta"]
            g = params["gamma"]
            quantiles = []
            for p in exceedance_probs:
                log_q = float(
                    pearson3.isf(
                        p,
                        params["skew_log"],
                        loc=params["mean_log"],
                        scale=params["std_log"],
                    )
                )
                quantiles.append(10**log_q)
        else:
            from scipy.stats import norm

            quantiles = [
                float(
                    10 ** norm.isf(p, loc=params["mean_log"], scale=params["std_log"])
                )
                for p in exceedance_probs
            ]
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return pd.DataFrame(
        {
            "return_period": return_periods,
            "exceedance_prob": exceedance_probs,
            "quantile": quantiles,
        }
    )


def _compute_lmoments(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute first three L-moments of a sample.

    Parameters
    ----------
    x : np.ndarray
        Sample values.

    Returns
    -------
    l1, l2, t3 : float
        First L-moment (mean), second L-moment (L-scale),
        L-skewness ratio.
    """
    n = len(x)
    xs = np.sort(x)

    # Probability weighted moments
    b0 = np.mean(xs)
    b1 = np.sum(np.arange(1, n) * xs[1:]) / (n * (n - 1))
    b2 = np.sum(np.arange(1, n - 1) * np.arange(2, n) * xs[2:]) / (
        n * (n - 1) * (n - 2)
    )

    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0

    t3 = l3 / l2 if abs(l2) > 1e-10 else 0.0

    return float(l1), float(l2), float(t3)


def _gamma_func(x: float) -> float:
    """Wrapper for scipy gamma function."""
    from scipy.special import gamma

    return float(gamma(x))


class NormalScoreTransform:
    """Empirical normal score transform with Hazen plotting position.

    Maps observed values to standard normal scores via the empirical CDF,
    using the Hazen plotting position F(x) = (rank - 0.5) / n.  Supports
    grouped transforms (e.g. one mapping per calendar month per site).

    Parameters
    ----------
    None

    References
    ----------
    Kirsch, B.R., Characklis, G.W., and Zeff, H.B. (2013). JWRPM, 139(4).

    Examples
    --------
    >>> nst = NormalScoreTransform()
    >>> nst.fit({(1, 0): np.array([10, 20, 30, 40, 50])})
    >>> z = nst.transform(np.array([25]), group_key=(1, 0))
    """

    def __init__(self) -> None:
        self._sorted_values: Dict[tuple, np.ndarray] = {}
        self._normal_scores: Dict[tuple, np.ndarray] = {}
        self._is_fitted: bool = False

    def fit(self, values_by_group: Dict[tuple, np.ndarray]) -> "NormalScoreTransform":
        """Fit the transform by computing Hazen normal scores per group.

        Parameters
        ----------
        values_by_group : Dict[tuple, np.ndarray]
            Mapping from group key (e.g. (month, site_index)) to a 1-D
            array of observed values for that group.

        Returns
        -------
        NormalScoreTransform
            The fitted instance (for method chaining).

        Raises
        ------
        ValueError
            If any group has fewer than 2 observations.
        """
        from scipy.stats import norm as _norm

        self._sorted_values = {}
        self._normal_scores = {}

        for key, values in values_by_group.items():
            vals = np.asarray(values, dtype=float)
            if len(vals) < 2:
                raise ValueError(
                    f"Group {key} has fewer than 2 observations; "
                    "cannot compute normal scores."
                )
            n = len(vals)
            pp = (np.arange(1, n + 1) - 0.5) / n
            nscores = _norm.ppf(pp)

            sorted_vals = np.sort(vals)
            self._sorted_values[key] = sorted_vals
            self._normal_scores[key] = nscores

        self._is_fitted = True
        return self

    def transform(self, values: np.ndarray, group_key: tuple) -> np.ndarray:
        """Forward transform: observed values to normal scores.

        Parameters
        ----------
        values : np.ndarray
            Values to transform (1-D).
        group_key : tuple
            Key identifying which fitted group mapping to use.

        Returns
        -------
        np.ndarray
            Normal scores with the same shape as *values*.

        Raises
        ------
        RuntimeError
            If the transform has not been fitted.
        KeyError
            If *group_key* was not present during fitting.
        """
        self._check_fitted()
        sv = self._sorted_values[group_key]
        ns = self._normal_scores[group_key]
        return np.interp(np.asarray(values, dtype=float), sv, ns)

    def inverse_transform(self, scores: np.ndarray, group_key: tuple) -> np.ndarray:
        """Inverse transform: normal scores to original value space.

        Uses linear extrapolation at the tails so that synthetic values
        beyond the observed range are not clamped.

        Parameters
        ----------
        scores : np.ndarray
            Normal scores to invert (1-D).
        group_key : tuple
            Key identifying which fitted group mapping to use.

        Returns
        -------
        np.ndarray
            Values in the original space with the same shape as *scores*.

        Raises
        ------
        RuntimeError
            If the transform has not been fitted.
        KeyError
            If *group_key* was not present during fitting.
        """
        self._check_fitted()
        ns = self._normal_scores[group_key]
        sv = self._sorted_values[group_key]

        # Linear tail extrapolation (prevents clamping at observed extremes)
        slope_lo = (sv[1] - sv[0]) / (ns[1] - ns[0]) if ns[1] != ns[0] else 1.0
        slope_hi = (sv[-1] - sv[-2]) / (ns[-1] - ns[-2]) if ns[-1] != ns[-2] else 1.0
        ns_ext = np.concatenate([[-10.0], ns, [10.0]])
        sv_ext = np.concatenate(
            [
                [sv[0] + slope_lo * (-10.0 - ns[0])],
                sv,
                [sv[-1] + slope_hi * (10.0 - ns[-1])],
            ]
        )
        return np.interp(np.asarray(scores, dtype=float), ns_ext, sv_ext)

    def _check_fitted(self) -> None:
        """Raise if the transform has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "NormalScoreTransform has not been fitted. Call fit() first."
            )
