"""
Wavelet Auto-Regressive Method (WARM) for streamflow generation.

Implements the enhanced WARM methodology of Nowak et al. (2011) for univariate
synthetic annual streamflow simulation that preserves non-stationary spectral
features. Significance of spectral peaks is assessed using the chi-squared
red-noise / white-noise background framework of Torrence and Compo (1998).
"""

import logging
from typing import Optional, Union, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pywt
from numpy.typing import NDArray
from scipy import stats

from synhydro.core.base import Generator, FittedParams
from synhydro.core.ensemble import Ensemble, EnsembleMetadata

logger = logging.getLogger(__name__)


# Wavelet-specific reconstruction constants from Torrence and Compo (1998),
# Table 2. C_delta is the reconstruction factor used in the inverse CWT and
# in the SAWP normalization (Nowak et al. 2011, Eqs. 4-5). psi_0 is the
# value of the (real part of the) mother wavelet at zero, also required
# in the inverse CWT.
_WAVELET_CONSTANTS: Dict[str, Dict[str, float]] = {
    # Morlet, omega_0 = 6
    "morl": {"C_delta": 0.776, "psi_0": np.pi ** (-0.25), "gamma": 2.32},
    # Mexican Hat (DOG, m=2)
    "mexh": {
        "C_delta": 3.541,
        "psi_0": (2.0 / np.sqrt(3.0)) * np.pi ** (-0.25),
        "gamma": 1.43,
    },
}


class WARMGenerator(Generator):
    """
    Wavelet Auto-Regressive Method (WARM) for non-stationary streamflow generation.

    Implements the enhanced WARM framework of Nowak et al. (2011). The procedure
    decomposes an observed annual flow record into significant spectral bands
    via the continuous wavelet transform, removes time-varying envelope by
    dividing each band-reconstructed signal by the square root of its
    Scale-Averaged Wavelet Power (SAWP), fits AR(p) models to the resulting
    stationary signals (one per band plus a noise residual), and reverses the
    process to synthesize new traces with the same non-stationary spectral
    structure as the historic record.

    Significance of spectral peaks is assessed using the chi-squared
    background spectrum framework of Torrence and Compo (1998), with either a
    white-noise or AR(1) red-noise background.

    Notes
    -----
    The WARMGenerator is univariate. For multi-site simulation as described in
    Nowak et al. (2011, Section 2.4), apply this generator to an aggregate
    gauge time series and then disaggregate spatially using the proportional
    KNN method of Nowak et al. (2010), available in SynHydro as
    ``synhydro.methods.disaggregation.spatial.NowakDisaggregator``.

    Examples
    --------
    >>> import pandas as pd
    >>> from synhydro.methods.generation.parametric.warm import WARMGenerator
    >>> Q_annual = pd.read_csv('annual_flows.csv', index_col=0, parse_dates=True)
    >>> warm = WARMGenerator(wavelet='morl', background_spectrum='red')
    >>> warm.fit(Q_annual.iloc[:, [0]])
    >>> ensemble = warm.generate(n_years=100, n_realizations=50, seed=42)

    References
    ----------
    Nowak, K., Rajagopalan, B., and Zagona, E. (2011). A Wavelet Auto-Regressive
    Method (WARM) for multi-site streamflow simulation of data with
    non-stationary spectra. Journal of Hydrology, 410(1-2), 1-12.

    Torrence, C., and Compo, G.P. (1998). A practical guide to wavelet analysis.
    Bulletin of the American Meteorological Society, 79(1), 61-78.
    """

    supports_multisite: bool = False
    supported_frequencies: tuple = ("YS",)

    def __init__(
        self,
        *,
        wavelet: str = "morl",
        scales: Optional[NDArray] = None,
        n_octaves: Optional[float] = None,
        n_voices: int = 8,
        s0: Optional[float] = None,
        ar_order: int = 1,
        n_ar_max: int = 5,
        ar_select: str = "fixed",
        bands: Optional[List[Tuple[float, float]]] = None,
        background_spectrum: str = "red",
        significance_level: float = 0.95,
        min_band_scales: int = 1,
        lower_bound: Union[float, str] = 0.0,
        name: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ):
        """
        Initialize the WARM Generator.

        Parameters
        ----------
        wavelet : str, default='morl'
            Wavelet type for the continuous wavelet transform. Supported with
            tabulated reconstruction constants: 'morl' (Morlet) and 'mexh'
            (Mexican Hat). Other PyWavelets continuous wavelets are accepted
            but will fall back to Morlet constants and emit a warning.
        scales : array-like of float, optional
            Explicit scales (in units of the sampling period) at which to
            evaluate the CWT. If ``None``, scales are constructed
            geometrically a la Torrence and Compo (1998) using ``s0``,
            ``n_voices``, and ``n_octaves``.
        n_octaves : float, optional
            Number of powers-of-two of scale to span. If ``None``, defaults to
            ``log2(N / (2 * s0))`` where N is the record length, capping the
            largest scale at half the record length.
        n_voices : int, default=8
            Number of voices per octave. Setting ``delta_j = 1 / n_voices``
            controls scale resolution. Default of 8 matches the Torrence and
            Compo (1998) recommendation for the Morlet wavelet.
        s0 : float, optional
            Smallest scale, in units of the sampling period. Defaults to 2,
            corresponding to a Fourier period of approximately ``2 * dt``.
        ar_order : int, default=1
            Order of the autoregressive model fitted to each band's
            stationary component when ``ar_select='fixed'``. Per Nowak et al.
            (2011), low-order AR models are usually adequate for the smooth
            band reconstructions.
        n_ar_max : int, default=5
            Maximum AR order considered when ``ar_select='aic'``.
        ar_select : {'fixed', 'aic'}, default='fixed'
            Strategy for choosing AR order. ``'fixed'`` uses ``ar_order`` for
            every band. ``'aic'`` selects the order in ``[1, n_ar_max]``
            minimizing Akaike's information criterion.
        bands : list of (period_low, period_high) tuples, optional
            Explicit Fourier-period bands (in years) to model. Each tuple
            specifies the inclusive low and high period bounds of a band. If
            ``None`` (default), bands are auto-detected from contiguous
            significant peaks in the global wavelet spectrum at the chosen
            ``significance_level`` against the chosen ``background_spectrum``.
        background_spectrum : {'red', 'white'}, default='red'
            Background spectrum for the chi-squared significance test of
            Torrence and Compo (1998). ``'red'`` uses a theoretical AR(1)
            spectrum with lag-1 coefficient estimated from the record;
            ``'white'`` uses a flat spectrum.
        significance_level : float, default=0.95
            Confidence level (0 < level < 1) used to threshold the global
            wavelet spectrum for band detection.
        min_band_scales : int, default=1
            Minimum number of contiguous scales above the significance
            threshold required to declare a band. Increase to suppress narrow
            single-scale spurious peaks.
        lower_bound : float or {'obs_min'}, default=0.0
            Floor applied to synthetic annual values before returning. The
            Gaussian AR + bootstrapped-SAWP synthesis can occasionally produce
            sums that are more negative than the historical record supports,
            and Nowak et al. (2011) clamp these at zero. For perennial rivers
            where zero-flow years are not physically plausible, pass a
            positive float (interpreted in the units of the input series) or
            the string ``'obs_min'`` to use the observed minimum as the
            floor. Default of ``0.0`` reproduces the original paper.
        name : str, optional
            Name for this generator instance.
        debug : bool, default=False
            Enable debug logging.
        **kwargs : dict, optional
            Additional parameters; ignored.

        Raises
        ------
        ValueError
            If ``ar_order`` < 1, ``n_ar_max`` < 1, ``ar_select`` not in
            {'fixed', 'aic'}, ``background_spectrum`` not in {'red', 'white'},
            ``significance_level`` not in (0, 1), or ``wavelet`` not a
            recognized continuous wavelet.
        """
        super().__init__(name=name, debug=debug)

        if ar_order < 1:
            raise ValueError(f"ar_order must be >= 1, got {ar_order}")
        if n_ar_max < 1:
            raise ValueError(f"n_ar_max must be >= 1, got {n_ar_max}")
        if ar_select not in ("fixed", "aic"):
            raise ValueError(f"ar_select must be 'fixed' or 'aic', got {ar_select!r}")
        if background_spectrum not in ("red", "white"):
            raise ValueError(
                "background_spectrum must be 'red' or 'white', "
                f"got {background_spectrum!r}"
            )
        if not (0.0 < significance_level < 1.0):
            raise ValueError(
                f"significance_level must be in (0, 1), got {significance_level}"
            )
        if min_band_scales < 1:
            raise ValueError(f"min_band_scales must be >= 1, got {min_band_scales}")
        if isinstance(lower_bound, str):
            if lower_bound != "obs_min":
                raise ValueError(
                    "lower_bound string must be 'obs_min', got " f"{lower_bound!r}"
                )
        else:
            try:
                lower_bound = float(lower_bound)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "lower_bound must be a float or the string 'obs_min', "
                    f"got {lower_bound!r}"
                ) from exc
        if wavelet not in pywt.wavelist(kind="continuous"):
            raise ValueError(
                f"wavelet '{wavelet}' not recognized. "
                f"Must be one of {pywt.wavelist(kind='continuous')}"
            )

        # Handle the legacy ``scales`` argument: int meaning "this many
        # consecutive integer scales" (kept for backwards-compatible test
        # semantics) or array-like of floats meaning "use these exact scales".
        scales_user: Optional[NDArray]
        if scales is None:
            scales_user = None
        elif np.isscalar(scales):
            scale_int = int(scales)
            if scale_int < 2:
                raise ValueError(f"scales must be >= 2, got {scale_int}")
            scales_user = np.arange(1, scale_int + 1, dtype=float)
        else:
            scales_user = np.asarray(scales, dtype=float)
            if scales_user.ndim != 1 or scales_user.size < 2:
                raise ValueError("scales must be a 1-D array of length >= 2")
            if np.any(scales_user <= 0):
                raise ValueError("scales must be positive")

        self.wavelet = wavelet
        self._scales_user = scales_user
        self.n_octaves = n_octaves
        self.n_voices = int(n_voices)
        self.s0 = s0
        self.ar_order = int(ar_order)
        self.n_ar_max = int(n_ar_max)
        self.ar_select = ar_select
        self.bands_user = bands
        self.background_spectrum = background_spectrum
        self.significance_level = float(significance_level)
        self.min_band_scales = int(min_band_scales)
        self.lower_bound = lower_bound

        # Backwards-compatible attribute for tests / introspection.
        self.scales = scales if scales is not None else 64

        # Fitted-state placeholders.
        self.scales_used_: Optional[NDArray] = None
        self.fourier_periods_: Optional[NDArray] = None
        self.delta_j_: Optional[float] = None
        self.delta_t_: Optional[float] = None
        self.wavelet_coeffs_: Optional[NDArray] = None
        self.global_spectrum_: Optional[NDArray] = None
        self.background_spectrum_values_: Optional[NDArray] = None
        self.significance_threshold_: Optional[NDArray] = None
        self.significant_mask_: Optional[NDArray] = None
        self.lag1_: Optional[float] = None
        self.bands_: Optional[List[Dict[str, Any]]] = None
        # Per-band SAWP series; legacy ``sawp_`` retained for tests as the
        # SAWP of the first detected band (or the whole-spectrum SAWP if no
        # bands were found).
        self.sawp_: Optional[NDArray] = None
        self.ar_params_: Optional[Dict[int, Dict[str, Any]]] = None
        self.noise_ar_params_: Optional[Dict[str, Any]] = None
        self.noise_residual_: Optional[NDArray] = None
        self.flow_mean_: Optional[float] = None

        self.init_params.algorithm_params = {
            "wavelet": wavelet,
            "scales": self.scales,
            "ar_order": self.ar_order,
            "n_ar_max": self.n_ar_max,
            "ar_select": self.ar_select,
            "bands": self.bands_user,
            "background_spectrum": self.background_spectrum,
            "significance_level": self.significance_level,
            "n_voices": self.n_voices,
            "min_band_scales": self.min_band_scales,
            "lower_bound": self.lower_bound,
        }

    @property
    def output_frequency(self) -> str:
        """Pandas frequency string of generated output (annual, year-start)."""
        return "YS"

    # ------------------------------------------------------------------
    # Preprocessing / fitting / generation
    # ------------------------------------------------------------------

    def preprocessing(self, Q_obs, *, sites=None, **kwargs) -> None:
        """
        Preprocess observed data for WARM fitting.

        Validates input, ensures (or resamples to) annual frequency, and stores
        the resulting series on ``self.Q_obs_annual``.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame
            Observed streamflow with a DatetimeIndex.
        sites : list of str, optional
            Sites to keep. If ``None``, uses all columns; only one site is
            permitted because WARM is univariate.
        **kwargs : dict
            Ignored.
        """
        Q = self._store_obs_data(Q_obs, sites=sites)
        Q_series = Q.iloc[:, 0]

        freq = Q_series.index.freq or pd.infer_freq(Q_series.index)
        if freq is None:
            freq_str = None
        elif hasattr(freq, "freqstr"):
            freq_str = freq.freqstr
        else:
            freq_str = str(freq)
        # Normalize to a short alias for matching.
        if freq_str is not None:
            freq_alias = freq_str.split("-")[0]
        else:
            freq_alias = None
        annual_aliases = {"YS", "Y", "A", "AS", "YE"}
        monthly_aliases = {"MS", "M", "ME"}
        daily_aliases = {"D"}

        if freq_alias not in annual_aliases:
            if freq_alias in monthly_aliases:
                self.logger.info("Resampling from monthly to annual (sum)")
                Q_series = Q_series.resample("YS").sum()
            elif freq_alias in daily_aliases:
                self.logger.info("Resampling from daily to annual (sum)")
                Q_series = Q_series.resample("YS").sum()
            elif freq_alias is None:
                self.logger.warning("Could not infer frequency. Assuming annual data.")
            else:
                self.logger.warning(
                    "Unknown frequency %s. Assuming annual data.", freq_str
                )

        self.Q_obs_annual = Q_series

        if len(self.Q_obs_annual) < 20:
            self.logger.warning(
                "Only %d years of data. WARM may not perform well with < 20 years.",
                len(self.Q_obs_annual),
            )

        self.update_state(preprocessed=True)
        self.logger.info(
            "Preprocessing complete: %d annual values", len(self.Q_obs_annual)
        )

    def fit(self, Q_obs=None, *, sites=None, **kwargs) -> None:
        """
        Fit the WARM model to observed annual flows.

        Steps follow Nowak et al. (2011) Sections 2.1-2.3:

        1. Compute the continuous wavelet transform on the mean-centered flow
           series.
        2. Compute the global wavelet spectrum and its chi-squared
           significance threshold against the chosen background spectrum
           (Torrence and Compo 1998).
        3. Identify significant spectral bands as contiguous runs of scales
           exceeding the threshold (or use user-supplied bands).
        4. For each band, compute the band-restricted SAWP (Eq. 5) and the
           band-reconstructed time-domain signal via the inverse CWT (Eq. 4).
        5. Divide the band-reconstructed signal by the square root of SAWP to
           obtain a stationary series and fit an AR(p) model.
        6. Form the noise residual as the observed series minus the sum of all
           band reconstructions, and fit an AR model to it.

        Parameters
        ----------
        Q_obs : pd.Series or pd.DataFrame, optional
            If provided, ``preprocessing`` is called automatically.
        sites : list of str, optional
            Forwarded to ``preprocessing`` if ``Q_obs`` is given.
        **kwargs : dict
            Ignored.
        """
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)
        self.validate_preprocessing()

        Q = np.asarray(self.Q_obs_annual.values, dtype=float)
        n = len(Q)
        self.flow_mean_ = float(np.mean(Q))
        Q_centered = Q - self.flow_mean_

        # Step 1. Build scales and run the CWT.
        self.delta_t_ = 1.0
        scales = self._build_scales(n)
        self.logger.info(
            "Step 1/4: Continuous wavelet transform on %d scales.", len(scales)
        )
        coefficients, frequencies = pywt.cwt(
            Q_centered, scales, self.wavelet, sampling_period=self.delta_t_
        )
        self.wavelet_coeffs_ = coefficients
        self.scales_used_ = scales
        # Fourier periods corresponding to each scale (years).
        self.fourier_periods_ = 1.0 / np.asarray(frequencies, dtype=float)
        self.delta_j_ = float(np.mean(np.diff(np.log2(scales))))

        # Step 2. Global spectrum and significance threshold.
        self.logger.info(
            "Step 2/4: Global wavelet spectrum and %s-noise significance test.",
            self.background_spectrum,
        )
        global_spectrum = np.mean(np.abs(coefficients) ** 2, axis=1)
        self.global_spectrum_ = global_spectrum
        self.lag1_ = self._compute_lag1(Q_centered)
        threshold, background = self._significance_threshold(
            self.fourier_periods_, n, self.lag1_
        )
        self.significance_threshold_ = threshold
        self.background_spectrum_values_ = background
        self.significant_mask_ = global_spectrum > threshold

        # Step 3. Identify bands.
        self.logger.info("Step 3/4: Identifying significant spectral bands.")
        self.bands_ = self._identify_bands()
        self.logger.info(
            "Identified %d significant spectral band(s).", len(self.bands_)
        )

        # Step 4. Per-band SAWP, reconstruction, normalization, AR fit.
        self.logger.info(
            "Step 4/4: Per-band SAWP, time-domain reconstruction, AR fitting."
        )
        self.ar_params_ = {}
        total_band_signal = np.zeros(n)
        for band_idx, band in enumerate(self.bands_):
            indices = band["scale_indices"]
            sawp = self._band_sawp(coefficients, scales, indices)
            recon = self._inverse_cwt(coefficients, scales, indices)
            stationary = self._destandardize_envelope(recon, sawp)
            ar_params = self._fit_ar_model(stationary)
            band["sawp"] = sawp
            band["reconstruction"] = recon
            band["stationary"] = stationary
            band["ar_params"] = ar_params
            self.ar_params_[band_idx] = ar_params
            total_band_signal = total_band_signal + recon

        # Noise residual: everything not explained by significant bands.
        self.noise_residual_ = Q_centered - total_band_signal
        self.noise_ar_params_ = self._fit_ar_model(self.noise_residual_)

        # Backwards-compatible attribute: SAWP of the first band, falling
        # back to the whole-spectrum SAWP if no bands were detected.
        if self.bands_:
            self.sawp_ = self.bands_[0]["sawp"]
        else:
            self.sawp_ = self._band_sawp(coefficients, scales, np.arange(len(scales)))

        self.update_state(fitted=True)
        self.fitted_params_ = self._compute_fitted_params()

        self.logger.info(
            "Fit complete: %d band(s) plus noise component.", len(self.bands_)
        )

    def generate(
        self,
        n_years: Optional[int] = None,
        n_realizations: int = 1,
        n_timesteps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Ensemble:
        """
        Generate synthetic annual streamflows.

        Parameters
        ----------
        n_years : int, optional
            Number of years per realization. Defaults to the historical
            record length.
        n_realizations : int, default=1
            Number of synthetic realizations to produce.
        n_timesteps : int, optional
            Synonym for ``n_years``; if both are given, ``n_timesteps`` wins.
        seed : int, optional
            Seed for the random number generator (NumPy ``default_rng``).
        **kwargs : dict
            Ignored.

        Returns
        -------
        Ensemble
            Ensemble object containing all realizations.

        Raises
        ------
        ValueError
            If ``n_years`` resolves to a non-positive value.
        """
        self.validate_fit()
        rng = np.random.default_rng(seed)

        if n_timesteps is not None:
            n_years = int(n_timesteps)
        elif n_years is None:
            n_years = len(self.Q_obs_annual)

        if n_years <= 0:
            raise ValueError(f"n_years must be positive, got {n_years}")

        realizations: Dict[int, pd.DataFrame] = {}
        for r in range(n_realizations):
            Q_syn = self._generate_one(n_years, rng=rng)
            realizations[r] = Q_syn

        self.logger.info(
            "Generated %d realizations of %d years each.",
            n_realizations,
            n_years,
        )

        first = realizations[0]
        metadata = EnsembleMetadata(
            generator_class=self.__class__.__name__,
            n_realizations=n_realizations,
            n_sites=1,
            time_resolution=self.output_frequency,
            time_period=(str(first.index[0].date()), str(first.index[-1].date())),
        )
        return Ensemble(realizations, metadata=metadata)

    # ------------------------------------------------------------------
    # Scale construction and wavelet support
    # ------------------------------------------------------------------

    def _build_scales(self, n: int) -> NDArray:
        """
        Construct the geometric scale set for the CWT.

        Follows Torrence and Compo (1998) Eq. 9-10: ``s_j = s_0 * 2^(j *
        delta_j)`` for ``j = 0, ..., J``, with ``delta_j = 1 / n_voices``.

        Parameters
        ----------
        n : int
            Length of the input record.

        Returns
        -------
        NDArray
            1-D array of strictly positive scales, in units of ``delta_t``.
        """
        if self._scales_user is not None:
            return np.asarray(self._scales_user, dtype=float)

        s0 = self.s0 if self.s0 is not None else 2.0 * self.delta_t_
        delta_j = 1.0 / max(self.n_voices, 1)
        if self.n_octaves is None:
            n_octaves = max(np.log2(n / (2.0 * s0)), delta_j)
        else:
            n_octaves = max(self.n_octaves, delta_j)

        n_scales = int(np.floor(n_octaves / delta_j)) + 1
        j = np.arange(n_scales, dtype=float)
        return s0 * 2.0 ** (j * delta_j)

    def _wavelet_constants(self) -> Dict[str, float]:
        """
        Look up reconstruction constants for the chosen wavelet.

        Returns
        -------
        dict
            Dictionary with keys ``'C_delta'``, ``'psi_0'``, and ``'gamma'``
            (decorrelation factor for empirical significance).
        """
        if self.wavelet in _WAVELET_CONSTANTS:
            return _WAVELET_CONSTANTS[self.wavelet]
        self.logger.warning(
            "Reconstruction constants not tabulated for wavelet '%s'. "
            "Falling back to Morlet values; SAWP and reconstruction "
            "magnitudes will be approximate.",
            self.wavelet,
        )
        return _WAVELET_CONSTANTS["morl"]

    # ------------------------------------------------------------------
    # Significance testing (Torrence and Compo 1998, Section 4)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_lag1(x: NDArray) -> float:
        """
        Estimate the lag-1 autocorrelation of a centered series.

        Returns the Pearson correlation between ``x[1:]`` and ``x[:-1]``,
        clipped to [0, 0.999) to avoid degenerate red-noise spectra.

        Parameters
        ----------
        x : NDArray
            1-D mean-centered time series.

        Returns
        -------
        float
            Lag-1 autocorrelation in [0, 0.999).
        """
        if len(x) < 2:
            return 0.0
        num = float(np.sum(x[1:] * x[:-1]))
        den = float(np.sum(x * x))
        if den <= 0:
            return 0.0
        rho = num / den
        return float(np.clip(rho, 0.0, 0.999))

    def _significance_threshold(
        self, fourier_periods: NDArray, n: int, lag1: float
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute the chi-squared significance threshold for the global spectrum.

        Implements Torrence and Compo (1998) Eqs. 16-17 with the equivalent
        degrees of freedom for the global (time-averaged) spectrum from their
        Eq. 23. The local scale-wise DOF is

        .. math::
            \\nu = 2 \\sqrt{1 + \\left( \\frac{n \\, dt}{\\gamma s} \\right)^2}

        for the Morlet wavelet (their Table 2 ``gamma``), and the threshold
        on the global spectrum at confidence ``p`` is

        .. math::
            P_k \\cdot \\chi^2_\\nu(p) / \\nu

        where :math:`P_k` is the theoretical background spectrum at the
        Fourier wavenumber corresponding to scale :math:`s`.

        Parameters
        ----------
        fourier_periods : NDArray
            Fourier periods (in years) corresponding to each scale.
        n : int
            Length of the input record.
        lag1 : float
            Lag-1 autocorrelation used for the AR(1) red-noise spectrum.

        Returns
        -------
        threshold : NDArray
            Per-scale significance threshold for the global spectrum.
        background : NDArray
            Per-scale theoretical background spectrum (variance-normalized).
        """
        constants = self._wavelet_constants()
        gamma = constants["gamma"]

        if self.background_spectrum == "white":
            alpha = 0.0
        else:
            alpha = lag1

        # Discrete Fourier frequencies normalized to [0, 0.5].
        k = 1.0 / fourier_periods
        # Theoretical AR(1) (or white-noise when alpha=0) spectrum.
        # Torrence and Compo (1998) Eq. 16: P_k = (1 - a^2) /
        # (1 + a^2 - 2 a cos(2 pi k)). Variance is normalized to unity.
        background = (1.0 - alpha**2) / (
            1.0 + alpha**2 - 2.0 * alpha * np.cos(2.0 * np.pi * k)
        )

        # Equivalent DOF for the global (time-averaged) spectrum,
        # Torrence and Compo (1998) Eq. 23.
        scales = self.scales_used_.astype(float)
        ratio = (n * self.delta_t_) / (gamma * scales)
        dof = 2.0 * np.sqrt(1.0 + ratio**2)
        dof = np.clip(dof, 2.0, None)

        chi2_p = stats.chi2.ppf(self.significance_level, dof)
        threshold = background * (chi2_p / dof)
        return threshold, background

    # ------------------------------------------------------------------
    # Band identification
    # ------------------------------------------------------------------

    def _identify_bands(self) -> List[Dict[str, Any]]:
        """
        Identify spectral bands either from user input or auto-detection.

        For auto-detection, contiguous runs of scales whose global spectrum
        exceeds the significance threshold are grouped together as bands;
        runs shorter than ``min_band_scales`` are discarded.

        Returns
        -------
        list of dict
            One entry per band, with keys ``'scale_indices'`` (NDArray of
            integer scale indices), ``'period_min'``, ``'period_max'``, and
            ``'auto_detected'`` (bool).
        """
        scales = self.scales_used_
        periods = self.fourier_periods_

        if self.bands_user is not None:
            bands: List[Dict[str, Any]] = []
            for low, high in self.bands_user:
                if low > high:
                    low, high = high, low
                mask = (periods >= low) & (periods <= high)
                idx = np.where(mask)[0]
                if idx.size == 0:
                    self.logger.warning(
                        "User-specified band [%g, %g] years contains no scales; "
                        "skipping.",
                        low,
                        high,
                    )
                    continue
                bands.append(
                    {
                        "scale_indices": idx,
                        "period_min": float(periods[idx].min()),
                        "period_max": float(periods[idx].max()),
                        "auto_detected": False,
                    }
                )
            return bands

        # Auto-detect by grouping contiguous significant scales.
        mask = self.significant_mask_
        bands_auto: List[Dict[str, Any]] = []
        in_run = False
        run_start = 0
        for i, flag in enumerate(mask):
            if flag and not in_run:
                in_run = True
                run_start = i
            elif not flag and in_run:
                in_run = False
                idx = np.arange(run_start, i)
                if idx.size >= self.min_band_scales:
                    bands_auto.append(
                        {
                            "scale_indices": idx,
                            "period_min": float(periods[idx].min()),
                            "period_max": float(periods[idx].max()),
                            "auto_detected": True,
                        }
                    )
        if in_run:
            idx = np.arange(run_start, len(mask))
            if idx.size >= self.min_band_scales:
                bands_auto.append(
                    {
                        "scale_indices": idx,
                        "period_min": float(periods[idx].min()),
                        "period_max": float(periods[idx].max()),
                        "auto_detected": True,
                    }
                )
        return bands_auto

    # ------------------------------------------------------------------
    # SAWP and inverse CWT (Nowak et al. 2011, Eqs. 4-5)
    # ------------------------------------------------------------------

    def _band_sawp(
        self,
        coefficients: NDArray,
        scales: NDArray,
        scale_indices: NDArray,
    ) -> NDArray:
        """
        Compute the band-restricted Scale-Averaged Wavelet Power.

        Implements Nowak et al. (2011) Eq. 5 with summation limits j1..j2
        equal to ``scale_indices`` (Torrence and Compo 1998, Eq. 24).

        Parameters
        ----------
        coefficients : NDArray
            Wavelet coefficients, shape ``(n_scales, n_time)``.
        scales : NDArray
            1-D array of scales (length ``n_scales``).
        scale_indices : NDArray
            Integer indices selecting the scales of the band.

        Returns
        -------
        NDArray
            Per-time-step SAWP series of length ``n_time``.
        """
        constants = self._wavelet_constants()
        C_delta = constants["C_delta"]
        sub_coefs = coefficients[scale_indices, :]
        sub_scales = scales[scale_indices].astype(float)
        power = np.abs(sub_coefs) ** 2
        weighted = power / sub_scales[:, np.newaxis]
        sawp = (self.delta_j_ * self.delta_t_ / C_delta) * np.sum(weighted, axis=0)
        # Numerical safety: SAWP is a power and must be non-negative.
        return np.maximum(sawp, 0.0)

    def _inverse_cwt(
        self,
        coefficients: NDArray,
        scales: NDArray,
        scale_indices: NDArray,
    ) -> NDArray:
        """
        Band-restricted inverse CWT in the variance-preserving form.

        Implements Nowak et al. (2011) Eq. 4 with the wavelet-specific
        constants from Torrence and Compo (1998) Table 2:

        .. math::
            x_t^{(b)} = \\frac{\\delta_j \\, \\delta_t^{1/2}}
                              {C_\\delta \\, \\psi_0(0)}
                       \\sum_{j \\in b} \\frac{\\Re[W(a_j, t)]}{a_j^{1/2}}

        Parameters
        ----------
        coefficients : NDArray
            Wavelet coefficients, shape ``(n_scales, n_time)``.
        scales : NDArray
            1-D array of scales.
        scale_indices : NDArray
            Integer indices selecting the scales of the band.

        Returns
        -------
        NDArray
            Time-domain reconstruction of the band, length ``n_time``.
        """
        constants = self._wavelet_constants()
        C_delta = constants["C_delta"]
        psi_0 = constants["psi_0"]

        sub_coefs = coefficients[scale_indices, :]
        sub_scales = scales[scale_indices].astype(float)

        prefactor = self.delta_j_ * np.sqrt(self.delta_t_) / (C_delta * psi_0)
        contrib = np.real(sub_coefs) / np.sqrt(sub_scales)[:, np.newaxis]
        return prefactor * np.sum(contrib, axis=0)

    @staticmethod
    def _destandardize_envelope(
        recon: NDArray, sawp: NDArray, eps: float = 1e-12
    ) -> NDArray:
        """
        Remove the time-varying SAWP envelope from a band reconstruction.

        Returns ``recon / sqrt(sawp)`` with a small floor on the denominator
        to avoid divide-by-zero where the band power vanishes.

        Parameters
        ----------
        recon : NDArray
            Band-reconstructed time-domain signal.
        sawp : NDArray
            Per-time-step SAWP for the same band.
        eps : float, default=1e-12
            Floor applied to ``sawp`` before taking the square root.

        Returns
        -------
        NDArray
            Approximately stationary series of the same length.
        """
        return recon / np.sqrt(np.maximum(sawp, eps))

    # ------------------------------------------------------------------
    # AR fitting
    # ------------------------------------------------------------------

    def _fit_ar_model(self, data: NDArray) -> Dict[str, Any]:
        """
        Fit an AR(p) model to a 1-D series via Yule-Walker.

        When ``ar_select == 'fixed'``, ``ar_order`` is used. When ``ar_select
        == 'aic'``, the order minimizing AIC over ``[1, n_ar_max]`` is
        selected.

        Parameters
        ----------
        data : NDArray
            Time series to fit.

        Returns
        -------
        dict
            Dictionary with keys ``'order'`` (int), ``'coeffs'`` (NDArray of
            length ``order``), ``'sigma'`` (float, innovation standard
            deviation), and ``'mean'`` (float, sample mean of ``data``).
        """
        if self.ar_select == "fixed":
            return self._fit_ar_fixed(data, self.ar_order)

        best: Optional[Dict[str, Any]] = None
        best_aic = np.inf
        n = len(data)
        for p in range(1, self.n_ar_max + 1):
            if n <= p + 1:
                break
            params = self._fit_ar_fixed(data, p)
            sigma2 = max(params["sigma"] ** 2, 1e-30)
            # AIC for AR(p) with Gaussian innovations.
            aic = n * np.log(sigma2) + 2.0 * p
            if aic < best_aic:
                best_aic = aic
                best = params
        if best is None:
            best = self._fit_ar_fixed(data, max(self.ar_order, 1))
        return best

    def _fit_ar_fixed(self, data: NDArray, order: int) -> Dict[str, Any]:
        """
        Fit an AR(p) model of fixed order via Yule-Walker equations.

        Parameters
        ----------
        data : NDArray
            Time series to fit.
        order : int
            AR order ``p >= 1``.

        Returns
        -------
        dict
            See :meth:`_fit_ar_model`.
        """
        mean = float(np.mean(data))
        x = np.asarray(data, dtype=float) - mean
        n = len(x)

        if n <= order + 1:
            self.logger.warning(
                "Data length (%d) <= AR order (%d) + 1; falling back to "
                "zero-coefficient model.",
                n,
                order,
            )
            sigma = float(np.std(x)) if n > 1 else 1.0
            return {
                "order": order,
                "coeffs": np.zeros(order),
                "sigma": max(sigma, 1e-12),
                "mean": mean,
            }

        # Biased autocovariance, normalized to autocorrelation.
        max_lag = order
        autocov = np.array(
            [float(np.sum(x[: n - k] * x[k:])) / n for k in range(max_lag + 1)]
        )
        gamma_0 = autocov[0]
        if gamma_0 <= 0:
            return {
                "order": order,
                "coeffs": np.zeros(order),
                "sigma": 1e-12,
                "mean": mean,
            }
        rho = autocov / gamma_0

        # Yule-Walker: R phi = r.
        R = np.array([[rho[abs(i - j)] for j in range(order)] for i in range(order)])
        r = rho[1 : order + 1]
        try:
            phi = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            self.logger.warning(
                "Singular Yule-Walker matrix; using ridge regularization."
            )
            phi = np.linalg.solve(R + 1e-6 * np.eye(order), r)

        innovation_var = gamma_0 * (1.0 - float(np.dot(phi, r)))
        innovation_var = max(innovation_var, 1e-12)
        return {
            "order": order,
            "coeffs": phi,
            "sigma": float(np.sqrt(innovation_var)),
            "mean": mean,
        }

    @staticmethod
    def _simulate_ar(
        ar_params: Dict[str, Any],
        n: int,
        rng: np.random.Generator,
        burn_in: int = 100,
    ) -> NDArray:
        """
        Simulate an AR(p) process given fitted parameters.

        Parameters
        ----------
        ar_params : dict
            Parameters as returned by :meth:`_fit_ar_model`.
        n : int
            Number of samples to return.
        rng : np.random.Generator
            Random number generator.
        burn_in : int, default=100
            Number of leading samples to discard so the simulation reaches
            stationarity.

        Returns
        -------
        NDArray
            Simulated series of length ``n``.
        """
        order = int(ar_params["order"])
        coeffs = np.asarray(ar_params["coeffs"], dtype=float)
        sigma = float(ar_params["sigma"])
        mean = float(ar_params["mean"])

        total = n + burn_in
        innovations = rng.normal(0.0, sigma, total)
        x = np.zeros(total)
        for t in range(total):
            ar_term = 0.0
            for lag in range(1, order + 1):
                if t - lag >= 0:
                    ar_term += coeffs[lag - 1] * x[t - lag]
            x[t] = ar_term + innovations[t]
        return x[burn_in:] + mean

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _generate_one(self, n_years: int, *, rng: np.random.Generator) -> pd.DataFrame:
        """
        Generate a single synthetic realization.

        For each significant band, an AR-simulated stationary series is
        re-multiplied by the square root of a bootstrapped SAWP series to
        restore the non-stationary envelope. The noise component is
        AR-simulated independently. The resulting band signals plus noise
        are summed in the time domain and the historical mean is added back.

        Parameters
        ----------
        n_years : int
            Length of the realization.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        pd.DataFrame
            One-column DataFrame with annual DatetimeIndex.
        """
        Q_syn = np.zeros(n_years)

        for band in self.bands_:
            stationary_syn = self._simulate_ar(band["ar_params"], n_years, rng)
            sawp_obs = band["sawp"]
            n_obs = len(sawp_obs)
            # Bootstrap a SAWP series of the requested length cyclically;
            # using sampling with replacement preserves the marginal
            # distribution of historical SAWP values per Nowak et al. (2011).
            idx = rng.integers(0, n_obs, size=n_years)
            sawp_syn = sawp_obs[idx]
            band_signal = stationary_syn * np.sqrt(np.maximum(sawp_syn, 0.0))
            Q_syn = Q_syn + band_signal

        if self.noise_ar_params_ is not None:
            noise_syn = self._simulate_ar(self.noise_ar_params_, n_years, rng)
            Q_syn = Q_syn + noise_syn

        Q_syn = Q_syn + self.flow_mean_
        Q_syn = np.maximum(Q_syn, self._effective_lower_bound())

        start_year = self.Q_obs_annual.index[0].year
        dates = pd.date_range(start=f"{start_year}-01-01", periods=n_years, freq="YS")
        return pd.DataFrame(Q_syn, index=dates, columns=[self._sites[0]])

    def _effective_lower_bound(self) -> float:
        """Resolve the lower_bound option to a numeric floor in flow units.

        'obs_min' resolves at synthesis time to the observed annual minimum
        so the floor remains attached to the fitted record rather than the
        constructor argument.
        """
        if isinstance(self.lower_bound, str):
            return float(np.min(self.Q_obs_annual.values))
        return float(self.lower_bound)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _compute_fitted_params(self) -> FittedParams:
        """
        Package fitted parameters into a :class:`FittedParams` dataclass.

        Returns
        -------
        FittedParams
            Summary of the fitted WARM model.
        """
        # Count: per band -> AR order + 1 (innovation std) + 1 (mean) + N
        # SAWP values; plus noise AR params; plus 1 for flow mean.
        n_params = 1
        for band in self.bands_ or []:
            n_params += int(band["ar_params"]["order"]) + 2
            n_params += len(band["sawp"])
        if self.noise_ar_params_ is not None:
            n_params += int(self.noise_ar_params_["order"]) + 2

        training_period = (
            str(self.Q_obs_annual.index[0].date()),
            str(self.Q_obs_annual.index[-1].date()),
        )

        bands_summary = []
        for band in self.bands_ or []:
            bands_summary.append(
                {
                    "period_min_years": band["period_min"],
                    "period_max_years": band["period_max"],
                    "n_scales": int(len(band["scale_indices"])),
                    "ar_order": int(band["ar_params"]["order"]),
                    "ar_coeffs": np.asarray(band["ar_params"]["coeffs"]),
                    "ar_sigma": float(band["ar_params"]["sigma"]),
                    "auto_detected": bool(band["auto_detected"]),
                }
            )

        noise_summary = None
        if self.noise_ar_params_ is not None:
            noise_summary = {
                "ar_order": int(self.noise_ar_params_["order"]),
                "ar_coeffs": np.asarray(self.noise_ar_params_["coeffs"]),
                "ar_sigma": float(self.noise_ar_params_["sigma"]),
            }

        return FittedParams(
            means_=pd.Series({"annual_mean": float(self.flow_mean_)}),
            stds_=pd.Series({"annual_std": float(np.std(self.Q_obs_annual.values))}),
            correlations_=None,
            distributions_={
                "type": "Wavelet AR with per-band SAWP envelope",
                "wavelet": self.wavelet,
                "n_scales": int(len(self.scales_used_)),
                "n_bands": len(self.bands_ or []),
                "background_spectrum": self.background_spectrum,
                "significance_level": self.significance_level,
                "lag1": float(self.lag1_) if self.lag1_ is not None else None,
            },
            transformations_={
                "scales": self.scales_used_,
                "fourier_periods": self.fourier_periods_,
                "delta_j": self.delta_j_,
                "delta_t": self.delta_t_,
                "global_spectrum": self.global_spectrum_,
                "significance_threshold": self.significance_threshold_,
                "bands": bands_summary,
                "noise_ar": noise_summary,
            },
            n_parameters_=int(n_params),
            sample_size_=len(self.Q_obs_annual),
            n_sites_=1,
            training_period_=training_period,
        )
