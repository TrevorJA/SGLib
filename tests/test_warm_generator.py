"""
Tests for WARM (Wavelet Auto-Regressive Method) streamflow generator.

Comprehensive test suite following SynHydro standards.
"""

import pytest
import pickle
import numpy as np
import pandas as pd

from synhydro.methods.generation.hybrid.warm import WARMGenerator
from synhydro.core.ensemble import Ensemble


@pytest.fixture
def sample_annual_series():
    """Generate a sample annual time series for testing WARM."""
    dates = pd.date_range(start="1950-01-01", end="2020-12-31", freq="YS")
    np.random.seed(42)

    # Generate realistic annual streamflow with some low-frequency variation
    n_years = len(dates)
    # Base flow with trend
    base = 500 + np.linspace(0, 50, n_years)
    # Add low-frequency component (decadal variation)
    low_freq = 100 * np.sin(2 * np.pi * np.arange(n_years) / 20)
    # Add high-frequency component (annual variation)
    high_freq = 50 * np.sin(2 * np.pi * np.arange(n_years) / 5)
    # Add noise
    noise = np.random.normal(0, 50, n_years)

    values = base + low_freq + high_freq + noise
    values = np.maximum(values, 10)  # Ensure positive

    return pd.Series(values, index=dates, name="site_1")


@pytest.fixture
def sample_annual_dataframe():
    """Generate a sample annual multi-site DataFrame for testing."""
    dates = pd.date_range(start="1950-01-01", end="2020-12-31", freq="YS")
    np.random.seed(42)
    n_years = len(dates)

    data = {}
    for i in range(3):
        base = 500 + np.linspace(0, 50, n_years)
        low_freq = 100 * np.sin(2 * np.pi * np.arange(n_years) / 20 + i)
        noise = np.random.normal(0, 50, n_years)
        data[f"site_{i+1}"] = base + low_freq + noise

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def short_annual_series():
    """Generate a short annual series (30 years) for faster testing."""
    dates = pd.date_range(start="1990-01-01", end="2020-12-31", freq="YS")
    np.random.seed(123)

    n_years = len(dates)
    values = 500 + 100 * np.sin(2 * np.pi * np.arange(n_years) / 10)
    values += np.random.normal(0, 30, n_years)
    values = np.maximum(values, 10)

    return pd.Series(values, index=dates, name="site_1")


class TestWARMInitialization:
    """Tests for WARMGenerator initialization."""

    def test_initialization_default_params(self, sample_annual_series):
        """Test initialization with default parameters."""
        gen = WARMGenerator()

        assert gen.is_preprocessed is False
        assert gen.is_fitted is False
        assert gen.debug is False
        assert gen.wavelet == "cmor1.5-1.0"
        assert gen.scales == 64
        assert gen.ar_select == "aic"
        assert gen.ar_method == "burg"
        # Default follows Nowak et al. (2011) step (iii): SAWP in observed order.
        assert gen.sawp_resampling == "historical"
        assert gen.noise_model == "ar_bootstrap"
        assert gen.lower_bound == 0.0
        # Nowak et al. (2011) use the 95% white-noise test of Torrence and Compo.
        assert gen.background_spectrum == "white"

    def test_initialization_custom_params(self, sample_annual_series):
        """Test initialization with custom parameters."""
        gen = WARMGenerator(wavelet="cmor1.0-0.5", scales=32, ar_order=2, debug=True)

        assert gen.wavelet == "cmor1.0-0.5"
        assert gen.scales == 32
        assert gen.ar_order == 2
        # An explicit ar_order without ar_select implies a fixed order.
        assert gen.ar_select == "fixed"
        assert gen.debug is True

    def test_initialization_invalid_scales(self, sample_annual_series):
        """Test initialization raises error for invalid scales."""
        with pytest.raises(ValueError, match="scales must be >= 2"):
            WARMGenerator(scales=1)

    def test_initialization_invalid_ar_order(self, sample_annual_series):
        """Test initialization raises error for invalid AR order."""
        with pytest.raises(ValueError, match="ar_order must be >= 1"):
            WARMGenerator(ar_order=0)

    def test_initialization_invalid_wavelet(self, sample_annual_series):
        """Test initialization raises error for invalid wavelet."""
        with pytest.raises(ValueError, match="not recognized"):
            WARMGenerator(wavelet="invalid_wavelet")

    def test_initialization_stores_algorithm_params(self, sample_annual_series):
        """Test that initialization stores algorithm parameters."""
        gen = WARMGenerator()

        assert "algorithm_params" in gen.init_params.__dict__
        params = gen.init_params.algorithm_params
        assert params["wavelet"] == "cmor1.5-1.0"
        assert params["scales"] == 64
        assert params["ar_select"] == "aic"
        assert params["ar_method"] == "burg"
        assert params["noise_model"] == "ar_bootstrap"
        assert params["lower_bound"] == 0.0


class TestWARMPreprocessing:
    """Tests for WARMGenerator preprocessing."""

    def test_preprocessing_annual_series(self, sample_annual_series):
        """Test preprocessing with annual Series."""
        gen = WARMGenerator()
        gen.preprocessing(sample_annual_series)

        assert gen.is_preprocessed is True
        assert hasattr(gen, "Q_obs_annual")
        assert isinstance(gen.Q_obs_annual, pd.Series)

    def test_preprocessing_multisite_raises(self, sample_annual_dataframe):
        """Test that multi-site DataFrame raises ValueError."""
        gen = WARMGenerator()
        with pytest.raises(ValueError, match="univariate"):
            gen.preprocessing(sample_annual_dataframe)

    def test_preprocessing_monthly_to_annual(self, sample_monthly_series):
        """Test preprocessing resamples monthly to annual."""
        gen = WARMGenerator()
        gen.preprocessing(sample_monthly_series)

        assert gen.is_preprocessed is True
        # Should have roughly 1/12 the number of observations
        assert len(gen.Q_obs_annual) < len(sample_monthly_series) / 6

    def test_preprocessing_validates_data_length(self, sample_annual_series):
        """Test preprocessing warns for short data."""
        # Create very short series
        short_data = sample_annual_series.iloc[:15]
        gen = WARMGenerator()

        # Should still preprocess but with warning
        gen.preprocessing(short_data)
        assert gen.is_preprocessed is True


class TestWARMFitting:
    """Tests for WARMGenerator fitting."""

    def test_fit_basic(self, short_annual_series):
        """Test basic fitting functionality."""
        gen = WARMGenerator(scales=16)
        gen.fit(short_annual_series)

        assert gen.is_fitted is True
        assert gen.wavelet_coeffs_ is not None
        assert gen.sawp_ is not None
        assert gen.ar_params_ is not None

    def test_fit_wavelet_coeffs_shape(self, short_annual_series):
        """Test wavelet coefficients have correct shape."""
        gen = WARMGenerator(scales=16)
        gen.fit(short_annual_series)

        n_years = len(gen.Q_obs_annual)
        assert gen.wavelet_coeffs_.shape == (16, n_years)

    def test_fit_sawp_shape(self, short_annual_series):
        """Test SAWP has correct shape."""
        gen = WARMGenerator(scales=16)
        gen.fit(short_annual_series)

        n_years = len(gen.Q_obs_annual)
        assert gen.sawp_.shape == (n_years,)
        assert np.all(gen.sawp_ >= 0)  # Power should be non-negative

    def test_fit_ar_params_structure(self, short_annual_series):
        """Test AR parameters have correct structure (per band, plus noise)."""
        gen = WARMGenerator(scales=16, ar_order=2)
        gen.fit(short_annual_series)

        # New paradigm: one AR fit per significant band, plus a noise
        # AR fit. There must be at least one of each.
        assert len(gen.ar_params_) >= 1
        assert gen.noise_ar_params_ is not None

        for band_idx, params in gen.ar_params_.items():
            assert "coeffs" in params
            assert "sigma" in params
            assert "mean" in params
            assert "order" in params
            assert len(params["coeffs"]) == 2  # AR(2)
            assert params["sigma"] > 0

        # Noise AR has the same fields.
        for key in ("coeffs", "sigma", "mean", "order"):
            assert key in gen.noise_ar_params_
        assert len(gen.noise_ar_params_["coeffs"]) == 2
        assert gen.noise_ar_params_["sigma"] > 0

    def test_fit_creates_fitted_params(self, short_annual_series):
        """Test that fit creates FittedParams object."""
        gen = WARMGenerator(scales=16)
        gen.fit(short_annual_series)

        assert hasattr(gen, "fitted_params_")
        assert gen.fitted_params_.n_sites_ == 1

    def test_fit_without_preprocessing_raises(self, sample_annual_series):
        """Test fit raises error without preprocessing (and no Q_obs)."""
        gen = WARMGenerator()

        with pytest.raises(ValueError, match="preprocessing"):
            gen.fit()

    def test_fit_different_wavelets(self, short_annual_series):
        """Test fitting with different complex Morlet bandwidth/center frequencies."""
        wavelets = ["cmor1.5-1.0", "cmor1.0-1.0", "cmor2.0-0.5"]

        for wavelet in wavelets:
            gen = WARMGenerator(wavelet=wavelet, scales=8)
            gen.fit(short_annual_series)

            assert gen.is_fitted is True
            assert gen.wavelet_coeffs_ is not None


class TestWARMGeneration:
    """Tests for WARMGenerator generation."""

    def test_generate_basic(self, short_annual_series):
        """Test basic generation functionality."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        result = gen.generate(n_years=20, n_realizations=3, seed=42)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 3

    def test_generate_shape(self, short_annual_series):
        """Test generated data has correct shape."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        n_years = 25
        result = gen.generate(n_years=n_years, n_realizations=2, seed=42)

        for r in range(2):
            df = result.data_by_realization[r]
            assert df.shape == (n_years, 1)  # Annual data, 1 site

    def test_generate_non_negative(self, short_annual_series):
        """Test that generated flows are non-negative."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        result = gen.generate(n_years=30, n_realizations=5, seed=42)

        for r in range(5):
            df = result.data_by_realization[r]
            assert np.all(df.values >= 0)

    def test_generate_reproducible(self, short_annual_series):
        """Test generation is reproducible with seed."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        result1 = gen.generate(n_years=20, n_realizations=2, seed=123)
        result2 = gen.generate(n_years=20, n_realizations=2, seed=123)

        for r in range(2):
            df1 = result1.data_by_realization[r]
            df2 = result2.data_by_realization[r]
            pd.testing.assert_frame_equal(df1, df2)

    def test_generate_without_fit_raises(self, sample_annual_series):
        """Test generation raises error without fitting."""
        gen = WARMGenerator()
        gen.preprocessing(sample_annual_series)

        with pytest.raises(ValueError, match="fit"):
            gen.generate(n_years=10)

    def test_generate_has_datetime_index(self, short_annual_series):
        """Test generated data has DatetimeIndex."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        result = gen.generate(n_years=15, n_realizations=1, seed=42)
        df = result.data_by_realization[0]

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_generate_annual_frequency(self, short_annual_series):
        """Test generated data has annual frequency."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        result = gen.generate(n_years=15, n_realizations=1, seed=42)
        df = result.data_by_realization[0]

        # Check frequency is annual
        assert df.index.freq in ["YS", "<YearBegin>"]

    def test_generate_default_n_years(self, short_annual_series):
        """Test generation with default n_years."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        # Default should use length of historical data
        result = gen.generate(n_realizations=1, seed=42)

        df = result.data_by_realization[0]
        assert len(df) == len(gen.Q_obs_annual)

    def test_generate_n_timesteps(self, short_annual_series):
        """Test generation with n_timesteps parameter."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        n_timesteps = 25
        result = gen.generate(n_timesteps=n_timesteps, n_realizations=1, seed=42)

        df = result.data_by_realization[0]
        assert len(df) == n_timesteps


class TestWARMWaveletProperties:
    """Tests for wavelet-specific properties."""

    def test_sawp_captures_variability(self, short_annual_series):
        """Test that SAWP varies over time."""
        gen = WARMGenerator(scales=16)
        gen.fit(short_annual_series)

        # SAWP should have variation (not constant)
        assert np.std(gen.sawp_) > 0
        assert np.max(gen.sawp_) > np.min(gen.sawp_)

    def test_multiple_realizations_differ(self, short_annual_series):
        """Test that multiple realizations are different."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        result = gen.generate(n_years=20, n_realizations=5, seed=42)

        # Compare realizations pairwise
        different_count = 0
        for i in range(5):
            for j in range(i + 1, 5):
                df_i = result.data_by_realization[i]
                df_j = result.data_by_realization[j]
                if not df_i.equals(df_j):
                    different_count += 1

        assert different_count > 0  # At least some should differ

    def test_wavelet_scales_used(self, short_annual_series):
        """Test that scales are correctly stored."""
        gen = WARMGenerator(scales=16)
        gen.fit(short_annual_series)

        assert gen.scales_used_ is not None
        assert len(gen.scales_used_) == 16
        assert np.all(gen.scales_used_ == np.arange(1, 17))


class TestWARMSerialization:
    """Tests for saving and loading WARMGenerator."""

    def test_pickle_save_load(self, short_annual_series, tmp_path):
        """Test saving and loading via pickle."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        # Save
        filepath = tmp_path / "warm_generator.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(gen, f)

        # Load
        with open(filepath, "rb") as f:
            gen_loaded = pickle.load(f)

        # Verify attributes preserved
        assert gen_loaded.is_fitted is True
        assert gen_loaded.is_preprocessed is True
        assert gen_loaded.scales == 8

    def test_pickle_generate_after_load(self, short_annual_series, tmp_path):
        """Test generation works after loading from pickle."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        # Save and load
        filepath = tmp_path / "warm_generator.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(gen, f)

        with open(filepath, "rb") as f:
            gen_loaded = pickle.load(f)

        # Generate from loaded generator
        result = gen_loaded.generate(n_years=15, n_realizations=2, seed=99)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 2


class TestWARMStatisticalProperties:
    """Tests for statistical properties of generated data."""

    def test_generated_mean_reasonable(self, short_annual_series):
        """Test generated data has reasonable mean."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        result = gen.generate(n_years=30, n_realizations=20, seed=42)

        # Compute ensemble mean
        all_data = []
        for r in range(20):
            all_data.append(result.data_by_realization[r].values)
        all_data = np.concatenate(all_data, axis=0)

        gen_mean = all_data.mean()
        obs_mean = gen.Q_obs_annual.mean()

        # Generated mean should be in reasonable range
        # WARM uses scale factors so allow wider tolerance
        ratio = gen_mean / obs_mean
        assert 0.1 < ratio < 10.0

    def test_generated_has_variability(self, short_annual_series):
        """Test generated data has reasonable variability."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        result = gen.generate(n_years=30, n_realizations=10, seed=42)

        # Each realization should have variability
        for r in range(10):
            df = result.data_by_realization[r]
            assert np.std(df.values) > 0


class TestWARMOutputFrequency:
    """Tests for output frequency property."""

    def test_output_frequency_annual(self, sample_annual_series):
        """Test output frequency is annual."""
        gen = WARMGenerator()

        freq = gen.output_frequency
        assert freq == "YS"


class TestWARMARModels:
    """Tests for AR model fitting components."""

    def test_ar_model_fitting_ar1(self, short_annual_series):
        """Test AR(1) model fitting (per band, plus noise residual)."""
        gen = WARMGenerator(scales=4, ar_order=1)
        gen.fit(short_annual_series)

        # AR fit lives on each detected band, plus the noise residual.
        for band_idx, params in gen.ar_params_.items():
            assert len(params["coeffs"]) == 1
            assert params["order"] == 1
        assert gen.noise_ar_params_ is not None
        assert len(gen.noise_ar_params_["coeffs"]) == 1

    def test_ar_model_fitting_ar2(self, short_annual_series):
        """Test AR(2) model fitting (per band, plus noise residual)."""
        gen = WARMGenerator(scales=4, ar_order=2)
        gen.fit(short_annual_series)

        for band_idx, params in gen.ar_params_.items():
            assert len(params["coeffs"]) == 2
            assert params["order"] == 2
        assert gen.noise_ar_params_ is not None
        assert len(gen.noise_ar_params_["coeffs"]) == 2


class TestWARMIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, short_annual_series):
        """Test complete workflow."""
        gen = WARMGenerator(scales=8)

        # Preprocessing
        gen.preprocessing(short_annual_series)
        assert gen.is_preprocessed is True

        # Fit
        gen.fit()
        assert gen.is_fitted is True

        # Generate
        result = gen.generate(n_years=25, n_realizations=5, seed=42)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 5

        # Check data quality
        for r in range(5):
            df = result.data_by_realization[r]
            assert not df.isna().any().any()
            assert (df >= 0).all().all()

    def test_workflow_different_ar_orders(self, short_annual_series):
        """Test workflow with different AR orders."""
        for ar_order in [1, 2, 3]:
            gen = WARMGenerator(scales=8, ar_order=ar_order)
            gen.fit(short_annual_series)
            result = gen.generate(n_years=15, n_realizations=2, seed=42)

            assert isinstance(result, Ensemble)

    def test_get_params(self, short_annual_series):
        """Test get_params method."""
        gen = WARMGenerator(scales=8)
        gen.fit(short_annual_series)

        params = gen.get_params()
        assert isinstance(params, dict)
        # Check for expected parameter keys
        assert "debug" in params or "wavelet" in params


class TestWARMBandIdentification:
    """Tests for the Nowak 2011 band-identification machinery."""

    def test_invalid_background_spectrum(self):
        """Background spectrum must be 'red' or 'white'."""
        with pytest.raises(ValueError, match="background_spectrum"):
            WARMGenerator(background_spectrum="pink")

    def test_invalid_significance_level(self):
        """Significance level must be in (0, 1)."""
        with pytest.raises(ValueError, match="significance_level"):
            WARMGenerator(significance_level=1.5)

    def test_invalid_ar_select(self):
        """ar_select must be 'fixed' or 'aic'."""
        with pytest.raises(ValueError, match="ar_select"):
            WARMGenerator(ar_select="bic")

    def test_invalid_ar_method(self):
        """ar_method must be 'burg' or 'yule_walker'."""
        with pytest.raises(ValueError, match="ar_method"):
            WARMGenerator(ar_method="ols")

    def test_invalid_sawp_resampling(self):
        """sawp_resampling must be 'random_offset' or 'historical'."""
        with pytest.raises(ValueError, match="sawp_resampling"):
            WARMGenerator(sawp_resampling="iid")

    def test_explicit_ar_select_overrides_ar_order_default(self):
        """ar_select='aic' with ar_order given still selects by AIC."""
        gen = WARMGenerator(ar_order=1, ar_select="aic")
        assert gen.ar_select == "aic"

    def test_auto_band_detection_attributes(self, sample_annual_series):
        """Auto-detected bands populate the bands_ attribute."""
        gen = WARMGenerator(background_spectrum="white")
        gen.fit(sample_annual_series)

        assert gen.bands_ is not None
        assert gen.global_spectrum_ is not None
        assert gen.significance_threshold_ is not None
        assert gen.fourier_periods_ is not None

        # Each band reports its scale indices, period range, AR fit, and SAWP.
        for band in gen.bands_:
            assert "scale_indices" in band
            assert "period_min" in band
            assert "period_max" in band
            assert "ar_params" in band
            assert "sawp" in band
            assert band["sawp"].shape == (len(gen.Q_obs_annual),)
            assert band["auto_detected"] is True

    def test_user_specified_bands(self, sample_annual_series):
        """User-supplied bands are honored exactly."""
        gen = WARMGenerator(bands=[(8.0, 32.0)])
        gen.fit(sample_annual_series)

        assert gen.bands_ is not None
        assert len(gen.bands_) == 1
        assert gen.bands_[0]["auto_detected"] is False
        # Band period range must lie within (or be the inner subset of) the
        # requested window.
        assert gen.bands_[0]["period_min"] >= 8.0 - 1e-6
        assert gen.bands_[0]["period_max"] <= 32.0 + 1e-6

    def test_red_vs_white_thresholds_differ(self, sample_annual_series):
        """Red-noise and white-noise backgrounds yield different thresholds."""
        gen_white = WARMGenerator(background_spectrum="white")
        gen_white.fit(sample_annual_series)
        gen_red = WARMGenerator(background_spectrum="red")
        gen_red.fit(sample_annual_series)

        # Background spectra differ only when lag-1 > 0; the synthetic series
        # has substantial persistence so the two backgrounds must not match.
        assert not np.allclose(
            gen_white.background_spectrum_values_,
            gen_red.background_spectrum_values_,
        )

    def test_aic_order_selection(self, sample_annual_series):
        """ar_select='aic' returns a valid order in [1, n_ar_max]."""
        gen = WARMGenerator(ar_select="aic", n_ar_max=4)
        gen.fit(sample_annual_series)
        for params in gen.ar_params_.values():
            assert 1 <= params["order"] <= 4
        assert 1 <= gen.noise_ar_params_["order"] <= 4

    def test_geometric_scales_default(self, sample_annual_series):
        """Default scales follow a geometric Torrence-Compo grid."""
        gen = WARMGenerator()
        gen.fit(sample_annual_series)
        # Geometric grid has nearly constant log2 spacing.
        log_diffs = np.diff(np.log2(gen.scales_used_))
        assert np.allclose(log_diffs, log_diffs[0], rtol=1e-6)


class TestWARMReconstruction:
    """Tests for the variance-preserving inverse CWT reconstruction."""

    def test_reconstruction_recovers_mean(self, sample_annual_series):
        """
        Generated ensemble mean approximately matches the historic mean
        without any post-hoc moment matching.
        """
        gen = WARMGenerator(background_spectrum="red")
        gen.fit(sample_annual_series)
        ensemble = gen.generate(
            n_years=len(sample_annual_series),
            n_realizations=50,
            seed=2026,
        )
        all_values = np.concatenate(
            [ensemble.data_by_realization[r].values.flatten() for r in range(50)]
        )
        ratio = np.mean(all_values) / sample_annual_series.mean()
        # Mean preservation is tight because the historical mean is added
        # back in synthesis after band/noise summation.
        assert 0.85 < ratio < 1.15

    def test_band_reconstruction_in_observed_units(self, sample_annual_series):
        """Per-band time-domain reconstructions are in flow units."""
        gen = WARMGenerator()
        gen.fit(sample_annual_series)
        for band in gen.bands_:
            recon = band["reconstruction"]
            assert recon.shape == (len(sample_annual_series),)
            # Reconstruction magnitudes should be on the same order as the
            # observed centered series, not normalized.
            assert np.std(recon) <= np.std(sample_annual_series.values) * 5


class TestWARMLowerBound:
    """Tests for the numeric lower_bound on synthesis output."""

    def test_default_lower_bound_is_zero(self):
        gen = WARMGenerator()
        assert gen.lower_bound == 0.0

    def test_default_clamps_at_zero(self, sample_annual_series):
        gen = WARMGenerator()
        gen.fit(sample_annual_series)
        ensemble = gen.generate(n_years=80, n_realizations=10, seed=42)
        for r in range(10):
            values = ensemble.data_by_realization[r].values
            assert float(values.min()) >= -1e-9

    def test_numeric_lower_bound_clamps_to_value(self, sample_annual_series):
        floor = 300.0
        gen = WARMGenerator(lower_bound=floor)
        gen.fit(sample_annual_series)
        ensemble = gen.generate(n_years=80, n_realizations=10, seed=42)
        for r in range(10):
            values = ensemble.data_by_realization[r].values
            assert float(values.min()) >= floor - 1e-9

    def test_string_lower_bound_raises(self):
        with pytest.raises(ValueError, match="lower_bound"):
            WARMGenerator(lower_bound="obs_min")


class TestWARMNowak2011Compliance:
    """Tests for the Nowak et al. (2011) mathematical fixes."""

    def test_variance_correction_factor_is_set(self, sample_annual_series):
        """Eq. 7 variance correction factor must be computed at fit time."""
        gen = WARMGenerator()
        gen.fit(sample_annual_series)
        assert gen.variance_correction_ is not None
        # Independent-component variance under-estimates total variance when
        # bands and noise carry a small positive cross-covariance, so the
        # correction factor is typically >= 1. A tiny tolerance allows for
        # near-zero cross-covariance on synthetic inputs.
        assert gen.variance_correction_ > 0.5
        assert gen.variance_correction_ < 5.0

    def test_synthetic_variance_matches_observed(self, sample_annual_series):
        """After Eq. 7 correction, ensemble variance approximates observed."""
        gen = WARMGenerator()
        gen.fit(sample_annual_series)
        ensemble = gen.generate(
            n_years=len(sample_annual_series), n_realizations=50, seed=2026
        )
        obs_var = float(np.var(sample_annual_series.values, ddof=0))
        ens_vars = [
            float(np.var(ensemble.data_by_realization[r].values, ddof=0))
            for r in range(50)
        ]
        ratio = float(np.mean(ens_vars)) / obs_var
        assert 0.7 < ratio < 1.4

    def test_sawp_resampling_preserves_autocorrelation(self):
        """Historical SAWP resampling preserves lag-1 autocorrelation."""
        rng = np.random.default_rng(0)
        # Smooth SAWP with strong autocorrelation.
        t = np.arange(100)
        sawp_obs = 1.0 + 0.5 * np.sin(2 * np.pi * t / 25.0)
        sawp_syn = WARMGenerator._resample_sawp(sawp_obs, 200, rng)
        lag1 = float(np.corrcoef(sawp_syn[:-1], sawp_syn[1:])[0, 1])
        assert lag1 > 0.9

    def test_noise_bootstrap_stores_residuals(self, sample_annual_series):
        """Bootstrap noise mode stores standardized empirical residuals."""
        gen = WARMGenerator(noise_model="ar_bootstrap")
        gen.fit(sample_annual_series)
        assert "std_residuals" in gen.noise_ar_params_
        std_resid = gen.noise_ar_params_["std_residuals"]
        assert std_resid.ndim == 1
        assert len(std_resid) > 0

    def test_noise_model_validation(self):
        with pytest.raises(ValueError, match="noise_model"):
            WARMGenerator(noise_model="kde")

    def test_significance_threshold_scale_invariance(self):
        """
        T&C 1998 Eq. 18 normalizes the significance threshold by the data
        variance and the wavelet integrated squared modulus. A regression
        test against an earlier implementation that omitted those factors:
        the fraction of scales flagged as significant on pure white noise
        must not depend on the input variance scale.
        """
        rates_by_sigma = {}
        for sigma in (1.0, 1000.0):
            rates = []
            for trial in range(20):
                rng = np.random.default_rng(trial)
                Q = pd.DataFrame(
                    {"site": rng.normal(0.0, sigma, 200)},
                    index=pd.date_range("1800-01-01", periods=200, freq="YS"),
                )
                gen = WARMGenerator(background_spectrum="white")
                gen.fit(Q)
                rates.append(float(np.mean(gen.significant_mask_)))
            rates_by_sigma[sigma] = float(np.mean(rates))
        # Mean FPR should be near zero for white noise (a strict test on the
        # global spectrum) and must be invariant under rescaling of sigma.
        assert (
            rates_by_sigma[1.0] < 0.30
        ), f"FPR too high on sigma=1 white noise: {rates_by_sigma[1.0]}"
        assert (
            rates_by_sigma[1000.0] < 0.30
        ), f"FPR too high on sigma=1000 white noise: {rates_by_sigma[1000.0]}"
        assert abs(rates_by_sigma[1.0] - rates_by_sigma[1000.0]) < 0.05, (
            "Significance FPR must not depend on sigma scale; got " f"{rates_by_sigma}"
        )

    def test_noise_gaussian_and_bootstrap_differ(self, sample_annual_series):
        """Gaussian vs bootstrap noise modes produce different ensembles."""
        gen_g = WARMGenerator(noise_model="ar_gaussian")
        gen_g.fit(sample_annual_series)
        ens_g = gen_g.generate(n_years=30, n_realizations=2, seed=99)

        gen_b = WARMGenerator(noise_model="ar_bootstrap")
        gen_b.fit(sample_annual_series)
        ens_b = gen_b.generate(n_years=30, n_realizations=2, seed=99)

        for r in range(2):
            assert not ens_g.data_by_realization[r].equals(ens_b.data_by_realization[r])


class TestWARMEnsembleMetadata:
    """Regression tests for proper EnsembleMetadata on generated output."""

    def test_ensemble_has_annual_frequency_attribute(self, sample_annual_series):
        """ensemble.frequency must reflect the output_frequency 'YS'.

        Regression test: previously WARMGenerator.generate() returned
        Ensemble(realizations) with no metadata, so ensemble.frequency was
        None and any downstream consumer that branches on frequency would
        break.
        """
        gen = WARMGenerator()
        gen.fit(sample_annual_series)
        ensemble = gen.generate(n_years=20, n_realizations=2, seed=1)

        assert ensemble.frequency == "YS"
        assert ensemble.metadata.generator_class == "WARMGenerator"
        assert ensemble.metadata.n_realizations == 2
        assert ensemble.metadata.n_sites == 1


class TestWARMSpectralReproduction:
    """Tests for AR estimation and spectral-peak reproduction."""

    @staticmethod
    def _sinusoid_plus_noise(seed: int = 1, n: int = 101) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        t = np.arange(n)
        amp = np.where(t < 50, 1.0, 0.2)
        x = amp * np.sin(2 * np.pi * t / 25.0) + rng.normal(0.0, 1.0, n)
        idx = pd.date_range("1900-01-01", periods=n, freq="YS")
        return pd.DataFrame({"Q": x + 10.0}, index=idx)

    @staticmethod
    def _gws_at_period(x, scales, wavelet, period):
        import pywt

        c, f = pywt.cwt(
            np.asarray(x, float) - np.mean(x), scales, wavelet, sampling_period=1.0
        )
        G = np.mean(np.abs(c) ** 2, axis=1)
        i = int(np.argmin(np.abs(1.0 / f - period)))
        return float(G[i])

    def test_burg_recovers_known_ar2(self):
        """Burg estimates of a long AR(2) realization match the truth."""
        rng = np.random.default_rng(0)
        x = np.zeros(5000)
        for t in range(2, len(x)):
            x[t] = 1.5 * x[t - 1] - 0.8 * x[t - 2] + rng.normal()
        gen = WARMGenerator()
        params = gen._fit_ar_burg(x, 2)
        assert np.allclose(params["coeffs"], [1.5, -0.8], atol=0.03)
        assert abs(params["sigma"] - 1.0) < 0.05
        # Stationary variance implied by (phi, sigma) equals the sample
        # variance by construction.
        rho = gen._ar_autocorrelation(params["coeffs"])
        implied = params["sigma"] ** 2 / (1.0 - np.dot(params["coeffs"], rho))
        assert np.isclose(implied, np.var(x), rtol=1e-6)

    def test_sawp_in_variance_units(self):
        """Whole-spectrum SAWP averages to the series variance (T&C Eq. 14)."""
        df = self._sinusoid_plus_noise()
        gen = WARMGenerator(background_spectrum="white")
        gen.fit(df)
        all_idx = np.arange(len(gen.scales_used_))
        sawp_all = gen._band_sawp(gen.wavelet_coeffs_, gen.scales_used_, all_idx)
        var_obs = float(np.var(df["Q"].values))
        assert 0.8 < float(np.mean(sawp_all)) / var_obs < 1.2

    def test_default_reproduces_spectral_peak(self):
        """Synthetic peak power at 25 yr lies within a reasonable band of observed.

        A sinusoid of period 25 yr with a non-stationary amplitude plus
        N(0, 1) noise is the Nowak et al. (2011) Section 2 test case. With
        the defaults (AIC order selection, Burg estimation) the ensemble
        median global wavelet power at the dominant period must be a large
        fraction of the observed power; the old AR(1) Yule-Walker default
        gave roughly 0.1-0.2 of it.
        """
        df = self._sinusoid_plus_noise(seed=1)
        gen = WARMGenerator(background_spectrum="white")
        gen.fit(df)
        assert len(gen.bands_) >= 1
        assert all(b["ar_params"]["order"] >= 2 for b in gen.bands_)
        scales = gen.scales_used_
        obs_pk = self._gws_at_period(df["Q"].values, scales, gen.wavelet, 25.0)
        ens = gen.generate(n_years=101, n_realizations=100, seed=3)
        X = ens.data_by_site["Q"].values
        syn = [
            self._gws_at_period(X[:, r], scales, gen.wavelet, 25.0)
            for r in range(X.shape[1])
        ]
        ratio = float(np.median(syn)) / obs_pk
        assert 0.4 < ratio < 1.6, f"peak power ratio {ratio:.2f}"

    def test_ar1_fixed_under_reproduces_peak(self):
        """A fixed AR(1) band model cannot carry the spectral peak."""
        df = self._sinusoid_plus_noise(seed=1)
        gen = WARMGenerator(background_spectrum="white", ar_order=1)
        gen.fit(df)
        scales = gen.scales_used_
        obs_pk = self._gws_at_period(df["Q"].values, scales, gen.wavelet, 25.0)
        ens = gen.generate(n_years=101, n_realizations=60, seed=3)
        X = ens.data_by_site["Q"].values
        syn = [
            self._gws_at_period(X[:, r], scales, gen.wavelet, 25.0)
            for r in range(X.shape[1])
        ]
        assert float(np.median(syn)) / obs_pk < 0.5

    def test_historical_sawp_preserves_epoch_timing(self):
        """sawp_resampling='historical' keeps high power in the first half."""
        df = self._sinusoid_plus_noise(seed=1)
        gen = WARMGenerator(background_spectrum="white", sawp_resampling="historical")
        gen.fit(df)
        ens = gen.generate(n_years=101, n_realizations=40, seed=5)
        X = ens.data_by_site["Q"].values
        first = np.mean(np.var(X[:50], axis=0))
        second = np.mean(np.var(X[50:], axis=0))
        assert first > 1.2 * second

    def test_random_offset_sawp_randomizes_epoch_timing(self):
        """sawp_resampling='random_offset' spreads power over the record.

        The ensemble-mean envelope is stationary, so the first-half to
        second-half variance ratio is much smaller than with 'historical'.
        """
        df = self._sinusoid_plus_noise(seed=1)
        gen = WARMGenerator(
            background_spectrum="white", sawp_resampling="random_offset"
        )
        gen.fit(df)
        assert gen.sawp_resampling == "random_offset"
        ens = gen.generate(n_years=101, n_realizations=60, seed=5)
        X = ens.data_by_site["Q"].values
        first = np.mean(np.var(X[:50], axis=0))
        second = np.mean(np.var(X[50:], axis=0))
        assert 0.7 < first / second < 1.4

    def test_yule_walker_option_still_available(self):
        """ar_method='yule_walker' fits and generates."""
        df = self._sinusoid_plus_noise(seed=1)
        gen = WARMGenerator(ar_method="yule_walker", ar_order=2)
        gen.fit(df)
        ens = gen.generate(n_years=50, n_realizations=2, seed=0)
        assert ens.data_by_site["Q"].shape == (50, 2)
