"""
Tests for Phase Randomization Generator (Brunner et al. 2019).
"""

import pytest
import numpy as np
import pandas as pd

from synhydro.methods.generation.hybrid.phase_randomization import (
    PhaseRandomizationGenerator,
)
from synhydro.core.ensemble import Ensemble


@pytest.fixture
def sample_daily_series_long():
    """Generate a sample daily time series with at least 2 full years (730+ days)."""
    # Create exactly 3 years of data (no leap days will be handled by preprocessing)
    dates = pd.date_range(start="2010-01-01", end="2012-12-31", freq="D")
    np.random.seed(42)

    # Generate seasonal flow data
    n = len(dates)
    seasonal = 100 + 50 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = np.random.gamma(shape=2.0, scale=20.0, size=n)
    values = seasonal + noise

    return pd.Series(values, index=dates, name="site_1")


@pytest.fixture
def sample_daily_dataframe_long():
    """Generate a sample daily multi-site DataFrame with at least 2 full years."""
    dates = pd.date_range(start="2010-01-01", end="2012-12-31", freq="D")
    np.random.seed(42)

    n = len(dates)
    n_sites = 3
    data = {}

    for i in range(n_sites):
        # Generate seasonal flow data with noise
        seasonal = 100 + 50 * np.sin(2 * np.pi * np.arange(n) / 365)
        noise = np.random.gamma(shape=2.0, scale=20.0, size=n)
        data[f"site_{i+1}"] = seasonal + noise

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_daily_series_short():
    """Generate a short daily time series (less than 2 years) for error testing."""
    dates = pd.date_range(start="2010-01-01", end="2010-12-31", freq="D")
    np.random.seed(42)
    values = np.random.gamma(shape=2.0, scale=50.0, size=len(dates))
    return pd.Series(values, index=dates, name="site_1")


class TestPhaseRandomizationGeneratorInit:
    """Tests for PhaseRandomizationGenerator initialization."""

    def test_initialization_default_params(self, sample_daily_series_long):
        """Test initialization with default parameters."""
        gen = PhaseRandomizationGenerator()

        assert gen.is_preprocessed is False
        assert gen.is_fitted is False
        assert gen.debug is False
        assert gen.marginal == "kappa"
        assert gen.win_h_length == 15

    def test_initialization_with_empirical_marginal(self, sample_daily_series_long):
        """Test initialization with empirical marginal distribution."""
        gen = PhaseRandomizationGenerator(marginal="empirical")

        assert gen.marginal == "empirical"

    def test_initialization_with_custom_window(self, sample_daily_series_long):
        """Test initialization with custom window length."""
        gen = PhaseRandomizationGenerator(win_h_length=20)

        assert gen.win_h_length == 20

    def test_initialization_invalid_marginal(self, sample_daily_series_long):
        """Test that invalid marginal raises ValueError."""
        with pytest.raises(ValueError, match="marginal must be"):
            PhaseRandomizationGenerator(marginal="invalid")

    def test_initialization_with_dataframe(self, sample_daily_dataframe_long):
        """Test initialization without data (data provided later)."""
        gen = PhaseRandomizationGenerator()

        assert gen.is_preprocessed is False

    def test_output_frequency(self, sample_daily_series_long):
        """Test that output frequency is daily."""
        gen = PhaseRandomizationGenerator()
        assert gen.output_frequency == "D"


class TestPhaseRandomizationGeneratorPreprocessing:
    """Tests for PhaseRandomizationGenerator preprocessing."""

    def test_preprocessing_basic(self, sample_daily_series_long):
        """Test basic preprocessing."""
        gen = PhaseRandomizationGenerator()
        gen.preprocessing(sample_daily_series_long)

        assert gen.is_preprocessed is True
        assert hasattr(gen, "Q_obs_")
        assert hasattr(gen, "day_index_")
        assert hasattr(gen, "n_years_")

    def test_preprocessing_removes_leap_days(self, sample_daily_series_long):
        """Test that preprocessing removes February 29."""
        gen = PhaseRandomizationGenerator()
        gen.preprocessing(sample_daily_series_long)

        # Data length should be multiple of 365
        assert len(gen.Q_obs_) % 365 == 0

    def test_preprocessing_day_index_range(self, sample_daily_series_long):
        """Test that day index is in range 1-365."""
        gen = PhaseRandomizationGenerator()
        gen.preprocessing(sample_daily_series_long)

        assert gen.day_index_.min() >= 1
        assert gen.day_index_.max() <= 365

    def test_preprocessing_minimum_data_requirement(self, sample_daily_series_short):
        """Test that preprocessing fails with insufficient data."""
        gen = PhaseRandomizationGenerator()

        with pytest.raises(ValueError, match="At least 730 days"):
            gen.preprocessing(sample_daily_series_short)

    def test_preprocessing_multisite_raises(self, sample_daily_dataframe_long):
        """Test that multi-site data raises ValueError."""
        gen = PhaseRandomizationGenerator()
        with pytest.raises(ValueError, match="univariate"):
            gen.preprocessing(sample_daily_dataframe_long)


class TestPhaseRandomizationGeneratorFit:
    """Tests for PhaseRandomizationGenerator fitting."""

    def test_fit_kappa_marginal(self, sample_daily_series_long):
        """Test fitting with kappa marginal distribution."""
        gen = PhaseRandomizationGenerator(marginal="kappa")
        gen.fit(sample_daily_series_long)

        assert gen.is_fitted is True
        assert hasattr(gen, "par_day_")
        assert hasattr(gen, "norm_")
        assert hasattr(gen, "modulus_")
        assert hasattr(gen, "phases_")

    def test_fit_empirical_marginal(self, sample_daily_series_long):
        """Test fitting with empirical marginal distribution."""
        gen = PhaseRandomizationGenerator(marginal="empirical")
        gen.fit(sample_daily_series_long)

        assert gen.is_fitted is True
        # Empirical marginal doesn't fit kappa params
        assert gen.par_day_ == {}

    def test_fit_kappa_params_structure(self, sample_daily_series_long):
        """Test that kappa parameters have correct structure."""
        gen = PhaseRandomizationGenerator(marginal="kappa")
        gen.fit(sample_daily_series_long)

        # Should have parameters for most days
        assert len(gen.par_day_) > 0

        # Check structure of a valid parameter set
        for day, params in gen.par_day_.items():
            if params is not None:
                assert "xi" in params
                assert "alfa" in params
                assert "k" in params
                assert "h" in params
                break

    def test_fit_normal_score_transform(self, sample_daily_series_long):
        """Test that normal score transform is applied."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        # Normalized data should have zero mean approximately
        assert gen.norm_ is not None
        assert len(gen.norm_) == len(gen.Q_obs_)

    def test_fit_fft_computation(self, sample_daily_series_long):
        """Test that FFT is computed correctly."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        # FFT should have same length as data
        assert len(gen.modulus_) == len(gen.Q_obs_)
        assert len(gen.phases_) == len(gen.Q_obs_)

        # Modulus should be non-negative
        assert np.all(gen.modulus_ >= 0)

        # Phases should be in [-pi, pi]
        assert np.all(gen.phases_ >= -np.pi)
        assert np.all(gen.phases_ <= np.pi)

    def test_fit_without_preprocessing_raises(self, sample_daily_series_long):
        """Test that fit without preprocessing raises error."""
        gen = PhaseRandomizationGenerator()

        with pytest.raises(Exception):  # Will raise due to validation
            gen.fit()

    def test_fit_creates_fitted_params(self, sample_daily_series_long):
        """Test that fit creates FittedParams object."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        assert hasattr(gen, "fitted_params_")
        assert gen.fitted_params_.n_sites_ == 1


class TestPhaseRandomizationGeneratorGenerate:
    """Tests for PhaseRandomizationGenerator generation."""

    def test_generate_single_realization(self, sample_daily_series_long):
        """Test generating a single realization."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=1, seed=42)

        assert isinstance(result, Ensemble)
        assert len(result.realization_ids) == 1

    def test_generate_multiple_realizations(self, sample_daily_series_long):
        """Test generating multiple realizations."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=5, seed=42)

        assert isinstance(result, Ensemble)
        assert len(result.realization_ids) == 5

    def test_generate_reproducibility(self, sample_daily_series_long):
        """Test that seed produces reproducible results."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result1 = gen.generate(n_realizations=1, seed=42)
        result2 = gen.generate(n_realizations=1, seed=42)

        # Same seed should produce same results
        np.testing.assert_array_almost_equal(
            result1.data_by_realization[0].values, result2.data_by_realization[0].values
        )

    def test_generate_different_seeds(self, sample_daily_series_long):
        """Test that different seeds produce different results."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result1 = gen.generate(n_realizations=1, seed=42)
        result2 = gen.generate(n_realizations=1, seed=123)

        # Different seeds should produce different results
        assert not np.allclose(
            result1.data_by_realization[0].values, result2.data_by_realization[0].values
        )

    def test_generate_output_length_default(self, sample_daily_series_long):
        """Test that output length equals observed length when n_years is not given."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=1, seed=42)

        assert len(result.data_by_realization[0]) == len(gen.Q_obs_)

    def test_generate_n_years(self, sample_daily_series_long):
        """Test that n_years produces the correct output length."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        n_years = 10
        result = gen.generate(n_realizations=2, n_years=n_years, seed=42)

        expected_len = n_years * 365
        for r in result.realization_ids:
            assert len(result.data_by_realization[r]) == expected_len

    def test_generate_n_years_longer_than_obs(self, sample_daily_series_long):
        """Test n_years greater than the observed record length."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        n_years = 50
        result = gen.generate(n_realizations=1, n_years=n_years, seed=42)

        expected_len = n_years * 365
        assert len(result.data_by_realization[0]) == expected_len

    def test_generate_n_years_noleap_index(self, sample_daily_series_long):
        """Test that output index contains no Feb 29 entries."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=1, n_years=5, seed=42)
        idx = result.data_by_realization[0].index
        feb29 = idx[(idx.month == 2) & (idx.day == 29)]
        assert len(feb29) == 0

    def test_generate_non_negative(self, sample_daily_series_long):
        """Test that generated flows are non-negative."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=10, seed=42)

        for r in result.realization_ids:
            assert (result.data_by_realization[r].values >= 0).all()

    def test_generate_kappa_marginal(self, sample_daily_series_long):
        """Test generation with kappa marginal."""
        gen = PhaseRandomizationGenerator(marginal="kappa")
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=1, seed=42)

        assert isinstance(result, Ensemble)
        assert not result.data_by_realization[0].isna().any().any()

    def test_generate_empirical_marginal(self, sample_daily_series_long):
        """Test generation with empirical marginal."""
        gen = PhaseRandomizationGenerator(marginal="empirical")
        gen.fit(sample_daily_series_long)

        result = gen.generate(n_realizations=1, seed=42)

        assert isinstance(result, Ensemble)
        assert not result.data_by_realization[0].isna().any().any()

    def test_generate_without_fit_raises(self, sample_daily_series_long):
        """Test that generate without fit raises error."""
        gen = PhaseRandomizationGenerator()
        gen.preprocessing(sample_daily_series_long)

        with pytest.raises(Exception):  # Will raise due to validation
            gen.generate(n_realizations=1)


class TestLMomentsComputation:
    """Tests for L-moments computation."""

    def test_lmoments_basic(self, sample_daily_series_long):
        """Test basic L-moments computation."""
        gen = PhaseRandomizationGenerator()

        # Generate test data
        np.random.seed(42)
        data = np.random.gamma(shape=2.0, scale=50.0, size=100)

        lmom = gen._compute_lmoments(data)

        assert "l1" in lmom
        assert "l2" in lmom
        assert "lcv" in lmom
        assert "lca" in lmom
        assert "lkur" in lmom

    def test_lmoments_l1_is_mean(self, sample_daily_series_long):
        """Test that L1 is approximately the sample mean."""
        gen = PhaseRandomizationGenerator()

        np.random.seed(42)
        data = np.random.gamma(shape=2.0, scale=50.0, size=1000)

        lmom = gen._compute_lmoments(data)

        # L1 should be the mean
        np.testing.assert_almost_equal(lmom["l1"], np.mean(data), decimal=5)

    def test_lmoments_insufficient_data(self, sample_daily_series_long):
        """Test that L-moments computation fails with insufficient data."""
        gen = PhaseRandomizationGenerator()

        data = np.array([1.0, 2.0, 3.0])  # Only 3 observations

        with pytest.raises(ValueError, match="at least 4 observations"):
            gen._compute_lmoments(data)


class TestKappaDistribution:
    """Tests for kappa distribution functions."""

    def test_invF_kappa_basic(self, sample_daily_series_long):
        """Test inverse kappa CDF basic functionality."""
        gen = PhaseRandomizationGenerator()

        F = np.array([0.1, 0.5, 0.9])
        x = gen._invF_kappa(F, xi=0, alfa=1, k=0.5, h=0.5)

        # Output should be finite
        assert np.all(np.isfinite(x))

        # Values should be monotonically increasing with F
        assert x[0] < x[1] < x[2]

    def test_invF_kappa_gev_case(self, sample_daily_series_long):
        """Test inverse kappa CDF when h=0 (GEV case)."""
        gen = PhaseRandomizationGenerator()

        F = np.array([0.1, 0.5, 0.9])
        x = gen._invF_kappa(F, xi=0, alfa=1, k=0.5, h=0)

        assert np.all(np.isfinite(x))

    def test_rand_kappa_basic(self, sample_daily_series_long):
        """Test random kappa generation."""
        gen = PhaseRandomizationGenerator()

        np.random.seed(42)
        samples = gen._rand_kappa(n=1000, xi=0, alfa=1, k=0.5, h=0.5)

        assert len(samples) == 1000
        assert np.all(np.isfinite(samples))


class TestStatisticalProperties:
    """Tests for statistical properties of generated data."""

    def test_mean_within_tolerance(self, sample_daily_series_long):
        """Test that generated mean is within tolerance of observed."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        # Generate many realizations
        result = gen.generate(n_realizations=50, seed=42)

        obs_mean = gen.Q_obs_.mean()

        # Compute ensemble mean
        ensemble_means = [
            result.data_by_realization[r].values.mean() for r in result.realization_ids
        ]
        sim_mean = np.mean(ensemble_means)

        # Should be within 20% of observed mean
        relative_error = abs(sim_mean - obs_mean) / obs_mean
        assert relative_error < 0.2

    def test_std_within_tolerance(self, sample_daily_series_long):
        """Test that generated std is within tolerance of observed."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        # Generate many realizations
        result = gen.generate(n_realizations=50, seed=42)

        obs_std = gen.Q_obs_.std()

        # Compute ensemble std
        ensemble_stds = [
            result.data_by_realization[r].values.std() for r in result.realization_ids
        ]
        sim_std = np.mean(ensemble_stds)

        # Should be within 30% of observed std
        relative_error = abs(sim_std - obs_std) / obs_std
        assert relative_error < 0.3


class TestPhaseRandomizationGeneratorSaveLoad:
    """Tests for PhaseRandomizationGenerator save and load."""

    def test_save_and_load(self, sample_daily_series_long, tmp_path):
        """Test saving and loading generator."""
        gen = PhaseRandomizationGenerator(marginal="kappa", win_h_length=15)
        gen.fit(sample_daily_series_long)

        # Save
        save_path = tmp_path / "phase_rand_gen.pkl"
        gen.save(str(save_path))

        # Load
        loaded_gen = PhaseRandomizationGenerator.load(str(save_path))

        assert loaded_gen.is_preprocessed is True
        assert loaded_gen.is_fitted is True
        assert loaded_gen.marginal == "kappa"
        assert loaded_gen.win_h_length == 15

    def test_load_and_generate(self, sample_daily_series_long, tmp_path):
        """Test that loaded generator can generate."""
        gen = PhaseRandomizationGenerator()
        gen.fit(sample_daily_series_long)

        # Generate before saving
        original_result = gen.generate(n_realizations=1, seed=42)

        # Save and load
        save_path = tmp_path / "phase_rand_gen.pkl"
        gen.save(str(save_path))
        loaded_gen = PhaseRandomizationGenerator.load(str(save_path))

        # Generate from loaded
        loaded_result = loaded_gen.generate(n_realizations=1, seed=42)

        # Results should have same shape
        assert (
            original_result.data_by_realization[0].shape
            == loaded_result.data_by_realization[0].shape
        )


class TestWindowDays:
    """Tests for window days computation."""

    def test_get_window_days_middle(self, sample_daily_series_long):
        """Test window days for middle of year."""
        gen = PhaseRandomizationGenerator(win_h_length=15)

        window = gen._get_window_days(100)  # Day 100

        # Should include day 100 and days within +/-15
        assert 100 in window
        assert len(window) == 31  # 15 before + 15 after + target

    def test_get_window_days_wrap_around_start(self, sample_daily_series_long):
        """Test window days wraps around at start of year."""
        gen = PhaseRandomizationGenerator(win_h_length=15)

        window = gen._get_window_days(5)  # Day 5

        # Should wrap around to include days from end of previous year
        assert 5 in window
        assert 365 in window or 364 in window  # Should include some Dec days

    def test_get_window_days_wrap_around_end(self, sample_daily_series_long):
        """Test window days wraps around at end of year."""
        gen = PhaseRandomizationGenerator(win_h_length=15)

        window = gen._get_window_days(360)  # Day 360

        # Should wrap around to include days from start of next year
        assert 360 in window
        assert 1 in window or 5 in window  # Should include some Jan days
