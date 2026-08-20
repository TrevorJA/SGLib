"""
Tests for Multi-Site Hidden Markov Model (HMM) Generator.

Based on Gold et al. (2024) methodology for multi-site streamflow generation
using a Gaussian HMM (one multivariate Gaussian emission per state).
"""

import pytest
import pickle
import warnings
import numpy as np
import pandas as pd

from synhydro.methods.generation.parametric.multisite_hmm import MultiSiteHMMGenerator
from synhydro.core.ensemble import Ensemble


@pytest.fixture
def sample_annual_dataframe():
    """Generate sample annual multi-site DataFrame for HMM testing."""
    dates = pd.date_range(start="2000-01-01", end="2029-12-31", freq="YS")
    np.random.seed(42)
    n_sites = 3
    n_years = len(dates)

    # Generate synthetic annual flows with state-like behavior
    # Alternate between dry and wet states
    data = {}
    for i in range(n_sites):
        base_dry = np.random.gamma(shape=2.0, scale=50.0, size=n_years)
        base_wet = np.random.gamma(shape=3.0, scale=100.0, size=n_years)

        # Create state sequence (0=dry, 1=wet)
        states = (np.arange(n_years) % 3 > 0).astype(int)  # Mostly wet with some dry

        # Blend based on states
        flows = np.where(states == 0, base_dry, base_wet)
        data[f"site_{i+1}"] = flows

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_annual_series():
    """Generate sample annual single-site Series for testing."""
    dates = pd.date_range(start="2000-01-01", end="2029-12-31", freq="YS")
    np.random.seed(42)

    base_dry = np.random.gamma(shape=2.0, scale=50.0, size=len(dates))
    base_wet = np.random.gamma(shape=3.0, scale=100.0, size=len(dates))
    states = (np.arange(len(dates)) % 3 > 0).astype(int)
    flows = np.where(states == 0, base_dry, base_wet)

    return pd.Series(flows, index=dates, name="site_1")


class TestMultiSiteHMMInitialization:
    """Tests for MultiSiteHMMGenerator initialization."""

    def test_initialization_default_params(self, sample_annual_dataframe):
        """Test initialization with default parameters."""
        gen = MultiSiteHMMGenerator()

        assert gen.n_states == 2
        assert gen.offset == 1.0
        assert gen.max_iterations == 1000
        assert gen.covariance_type == "full"
        assert gen.is_preprocessed is False
        assert gen.is_fitted is False
        assert gen.debug is False

    def test_initialization_custom_params(self, sample_annual_dataframe):
        """Test initialization with custom parameters."""
        gen = MultiSiteHMMGenerator(
            n_states=3,
            offset=0.5,
            max_iterations=500,
            covariance_type="diag",
            name="test_hmm",
            debug=True,
        )

        assert gen.n_states == 3
        assert gen.offset == 0.5
        assert gen.max_iterations == 500
        assert gen.covariance_type == "diag"
        assert gen.name == "test_hmm"
        assert gen.debug is True

    def test_initialization_invalid_n_states(self, sample_annual_dataframe):
        """Test initialization with invalid n_states."""
        with pytest.raises(ValueError, match="n_states must be >= 2"):
            MultiSiteHMMGenerator(n_states=1)

    def test_initialization_invalid_offset(self, sample_annual_dataframe):
        """Test initialization with invalid offset."""
        with pytest.raises(ValueError, match="offset must be positive"):
            MultiSiteHMMGenerator(offset=-1.0)

        with pytest.raises(ValueError, match="offset must be positive"):
            MultiSiteHMMGenerator(offset=0.0)

    def test_initialization_invalid_covariance_type(self, sample_annual_dataframe):
        """Test initialization with invalid covariance type."""
        with pytest.raises(ValueError, match="covariance_type must be"):
            MultiSiteHMMGenerator(covariance_type="invalid")

    def test_initialization_stores_algorithm_params(self, sample_annual_dataframe):
        """Test that initialization stores algorithm parameters."""
        gen = MultiSiteHMMGenerator(n_states=3, offset=2.0)

        assert "algorithm_params" in gen.init_params.__dict__
        params = gen.init_params.algorithm_params
        assert params["n_states"] == 3
        assert params["offset"] == 2.0
        assert params["method"] == "Multi-Site Hidden Markov Model (Gold et al. 2024)"


class TestMultiSiteHMMPreprocessing:
    """Tests for MultiSiteHMMGenerator preprocessing."""

    def test_preprocessing_multi_site(self, sample_annual_dataframe):
        """Test preprocessing with multi-site DataFrame."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)

        assert gen.is_preprocessed is True
        assert hasattr(gen, "Q_log_")
        assert hasattr(gen, "_Q_obs")
        assert hasattr(gen, "_sites")
        assert len(gen._sites) == 3
        assert gen.Q_log_.shape == (30, 3)

    def test_preprocessing_site_subset(self, sample_annual_dataframe):
        """Test preprocessing with site subset."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe, sites=["site_1", "site_2"])

        assert gen.is_preprocessed is True
        assert len(gen._sites) == 2
        assert gen.Q_log_.shape == (30, 2)

    def test_preprocessing_log_transformation(self, sample_annual_dataframe):
        """Test that log transformation is applied correctly."""
        gen = MultiSiteHMMGenerator(offset=1.0)
        gen.preprocessing(sample_annual_dataframe)

        # Manually compute expected log values
        Q_adj = gen._Q_obs + gen.offset
        Q_log_expected = np.log(Q_adj).values

        np.testing.assert_array_almost_equal(gen.Q_log_, Q_log_expected)

    def test_preprocessing_handles_zeros(self, sample_annual_dataframe):
        """Test preprocessing handles zero values correctly."""
        # Add some zeros
        data = sample_annual_dataframe.copy()
        data.iloc[0, 0] = 0.0
        data.iloc[5, 1] = 0.0

        gen = MultiSiteHMMGenerator(offset=1.0)
        gen.preprocessing(data)

        # Should not have any non-finite values
        assert np.all(np.isfinite(gen.Q_log_))

    def test_preprocessing_invalid_sites(self, sample_annual_dataframe):
        """Test preprocessing with invalid site names."""
        gen = MultiSiteHMMGenerator()

        with pytest.raises(ValueError, match="Sites not found"):
            gen.preprocessing(sample_annual_dataframe, sites=["invalid_site"])

    def test_preprocessing_single_site_warning(self, sample_annual_series):
        """Test preprocessing warns for single-site data."""
        df = sample_annual_series.to_frame()
        gen = MultiSiteHMMGenerator()

        # Should process but warn
        gen.preprocessing(df)
        assert gen.is_preprocessed is True
        assert len(gen._sites) == 1

    def test_preprocessing_negative_flows(self, sample_annual_dataframe):
        """Test preprocessing rejects negative flows (with small offset)."""
        # Add negative value
        data = sample_annual_dataframe.copy()
        data.iloc[0, 0] = -5.0

        gen = MultiSiteHMMGenerator(offset=1.0)

        with pytest.raises(ValueError, match="non-finite values"):
            gen.preprocessing(data)


class TestMultiSiteHMMFitting:
    """Tests for MultiSiteHMMGenerator fitting."""

    def test_fit_basic(self, sample_annual_dataframe):
        """Test basic fitting functionality."""
        gen = MultiSiteHMMGenerator(n_states=2)
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        assert gen.is_fitted is True
        assert hasattr(gen, "means_")
        assert hasattr(gen, "covariances_")
        assert hasattr(gen, "transition_matrix_")
        assert hasattr(gen, "stationary_distribution_")

    def test_fit_parameter_shapes(self, sample_annual_dataframe):
        """Test fitted parameter shapes are correct."""
        n_states = 2
        n_sites = 3

        gen = MultiSiteHMMGenerator(n_states=n_states)
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        assert gen.means_.shape == (n_states, n_sites)
        assert gen.covariances_.shape == (n_states, n_sites, n_sites)
        assert gen.transition_matrix_.shape == (n_states, n_states)
        assert gen.stationary_distribution_.shape == (n_states,)

    def test_fit_state_ordering(self, sample_annual_dataframe):
        """Test that states are ordered by mean (dry to wet)."""
        gen = MultiSiteHMMGenerator(n_states=2)
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        # State 0 should have lower mean than state 1 for first site
        assert gen.means_[0, 0] < gen.means_[1, 0]

    def test_fit_transition_matrix_valid(self, sample_annual_dataframe):
        """Test transition matrix is valid (rows sum to 1)."""
        gen = MultiSiteHMMGenerator(n_states=2)
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        # Each row should sum to 1
        row_sums = gen.transition_matrix_.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(2))

        # All values should be in [0, 1]
        assert np.all(gen.transition_matrix_ >= 0)
        assert np.all(gen.transition_matrix_ <= 1)

    def test_fit_stationary_distribution_valid(self, sample_annual_dataframe):
        """Test stationary distribution is valid (sums to 1)."""
        gen = MultiSiteHMMGenerator(n_states=2)
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        # Should sum to 1
        assert abs(gen.stationary_distribution_.sum() - 1.0) < 1e-10

        # All values should be in [0, 1]
        assert np.all(gen.stationary_distribution_ >= 0)
        assert np.all(gen.stationary_distribution_ <= 1)

    def test_fit_covariance_matrices_psd(self, sample_annual_dataframe):
        """Test covariance matrices are positive semi-definite."""
        gen = MultiSiteHMMGenerator(n_states=2)
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        # Check each state's covariance matrix
        for state in range(gen.n_states):
            cov = gen.covariances_[state]
            # All eigenvalues should be non-negative
            eigenvalues = np.linalg.eigvalsh(cov)
            assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

    def test_fit_creates_fitted_params(self, sample_annual_dataframe):
        """Test that fit creates FittedParams object."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        assert hasattr(gen, "fitted_params_")
        assert gen.fitted_params_.n_parameters_ > 0
        assert gen.fitted_params_.sample_size_ == 30
        assert gen.fitted_params_.n_sites_ == 3

    def test_fit_without_preprocessing(self, sample_annual_dataframe):
        """Test fit raises error without preprocessing."""
        gen = MultiSiteHMMGenerator()

        with pytest.raises(ValueError, match="preprocessing"):
            gen.fit()

    def test_fit_with_q_obs(self, sample_annual_dataframe):
        """Test fit with Q_obs runs preprocessing automatically."""
        gen = MultiSiteHMMGenerator()
        gen.fit(sample_annual_dataframe, random_state=42)

        assert gen.is_preprocessed is True
        assert gen.is_fitted is True

    def test_fit_reproducible_with_seed(self, sample_annual_dataframe):
        """Test fit is reproducible with same random seed."""
        gen1 = MultiSiteHMMGenerator()
        gen1.preprocessing(sample_annual_dataframe)
        gen1.fit(random_state=42)

        gen2 = MultiSiteHMMGenerator()
        gen2.preprocessing(sample_annual_dataframe)
        gen2.fit(random_state=42)

        np.testing.assert_array_almost_equal(gen1.means_, gen2.means_)
        np.testing.assert_array_almost_equal(
            gen1.transition_matrix_, gen2.transition_matrix_
        )

    def test_fit_different_n_states(self, sample_annual_dataframe):
        """Test fitting with different number of states."""
        gen = MultiSiteHMMGenerator(n_states=3)
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        assert gen.means_.shape[0] == 3
        assert gen.transition_matrix_.shape == (3, 3)
        assert len(gen.stationary_distribution_) == 3


class TestMultiSiteHMMGeneration:
    """Tests for MultiSiteHMMGenerator generation."""

    def test_generate_basic(self, sample_annual_dataframe):
        """Test basic generation functionality."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=5, n_years=20, seed=42)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 5

    def test_ensemble_metadata(self, sample_annual_dataframe):
        """Ensemble carries the metadata needed for pipeline chaining."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=2, n_years=10, seed=42)

        assert result.frequency == "YS"
        assert result.metadata.time_resolution == "YS"
        assert result.metadata.generator_class == "MultiSiteHMMGenerator"
        assert result.metadata.n_sites == 3
        assert result.metadata.time_period is not None

    def test_generate_shape(self, sample_annual_dataframe):
        """Test generated data has correct shape."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        n_years = 15
        result = gen.generate(n_realizations=3, n_years=n_years, seed=42)

        for r in range(3):
            df = result.data_by_realization[r]
            assert df.shape == (n_years, 3)  # 3 sites

    def test_generate_n_timesteps(self, sample_annual_dataframe):
        """Test generation with n_timesteps parameter."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=2, n_timesteps=25, seed=42)

        df = result.data_by_realization[0]
        assert len(df) == 25

    def test_generate_non_negative(self, sample_annual_dataframe):
        """Test that generated flows are non-negative."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=10, n_years=20, seed=42)

        for r in range(10):
            df = result.data_by_realization[r]
            assert np.all(df.values >= 0)

    def test_generate_reproducible(self, sample_annual_dataframe):
        """Test generation is reproducible with seed."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        result1 = gen.generate(n_realizations=3, n_years=10, seed=123)
        result2 = gen.generate(n_realizations=3, n_years=10, seed=123)

        for r in range(3):
            df1 = result1.data_by_realization[r]
            df2 = result2.data_by_realization[r]
            pd.testing.assert_frame_equal(df1, df2)

    def test_generate_without_fit(self, sample_annual_dataframe):
        """Test generation raises error without fitting."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)

        with pytest.raises(ValueError, match="fit"):
            gen.generate(n_realizations=1, n_years=10)

    def test_generate_missing_n_years_and_timesteps(self, sample_annual_dataframe):
        """Test generation raises error without n_years or n_timesteps."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        with pytest.raises(ValueError, match="Must provide either"):
            gen.generate(n_realizations=1)

    def test_generate_has_datetime_index(self, sample_annual_dataframe):
        """Test generated data has DatetimeIndex."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=1, n_years=10, seed=42)
        df = result.data_by_realization[0]

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_generate_preserves_site_names(self, sample_annual_dataframe):
        """Test generated data preserves site names."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=1, n_years=10, seed=42)
        df = result.data_by_realization[0]

        assert list(df.columns) == ["site_1", "site_2", "site_3"]


class TestMultiSiteHMMStationary:
    """Tests for stationary distribution computation."""

    def test_stationary_eigenvector_method(self, sample_annual_dataframe):
        """Test stationary distribution via eigenvector method."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        # Verify it's the stationary distribution
        # pi * P = pi
        pi = gen.stationary_distribution_
        P = gen.transition_matrix_

        result = pi @ P
        np.testing.assert_array_almost_equal(result, pi, decimal=5)

    def test_stationary_distribution_uniqueness(self, sample_annual_dataframe):
        """Test stationary distribution is consistent."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        # Recompute manually
        pi = gen._compute_stationary_distribution()

        np.testing.assert_array_almost_equal(
            pi, gen.stationary_distribution_, decimal=10
        )


class TestMultiSiteHMMStateTrajectory:
    """Tests for state trajectory generation."""

    def test_state_trajectory_length(self, sample_annual_dataframe):
        """Test state trajectory has correct length."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        rng = np.random.default_rng(42)
        states = gen._generate_state_trajectory(50, rng=rng)

        assert len(states) == 50

    def test_state_trajectory_valid_states(self, sample_annual_dataframe):
        """Test state trajectory contains only valid state indices."""
        gen = MultiSiteHMMGenerator(n_states=3)
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        rng = np.random.default_rng(42)
        states = gen._generate_state_trajectory(100, rng=rng)

        assert all(s in [0, 1, 2] for s in states)

    def test_state_trajectory_uses_transition_matrix(self, sample_annual_dataframe):
        """Test state trajectory respects transition probabilities."""
        gen = MultiSiteHMMGenerator(n_states=2)
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        # Generate many trajectories
        rng = np.random.default_rng(42)
        n_trajectories = 1000
        n_steps = 100

        transitions = np.zeros((2, 2))

        for _ in range(n_trajectories):
            states = gen._generate_state_trajectory(n_steps, rng=rng)
            for i in range(len(states) - 1):
                transitions[states[i], states[i + 1]] += 1

        # Normalize to probabilities
        transition_probs = transitions / transitions.sum(axis=1, keepdims=True)

        # Should be close to fitted transition matrix
        np.testing.assert_array_almost_equal(
            transition_probs,
            gen.transition_matrix_,
            decimal=1,  # Allow some sampling variation
        )


class TestMultiSiteHMMSerialization:
    """Tests for saving and loading MultiSiteHMMGenerator."""

    def test_pickle_save_load(self, sample_annual_dataframe, tmp_path):
        """Test saving and loading via pickle."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        # Save
        filepath = tmp_path / "hmm_generator.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(gen, f)

        # Load
        with open(filepath, "rb") as f:
            gen_loaded = pickle.load(f)

        # Verify attributes preserved
        assert gen_loaded.is_fitted is True
        assert gen_loaded.is_preprocessed is True
        np.testing.assert_array_almost_equal(gen.means_, gen_loaded.means_)
        np.testing.assert_array_almost_equal(
            gen.transition_matrix_, gen_loaded.transition_matrix_
        )

    def test_pickle_generate_after_load(self, sample_annual_dataframe, tmp_path):
        """Test generation works after loading from pickle."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        # Save and load
        filepath = tmp_path / "hmm_generator.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(gen, f)

        with open(filepath, "rb") as f:
            gen_loaded = pickle.load(f)

        # Generate from loaded generator
        result = gen_loaded.generate(n_realizations=2, n_years=10, seed=99)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 2


class TestMultiSiteHMMStatisticalProperties:
    """Tests for statistical properties of generated data."""

    def test_generated_mean_reasonable(self, sample_annual_dataframe):
        """Test generated data has reasonable mean."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=100, n_years=30, seed=42)

        # Compute ensemble mean
        all_data = []
        for r in range(100):
            all_data.append(result.data_by_realization[r].values)
        all_data = np.concatenate(all_data, axis=0)

        gen_mean = all_data.mean(axis=0)
        obs_mean = sample_annual_dataframe.mean(axis=0).values

        # Generated mean should be within reasonable range of observed
        # (not exact due to log transformation and limited sample)
        ratio = gen_mean / obs_mean
        assert np.all((ratio > 0.5) & (ratio < 2.0))

    def test_generated_variance_reasonable(self, sample_annual_dataframe):
        """Test generated data has reasonable variance."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        result = gen.generate(n_realizations=100, n_years=30, seed=42)

        # Compute ensemble variance
        all_data = []
        for r in range(100):
            all_data.append(result.data_by_realization[r].values)
        all_data = np.concatenate(all_data, axis=0)

        gen_std = all_data.std(axis=0)
        obs_std = sample_annual_dataframe.std(axis=0).values

        # Generated std should be in reasonable range
        ratio = gen_std / obs_std
        assert np.all((ratio > 0.3) & (ratio < 3.0))

    def test_spatial_correlation_preserved(self, sample_annual_dataframe):
        """Test spatial correlations are approximately preserved."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)
        gen.fit(random_state=42)

        # Generate large ensemble
        result = gen.generate(n_realizations=200, n_years=30, seed=42)

        # Compute observed correlation
        obs_corr = sample_annual_dataframe.corr().values

        # Compute generated correlation
        all_data = []
        for r in range(200):
            all_data.append(result.data_by_realization[r])
        all_data = pd.concat(all_data, axis=0)
        gen_corr = all_data.corr().values

        # Correlations should be in same ballpark
        # (not exact due to limited sample and log transformation)
        diff = np.abs(obs_corr - gen_corr)
        assert np.all(diff < 0.5)  # Allow substantial difference due to small sample


class TestMultiSiteHMMOutputFrequency:
    """Tests for output frequency property."""

    def test_output_frequency_annual(self, sample_annual_dataframe):
        """Test output frequency matches input (annual)."""
        gen = MultiSiteHMMGenerator()
        gen.preprocessing(sample_annual_dataframe)

        freq = gen.output_frequency
        # Anchored annual aliases (YS-JAN, AS-JAN, ...) are normalized to "YS"
        # so the generator can be chained into an annual disaggregator.
        assert freq == "YS"

    def test_output_frequency_before_preprocessing(self, sample_annual_dataframe):
        """Test output frequency returns default before preprocessing."""
        gen = MultiSiteHMMGenerator()

        freq = gen.output_frequency
        assert freq == "YS"  # Default


def _simulate_hmm(n_years, A, mu, Sigma, seed):
    """Simulate log-space flows from a known Gaussian HMM and return a DataFrame."""
    rng = np.random.default_rng(seed)
    n_states = A.shape[0]
    evals, evecs = np.linalg.eig(A.T)
    pi = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))])
    pi = pi / pi.sum()
    states = np.empty(n_years, dtype=int)
    states[0] = rng.choice(n_states, p=pi)
    for t in range(1, n_years):
        states[t] = rng.choice(n_states, p=A[states[t - 1]])
    Y = np.array([rng.multivariate_normal(mu[k], Sigma[k]) for k in states])
    dates = pd.date_range("1700-01-01", periods=n_years, freq="YS")
    # offset=1.0 in the generator, so Q = exp(Y) - 1 makes log(Q + 1) == Y
    return pd.DataFrame(np.exp(Y) - 1.0, index=dates, columns=["a", "b", "c"]), Y


class TestMultiSiteHMMConvergence:
    """Tests for EM convergence reporting and multi-restart fitting."""

    def test_non_convergence_warning(self, sample_annual_dataframe):
        """A single EM iteration cannot converge and must emit a warning."""
        gen = MultiSiteHMMGenerator(max_iterations=1)
        gen.preprocessing(sample_annual_dataframe)
        with pytest.warns(UserWarning, match="did not converge"):
            gen.fit(random_state=42)
        assert gen.converged_ is False

    def test_converged_fit_no_warning(self, sample_annual_dataframe):
        """A fully converged single fit emits no warnings."""
        gen = MultiSiteHMMGenerator(max_iterations=1000)
        gen.preprocessing(sample_annual_dataframe)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gen.fit(random_state=42)
        assert gen.converged_ is True
        assert np.isfinite(gen.log_likelihood_)
        assert len(gen.log_likelihoods_) == 1

    def test_n_init_keeps_best_log_likelihood(self, sample_annual_dataframe):
        """With restarts, the retained fit has the maximum log-likelihood."""
        gen = MultiSiteHMMGenerator(n_init=8)
        gen.preprocessing(sample_annual_dataframe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # spread warning may fire
            gen.fit(random_state=0)
        assert len(gen.log_likelihoods_) == 8
        assert gen.log_likelihood_ == np.nanmax(gen.log_likelihoods_)
        assert np.isclose(gen._hmm_model.score(gen.Q_log_), gen.log_likelihood_)

    def test_n_init_beats_or_matches_single_fit(self, sample_annual_dataframe):
        """Best-of-n restarts is never worse than any single restart seed."""
        gen_multi = MultiSiteHMMGenerator(n_init=5)
        gen_multi.preprocessing(sample_annual_dataframe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gen_multi.fit(random_state=7)
        assert gen_multi.log_likelihood_ >= max(gen_multi.log_likelihoods_) - 1e-9

    def test_spread_warning_fires_on_disagreeing_restarts(
        self, sample_annual_dataframe, monkeypatch
    ):
        """If restart log-likelihoods differ by more than the tolerance, warn."""
        gen = MultiSiteHMMGenerator(n_init=3)
        gen.preprocessing(sample_annual_dataframe)
        # Force the tolerance to zero so any disagreement triggers the warning
        monkeypatch.setattr(MultiSiteHMMGenerator, "_LOGLIK_SPREAD_TOL", -1.0)
        with pytest.warns(UserWarning, match="local optima"):
            gen.fit(random_state=3)

    def test_fit_reproducible_with_restarts(self, sample_annual_dataframe):
        """Same random_state and n_init gives identical fits."""
        fits = []
        for _ in range(2):
            gen = MultiSiteHMMGenerator(n_init=4)
            gen.preprocessing(sample_annual_dataframe)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gen.fit(random_state=123)
            fits.append(gen)
        np.testing.assert_array_equal(fits[0].means_, fits[1].means_)
        np.testing.assert_array_equal(fits[0].covariances_, fits[1].covariances_)
        np.testing.assert_array_equal(
            fits[0].transition_matrix_, fits[1].transition_matrix_
        )
        assert fits[0].log_likelihoods_ == fits[1].log_likelihoods_

    def test_invalid_n_init(self):
        """n_init must be at least 1."""
        with pytest.raises(ValueError, match="n_init"):
            MultiSiteHMMGenerator(n_init=0)


class TestMultiSiteHMMParameterCount:
    """n_parameters_ must reflect covariance_type."""

    @pytest.mark.parametrize(
        "cov_type, n_cov",
        [("full", 2 * 6), ("diag", 2 * 3), ("spherical", 2), ("tied", 6)],
    )
    def test_n_parameters_by_covariance_type(
        self, sample_annual_dataframe, cov_type, n_cov
    ):
        gen = MultiSiteHMMGenerator(n_states=2, covariance_type=cov_type)
        gen.preprocessing(sample_annual_dataframe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gen.fit(random_state=42)
        expected = 2 * 3 + n_cov + 2 * 1
        assert gen.fitted_params_.n_parameters_ == expected
        assert gen.covariances_.shape == (2, 3, 3)


class TestMultiSiteHMMParameterRecovery:
    """Recover known parameters from a simulated 2-state, 3-site HMM."""

    A_TRUE = np.array([[0.8, 0.2], [0.3, 0.7]])
    MU_TRUE = np.array([[4.0, 4.5, 3.5], [5.5, 6.0, 5.0]])
    SIGMA_TRUE = np.array(
        [
            [[0.10, 0.07, 0.05], [0.07, 0.12, 0.06], [0.05, 0.06, 0.09]],
            [[0.08, 0.05, 0.03], [0.05, 0.10, 0.04], [0.03, 0.04, 0.07]],
        ]
    )

    def test_recovers_known_parameters(self):
        n_years = 300
        Q, Y_true = _simulate_hmm(
            n_years, self.A_TRUE, self.MU_TRUE, self.SIGMA_TRUE, seed=2024
        )

        gen = MultiSiteHMMGenerator(n_states=2, n_init=5)
        gen.preprocessing(Q)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gen.fit(random_state=1)

        # States are ordered dry -> wet, matching the truth ordering
        np.testing.assert_allclose(gen.means_, self.MU_TRUE, atol=0.1)
        np.testing.assert_allclose(gen.transition_matrix_, self.A_TRUE, atol=0.1)
        assert gen.converged_

        # Cross-site correlation of generated flows matches the simulated truth
        ens = gen.generate(n_realizations=50, n_years=n_years, seed=5)
        Q_syn = pd.concat(
            [ens.data_by_realization[r] for r in range(50)], axis=0
        )
        true_corr = np.corrcoef(np.exp(Y_true).T - 1.0)
        syn_corr = Q_syn.corr().values
        assert np.all(np.abs(true_corr - syn_corr) < 0.1)
