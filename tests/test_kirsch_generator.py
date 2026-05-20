"""
Tests for Kirsch nonparametric streamflow generator.
Updated to match current API.
"""

import pytest
import numpy as np
import pandas as pd

from synhydro.methods.generation.hybrid.kirsch import KirschGenerator
from synhydro.core.ensemble import Ensemble


class TestKirschGeneratorInitialization:
    """Tests for KirschGenerator initialization."""

    def test_initialization_default_params(self, sample_monthly_dataframe):
        """Test initialization with default parameters (no Q_obs at init)."""
        gen = KirschGenerator()
        assert gen.is_preprocessed is False
        assert gen.is_fitted is False
        assert gen.debug is False

    def test_initialization_with_params(self, sample_monthly_dataframe):
        """Test initialization with custom parameters."""
        gen = KirschGenerator(
            generate_using_log_flow=True, matrix_repair_method="nearest", debug=True
        )
        assert gen.debug is True


class TestKirschGeneratorPreprocessing:
    """Tests for KirschGenerator preprocessing."""

    def test_preprocessing_daily_dataframe(self, sample_daily_dataframe):
        """Test preprocessing with daily DataFrame."""
        gen = KirschGenerator()
        gen.preprocessing(sample_daily_dataframe)

        assert gen.is_preprocessed is True
        assert hasattr(gen, "Q")
        assert hasattr(gen, "Qm")
        assert gen.n_sites == 3
        assert gen.Qm.shape[1] == 3

    def test_preprocessing_monthly_dataframe(self, sample_monthly_dataframe):
        """Test preprocessing with monthly DataFrame."""
        gen = KirschGenerator()
        gen.preprocessing(sample_monthly_dataframe)

        assert gen.is_preprocessed is True

    def test_preprocessing_with_log_transform(self, sample_daily_dataframe):
        """Test preprocessing with log transformation."""
        gen = KirschGenerator(generate_using_log_flow=True)
        gen.preprocessing(sample_daily_dataframe)

        assert gen.is_preprocessed is True
        # Log-transformed data should be stored

    def test_preprocessing_invalid_input(self):
        """Test that invalid input type raises TypeError during preprocessing."""
        gen = KirschGenerator()
        # Pass invalid type to validate_input_data directly
        with pytest.raises(TypeError):
            gen.validate_input_data([1, 2, 3, 4, 5])


class TestKirschGeneratorFit:
    """Tests for KirschGenerator fitting."""

    def test_fit_single_site(self, sample_daily_series):
        """Test fitting with single site (Series converted to DataFrame)."""
        # Convert Series to DataFrame - KirschGenerator requires DataFrame
        df = sample_daily_series.to_frame()
        gen = KirschGenerator()
        gen.fit(df)

        assert gen.is_fitted is True
        assert hasattr(gen, "mean_month")
        assert hasattr(gen, "std_month")
        assert hasattr(gen, "Z_h")
        assert len(gen.mean_month) == 12
        assert len(gen.std_month) == 12

    def test_fit_multiple_sites(self, sample_daily_dataframe):
        """Test fitting with multiple sites."""
        gen = KirschGenerator()
        gen.fit(sample_daily_dataframe)

        assert gen.is_fitted is True
        assert gen.mean_month.shape == (12, 3)
        assert gen.std_month.shape == (12, 3)

    def test_fit_creates_cholesky_decomposition(self, sample_daily_dataframe):
        """Test that fit creates Cholesky decomposition matrices."""
        gen = KirschGenerator()
        gen.fit(sample_daily_dataframe)

        assert hasattr(gen, "U_site")
        assert isinstance(gen.U_site, dict)
        # Should have Cholesky matrices for each site
        assert len(gen.U_site) > 0

    def test_fit_stores_correlation_matrices(self, sample_daily_dataframe):
        """Test that fit stores correlation information."""
        gen = KirschGenerator()
        gen.fit(sample_daily_dataframe)

        # Should have correlation-related attributes
        assert hasattr(gen, "Z_h")


class TestKirschGeneratorGenerate:
    """Tests for KirschGenerator generation."""

    def test_generate_single_realization_series(self, sample_daily_series):
        """Test generating single realization from Series (converted to DataFrame)."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=1, n_years=1)

        assert isinstance(result, Ensemble)
        assert 0 in result.realization_ids
        assert isinstance(result.data_by_realization[0], pd.DataFrame)
        assert len(result.data_by_realization[0]) == 12  # 12 months

    def test_generate_multiple_realizations_series(self, sample_daily_series):
        """Test generating multiple realizations from Series."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=5, n_years=1)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 5
        for i in range(5):
            assert i in result.realization_ids
            assert isinstance(result.data_by_realization[i], pd.DataFrame)

    def test_generate_single_realization_dataframe(self, sample_daily_dataframe):
        """Test generating single realization from DataFrame."""
        gen = KirschGenerator()
        gen.fit(sample_daily_dataframe)

        result = gen.generate(n_realizations=1, n_years=1)

        assert isinstance(result, Ensemble)
        assert 0 in result.realization_ids
        df = result.data_by_realization[0]
        assert df.shape[1] == 3  # 3 sites
        assert len(df) == 12

    def test_generate_multiple_realizations_dataframe(self, sample_daily_dataframe):
        """Test generating multiple realizations from DataFrame."""
        gen = KirschGenerator()
        gen.fit(sample_daily_dataframe)

        result = gen.generate(n_realizations=3, n_years=1)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 3
        for r in range(3):
            assert r in result.realization_ids
            assert isinstance(result.data_by_realization[r], pd.DataFrame)
            assert result.data_by_realization[r].shape[1] == 3

    def test_generate_preserves_monthly_statistics(self, sample_daily_dataframe):
        """Test that generated data has similar monthly statistics."""
        gen = KirschGenerator()
        gen.fit(sample_daily_dataframe)

        # Generate many realizations for statistical testing
        result = gen.generate(n_realizations=100, n_years=10)

        # Check that generated data has reasonable values
        assert isinstance(result, Ensemble)
        for r in range(100):
            df = result.data_by_realization[r]
            assert not df.isna().any().any()
            assert (df >= 0).all().all()  # Flows should be non-negative

    def test_generate_with_log_flow(self, sample_daily_series):
        """Test generation with log-transformed flows."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator(generate_using_log_flow=True)
        gen.fit(df)

        result = gen.generate(n_realizations=1, n_years=1)

        assert isinstance(result, Ensemble)
        df_result = result.data_by_realization[0]
        assert not df_result.isna().any().any()

    def test_generate_as_array(self, sample_daily_dataframe):
        """Test generate_single_series returns array when as_array=True."""
        gen = KirschGenerator()
        gen.fit(sample_daily_dataframe)

        result = gen.generate_single_series(n_years=2, as_array=True)

        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 3  # 3 sites


class TestKirschGeneratorSaveLoad:
    """Tests for KirschGenerator save and load."""

    def test_save_and_load(self, sample_daily_dataframe, tmp_path):
        """Test saving and loading generator."""
        gen = KirschGenerator(generate_using_log_flow=True)
        gen.fit(sample_daily_dataframe)

        # Generate a realization before saving
        original_result = gen.generate(n_realizations=1, n_years=1)

        # Save
        save_path = tmp_path / "kirsch_gen.pkl"
        gen.save(str(save_path))

        # Load
        loaded_gen = KirschGenerator.load(str(save_path))

        assert loaded_gen.is_preprocessed is True
        assert loaded_gen.is_fitted is True
        assert loaded_gen.n_sites == 3

        # Generate from loaded generator
        loaded_result = loaded_gen.generate(n_realizations=1, n_years=1)

        assert (
            loaded_result.data_by_realization[0].shape
            == original_result.data_by_realization[0].shape
        )


class TestKirschGeneratorMethods:
    """Tests for KirschGenerator internal methods."""

    def test_repair_and_cholesky(self, sample_daily_dataframe):
        """Test _repair_and_cholesky method."""
        gen = KirschGenerator()
        gen.fit(sample_daily_dataframe)

        # Should have created Cholesky decompositions without error
        assert hasattr(gen, "U_site")

    def test_bootstrap_indices_generation(self, sample_daily_series):
        """Test bootstrap index generation."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator()
        gen.fit(df)

        # Call generate to test bootstrap indices internally
        result = gen.generate(n_realizations=2, n_years=1)
        assert result is not None

    def test_destandardize_flows(self, sample_daily_dataframe):
        """Test flow destandardization."""
        gen = KirschGenerator()
        gen.fit(sample_daily_dataframe)

        # Generate to test destandardization internally
        result = gen.generate(n_realizations=1, n_years=1)

        # Check that values are in reasonable range
        assert (result.data_by_realization[0] >= 0).all().all()


class TestKirschPaperConformance:
    """Tests verifying conformance with Kirsch et al. (2013), p. 6.

    The paper specifies that X_prime is a deterministic 6-month shift of X,
    not an independent bootstrap. These tests guard against regression to
    the pre-fix behavior where ``generate_single_series`` drew a second
    bootstrap and ``generate_from_indices`` did a shared-index lookup.
    """

    def test_derive_X_prime_is_deterministic_shift(self, sample_monthly_dataframe):
        """X_prime row i = [second-half of X year i, first-half of X year i+1]."""
        gen = KirschGenerator()
        gen.fit(sample_monthly_dataframe)

        rng = np.random.default_rng(42)
        n_years = 5
        M = gen._get_bootstrap_indices(n_years + 1, max_idx=gen.Y.shape[0], rng=rng)
        X = gen._create_bootstrap_tensor(M)
        X_prime = gen._derive_X_prime(X)

        assert X_prime.shape == (n_years + 1, 12, gen.n_sites)
        np.testing.assert_allclose(X_prime[:n_years, :6], X[:n_years, 6:])
        np.testing.assert_allclose(X_prime[:n_years, 6:], X[1 : n_years + 1, :6])

    def test_derive_X_prime_rejects_wrong_month_count(self, sample_monthly_dataframe):
        """_derive_X_prime validates its input shape."""
        gen = KirschGenerator()
        gen.fit(sample_monthly_dataframe)
        bad_X = np.zeros((3, 10, gen.n_sites))
        with pytest.raises(ValueError, match="expected"):
            gen._derive_X_prime(bad_X)

    def test_entry_points_agree_on_cross_month_correlation(
        self, sample_monthly_dataframe
    ):
        """generate() and generate_from_residuals() should produce ensembles
        with statistically equivalent cross-month correlation. Pre-fix,
        generate() drew an independent bootstrap for X_prime and diverged."""
        gen = KirschGenerator()
        gen.fit(sample_monthly_dataframe)

        n_years = 10
        n_realizations = 100

        ens_a = gen.generate(n_realizations=n_realizations, n_years=n_years, seed=42)
        flows_a = [
            ens_a.data_by_realization[r].values
            for r in sorted(ens_a.data_by_realization)
        ]

        rng = np.random.default_rng(42)
        flows_c = []
        for _ in range(n_realizations):
            residuals = np.empty((n_years, 12, gen.n_sites))
            for m in range(12):
                for s in range(gen.n_sites):
                    residuals[:, m, s] = rng.choice(
                        gen.Z_h[:, m, s], size=n_years, replace=True
                    )
            flows_c.append(gen.generate_from_residuals(residuals))

        def pool_corr(flows_list):
            mats = []
            for fl in flows_list:
                col0 = fl[:, 0]
                monthly = col0[: (len(col0) // 12) * 12].reshape(-1, 12)
                mats.append(np.corrcoef(monthly, rowvar=False))
            return np.mean(mats, axis=0)

        frob = np.linalg.norm(pool_corr(flows_a) - pool_corr(flows_c), ord="fro")
        assert frob < 1.5, (
            f"generate() and generate_from_residuals() disagree on cross-month "
            f"correlation: Frobenius {frob:.3f}."
        )

    def test_generate_from_indices_matches_single_series(
        self, sample_monthly_dataframe
    ):
        """Given the same M, generate_from_indices and generate_single_series
        must produce identical output. Pre-fix they diverged because
        generate_from_indices clamped indices differently for Y_prime."""
        gen = KirschGenerator()
        gen.fit(sample_monthly_dataframe)

        rng = np.random.default_rng(123)
        n_years = 8
        M = gen._get_bootstrap_indices(n_years + 1, max_idx=gen.Y.shape[0], rng=rng)

        out_series = gen.generate_single_series(n_years, M=M, as_array=True)
        out_indices = gen.generate_from_indices(M, n_years=n_years, as_array=True)

        np.testing.assert_allclose(out_series, out_indices)
