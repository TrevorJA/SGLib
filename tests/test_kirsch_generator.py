"""
Tests for Kirsch hybrid bootstrap streamflow generator.

Core round-trip, fit, generation, and paper-conformance tests are parametrized
across monthly and weekly fixtures. Tests that depend on daily input remain
monthly-only because daily-to-weekly aggregation is opt-in (requires
``timestep='weekly'``).
"""

import pytest
import numpy as np
import pandas as pd

from synhydro.methods.generation.hybrid.kirsch import KirschGenerator
from synhydro.core.ensemble import Ensemble


AGGREGATED_FIXTURES = ("sample_monthly_dataframe", "sample_weekly_dataframe")


class TestKirschGeneratorInitialization:
    """Tests for KirschGenerator initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters (no Q_obs at init)."""
        gen = KirschGenerator()
        assert gen.is_preprocessed is False
        assert gen.is_fitted is False
        assert gen.debug is False

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        gen = KirschGenerator(
            generate_using_log_flow=True, matrix_repair_method="nearest", debug=True
        )
        assert gen.debug is True


class TestKirschGeneratorPreprocessing:
    """Tests for KirschGenerator preprocessing."""

    def test_preprocessing_daily_dataframe(self, sample_daily_dataframe):
        """Daily input aggregates to monthly by default."""
        gen = KirschGenerator()
        gen.preprocessing(sample_daily_dataframe)

        assert gen.is_preprocessed is True
        assert hasattr(gen, "Q")
        assert hasattr(gen, "Qm")
        assert gen.n_sites == 3
        assert gen.Qm.shape[1] == 3
        assert gen.output_frequency == "MS"
        assert gen.n_periods_per_year == 12

    def test_preprocessing_daily_with_weekly_timestep(self, sample_daily_dataframe):
        """Daily input aggregates to weekly when timestep='weekly'."""
        gen = KirschGenerator()
        gen.preprocessing(sample_daily_dataframe, timestep="weekly")

        assert gen.is_preprocessed is True
        assert gen.output_frequency == "W-SUN"
        assert gen.n_periods_per_year == 52

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_preprocessing_aggregated_dataframe(self, fixture_name, request):
        """Pre-aggregated monthly/weekly input is auto-detected."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.preprocessing(df)

        assert gen.is_preprocessed is True
        if fixture_name == "sample_monthly_dataframe":
            assert gen.output_frequency == "MS"
            assert gen.n_periods_per_year == 12
        else:
            assert gen.output_frequency == "W-SUN"
            assert gen.n_periods_per_year == 52

    def test_preprocessing_contradictory_timestep_raises(
        self, sample_monthly_dataframe
    ):
        """Asking for weekly on monthly input must raise."""
        gen = KirschGenerator()
        with pytest.raises(ValueError, match="weekly"):
            gen.preprocessing(sample_monthly_dataframe, timestep="weekly")

    def test_preprocessing_with_log_transform(self, sample_daily_dataframe):
        """Log transformation is applied without error."""
        gen = KirschGenerator(generate_using_log_flow=True)
        gen.preprocessing(sample_daily_dataframe)

        assert gen.is_preprocessed is True

    def test_preprocessing_invalid_input(self):
        """Invalid input type raises TypeError during validation."""
        gen = KirschGenerator()
        with pytest.raises(TypeError):
            gen.validate_input_data([1, 2, 3, 4, 5])


class TestKirschGeneratorFit:
    """Tests for KirschGenerator fitting."""

    def test_fit_single_site(self, sample_daily_series):
        """Fit on a single-site DataFrame derived from a Series."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator()
        gen.fit(df)

        assert gen.is_fitted is True
        assert hasattr(gen, "mean_period")
        assert hasattr(gen, "std_period")
        assert hasattr(gen, "Z_h")
        assert len(gen.mean_period) == 12
        assert len(gen.std_period) == 12

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_fit_multiple_sites(self, fixture_name, request):
        """Fit produces per-period mean/std with the right shape."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        assert gen.is_fitted is True
        n_per = gen.n_periods_per_year
        assert gen.mean_period.shape == (n_per, 3)
        assert gen.std_period.shape == (n_per, 3)

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_fit_creates_cholesky_decomposition(self, fixture_name, request):
        """Fit populates per-site Cholesky factors."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        assert hasattr(gen, "U_site")
        assert isinstance(gen.U_site, dict)
        assert len(gen.U_site) == gen.n_sites
        for s in range(gen.n_sites):
            assert gen.U_site[s].shape == (
                gen.n_periods_per_year,
                gen.n_periods_per_year,
            )

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_fit_stores_correlation_matrices(self, fixture_name, request):
        """Fit stores Z_h with the expected (n_years, n_per, n_sites) shape."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        assert hasattr(gen, "Z_h")
        assert gen.Z_h.shape[1] == gen.n_periods_per_year
        assert gen.Z_h.shape[2] == gen.n_sites


class TestKirschGeneratorGenerate:
    """Tests for KirschGenerator generation."""

    def test_generate_single_realization_series(self, sample_daily_series):
        """Single-realization round-trip on monthly-from-daily input."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=1, n_years=1)

        assert isinstance(result, Ensemble)
        assert 0 in result.realization_ids
        assert isinstance(result.data_by_realization[0], pd.DataFrame)
        assert len(result.data_by_realization[0]) == gen.n_periods_per_year

    def test_generate_multiple_realizations_series(self, sample_daily_series):
        """Multiple realizations on monthly-from-daily input."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=5, n_years=1)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 5
        for i in range(5):
            assert i in result.realization_ids
            assert isinstance(result.data_by_realization[i], pd.DataFrame)

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_generate_single_realization_dataframe(self, fixture_name, request):
        """Single realization has shape (n_periods_per_year, n_sites)."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=1, n_years=1)

        assert isinstance(result, Ensemble)
        assert 0 in result.realization_ids
        out = result.data_by_realization[0]
        assert out.shape[1] == 3
        assert len(out) == gen.n_periods_per_year

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_generate_multiple_realizations_dataframe(self, fixture_name, request):
        """Multiple realizations each have n_sites columns and the right length."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=3, n_years=1)

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 3
        for r in range(3):
            assert r in result.realization_ids
            out = result.data_by_realization[r]
            assert out.shape[1] == 3
            assert len(out) == gen.n_periods_per_year

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_generate_preserves_period_statistics(self, fixture_name, request):
        """Generated flows are finite and non-negative."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=20, n_years=5, seed=0)

        assert isinstance(result, Ensemble)
        for r in range(20):
            d = result.data_by_realization[r]
            assert not d.isna().any().any()
            assert (d >= 0).all().all()

    def test_generate_with_log_flow(self, sample_daily_series):
        """Log-flow generation produces finite output."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator(generate_using_log_flow=True)
        gen.fit(df)

        result = gen.generate(n_realizations=1, n_years=1)

        assert isinstance(result, Ensemble)
        d = result.data_by_realization[0]
        assert not d.isna().any().any()

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_generate_as_array(self, fixture_name, request):
        """generate_single_series returns an array of shape (n_per*n_years, n_sites)."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate_single_series(n_years=2, as_array=True)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2 * gen.n_periods_per_year, 3)

    def test_generate_default_index_anchors_after_record(
        self, sample_monthly_dataframe
    ):
        """Default output index starts January 1 of the year after the record."""
        gen = KirschGenerator()
        gen.fit(sample_monthly_dataframe)

        result = gen.generate(n_realizations=1, n_years=2, seed=0)

        idx = result.data_by_realization[0].index
        assert idx[0] == pd.Timestamp("2021-01-01")
        assert len(idx) == 24

    def test_generate_start_year_anchors_index(self, sample_monthly_dataframe):
        """start_year anchors the output index at January 1 of that year, with
        month labels running Jan..Dec within each synthetic year."""
        gen = KirschGenerator()
        gen.fit(sample_monthly_dataframe)

        result = gen.generate(n_realizations=2, n_years=3, seed=0, start_year=1945)

        for r in range(2):
            idx = result.data_by_realization[r].index
            assert idx[0] == pd.Timestamp("1945-01-01")
            assert list(idx.month) == list(range(1, 13)) * 3
            assert list(np.unique(idx.year)) == [1945, 1946, 1947]

    def test_generate_start_year_relabels_only(self, sample_monthly_dataframe):
        """start_year changes labels, never values (same seed, same content)."""
        gen = KirschGenerator()
        gen.fit(sample_monthly_dataframe)

        default = gen.generate(n_realizations=1, n_years=2, seed=7)
        anchored = gen.generate(n_realizations=1, n_years=2, seed=7, start_year=1945)

        np.testing.assert_array_equal(
            default.data_by_realization[0].to_numpy(),
            anchored.data_by_realization[0].to_numpy(),
        )


class TestKirschGeneratorSaveLoad:
    """Tests for KirschGenerator save and load."""

    def test_save_and_load(self, sample_daily_dataframe, tmp_path):
        """Save then load reproduces the same output shape."""
        gen = KirschGenerator(generate_using_log_flow=True)
        gen.fit(sample_daily_dataframe)

        original_result = gen.generate(n_realizations=1, n_years=1)

        save_path = tmp_path / "kirsch_gen.pkl"
        gen.save(str(save_path))

        loaded_gen = KirschGenerator.load(str(save_path))

        assert loaded_gen.is_preprocessed is True
        assert loaded_gen.is_fitted is True
        assert loaded_gen.n_sites == 3

        loaded_result = loaded_gen.generate(n_realizations=1, n_years=1)

        assert (
            loaded_result.data_by_realization[0].shape
            == original_result.data_by_realization[0].shape
        )


class TestKirschGeneratorMethods:
    """Tests for KirschGenerator internal methods."""

    def test_repair_and_cholesky(self, sample_daily_dataframe):
        """_repair_and_cholesky is exercised by fit without error."""
        gen = KirschGenerator()
        gen.fit(sample_daily_dataframe)

        assert hasattr(gen, "U_site")

    def test_bootstrap_indices_generation(self, sample_daily_series):
        """Bootstrap index path produces output."""
        df = sample_daily_series.to_frame()
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=2, n_years=1)
        assert result is not None

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_destandardize_flows(self, fixture_name, request):
        """Destandardization keeps output non-negative."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        result = gen.generate(n_realizations=1, n_years=1)

        assert (result.data_by_realization[0] >= 0).all().all()


class TestKirschPaperConformance:
    """Tests verifying conformance with Kirsch et al. (2013), p. 6.

    The paper specifies that X_prime is a deterministic half-year shift of X,
    not an independent bootstrap. These tests guard against regression to
    the pre-fix behavior where ``generate_single_series`` drew a second
    bootstrap and ``generate_from_indices`` did a shared-index lookup.
    """

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_derive_X_prime_is_deterministic_shift(self, fixture_name, request):
        """X_prime row i = [second-half of X year i, first-half of X year i+1]."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        rng = np.random.default_rng(42)
        n_years = 5
        n_per = gen.n_periods_per_year
        half = n_per // 2
        M = gen._get_bootstrap_indices(n_years + 1, max_idx=gen.Y.shape[0], rng=rng)
        X = gen._create_bootstrap_tensor(M)
        X_prime = gen._derive_X_prime(X)

        assert X_prime.shape == (n_years + 1, n_per, gen.n_sites)
        np.testing.assert_allclose(X_prime[:n_years, :half], X[:n_years, half:])
        np.testing.assert_allclose(X_prime[:n_years, half:], X[1 : n_years + 1, :half])

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_derive_X_prime_rejects_wrong_period_count(self, fixture_name, request):
        """_derive_X_prime validates its input shape."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)
        bad_X = np.zeros((3, gen.n_periods_per_year - 2, gen.n_sites))
        with pytest.raises(ValueError, match="expected"):
            gen._derive_X_prime(bad_X)

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_entry_points_agree_on_cross_period_correlation(
        self, fixture_name, request
    ):
        """generate() and generate_from_residuals() must agree on cross-period
        correlation. Pre-fix, generate() drew an independent bootstrap for
        X_prime and diverged."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        n_years = 10
        n_realizations = 50
        n_per = gen.n_periods_per_year

        ens_a = gen.generate(n_realizations=n_realizations, n_years=n_years, seed=42)
        flows_a = [
            ens_a.data_by_realization[r].values
            for r in sorted(ens_a.data_by_realization)
        ]

        rng = np.random.default_rng(42)
        flows_c = []
        for _ in range(n_realizations):
            residuals = np.empty((n_years, n_per, gen.n_sites))
            for m in range(n_per):
                for s in range(gen.n_sites):
                    residuals[:, m, s] = rng.choice(
                        gen.Z_h[:, m, s], size=n_years, replace=True
                    )
            flows_c.append(gen.generate_from_residuals(residuals))

        def pool_corr(flows_list):
            mats = []
            for fl in flows_list:
                col0 = fl[:, 0]
                grid = col0[: (len(col0) // n_per) * n_per].reshape(-1, n_per)
                mats.append(np.corrcoef(grid, rowvar=False))
            return np.mean(mats, axis=0)

        diff = pool_corr(flows_a) - pool_corr(flows_c)
        # Frobenius norm scales with sqrt(n_per^2); allow more slack for weekly.
        threshold = 1.5 if n_per == 12 else 6.0
        frob = np.linalg.norm(diff, ord="fro")
        assert frob < threshold, (
            f"generate() and generate_from_residuals() disagree on cross-period "
            f"correlation: Frobenius {frob:.3f} (threshold {threshold})."
        )

    @pytest.mark.parametrize("fixture_name", AGGREGATED_FIXTURES)
    def test_generate_from_indices_matches_single_series(self, fixture_name, request):
        """Given the same M, generate_from_indices and generate_single_series
        must produce identical output."""
        df = request.getfixturevalue(fixture_name)
        gen = KirschGenerator()
        gen.fit(df)

        rng = np.random.default_rng(123)
        n_years = 8
        M = gen._get_bootstrap_indices(n_years + 1, max_idx=gen.Y.shape[0], rng=rng)

        out_series = gen.generate_single_series(n_years, M=M, as_array=True)
        out_indices = gen.generate_from_indices(M, n_years=n_years, as_array=True)

        np.testing.assert_allclose(out_series, out_indices)
