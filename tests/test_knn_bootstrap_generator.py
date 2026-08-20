"""
Tests for KNNBootstrapGenerator.

Tests the initialization, preprocessing, fitting, and generation workflow
for the KNN Bootstrap nonparametric generator.
"""

import pytest
import numpy as np
import pandas as pd
from synhydro.methods.generation.nonparametric.knn_bootstrap import (
    KNNBootstrapGenerator,
)
from synhydro.core.ensemble import Ensemble


class TestKNNBootstrapGeneratorInitialization:
    """Test generator initialization."""

    def test_initialization_with_dataframe(self, sample_monthly_dataframe):
        """Test generator initialization."""
        gen = KNNBootstrapGenerator()
        assert gen is not None
        assert not gen.is_fitted
        assert not gen.is_preprocessed
        assert gen.name == "KNNBootstrapGenerator"

    def test_initialization_with_custom_name(self, sample_monthly_dataframe):
        """Test initialization with custom name."""
        gen = KNNBootstrapGenerator(name="MyKNNGen")
        assert gen.name == "MyKNNGen"

    def test_initialization_with_custom_parameters(self, sample_monthly_dataframe):
        """Test initialization with custom algorithm parameters."""
        gen = KNNBootstrapGenerator(n_neighbors=15, block_size=2, name="CustomKNN")
        assert gen.n_neighbors == 15
        assert gen.block_size == 2

    def test_initialization_with_debug(self, sample_monthly_dataframe):
        """Test initialization with debug flag."""
        gen = KNNBootstrapGenerator(debug=True)
        assert gen.debug is True


class TestKNNBootstrapGeneratorPreprocessing:
    """Test preprocessing workflow."""

    def test_preprocessing_basic(self, sample_monthly_dataframe):
        """Test basic preprocessing."""
        gen = KNNBootstrapGenerator()
        gen.preprocessing(sample_monthly_dataframe)
        assert gen.is_preprocessed
        assert gen.n_sites == 3
        assert gen.sites == ["site_1", "site_2", "site_3"]

    def test_preprocessing_with_site_subset(self, sample_monthly_dataframe):
        """Test preprocessing with site subset."""
        gen = KNNBootstrapGenerator()
        gen.preprocessing(sample_monthly_dataframe, sites=["site_1", "site_2"])
        assert gen.n_sites == 2
        assert gen.sites == ["site_1", "site_2"]

    def test_preprocessing_detects_monthly_frequency(self, sample_monthly_dataframe):
        """Test frequency detection for monthly data."""
        gen = KNNBootstrapGenerator()
        gen.preprocessing(sample_monthly_dataframe)
        assert gen.output_frequency == "MS"

    def test_preprocessing_rejects_daily_frequency(self, sample_daily_dataframe):
        """Test that daily input is rejected.

        KNNBootstrapGenerator targets monthly and annual streamflow only
        (Lall & Sharma 1996; Prairie et al. 2006, 2008). Daily streamflow
        synthesis is not established for KNN bootstrap in the primary
        literature, so sub-monthly input raises ValueError.
        """
        gen = KNNBootstrapGenerator()
        with pytest.raises(ValueError, match="Sub-monthly input"):
            gen.preprocessing(sample_daily_dataframe)

    def test_preprocessing_detects_annual_frequency(self, sample_annual_dataframe):
        """Test frequency detection for annual data (Prairie et al. 2008)."""
        gen = KNNBootstrapGenerator()
        gen.preprocessing(sample_annual_dataframe)
        assert gen.output_frequency == "YS"

    def test_preprocessing_n_neighbors_heuristic(self, sample_monthly_dataframe):
        """Global n_neighbors is ceil(sqrt(n_pairs)) with n_pairs = N - 1 pairs."""
        gen = KNNBootstrapGenerator()
        gen.preprocessing(sample_monthly_dataframe)
        n_pairs = len(sample_monthly_dataframe) - 1
        expected_k = max(1, int(np.ceil(np.sqrt(n_pairs))))
        assert gen._n_neighbors == expected_k

    def test_annual_default_k_uses_pair_count(self):
        """Annual default K is ceil(sqrt(N - 1)), the searched pool size.

        N = 50 gives ceil(sqrt(50)) = 8 but ceil(sqrt(49)) = 7, so the two
        conventions are distinguishable.
        """
        dates = pd.date_range("1950-01-01", periods=50, freq="YS")
        df = pd.DataFrame({"s": np.random.default_rng(0).normal(size=50)}, index=dates)
        gen = KNNBootstrapGenerator()
        gen.fit(df)
        assert len(gen._feature_vectors) == 49
        assert gen._n_neighbors == 7
        assert gen._knn_model.n_neighbors == 7

    def test_preprocessing_rejects_non_monthly_spacing(self):
        """Spacing that is neither monthly nor annual raises ValueError."""
        for freq in ("14D", "QS"):
            dates = pd.date_range("2000-01-01", periods=60, freq=freq)
            df = pd.DataFrame(
                {"s": np.random.default_rng(0).normal(size=60)}, index=dates
            )
            with pytest.raises(ValueError, match="Unsupported input spacing"):
                KNNBootstrapGenerator().preprocessing(df)

    def test_preprocessing_accepts_month_end_frequency(self):
        """Month-end stamps (28-31 day spacing) are treated as monthly."""
        dates = pd.date_range("2000-01-31", periods=120, freq="ME")
        df = pd.DataFrame({"s": np.random.default_rng(0).normal(size=120)}, index=dates)
        gen = KNNBootstrapGenerator()
        gen.preprocessing(df)
        assert gen.output_frequency == "MS"

    def test_monthly_default_k_uses_pool_size(self, sample_monthly_dataframe):
        """Default K for each monthly pool is ceil(sqrt(n_m)) of that pool.

        Lall and Sharma (1996) take K = sqrt(n) with n the size of the
        searched sample, which for month-conditioned pools is the number of
        feature-successor pairs in that month, not the full record length.
        """
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)
        n_total = len(sample_monthly_dataframe)
        for m in range(1, 13):
            n_m = len(gen._monthly_features[m])
            expected = min(max(1, int(np.ceil(np.sqrt(n_m)))), n_m - 1)
            assert gen._monthly_k[m] == expected
            assert gen._monthly_knn[m].n_neighbors == expected
            assert len(gen._monthly_weights[m]) == expected
            assert expected < int(np.ceil(np.sqrt(n_total)))

    def test_monthly_explicit_k_respected(self, sample_monthly_dataframe):
        """Explicit n_neighbors is used for every monthly pool."""
        gen = KNNBootstrapGenerator(n_neighbors=4)
        gen.fit(sample_monthly_dataframe)
        for m in range(1, 13):
            assert gen._monthly_k[m] == 4

    def test_preprocessing_n_neighbors_custom(self, sample_monthly_dataframe):
        """Test n_neighbors is used if provided."""
        gen = KNNBootstrapGenerator(n_neighbors=5)
        gen.preprocessing(sample_monthly_dataframe)
        assert gen._n_neighbors == 5

    def test_preprocessing_n_neighbors_clamped(self, sample_monthly_dataframe):
        """Test n_neighbors is clamped to n_timesteps - 1."""
        gen = KNNBootstrapGenerator(n_neighbors=1000)
        gen.preprocessing(sample_monthly_dataframe)
        # N - 1 pairs are searched, so K is clamped to n_pairs - 1 = N - 2
        assert gen._n_neighbors == len(sample_monthly_dataframe) - 2

    def test_preprocessing_invalid_site_raises_error(self, sample_monthly_dataframe):
        """Test that invalid site name raises error."""
        gen = KNNBootstrapGenerator()
        with pytest.raises(ValueError):
            gen.preprocessing(sample_monthly_dataframe, sites=["nonexistent_site"])

    def test_preprocessing_builds_feature_vectors(self, sample_monthly_dataframe):
        """Test that feature vectors and successor pairs are built."""
        gen = KNNBootstrapGenerator()
        gen.preprocessing(sample_monthly_dataframe)
        assert gen._feature_vectors is not None
        assert gen._successor_values is not None
        assert len(gen._feature_vectors) == len(sample_monthly_dataframe) - 1
        assert len(gen._successor_values) == len(sample_monthly_dataframe) - 1

    def test_preprocessing_with_feature_cols(self, sample_monthly_dataframe):
        """Test preprocessing with specified feature columns."""
        gen = KNNBootstrapGenerator(feature_cols=["site_1"])
        gen.preprocessing(sample_monthly_dataframe)
        assert gen._feature_cols == ["site_1"]
        assert gen._feature_vectors.shape[1] == 1

    def test_preprocessing_with_invalid_feature_cols(self, sample_monthly_dataframe):
        """Test that invalid feature_cols raises error."""
        gen = KNNBootstrapGenerator(feature_cols=["nonexistent"])
        with pytest.raises(ValueError):
            gen.preprocessing(sample_monthly_dataframe)

    def test_preprocessing_with_index_site(self, sample_monthly_dataframe):
        """Test preprocessing with specified index site."""
        gen = KNNBootstrapGenerator(index_site="site_1")
        gen.preprocessing(sample_monthly_dataframe)
        assert gen.index_site == "site_1"

    def test_preprocessing_with_invalid_index_site(self, sample_monthly_dataframe):
        """Test that invalid index_site raises error."""
        gen = KNNBootstrapGenerator(index_site="nonexistent")
        with pytest.raises(ValueError):
            gen.preprocessing(sample_monthly_dataframe)


class TestKNNBootstrapGeneratorFitting:
    """Test fitting workflow."""

    def test_fit_without_preprocessing_raises_error(self, sample_monthly_dataframe):
        """Test that fit without preprocessing raises error."""
        gen = KNNBootstrapGenerator()
        with pytest.raises(ValueError):
            gen.fit()

    def test_fit_basic(self, sample_monthly_dataframe):
        """Test basic fitting."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)
        assert gen.is_fitted
        assert gen._is_monthly_conditioned
        assert len(gen._monthly_knn) == 12

    def test_fit_computes_kernel_weights(self, sample_monthly_dataframe):
        """Test that kernel weights are computed correctly."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)
        assert gen._kernel_weights is not None
        assert len(gen._kernel_weights) == gen._n_neighbors
        assert np.isclose(np.sum(gen._kernel_weights), 1.0)

    def test_fit_kernel_weights_are_decreasing(self, sample_monthly_dataframe):
        """Test that Lall-Sharma kernel weights are decreasing."""
        gen = KNNBootstrapGenerator(n_neighbors=10)
        gen.fit(sample_monthly_dataframe)
        weights = gen._kernel_weights
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]

    def test_fit_kernel_weights_exact(self):
        """Kernel weights are exactly (1/j) / H_K for rank j = 1..K."""
        gen = KNNBootstrapGenerator()
        for k in (1, 2, 5, 10):
            w = gen._make_kernel_weights(k)
            H = sum(1.0 / j for j in range(1, k + 1))
            expected = np.array([(1.0 / j) / H for j in range(1, k + 1)])
            np.testing.assert_allclose(w, expected, rtol=0, atol=1e-15)
            assert np.isclose(w.sum(), 1.0)
            if k >= 2:
                assert np.isclose(w[0] / w[1], 2.0)

    def test_fit_empty_month_pool_raises(self):
        """A record missing a calendar month raises a clear ValueError."""
        dates = pd.date_range("2000-01-01", periods=120, freq="MS")
        df = pd.DataFrame({"s": np.random.default_rng(0).normal(size=120)}, index=dates)
        df = df[df.index.month != 7]
        gen = KNNBootstrapGenerator()
        with pytest.raises(ValueError, match=r"calendar month 7 \(July\)"):
            gen.fit(df)

    def test_fit_sets_fitted_params(self, sample_monthly_dataframe):
        """Test that fitted_params_ is set after fit."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)
        assert gen.fitted_params_ is not None
        assert gen.fitted_params_.sample_size_ == len(sample_monthly_dataframe) - 1
        assert gen.fitted_params_.n_sites_ == 3


class TestKNNBootstrapGeneratorGeneration:
    """Test generation workflow."""

    def test_generate_without_fit_raises_error(self, sample_monthly_dataframe):
        """Test that generate without fit raises error."""
        gen = KNNBootstrapGenerator()
        gen.preprocessing(sample_monthly_dataframe)
        with pytest.raises(ValueError):
            gen.generate()

    def test_generate_single_realization(self, sample_monthly_dataframe):
        """Test generation of single realization."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)
        ensemble = gen.generate(n_realizations=1, n_years=5)
        assert isinstance(ensemble, Ensemble)
        assert len(ensemble.realization_ids) == 1
        assert ensemble.metadata.n_sites == 3

    def test_generate_multiple_realizations(self, sample_monthly_dataframe):
        """Test generation of multiple realizations."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)
        ensemble = gen.generate(n_realizations=10, n_years=5)
        assert len(ensemble.realization_ids) == 10
        assert ensemble.metadata.n_sites == 3

    def test_generate_with_n_years(self, sample_monthly_dataframe):
        """Test generation with n_years parameter."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)
        n_years = 3
        ensemble = gen.generate(n_realizations=1, n_years=n_years)
        realization_df = ensemble.data_by_realization[0]
        assert len(realization_df) == n_years * 12

    def test_generate_with_n_timesteps(self, sample_monthly_dataframe):
        """Test generation with explicit n_timesteps."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)
        n_timesteps = 100
        ensemble = gen.generate(n_realizations=1, n_timesteps=n_timesteps)
        realization_df = ensemble.data_by_realization[0]
        assert len(realization_df) == n_timesteps

    def test_generate_with_seed_reproducibility(self, sample_monthly_dataframe):
        """Test that seed produces reproducible results."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)

        ensemble1 = gen.generate(n_realizations=1, n_years=5, seed=42)
        ensemble2 = gen.generate(n_realizations=1, n_years=5, seed=42)

        df1 = ensemble1.data_by_realization[0]
        df2 = ensemble2.data_by_realization[0]

        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_values_in_historical_range(self, sample_monthly_dataframe):
        """Test that generated values are within historical range."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)
        ensemble = gen.generate(n_realizations=5, n_years=10)

        for real_id, df in ensemble.data_by_realization.items():
            for site in df.columns:
                # Generated values should be from successors, which are historical
                min_val = sample_monthly_dataframe[site].min()
                max_val = sample_monthly_dataframe[site].max()
                assert df[site].min() >= min_val - abs(min_val) * 0.01
                assert df[site].max() <= max_val + abs(max_val) * 0.01

    def test_generate_output_has_datetimeindex(self, sample_monthly_dataframe):
        """Test that generated output has DatetimeIndex."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)
        ensemble = gen.generate(n_realizations=1, n_years=5)
        realization_df = ensemble.data_by_realization[0]
        assert isinstance(realization_df.index, pd.DatetimeIndex)

    def test_generate_output_has_correct_columns(self, sample_monthly_dataframe):
        """Test that generated output has correct site columns."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)
        ensemble = gen.generate(n_realizations=1, n_years=5)
        realization_df = ensemble.data_by_realization[0]
        assert list(realization_df.columns) == gen.sites

    def test_generate_ensemble_metadata(self, sample_monthly_dataframe):
        """Test that ensemble has proper metadata."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_monthly_dataframe)
        ensemble = gen.generate(n_realizations=3, n_years=5)
        assert ensemble.metadata.generator_class == "KNNBootstrapGenerator"
        assert ensemble.metadata.n_realizations == 3
        assert ensemble.metadata.n_sites == 3

    def test_generate_with_annual_data(self, sample_annual_dataframe):
        """Test generation with annual data (Prairie et al. 2008)."""
        gen = KNNBootstrapGenerator()
        gen.fit(sample_annual_dataframe)
        ensemble = gen.generate(n_realizations=1, n_years=10)
        realization_df = ensemble.data_by_realization[0]
        assert len(realization_df) == 10


class TestKNNBootstrapGeneratorEdgeCases:
    """Test edge cases and error handling."""

    def test_single_site_data(self, sample_monthly_series):
        """Test with single-site data."""
        df = sample_monthly_series.to_frame()
        gen = KNNBootstrapGenerator()
        gen.fit(df)
        ensemble = gen.generate(n_realizations=2, n_years=5)
        assert ensemble.metadata.n_sites == 1
        assert len(ensemble.realization_ids) == 2

    def test_very_short_dataset(self):
        """Test with very short dataset."""
        dates = pd.date_range("2000-01-01", periods=20, freq="MS")
        df = pd.DataFrame({"site1": np.random.randn(20)}, index=dates)
        gen = KNNBootstrapGenerator(n_neighbors=5)
        gen.fit(df)
        ensemble = gen.generate(n_realizations=1, n_timesteps=50)
        assert len(ensemble.data_by_realization[0]) == 50

    def test_all_zero_flows(self):
        """Test with all-zero flows."""
        dates = pd.date_range("2000-01-01", periods=100, freq="MS")
        df = pd.DataFrame({"site1": np.zeros(100)}, index=dates)
        gen = KNNBootstrapGenerator()
        gen.fit(df)
        ensemble = gen.generate(n_realizations=1, n_years=3)
        # All generated values should be zero
        assert (ensemble.data_by_realization[0] == 0).all().all()

    def test_constant_flow(self):
        """Test with constant flow values."""
        dates = pd.date_range("2000-01-01", periods=100, freq="MS")
        df = pd.DataFrame({"site1": np.full(100, 100.0)}, index=dates)
        gen = KNNBootstrapGenerator()
        gen.fit(df)
        ensemble = gen.generate(n_realizations=1, n_years=3)
        # All generated values should be constant
        assert (ensemble.data_by_realization[0] == 100.0).all().all()

    def test_negative_values(self):
        """Test with negative flow values (e.g., anomalies)."""
        dates = pd.date_range("2000-01-01", periods=100, freq="MS")
        df = pd.DataFrame(
            {"site1": np.random.randn(100) * 50}, index=dates  # Can be negative
        )
        gen = KNNBootstrapGenerator()
        gen.fit(df)
        ensemble = gen.generate(n_realizations=1, n_years=3)
        # Should handle negative values without error
        assert len(ensemble.data_by_realization[0]) == 36

    def test_very_large_n_neighbors(self, sample_monthly_dataframe):
        """Test with n_neighbors larger than dataset."""
        gen = KNNBootstrapGenerator(n_neighbors=10000)
        gen.preprocessing(sample_monthly_dataframe)
        assert gen._n_neighbors < len(sample_monthly_dataframe)
        gen.fit()
        ensemble = gen.generate(n_realizations=1, n_years=5)
        assert len(ensemble.data_by_realization[0]) == 60


def _two_site_monthly(n: int = 600, rho: float = 0.9, seed: int = 0) -> pd.DataFrame:
    """Two sites on very different scales; only site B is autocorrelated."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1700-01-01", periods=n, freq="MS")
    a = rng.uniform(1.0, 10.0, n)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + np.sqrt(1.0 - rho**2) * rng.normal()
    b = np.exp(np.log(3000.0) + 0.8 * z)
    return pd.DataFrame({"A": a, "B": b}, index=dates)


class TestKNNBootstrapQueryColumns:
    """Regression tests for KNN queries built from the feature column positions."""

    def test_knn_col_idx_matches_index_site(self):
        """Feature column positions follow index_site, not column order."""
        df = _two_site_monthly()
        gen = KNNBootstrapGenerator(index_site="B")
        gen.preprocessing(df)
        assert gen._knn_cols == ["B"]
        assert gen._knn_col_idx == [1]

    def test_knn_col_idx_matches_feature_cols(self):
        """Feature column positions follow feature_cols order."""
        df = _two_site_monthly()
        gen = KNNBootstrapGenerator(feature_cols=["B", "A"])
        gen.preprocessing(df)
        assert gen._knn_col_idx == [1, 0]

    @pytest.mark.parametrize("freq", ["MS", "YS"])
    def test_non_leading_index_site_preserves_lag1(self, freq):
        """With index_site='B' (second column) the lag-1 of B is preserved.

        Before the fix the query used the first column (site A, range 1-10)
        against training features on site B's scale (hundreds to tens of
        thousands), so neighbors were effectively random and the synthetic
        lag-1 of B collapsed to ~0.03 against an observed ~0.9.
        """
        df = _two_site_monthly(n=600 if freq == "MS" else 200)
        if freq == "YS":
            df.index = pd.date_range("1700-01-01", periods=len(df), freq="YS")
        gen = KNNBootstrapGenerator(index_site="B")
        gen.fit(df)
        n_steps = 3000 if freq == "MS" else 300
        syn = gen.generate(n_realizations=1, n_timesteps=n_steps, seed=1)
        syn_df = syn.data_by_realization[0]
        obs_lag1 = df["B"].autocorr(1)
        syn_lag1 = syn_df["B"].autocorr(1)
        assert obs_lag1 > 0.7
        assert syn_lag1 > 0.5, f"synthetic lag-1 {syn_lag1:.3f} vs observed {obs_lag1:.3f}"
        assert syn_lag1 > 0.7 * obs_lag1

    def test_query_site_is_index_site(self):
        """Each KNN query equals the previously generated value of the index site."""
        df = _two_site_monthly()
        gen = KNNBootstrapGenerator(index_site="B")
        gen.fit(df)
        queries = []
        for m in range(1, 13):
            knn = gen._monthly_knn[m]
            orig = knn.kneighbors

            def spy(X, *args, _orig=orig, **kwargs):
                queries.append(float(X[0, 0]))
                return _orig(X, *args, **kwargs)

            knn.kneighbors = spy
        syn_df = gen._generate_single_realization(50, rng=np.random.default_rng(3))
        assert len(queries) == 49
        np.testing.assert_allclose(queries, syn_df["B"].values[:-1])


class TestKNNBootstrapPoolAlignment:
    """Monthly pools pair month-m features with calendar month m+1 successors."""

    def test_pool_month_alignment_with_dec_jan_wrap(self):
        df = _two_site_monthly(n=240)
        gen = KNNBootstrapGenerator()
        gen.fit(df)
        months = df.index.month[:-1]
        years = df.index.year[:-1]
        for m in range(1, 13):
            m_next = 1 if m == 12 else m + 1
            feats = gen._monthly_features[m]
            succ = gen._monthly_successors[m][:, 0, :]
            # Features are the observed rows of month m (excluding the final row)
            np.testing.assert_array_equal(feats, df.values[:-1][months == m])
            # Successors are the observed rows of calendar month m+1 in the
            # same year (Dec -> Jan of the following year)
            succ_years = years[months == m] + (1 if m == 12 else 0)
            expected = np.stack(
                [df.loc[pd.Timestamp(y, m_next, 1)].values for y in succ_years]
            )
            np.testing.assert_array_equal(succ, expected)
            assert len(feats) == len(succ)
        # 20 Decembers but the last has no successor
        assert len(gen._monthly_features[12]) == 240 // 12 - 1
        assert len(gen._monthly_features[1]) == 240 // 12


class TestKNNBootstrapInitialState:
    """The initial draw must come from the pool for the first synthetic month."""

    def test_initial_value_from_correct_month(self):
        """First synthetic timestep is an observed value of that calendar month."""
        df = _two_site_monthly()
        gen = KNNBootstrapGenerator(index_site="B")
        gen.fit(df)
        m0 = (df.index[-1] + pd.DateOffset(months=1)).month
        pool = df["B"][df.index.month == m0].values
        other = df["B"][df.index.month != m0].values
        for seed in range(50):
            first = gen._generate_single_realization(
                1, rng=np.random.default_rng(seed)
            )
            assert first.index[0].month == m0
            val = first.iloc[0]["B"]
            assert np.isclose(pool, val).any()
            assert not np.isclose(other, val).any()


class TestKNNBootstrapGeneratorWorkflow:
    """Test complete workflow."""

    def test_complete_workflow_monthly(self, sample_monthly_dataframe):
        """Test complete preprocessing -> fit -> generate workflow."""
        gen = KNNBootstrapGenerator(n_neighbors=10, name="TestKNN")

        # Preprocessing
        gen.preprocessing(sample_monthly_dataframe)
        assert gen.is_preprocessed

        # Fitting
        gen.fit()
        assert gen.is_fitted

        # Generation
        ensemble = gen.generate(n_realizations=5, n_years=10, seed=42)
        assert isinstance(ensemble, Ensemble)
        assert len(ensemble.realization_ids) == 5
        assert ensemble.metadata.n_sites == 3

        # Check data quality
        for real_id, df in ensemble.data_by_realization.items():
            assert df.shape[0] == 120  # 10 years * 12 months
            assert df.shape[1] == 3  # 3 sites
            assert df.notna().all().all()

    def test_complete_workflow_annual(self, sample_annual_dataframe):
        """Test complete workflow with annual data (Prairie et al. 2008)."""
        gen = KNNBootstrapGenerator(n_neighbors=5)
        gen.fit(sample_annual_dataframe)
        ensemble = gen.generate(n_realizations=2, n_years=10)

        for real_id, df in ensemble.data_by_realization.items():
            assert df.shape[0] == 10
            assert df.shape[1] == 3

    @pytest.mark.parametrize(
        "kwargs", [{}, {"index_site": "site_2"}, {"block_size": 3}]
    )
    def test_multisite_joint_resampling_rows_are_observed(
        self, sample_monthly_dataframe, kwargs
    ):
        """Every generated multi-site row equals some observed row.

        Joint resampling takes the full successor vector across sites, so
        each synthetic row must match an observed row exactly (no mixing of
        sites from different historical dates).
        """
        gen = KNNBootstrapGenerator(**kwargs)
        gen.fit(sample_monthly_dataframe)
        ensemble = gen.generate(n_realizations=2, n_years=5, seed=42)
        obs = sample_monthly_dataframe.values
        for df in ensemble.data_by_realization.values():
            assert df.shape[1] == 3
            assert not df.isnull().any().any()
            for row in df.values:
                matches = np.all(np.isclose(obs, row), axis=1)
                assert matches.any(), f"row {row} not found in observed record"
