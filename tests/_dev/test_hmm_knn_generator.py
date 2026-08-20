"""
Tests for HMMKNNGenerator (Prairie et al. 2008).
"""

import pickle
import numpy as np
import pandas as pd
import pytest

from synhydro.methods.generation._dev.hmm_knn import HMMKNNGenerator
from synhydro.core.ensemble import Ensemble


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def annual_multisite():
    """50 years of correlated lognormal annual flows at 3 sites."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("1970-01-01", periods=50, freq="YS")
    base = rng.standard_normal(50)
    data = {
        "A": np.exp(6.0 + 0.4 * base + 0.2 * rng.standard_normal(50)),
        "B": np.exp(5.8 + 0.4 * base + 0.2 * rng.standard_normal(50)),
        "C": np.exp(5.5 + 0.4 * base + 0.2 * rng.standard_normal(50)),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def annual_single_site(annual_multisite):
    """Single-site version of the annual fixture."""
    return annual_multisite[["A"]]


# ---------------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------------


class TestHMMKNNImportPaths:
    def test_exported_from_hybrid_package(self):
        from synhydro.methods.generation import hybrid

        assert hybrid.HMMKNNGenerator is HMMKNNGenerator
        assert "HMMKNNGenerator" in hybrid.__all__

    def test_exported_from_generation_package(self):
        from synhydro.methods import generation

        assert generation.HMMKNNGenerator is HMMKNNGenerator
        assert "HMMKNNGenerator" in generation.__all__

    def test_exported_from_top_level(self):
        import synhydro

        assert synhydro.HMMKNNGenerator is HMMKNNGenerator
        assert "HMMKNNGenerator" in synhydro.__all__


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestHMMKNNInit:
    def test_default_params(self):
        gen = HMMKNNGenerator()
        assert gen.n_states == 2
        assert gen.delta == 1.0
        assert gen.covariance_type == "full"
        assert gen.n_init == 10
        assert gen.is_preprocessed is False
        assert gen.is_fitted is False

    def test_custom_params(self):
        gen = HMMKNNGenerator(n_states=3, delta=0.1, covariance_type="diag", n_init=5)
        assert gen.n_states == 3
        assert gen.delta == 0.1
        assert gen.covariance_type == "diag"
        assert gen.n_init == 5

    def test_invalid_n_states(self):
        with pytest.raises(ValueError, match="n_states"):
            HMMKNNGenerator(n_states=1)

    def test_invalid_delta(self):
        with pytest.raises(ValueError, match="delta"):
            HMMKNNGenerator(delta=0.0)

    def test_invalid_covariance_type(self):
        with pytest.raises(ValueError, match="covariance_type"):
            HMMKNNGenerator(covariance_type="bad")

    def test_invalid_n_init(self):
        with pytest.raises(ValueError, match="n_init"):
            HMMKNNGenerator(n_init=0)

    def test_stores_algorithm_params(self):
        gen = HMMKNNGenerator()
        assert "Prairie" in gen.init_params.algorithm_params["method"]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestHMMKNNPreprocessing:
    def test_preprocessed_flag(self, annual_multisite):
        gen = HMMKNNGenerator()
        gen.preprocessing(annual_multisite)
        assert gen.is_preprocessed is True

    def test_sites_stored(self, annual_multisite):
        gen = HMMKNNGenerator()
        gen.preprocessing(annual_multisite)
        assert gen._sites == ["A", "B", "C"]
        assert gen._n_sites == 3

    def test_single_site(self, annual_single_site):
        gen = HMMKNNGenerator()
        gen.preprocessing(annual_single_site)
        assert gen._n_sites == 1

    def test_site_subset(self, annual_multisite):
        gen = HMMKNNGenerator()
        gen.preprocessing(annual_multisite, sites=["A", "B"])
        assert gen._sites == ["A", "B"]
        assert gen._n_sites == 2

    def test_log_transform_stored(self, annual_multisite):
        gen = HMMKNNGenerator()
        gen.preprocessing(annual_multisite)
        assert gen.Q_log_ is not None
        assert gen.Q_log_.shape == (50, 3)

    def test_log_values_finite(self, annual_multisite):
        gen = HMMKNNGenerator()
        gen.preprocessing(annual_multisite)
        assert np.all(np.isfinite(gen.Q_log_))

    def test_no_fit_before_preprocessing_raises(self):
        gen = HMMKNNGenerator()
        with pytest.raises(ValueError, match="preprocessing"):
            gen.fit()


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestHMMKNNFit:
    def test_fitted_flag(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        assert gen.is_fitted is True

    def test_transition_matrix_shape(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        assert gen.transition_matrix_.shape == (2, 2)

    def test_transition_rows_sum_to_one(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        row_sums = gen.transition_matrix_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_stationary_distribution_sums_to_one(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        assert abs(gen.stationary_distribution_.sum() - 1.0) < 1e-8

    def test_state_sequence_length(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        assert len(gen.state_sequence_) == 50

    def test_state_sequence_values(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        assert set(gen.state_sequence_).issubset(set(range(2)))

    def test_state_pools_built(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        assert len(gen._state_pools) == 2
        for s in range(2):
            assert s in gen._state_pools

    def test_category_pools_nonempty(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        assert len(gen._category_pools) > 0

    def test_log_std_shape(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        assert gen.log_std_.shape == (3,)
        assert np.all(gen.log_std_ > 0)

    def test_fitted_params_stored(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        assert gen.fitted_params_ is not None
        assert gen.fitted_params_.n_sites_ == 3

    def test_n_states_3(self, annual_multisite):
        gen = HMMKNNGenerator(n_states=3, n_init=2)
        gen.fit(annual_multisite)
        assert gen.transition_matrix_.shape == (3, 3)
        assert len(gen.stationary_distribution_) == 3

    def test_single_site_fit(self, annual_single_site):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_single_site)
        assert gen.is_fitted
        assert gen._n_sites == 1

    def test_no_generate_before_fit_raises(self, annual_multisite):
        gen = HMMKNNGenerator()
        gen.preprocessing(annual_multisite)
        with pytest.raises(ValueError, match="fit"):
            gen.generate(n_years=5)

    def test_fit_via_fit_shortcut(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        assert gen.is_fitted

    def test_diag_covariance(self, annual_multisite):
        gen = HMMKNNGenerator(covariance_type="diag", n_init=2)
        gen.fit(annual_multisite)
        assert gen.is_fitted


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class TestHMMKNNFittedParams:
    @pytest.mark.parametrize(
        "covariance_type,expected",
        [
            # 2 states, 3 sites: 2 transition + 6 means + covariances
            ("full", 2 + 6 + 2 * 3 * 4 // 2),
            ("diag", 2 + 6 + 2 * 3),
            ("spherical", 2 + 6 + 2),
        ],
    )
    def test_n_parameters_includes_covariances(
        self, annual_multisite, covariance_type, expected
    ):
        gen = HMMKNNGenerator(n_states=2, covariance_type=covariance_type, n_init=2)
        gen.fit(annual_multisite)
        assert gen.fitted_params_.n_parameters_ == expected


class TestHMMKNNGenerate:
    def test_returns_ensemble(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=20, n_realizations=2, seed=0)
        assert isinstance(result, Ensemble)

    def test_n_realizations(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=10, n_realizations=5, seed=0)
        assert result.metadata.n_realizations == 5

    def test_ensemble_metadata(self, annual_multisite):
        """Ensemble carries the metadata needed for pipeline chaining."""
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=10, n_realizations=2, seed=0)
        assert result.frequency == "YS"
        assert result.metadata.time_resolution == "YS"
        assert result.metadata.generator_class == "HMMKNNGenerator"
        assert result.metadata.n_sites == 3
        assert result.metadata.time_period is not None

    def test_shape_multisite(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=30, n_realizations=1, seed=0)
        df = result.data_by_realization[0]
        assert df.shape == (30, 3)

    def test_shape_single_site(self, annual_single_site):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_single_site)
        result = gen.generate(n_years=20, n_realizations=1, seed=0)
        df = result.data_by_realization[0]
        assert df.shape == (20, 1)

    def test_site_columns_match(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=10, n_realizations=1, seed=0)
        assert list(result.data_by_realization[0].columns) == ["A", "B", "C"]

    def test_datetime_index(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=10, n_realizations=1, seed=0)
        df = result.data_by_realization[0]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_non_negative_flows(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=50, n_realizations=5, seed=0)
        for i in range(5):
            assert (result.data_by_realization[i].values >= 0).all()

    def test_no_nans(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=30, n_realizations=3, seed=0)
        for i in range(3):
            assert not result.data_by_realization[i].isna().any().any()

    def test_seed_reproducibility(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        r1 = gen.generate(n_years=20, n_realizations=3, seed=42)
        r2 = gen.generate(n_years=20, n_realizations=3, seed=42)
        for i in range(3):
            pd.testing.assert_frame_equal(
                r1.data_by_realization[i], r2.data_by_realization[i]
            )

    def test_different_seeds_differ(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        r1 = gen.generate(n_years=20, n_realizations=1, seed=1)
        r2 = gen.generate(n_years=20, n_realizations=1, seed=2)
        assert not r1.data_by_realization[0].equals(r2.data_by_realization[0])

    def test_realizations_differ(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=20, n_realizations=5, seed=7)
        frames = [result.data_by_realization[i] for i in range(5)]
        n_unique = sum(1 for j in range(1, 5) if not frames[0].equals(frames[j]))
        assert n_unique >= 3

    def test_default_n_years(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_realizations=1, seed=0)
        assert len(result.data_by_realization[0]) == 50

    def test_n_timesteps_override(self, annual_multisite):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_timesteps=15, n_realizations=1, seed=0)
        assert len(result.data_by_realization[0]) == 15

    def test_n_states_3_generation(self, annual_multisite):
        gen = HMMKNNGenerator(n_states=3, n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=20, n_realizations=2, seed=0)
        assert result.data_by_realization[0].shape == (20, 3)
        assert (result.data_by_realization[0].values >= 0).all()

    def test_values_drawn_from_historical(self, annual_multisite):
        """Synthetic values must be rows present in the historical record."""
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=30, n_realizations=1, seed=0)
        df_syn = result.data_by_realization[0]
        obs_rows = set(map(tuple, annual_multisite.values.round(6)))
        for row in df_syn.values:
            assert (
                tuple(row.round(6)) in obs_rows
            ), f"Synthetic row {row} not found in historical record"


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------


class TestHMMKNNStatistics:
    def test_mean_approximately_preserved(self, annual_multisite):
        """Ensemble mean should approximate historical mean (wide tolerance)."""
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=50, n_realizations=20, seed=0)
        all_dfs = pd.concat([result.data_by_realization[i] for i in range(20)])
        syn_mean = all_dfs.mean()
        obs_mean = annual_multisite.mean()
        ratio = syn_mean / obs_mean
        assert (ratio > 0.5).all() and (ratio < 2.0).all()

    def test_spatial_correlation_sign_preserved(self, annual_multisite):
        """Sign of pairwise spatial correlations should match historical."""
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        result = gen.generate(n_years=50, n_realizations=20, seed=0)
        all_dfs = pd.concat([result.data_by_realization[i] for i in range(20)])
        syn_corr = all_dfs.corr()
        obs_corr = annual_multisite.corr()
        for si in ["A", "B", "C"]:
            for sj in ["A", "B", "C"]:
                if si != sj:
                    assert np.sign(syn_corr.loc[si, sj]) == np.sign(
                        obs_corr.loc[si, sj]
                    ), f"Correlation sign mismatch: {si} vs {sj}"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestHMMKNNSerialization:
    def test_pickle_roundtrip(self, annual_multisite, tmp_path):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        path = tmp_path / "hmm_knn.pkl"
        with open(path, "wb") as f:
            pickle.dump(gen, f)
        with open(path, "rb") as f:
            gen2 = pickle.load(f)
        assert gen2.is_fitted
        assert gen2._n_sites == 3

    def test_generate_after_pickle(self, annual_multisite, tmp_path):
        gen = HMMKNNGenerator(n_init=2)
        gen.fit(annual_multisite)
        path = tmp_path / "hmm_knn.pkl"
        with open(path, "wb") as f:
            pickle.dump(gen, f)
        with open(path, "rb") as f:
            gen2 = pickle.load(f)
        result = gen2.generate(n_years=10, n_realizations=2, seed=0)
        assert result.metadata.n_realizations == 2
