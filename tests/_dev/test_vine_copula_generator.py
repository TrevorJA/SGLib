"""Tests for the Vine Copula Generator."""

import pickle
import numpy as np
import pandas as pd
import pytest

pv = pytest.importorskip("pyvinecopulib")

from synhydro.methods.generation._dev.vine_copula import (
    VineCopulaGenerator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monthly_multisite():
    """30 years of monthly 3-site data with seasonal pattern and cross-correlation."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1990-01-01", periods=360, freq="MS")
    n = len(dates)
    base = rng.gamma(shape=2.0, scale=80.0, size=(n, 3))
    shared = rng.gamma(shape=2.0, scale=40.0, size=n)
    seasonal = 100 + 80 * np.sin(2 * np.pi * np.arange(n) / 12)
    data = {}
    for i, name in enumerate(["site_A", "site_B", "site_C"]):
        data[name] = np.maximum(base[:, i] + shared + seasonal, 1.0)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def monthly_single_site():
    """Single-site monthly series."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")
    vals = rng.gamma(2, 50, len(dates))
    return pd.DataFrame({"only_site": vals}, index=dates)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestVineCopulaInit:
    def test_default_params(self):
        gen = VineCopulaGenerator()
        assert gen.vine_type == "rvine"
        assert gen.family_set == "all"
        assert gen.selection_criterion == "aic"
        assert gen.marginal_method == "parametric"
        assert gen.log_transform is False
        assert gen.is_fitted is False

    def test_custom_params(self):
        gen = VineCopulaGenerator(
            vine_type="dvine",
            family_set=["gaussian", "clayton"],
            selection_criterion="bic",
            marginal_method="empirical",
            log_transform=True,
            offset=0.5,
            trunc_level=1,
        )
        assert gen.vine_type == "dvine"
        assert gen.family_set == ["gaussian", "clayton"]
        assert gen.selection_criterion == "bic"
        assert gen.marginal_method == "empirical"
        assert gen.log_transform is True
        assert gen.offset == 0.5
        assert gen.trunc_level == 1

    def test_invalid_vine_type_raises(self):
        with pytest.raises(ValueError, match="vine_type"):
            VineCopulaGenerator(vine_type="xvine")

    def test_invalid_marginal_method_raises(self):
        with pytest.raises(ValueError, match="marginal_method"):
            VineCopulaGenerator(marginal_method="kde")

    def test_invalid_selection_criterion_raises(self):
        with pytest.raises(ValueError, match="selection_criterion"):
            VineCopulaGenerator(selection_criterion="mle")

    def test_stores_algorithm_params(self):
        gen = VineCopulaGenerator(vine_type="dvine")
        assert gen.init_params.algorithm_params["vine_type"] == "dvine"
        assert "Vine Copula" in gen.init_params.algorithm_params["method"]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestVineCopulaPreprocessing:
    def test_preprocessing_monthly_dataframe(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.preprocessing(monthly_multisite)
        assert gen.is_preprocessed
        assert gen.n_sites == 3

    def test_preprocessing_site_subset(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.preprocessing(monthly_multisite, sites=["site_A", "site_C"])
        assert gen.n_sites == 2
        assert set(gen.sites) == {"site_A", "site_C"}

    def test_preprocessing_log_transform(self, monthly_multisite):
        gen = VineCopulaGenerator(log_transform=True)
        gen.preprocessing(monthly_multisite)
        assert gen.is_preprocessed
        assert gen._Q_monthly.values.max() < monthly_multisite.values.max()


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestVineCopulaFit:
    def test_fit_basic_parametric(self, monthly_multisite):
        gen = VineCopulaGenerator(marginal_method="parametric")
        gen.fit(monthly_multisite)
        assert gen.is_fitted
        assert len(gen._monthly_vines) == 12

    def test_fit_basic_empirical(self, monthly_multisite):
        gen = VineCopulaGenerator(marginal_method="empirical")
        gen.fit(monthly_multisite)
        assert gen.is_fitted
        assert gen._nst is not None

    def test_fit_with_q_obs(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        assert gen.is_fitted

    def test_fit_without_preprocessing_raises(self):
        gen = VineCopulaGenerator()
        with pytest.raises(Exception):
            gen.fit()

    def test_fit_creates_fitted_params(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        fp = gen.get_fitted_params()
        assert fp is not None
        assert fp["n_sites_"] == 3
        assert fp["n_parameters_"] > 0

    def test_fit_vine_per_month(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        for m in range(1, 13):
            vine = gen._monthly_vines[m]
            # Should be a Vinecop object for 3 sites
            assert vine is not None
            assert vine.dim == 3

    def test_fit_dvine(self, monthly_multisite):
        gen = VineCopulaGenerator(vine_type="dvine")
        gen.fit(monthly_multisite)
        assert gen.is_fitted

    def test_fit_cvine(self, monthly_multisite):
        gen = VineCopulaGenerator(vine_type="cvine")
        gen.fit(monthly_multisite)
        assert gen.is_fitted

    def test_fit_with_trunc_level(self, monthly_multisite):
        gen = VineCopulaGenerator(trunc_level=1)
        gen.fit(monthly_multisite)
        assert gen.is_fitted
        for m in range(1, 13):
            vine = gen._monthly_vines[m]
            if vine is not None:
                assert vine.trunc_lvl <= 1

    def test_fit_custom_family_set(self, monthly_multisite):
        gen = VineCopulaGenerator(family_set=["gaussian", "clayton"])
        gen.fit(monthly_multisite)
        assert gen.is_fitted

    def test_fit_bic_selection(self, monthly_multisite):
        gen = VineCopulaGenerator(selection_criterion="bic")
        gen.fit(monthly_multisite)
        assert gen.is_fitted

    def test_fit_marginal_params_stored(self, monthly_multisite):
        gen = VineCopulaGenerator(marginal_method="parametric")
        gen.fit(monthly_multisite)
        # 12 months * 3 sites = 36 entries
        assert len(gen._marginal_params) == 36
        for key, params in gen._marginal_params.items():
            assert params["dist"] in ("gamma", "lognorm")
            assert "shape" in params
            assert "scale" in params


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class TestVineCopulaGeneration:
    def test_generate_basic(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=2, n_years=10, seed=42)
        assert len(ens.data_by_realization) == 2

    def test_generate_shape(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, n_years=5, seed=42)
        df = ens.data_by_realization[0]
        assert df.shape == (60, 3)

    def test_generate_n_timesteps(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, n_timesteps=37, seed=42)
        df = ens.data_by_realization[0]
        assert len(df) == 37

    def test_generate_non_negative(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=5, n_years=20, seed=42)
        for r, df in ens.data_by_realization.items():
            assert (df.values >= 0).all(), f"Realization {r} has negative values"

    def test_generate_reproducible_with_seed(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        ens1 = gen.generate(n_realizations=1, n_years=10, seed=123)
        ens2 = gen.generate(n_realizations=1, n_years=10, seed=123)
        pd.testing.assert_frame_equal(
            ens1.data_by_realization[0], ens2.data_by_realization[0]
        )

    def test_generate_without_fit_raises(self):
        gen = VineCopulaGenerator()
        with pytest.raises(Exception):
            gen.generate(n_realizations=1, n_years=5)

    def test_generate_has_datetime_index(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, n_years=5, seed=42)
        df = ens.data_by_realization[0]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert pd.infer_freq(df.index) in ("MS", "<MonthBegin>")

    def test_generate_preserves_site_names(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, n_years=5, seed=42)
        df = ens.data_by_realization[0]
        assert list(df.columns) == ["site_A", "site_B", "site_C"]

    def test_generate_empirical_marginals(self, monthly_multisite):
        gen = VineCopulaGenerator(marginal_method="empirical")
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=2, n_years=10, seed=42)
        assert len(ens.data_by_realization) == 2

    def test_generate_default_n_years(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, seed=42)
        df = ens.data_by_realization[0]
        assert len(df) == len(monthly_multisite)


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------


class TestVineCopulaStatisticalProperties:
    def test_marginal_means_preserved(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=50, n_years=30, seed=42)

        obs_means = monthly_multisite.mean()
        syn_means = pd.concat(
            [df.mean() for df in ens.data_by_realization.values()], axis=1
        ).mean(axis=1)
        for site in monthly_multisite.columns:
            ratio = syn_means[site] / obs_means[site]
            assert 0.7 < ratio < 1.3, (
                f"Site {site}: obs={obs_means[site]:.1f}, " f"syn={syn_means[site]:.1f}"
            )

    def test_spatial_correlation_preserved(self, monthly_multisite):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=30, n_years=30, seed=42)

        obs_corr = monthly_multisite.corr()
        syn_corrs = [df.corr() for df in ens.data_by_realization.values()]
        avg_syn_corr = sum(syn_corrs) / len(syn_corrs)

        for i in range(3):
            for j in range(i + 1, 3):
                site_i = monthly_multisite.columns[i]
                site_j = monthly_multisite.columns[j]
                diff = abs(
                    avg_syn_corr.loc[site_i, site_j] - obs_corr.loc[site_i, site_j]
                )
                assert diff < 0.3, (
                    f"Correlation {site_i}-{site_j}: "
                    f"obs={obs_corr.loc[site_i, site_j]:.3f}, "
                    f"syn={avg_syn_corr.loc[site_i, site_j]:.3f}"
                )

    def test_univariate_special_case(self, monthly_single_site):
        gen = VineCopulaGenerator()
        gen.fit(monthly_single_site)
        ens = gen.generate(n_realizations=3, n_years=10, seed=42)
        df = ens.data_by_realization[0]
        assert df.shape[1] == 1
        assert (df.values >= 0).all()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestVineCopulaSerialization:
    def test_pickle_save_load(self, monthly_multisite, tmp_path):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)

        filepath = tmp_path / "vine_gen.pkl"
        gen.save(str(filepath))

        loaded = VineCopulaGenerator.load(str(filepath))
        assert loaded.is_fitted
        assert loaded.vine_type == "rvine"
        assert loaded.marginal_method == "parametric"
        assert len(loaded._monthly_vines) == 12

    def test_pickle_generate_after_load(self, monthly_multisite, tmp_path):
        gen = VineCopulaGenerator()
        gen.fit(monthly_multisite)

        filepath = tmp_path / "vine_gen.pkl"
        gen.save(str(filepath))
        loaded = VineCopulaGenerator.load(str(filepath))

        ens = loaded.generate(n_realizations=1, n_years=5, seed=42)
        assert len(ens.data_by_realization) == 1
        assert ens.data_by_realization[0].shape == (60, 3)

    def test_pickle_dvine_roundtrip(self, monthly_multisite, tmp_path):
        gen = VineCopulaGenerator(vine_type="dvine")
        gen.fit(monthly_multisite)

        filepath = tmp_path / "dvine_gen.pkl"
        gen.save(str(filepath))
        loaded = VineCopulaGenerator.load(str(filepath))

        assert loaded.vine_type == "dvine"
        ens = loaded.generate(n_realizations=1, n_years=5, seed=42)
        assert len(ens.data_by_realization) == 1
