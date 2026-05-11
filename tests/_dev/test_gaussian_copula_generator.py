"""Tests for the Gaussian/t-Copula Generator."""

import pickle
import numpy as np
import pandas as pd
import pytest

from synhydro.methods.generation._dev.gaussian_copula import (
    GaussianCopulaGenerator,
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
    # Correlated base
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


class TestGaussianCopulaInit:
    def test_default_params(self):
        gen = GaussianCopulaGenerator()
        assert gen.copula_type == "gaussian"
        assert gen.marginal_method == "parametric"
        assert gen.log_transform is False
        assert gen.is_fitted is False

    def test_custom_params(self):
        gen = GaussianCopulaGenerator(
            copula_type="t",
            marginal_method="empirical",
            log_transform=True,
            offset=0.5,
        )
        assert gen.copula_type == "t"
        assert gen.marginal_method == "empirical"
        assert gen.log_transform is True
        assert gen.offset == 0.5

    def test_invalid_copula_type_raises(self):
        with pytest.raises(ValueError, match="copula_type"):
            GaussianCopulaGenerator(copula_type="vine")

    def test_invalid_marginal_method_raises(self):
        with pytest.raises(ValueError, match="marginal_method"):
            GaussianCopulaGenerator(marginal_method="kde")

    def test_stores_algorithm_params(self):
        gen = GaussianCopulaGenerator(copula_type="t")
        assert gen.init_params.algorithm_params["copula_type"] == "t"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestGaussianCopulaPreprocessing:
    def test_preprocessing_monthly_dataframe(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.preprocessing(monthly_multisite)
        assert gen.is_preprocessed
        assert gen.n_sites == 3

    def test_preprocessing_site_subset(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.preprocessing(monthly_multisite, sites=["site_A", "site_C"])
        assert gen.n_sites == 2
        assert set(gen.sites) == {"site_A", "site_C"}

    def test_preprocessing_log_transform(self, monthly_multisite):
        gen = GaussianCopulaGenerator(log_transform=True)
        gen.preprocessing(monthly_multisite)
        assert gen.is_preprocessed
        # Log-transformed values should be smaller
        assert gen._Q_monthly.values.max() < monthly_multisite.values.max()


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestGaussianCopulaFit:
    def test_fit_basic_parametric(self, monthly_multisite):
        gen = GaussianCopulaGenerator(marginal_method="parametric")
        gen.fit(monthly_multisite)
        assert gen.is_fitted
        assert len(gen._monthly_correlations) == 12
        assert len(gen._monthly_cholesky) == 12

    def test_fit_basic_empirical(self, monthly_multisite):
        gen = GaussianCopulaGenerator(marginal_method="empirical")
        gen.fit(monthly_multisite)
        assert gen.is_fitted
        assert gen._nst is not None

    def test_fit_with_q_obs(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        assert gen.is_fitted

    def test_fit_without_preprocessing_raises(self):
        gen = GaussianCopulaGenerator()
        with pytest.raises(Exception):
            gen.fit()

    def test_fit_creates_fitted_params(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        fp = gen.get_fitted_params()
        assert fp is not None
        assert fp["n_sites_"] == 3
        assert fp["n_parameters_"] > 0

    def test_fit_correlation_matrices_shape(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        for m in range(1, 13):
            corr = gen._monthly_correlations[m]
            assert corr.shape == (3, 3)
            # Symmetric
            np.testing.assert_allclose(corr, corr.T, atol=1e-10)
            # Unit diagonal
            np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_fit_correlation_matrices_psd(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        for m in range(1, 13):
            eigvals = np.linalg.eigvalsh(gen._monthly_correlations[m])
            assert np.all(eigvals > -1e-8), f"Month {m} has negative eigenvalue"

    def test_fit_marginal_params_stored(self, monthly_multisite):
        gen = GaussianCopulaGenerator(marginal_method="parametric")
        gen.fit(monthly_multisite)
        # 12 months * 3 sites = 36 entries
        assert len(gen._marginal_params) == 36
        for key, params in gen._marginal_params.items():
            assert params["dist"] in ("gamma", "lognorm")
            assert "shape" in params
            assert "scale" in params

    def test_fit_t_copula_df_estimated(self, monthly_multisite):
        gen = GaussianCopulaGenerator(copula_type="t")
        gen.fit(monthly_multisite)
        assert gen._df is not None
        assert 2.0 <= gen._df <= 50.0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class TestGaussianCopulaGeneration:
    def test_generate_basic(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=2, n_years=10, seed=42)
        assert len(ens.data_by_realization) == 2

    def test_generate_shape(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, n_years=5, seed=42)
        df = ens.data_by_realization[0]
        assert df.shape == (60, 3)  # 5 years * 12 months, 3 sites

    def test_generate_n_timesteps(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, n_timesteps=37, seed=42)
        df = ens.data_by_realization[0]
        assert len(df) == 37

    def test_generate_non_negative(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=5, n_years=20, seed=42)
        for r, df in ens.data_by_realization.items():
            assert (df.values >= 0).all(), f"Realization {r} has negative values"

    def test_generate_reproducible_with_seed(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        ens1 = gen.generate(n_realizations=1, n_years=10, seed=123)
        ens2 = gen.generate(n_realizations=1, n_years=10, seed=123)
        pd.testing.assert_frame_equal(
            ens1.data_by_realization[0], ens2.data_by_realization[0]
        )

    def test_generate_without_fit_raises(self):
        gen = GaussianCopulaGenerator()
        with pytest.raises(Exception):
            gen.generate(n_realizations=1, n_years=5)

    def test_generate_has_datetime_index(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, n_years=5, seed=42)
        df = ens.data_by_realization[0]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert pd.infer_freq(df.index) in ("MS", "<MonthBegin>")

    def test_generate_preserves_site_names(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, n_years=5, seed=42)
        df = ens.data_by_realization[0]
        assert list(df.columns) == ["site_A", "site_B", "site_C"]

    def test_generate_t_copula(self, monthly_multisite):
        gen = GaussianCopulaGenerator(copula_type="t")
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=2, n_years=10, seed=42)
        assert len(ens.data_by_realization) == 2
        for r, df in ens.data_by_realization.items():
            assert (df.values >= 0).all()

    def test_generate_empirical_marginals(self, monthly_multisite):
        gen = GaussianCopulaGenerator(marginal_method="empirical")
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=2, n_years=10, seed=42)
        assert len(ens.data_by_realization) == 2

    def test_generate_default_n_years(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, seed=42)
        df = ens.data_by_realization[0]
        # Default should match historic length
        assert len(df) == len(monthly_multisite)


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------


class TestGaussianCopulaStatisticalProperties:
    def test_marginal_means_preserved(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=50, n_years=30, seed=42)

        obs_means = monthly_multisite.mean()
        syn_means = pd.concat(
            [df.mean() for df in ens.data_by_realization.values()], axis=1
        ).mean(axis=1)
        # Ensemble mean within 30% of observed
        for site in monthly_multisite.columns:
            ratio = syn_means[site] / obs_means[site]
            assert 0.7 < ratio < 1.3, (
                f"Site {site}: obs={obs_means[site]:.1f}, " f"syn={syn_means[site]:.1f}"
            )

    def test_spatial_correlation_preserved(self, monthly_multisite):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=30, n_years=30, seed=42)

        obs_corr = monthly_multisite.corr()
        # Average synthetic correlation across realizations
        syn_corrs = [df.corr() for df in ens.data_by_realization.values()]
        avg_syn_corr = sum(syn_corrs) / len(syn_corrs)

        # Off-diagonal correlations within 0.3 of observed
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
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_single_site)
        ens = gen.generate(n_realizations=3, n_years=10, seed=42)
        df = ens.data_by_realization[0]
        assert df.shape[1] == 1
        assert (df.values >= 0).all()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestGaussianCopulaSerialization:
    def test_pickle_save_load(self, monthly_multisite, tmp_path):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)

        filepath = tmp_path / "copula_gen.pkl"
        gen.save(str(filepath))

        loaded = GaussianCopulaGenerator.load(str(filepath))
        assert loaded.is_fitted
        assert loaded.copula_type == "gaussian"
        assert loaded.marginal_method == "parametric"
        assert len(loaded._monthly_correlations) == 12

    def test_pickle_generate_after_load(self, monthly_multisite, tmp_path):
        gen = GaussianCopulaGenerator()
        gen.fit(monthly_multisite)

        filepath = tmp_path / "copula_gen.pkl"
        gen.save(str(filepath))
        loaded = GaussianCopulaGenerator.load(str(filepath))

        ens = loaded.generate(n_realizations=1, n_years=5, seed=42)
        assert len(ens.data_by_realization) == 1
        assert ens.data_by_realization[0].shape == (60, 3)

    def test_pickle_t_copula_roundtrip(self, monthly_multisite, tmp_path):
        gen = GaussianCopulaGenerator(copula_type="t")
        gen.fit(monthly_multisite)

        filepath = tmp_path / "t_copula_gen.pkl"
        gen.save(str(filepath))
        loaded = GaussianCopulaGenerator.load(str(filepath))

        assert loaded.copula_type == "t"
        assert loaded._df is not None
        ens = loaded.generate(n_realizations=1, n_years=5, seed=42)
        assert len(ens.data_by_realization) == 1
