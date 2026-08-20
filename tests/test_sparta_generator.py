"""Tests for the SPARTAGenerator."""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble
from synhydro.methods.generation.parametric.sparta import SPARTAGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monthly_multisite():
    """Synthetic monthly data: 30 years, 3 correlated sites with seasonality."""
    rng = np.random.default_rng(42)
    n_years = 30
    n_months = n_years * 12
    dates = pd.date_range("1990-01-01", periods=n_months, freq="MS")

    from scipy.stats import gamma as gamma_dist, norm

    # Generate correlated Gaussian with month-to-month persistence
    z = np.zeros((n_months, 3))
    z[0] = rng.standard_normal(3)
    corr_mat = np.array([[1, 0.6, 0.4], [0.6, 1, 0.5], [0.4, 0.5, 1]])
    L = np.linalg.cholesky(corr_mat)
    for t in range(1, n_months):
        w = L @ rng.standard_normal(3)
        z[t] = 0.5 * z[t - 1] + np.sqrt(1 - 0.25) * w

    # Map to gamma with seasonal means
    months = dates.month
    data = {}
    for s_idx, (site, shape, base_scale) in enumerate(
        [("siteA", 3, 100), ("siteB", 5, 80), ("siteC", 2, 150)]
    ):
        # Season-varying scale
        seasonal_scale = base_scale * (1 + 0.5 * np.sin(2 * np.pi * (months - 4) / 12))
        u = norm.cdf(z[:, s_idx])
        data[site] = gamma_dist.ppf(u, a=shape, scale=seasonal_scale)

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def monthly_single_site(monthly_multisite):
    """Single-site monthly data."""
    return monthly_multisite[["siteA"]]


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestSPARTAInit:
    def test_default_params(self):
        gen = SPARTAGenerator()
        assert gen.nataf_method == "GH"
        assert gen.nataf_poly_deg == 8

    def test_custom_params(self):
        gen = SPARTAGenerator(nataf_method="MC", nataf_n_eval=11)
        assert gen.nataf_method == "MC"
        assert gen.nataf_n_eval == 11


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestSPARTAPreprocessing:
    def test_preprocessed_flag(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.preprocessing(monthly_multisite)
        assert gen.is_preprocessed

    def test_sites_stored(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.preprocessing(monthly_multisite)
        assert gen._n_sites == 3
        assert list(gen._sites) == ["siteA", "siteB", "siteC"]

    def test_monthly_data_shape(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.preprocessing(monthly_multisite)
        assert gen._Q_monthly.shape == (360, 3)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestSPARTAFit:
    def test_fitted_flag(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        assert gen.is_fitted

    def test_marginal_params_populated(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        # 12 months * 1 site = 12 entries
        assert len(gen._marginal_params) == 12
        for m in range(1, 13):
            assert (m, 0) in gen._marginal_params

    def test_equiv_auto_shape(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        assert gen._equiv_auto.shape == (1, 12)

    def test_multisite_fit(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.fit(monthly_multisite)
        assert gen.is_fitted
        assert gen._equiv_auto.shape == (3, 12)
        assert len(gen._equiv_cross) == 12
        assert len(gen._A_s) == 12
        assert len(gen._B_s) == 12

    def test_fitted_params_returned(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        fp = gen._compute_fitted_params()
        assert fp.n_sites_ == 1
        assert fp.sample_size_ == 360


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


class TestSPARTAGenerate:
    def test_generate_shape_univariate(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens = gen.generate(n_realizations=2, n_years=20)
        assert isinstance(ens, Ensemble)
        assert len(ens.data_by_realization) == 2
        df = ens.data_by_realization[0]
        assert df.shape == (240, 1)

    def test_generate_shape_multivariate(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, n_years=10)
        df = ens.data_by_realization[0]
        assert df.shape == (120, 3)

    def test_generate_default_length(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens = gen.generate(n_realizations=1)
        df = ens.data_by_realization[0]
        assert df.shape[0] == 360  # matches observed

    def test_seed_reproducibility(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens1 = gen.generate(n_realizations=1, n_years=10, seed=123)
        ens2 = gen.generate(n_realizations=1, n_years=10, seed=123)
        pd.testing.assert_frame_equal(
            ens1.data_by_realization[0],
            ens2.data_by_realization[0],
        )

    def test_different_seeds_differ(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens1 = gen.generate(n_realizations=1, n_years=10, seed=1)
        ens2 = gen.generate(n_realizations=1, n_years=10, seed=2)
        assert not np.allclose(
            ens1.data_by_realization[0].values,
            ens2.data_by_realization[0].values,
        )

    def test_output_has_datetime_index(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens = gen.generate(n_realizations=1, n_years=5)
        df = ens.data_by_realization[0]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_output_columns(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=1, n_years=5)
        df = ens.data_by_realization[0]
        assert list(df.columns) == ["siteA", "siteB", "siteC"]

    def test_positive_values(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens = gen.generate(n_realizations=1, n_years=50, seed=42)
        df = ens.data_by_realization[0]
        assert (df.values > 0).mean() > 0.95

    def test_n_timesteps(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        ens = gen.generate(n_realizations=1, n_timesteps=100)
        df = ens.data_by_realization[0]
        assert df.shape[0] == 100


# ---------------------------------------------------------------------------
# State validation
# ---------------------------------------------------------------------------


class TestSPARTAStateValidation:
    def test_generate_before_fit_raises(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.preprocessing(monthly_single_site)
        with pytest.raises(Exception):
            gen.generate()

    def test_fit_auto_preprocesses(self, monthly_single_site):
        gen = SPARTAGenerator()
        gen.fit(monthly_single_site)
        assert gen.is_preprocessed
        assert gen.is_fitted


# ---------------------------------------------------------------------------
# Statistical preservation
# ---------------------------------------------------------------------------


class TestSPARTAStatistics:
    def test_innovation_covariance_diagonal(self, monthly_multisite):
        """diag(B_s B_s^T) must equal 1 - r_s^2 for every season, including
        seasons where G_s had to be repaired."""
        gen = SPARTAGenerator()
        gen.fit(monthly_multisite)
        for m_idx in range(12):
            r = np.diag(gen._A_s[m_idx])
            d = np.diag(gen._B_s[m_idx] @ gen._B_s[m_idx].T)
            assert np.allclose(d, 1.0 - r**2, atol=0.05), (m_idx + 1, d, 1 - r**2)

    def test_repair_preserves_diagonal_on_indefinite_g(self):
        """Force the repair path with an indefinite G_s and check the scale."""
        gen = SPARTAGenerator()
        G = np.array([[0.5, 0.6], [0.6, 0.5]])
        rep = gen._repair_innovation_covariance(G)
        np.linalg.cholesky(rep)
        assert np.allclose(np.diag(rep), 0.5, atol=0.11)
        gen_n = SPARTAGenerator(matrix_repair_method="nearest")
        rep_n = gen_n._repair_innovation_covariance(G)
        assert np.allclose(np.diag(rep_n), 0.5)

    def test_gaussian_variance_unit(self, monthly_multisite):
        """Auxiliary Gaussian process must have unit variance per month."""
        gen = SPARTAGenerator()
        gen.fit(monthly_multisite)
        n_years = 400
        rng = np.random.default_rng(0)
        W = rng.standard_normal((n_years * 12, gen._n_sites))
        Z = np.empty_like(W)
        for t in range(n_years * 12):
            m_idx = t % 12
            prev = Z[t - 1] if t > 0 else np.zeros(gen._n_sites)
            Z[t] = gen._A_s[m_idx] @ prev + gen._B_s[m_idx] @ W[t]
        var = Z[120:].reshape(-1, 12, gen._n_sites).var(axis=0)
        # Eigenvalue clipping of an indefinite G_s perturbs the variance
        # slightly; the old unit-diagonal rescaling gave Var(z) ~ 1.4-1.5.
        assert np.all(np.abs(var - 1.0) < 0.2), var

    def test_monthly_std_preserved(self, monthly_multisite):
        gen = SPARTAGenerator()
        gen.fit(monthly_multisite)
        ens = gen.generate(n_realizations=20, n_years=30, seed=0)
        syn = pd.concat([ens.data_by_realization[i] for i in range(20)])
        obs = gen._Q_monthly
        ratio = syn.groupby(syn.index.month).std() / obs.groupby(obs.index.month).std()
        assert ((ratio > 0.7) & (ratio < 1.3)).all().all(), ratio


# ---------------------------------------------------------------------------
# Non-January start
# ---------------------------------------------------------------------------


class TestSPARTAStartMonth:
    def test_season_corr_independent_of_start_month(self, monthly_multisite):
        """Calendar-year matrix gives identical correlations for Jan and Apr
        starts once partial years are trimmed."""
        Q_jan = monthly_multisite
        Q_apr = monthly_multisite.loc["1990-04-01":]
        gen = SPARTAGenerator()
        mat_jan = gen._calendar_year_matrix(Q_jan)["siteA"][1:]
        mat_apr = gen._calendar_year_matrix(Q_apr)["siteA"]
        assert mat_apr.shape == (29, 12)
        np.testing.assert_allclose(mat_jan, mat_apr)
        r_jan = SPARTAGenerator._season_to_season_corr(mat_jan)
        r_apr = SPARTAGenerator._season_to_season_corr(mat_apr)
        np.testing.assert_allclose(r_jan, r_apr)

    def test_partial_years_warn(self, monthly_multisite, caplog):
        import logging

        gen = SPARTAGenerator()
        with caplog.at_level(logging.WARNING):
            gen.fit(monthly_multisite.loc["1990-04-01":])
        assert any("partial calendar year" in r.message for r in caplog.records)

    def test_april_start_fit_and_generate(self, monthly_multisite):
        """Output starts in January and monthly means by index.month match."""
        Q_apr = monthly_multisite.loc["1990-04-01":]
        gen = SPARTAGenerator()
        gen.fit(Q_apr)
        ens = gen.generate(n_realizations=10, n_years=30, seed=3)
        df0 = ens.data_by_realization[0]
        assert df0.index[0].month == 1
        assert df0.index[0].year == 1990
        syn = pd.concat([ens.data_by_realization[i] for i in range(10)])
        ratio = syn.groupby(syn.index.month).mean() / Q_apr.groupby(
            Q_apr.index.month
        ).mean()
        assert ((ratio > 0.85) & (ratio < 1.15)).all().all(), ratio

    def test_too_few_complete_years_raises(self, monthly_multisite):
        gen = SPARTAGenerator()
        with pytest.raises(ValueError, match="complete calendar years"):
            gen.fit(monthly_multisite.loc["1990-04-01":"1991-10-01"])


# ---------------------------------------------------------------------------
# Zero handling in marginal fitting
# ---------------------------------------------------------------------------


class TestSPARTAZeroFlows:
    def test_zeros_excluded_from_marginal_fit(self, monthly_single_site):
        """A month with ~30% zeros fits the same marginal as its positive values."""
        from scipy.stats import gamma as gamma_dist, lognorm

        Q = monthly_single_site.copy()
        jul = Q.index.month == 7
        jul_idx = np.where(jul)[0]
        rng = np.random.default_rng(7)
        n_zero = int(round(0.3 * len(jul_idx)))
        zero_idx = rng.choice(jul_idx, size=n_zero, replace=False)
        Q.iloc[zero_idx, 0] = 0.0
        positive = Q.loc[jul, "siteA"].values
        positive = positive[positive > 0]
        assert 0 < len(positive) < jul.sum()

        gen = SPARTAGenerator()
        gen.fit(Q)
        params = gen._marginal_params[(7, 0)]

        # Reference fit on the positive values only
        if params["dist"] == "gamma":
            shape, _, scale = gamma_dist.fit(positive, floc=0)
            assert params["shape"] == pytest.approx(shape, rel=1e-6)
            assert params["scale"] == pytest.approx(scale, rel=1e-6)
            ref = gamma_dist(a=shape, loc=0, scale=scale)
        else:
            s, _, scale = lognorm.fit(positive, floc=0)
            assert params["s"] == pytest.approx(s, rel=1e-6)
            assert params["scale"] == pytest.approx(scale, rel=1e-6)
            ref = lognorm(s=s, loc=0, scale=scale)

        # Simulated July statistics should match the positive-only marginal
        ens = gen.generate(n_realizations=20, n_years=30, seed=1)
        sim = np.concatenate(
            [df.loc[df.index.month == 7, "siteA"].values for df in ens.data_by_realization.values()]
        )
        assert np.all(sim > 0)
        assert np.median(sim) == pytest.approx(ref.median(), rel=0.1)
        assert np.std(sim) == pytest.approx(ref.std(), rel=0.15)
        # Fitting on the clipped (1e-6) values instead would shrink the median
        # and inflate the spread; guard against that regression explicitly.
        assert np.median(sim) > 0.5 * np.median(positive)

    def test_shape_not_distorted_by_zeros(self, monthly_single_site):
        """Gamma shape for a zero-inflated month stays near the perennial fit."""
        Q = monthly_single_site.copy()
        gen_ref = SPARTAGenerator()
        gen_ref.fit(Q)
        ref = gen_ref._marginal_params[(7, 0)]

        jul_idx = np.where(Q.index.month == 7)[0]
        Q.iloc[jul_idx[::3], 0] = 0.0
        gen = SPARTAGenerator()
        gen.fit(Q)
        got = gen._marginal_params[(7, 0)]

        assert got["dist"] == ref["dist"]
        key = "shape" if got["dist"] == "gamma" else "s"
        assert got[key] == pytest.approx(ref[key], rel=0.35)
