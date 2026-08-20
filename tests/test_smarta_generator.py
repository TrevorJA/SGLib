"""Tests for the SMARTAGenerator."""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble
from synhydro.methods.generation.parametric.smarta import SMARTAGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def annual_multisite():
    """Synthetic annual data: 80 years, 3 correlated sites."""
    rng = np.random.default_rng(42)
    n_years = 80
    dates = pd.date_range("1940-01-01", periods=n_years, freq="YS")

    # Correlated gamma-distributed flows
    z = rng.multivariate_normal(
        [0, 0, 0],
        [[1, 0.7, 0.5], [0.7, 1, 0.6], [0.5, 0.6, 1]],
        size=n_years,
    )
    from scipy.stats import gamma as gamma_dist, norm

    u = norm.cdf(z)
    site_a = gamma_dist.ppf(u[:, 0], a=3, scale=100)
    site_b = gamma_dist.ppf(u[:, 1], a=5, scale=80)
    site_c = gamma_dist.ppf(u[:, 2], a=2, scale=150)

    return pd.DataFrame(
        {"siteA": site_a, "siteB": site_b, "siteC": site_c},
        index=dates,
    )


@pytest.fixture
def annual_single_site(annual_multisite):
    """Single-site annual data."""
    return annual_multisite[["siteA"]]


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestSMARTAInit:
    def test_default_params(self):
        gen = SMARTAGenerator()
        assert gen.acf_model == "cas"
        assert gen.sma_order == 512
        assert gen.nataf_method == "GH"

    def test_custom_sma_order(self):
        gen = SMARTAGenerator(sma_order=64)
        assert gen.sma_order == 64

    def test_stores_kwargs(self):
        gen = SMARTAGenerator(nataf_method="MC", nataf_n_eval=11)
        assert gen.nataf_method == "MC"
        assert gen.nataf_n_eval == 11


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestSMARTAPreprocessing:
    def test_preprocessed_flag(self, annual_multisite):
        gen = SMARTAGenerator()
        gen.preprocessing(annual_multisite)
        assert gen.is_preprocessed

    def test_sites_stored(self, annual_multisite):
        gen = SMARTAGenerator()
        gen.preprocessing(annual_multisite)
        assert gen._n_sites == 3
        assert list(gen._sites) == ["siteA", "siteB", "siteC"]

    def test_annual_data_shape(self, annual_multisite):
        gen = SMARTAGenerator()
        gen.preprocessing(annual_multisite)
        assert gen._Q_annual.shape == (80, 3)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestSMARTAFit:
    def test_fitted_flag(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        assert gen.is_fitted

    def test_marginal_params_populated(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        assert len(gen._marginal_params) == 3
        for s_idx in range(3):
            assert "dist" in gen._marginal_params[s_idx]

    def test_sma_weights_shape(self, annual_multisite):
        q = 64
        gen = SMARTAGenerator(sma_order=q)
        gen.fit(annual_multisite)
        assert len(gen._sma_weights) == 3
        for w in gen._sma_weights:
            assert len(w) == 2 * q + 1

    def test_b_tilde_shape(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        assert gen._B_tilde.shape == (3, 3)

    def test_cas_params_stored(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        assert len(gen._cas_params) == 3

    def test_single_site_fit(self, annual_single_site):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_single_site)
        assert gen.is_fitted
        assert gen._B_tilde.shape == (1, 1)

    def test_fitted_params_returned(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        fp = gen._compute_fitted_params()
        assert fp.n_sites_ == 3
        assert fp.sample_size_ == 80


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


class TestSMARTAGenerate:
    def test_generate_shape(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=2, n_years=50)
        assert isinstance(ens, Ensemble)
        assert len(ens.data_by_realization) == 2
        df = ens.data_by_realization[0]
        assert df.shape == (50, 3)

    def test_generate_default_length(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=1)
        df = ens.data_by_realization[0]
        assert df.shape[0] == 80  # matches observed length

    def test_seed_reproducibility(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens1 = gen.generate(n_realizations=1, n_years=30, seed=123)
        ens2 = gen.generate(n_realizations=1, n_years=30, seed=123)
        pd.testing.assert_frame_equal(
            ens1.data_by_realization[0],
            ens2.data_by_realization[0],
        )

    def test_different_seeds_differ(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens1 = gen.generate(n_realizations=1, n_years=30, seed=1)
        ens2 = gen.generate(n_realizations=1, n_years=30, seed=2)
        assert not np.allclose(
            ens1.data_by_realization[0].values,
            ens2.data_by_realization[0].values,
        )

    def test_output_has_correct_columns(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=1, n_years=20)
        df = ens.data_by_realization[0]
        assert list(df.columns) == ["siteA", "siteB", "siteC"]

    def test_output_has_datetime_index(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=1, n_years=20)
        df = ens.data_by_realization[0]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_single_site_generate(self, annual_single_site):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_single_site)
        ens = gen.generate(n_realizations=1, n_years=30)
        df = ens.data_by_realization[0]
        assert df.shape == (30, 1)

    def test_positive_values(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=1, n_years=100, seed=42)
        df = ens.data_by_realization[0]
        # Most values should be positive (gamma/lognorm marginals)
        assert (df.values > 0).mean() > 0.95


# ---------------------------------------------------------------------------
# State validation
# ---------------------------------------------------------------------------


class TestSMARTAStateValidation:
    def test_generate_before_fit_raises(self, annual_multisite):
        gen = SMARTAGenerator()
        gen.preprocessing(annual_multisite)
        with pytest.raises(Exception):
            gen.generate()

    def test_fit_auto_preprocesses(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        assert gen.is_preprocessed
        assert gen.is_fitted


# ---------------------------------------------------------------------------
# Long runs (beyond the pandas nanosecond year-2262 limit)
# ---------------------------------------------------------------------------


class TestSMARTALongRuns:
    def test_generate_1000_years(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=1, n_years=1000, seed=0)
        df = ens.data_by_realization[0]
        assert df.shape == (1000, 3)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0].year == 1940
        assert df.index[-1].year == 1940 + 999
        assert df.index.is_monotonic_increasing
        assert len(set(df.index.year)) == 1000
        assert np.all(np.isfinite(df.values))
        assert ens.metadata.time_period == ("1940-01-01", "2939-01-01")

    def test_long_run_to_hdf5(self, annual_multisite, tmp_path):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        ens = gen.generate(n_realizations=2, n_years=1200, seed=0)
        fn = tmp_path / "smarta_long.h5"
        ens.to_hdf5(str(fn))
        assert fn.exists()

    def test_short_run_index_values_unchanged(self, annual_multisite):
        """Calendar values match the observed index; dtype is datetime64[s]."""
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        df = gen.generate(n_years=80, seed=0).data_by_realization[0]
        assert (df.index.values == annual_multisite.index.values).all()
        assert df.index.dtype == np.dtype("datetime64[s]")

    def test_long_index_matches_date_range_values(self, annual_multisite):
        gen = SMARTAGenerator(sma_order=64)
        gen.fit(annual_multisite)
        long_idx = gen.generate(n_years=1000, seed=0).data_by_realization[0].index
        short_ref = pd.date_range("1940-01-01", periods=300, freq="YS")
        assert (long_idx[:300].values == short_ref.values).all()
        assert long_idx[-1].year == 1940 + 999


# ---------------------------------------------------------------------------
# Hurst fallback warning
# ---------------------------------------------------------------------------


class TestSMARTAHurstFallback:
    def test_warns_when_beta_not_lrd(self, annual_multisite, caplog):
        import logging

        gen = SMARTAGenerator(sma_order=64, acf_model="hurst")
        with caplog.at_level(logging.WARNING):
            gen.fit(annual_multisite)
        # i.i.d. gamma input has no LRD, so the CAS fit gives beta <= 1
        # for at least one site and the H=0.6 fallback must be announced.
        fallback_sites = [i for i, (H, _) in gen._cas_params.items() if H == 0.6]
        assert fallback_sites
        assert any(
            "Falling back to the default H=0.6" in r.message for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Non-PD innovation covariance repair
# ---------------------------------------------------------------------------


class TestSMARTAInnovationRepair:
    def test_repair_preserves_diagonal_of_g_tilde(
        self, annual_multisite, monkeypatch, caplog
    ):
        """The G_tilde fallback must not rescale the diagonal to one.

        Forces the repair path by making the first Cholesky call fail and
        scales the SMA weights so that diag(G_tilde) = 1 / sum(a^2) != 1.
        The old repair_correlation_matrix() call would have returned a unit
        diagonal; the diagonal of B B^T must equal diag(G_tilde) instead.
        """
        import logging

        import synhydro.methods.generation.parametric.smarta as smarta_mod

        orig_chol = np.linalg.cholesky
        calls = {"n": 0}

        def failing_cholesky(x):
            calls["n"] += 1
            if calls["n"] == 1:
                raise np.linalg.LinAlgError("forced")
            return orig_chol(x)

        orig_sma = smarta_mod.sma_weights_fft
        monkeypatch.setattr(
            smarta_mod, "sma_weights_fft", lambda acf: orig_sma(acf) * 1.1
        )
        monkeypatch.setattr(np.linalg, "cholesky", failing_cholesky)

        gen = SMARTAGenerator(sma_order=64)
        captured = {}
        orig_repair = gen._repair_innovation_covariance

        def capturing_repair(G):
            captured["G"] = G.copy()
            return orig_repair(G)

        gen._repair_innovation_covariance = capturing_repair

        with caplog.at_level(logging.WARNING):
            gen.fit(annual_multisite)

        assert calls["n"] == 2
        assert any("not positive-definite" in r.message for r in caplog.records)
        G = captured["G"]
        expected_diag = np.full(3, 1.0 / 1.1**2)
        np.testing.assert_allclose(np.diag(G), expected_diag, rtol=1e-10)
        BBt = gen._B_tilde @ gen._B_tilde.T
        np.testing.assert_allclose(np.diag(BBt), np.diag(G), rtol=1e-10)
        # G_tilde was PD to begin with, so the repair must be a no-op
        np.testing.assert_allclose(BBt, G, atol=1e-12)

    def test_repair_fixes_indefinite_matrix_and_keeps_diagonal(self):
        gen = SMARTAGenerator()
        G = np.array([[0.8, 0.9, 0.9], [0.9, 0.8, -0.9], [0.9, -0.9, 0.8]])
        assert np.linalg.eigvalsh(G).min() < 0
        G_rep = gen._repair_innovation_covariance(G)
        np.testing.assert_allclose(np.diag(G_rep), np.diag(G), rtol=1e-12)
        np.linalg.cholesky(G_rep)

