"""
Tests for MultisitePhaseRandomizationGenerator (Brunner and Gilleland, 2020).
"""

import time

import pytest
import numpy as np
import pandas as pd

from synhydro.methods.generation.hybrid.multisite_phase_randomization import (
    MultisitePhaseRandomizationGenerator,
)
from synhydro.core.ensemble import Ensemble


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_multisite_df(n_years: int, n_sites: int, seed: int = 42) -> pd.DataFrame:
    """Build a no-leap daily multisite DataFrame of exact length n_years*365."""
    rng = np.random.default_rng(seed)
    n_days = n_years * 365
    # Build index without Feb 29
    index = MultisitePhaseRandomizationGenerator._build_noleap_index(
        n_days, start_year=2000
    )
    data = {}
    for i in range(n_sites):
        t = np.arange(n_days, dtype=float)
        seasonal = 100.0 + 50.0 * np.sin(2 * np.pi * t / 365 + i * 0.3)
        noise = rng.lognormal(mean=0.0, sigma=0.4, size=n_days) * 20.0
        # Introduce inter-site correlation
        common = rng.lognormal(mean=0.0, sigma=0.3, size=n_days) * 30.0
        data[f"site_{i + 1}"] = np.maximum(seasonal + noise + common, 1.0)
    return pd.DataFrame(data, index=index)


@pytest.fixture
def df_2sites_10yr():
    """Two-site DataFrame, 10 years (3650 days), no leap days."""
    return _make_multisite_df(n_years=10, n_sites=2, seed=42)


@pytest.fixture
def df_3sites_10yr():
    """Three-site DataFrame, 10 years (3650 days), no leap days."""
    return _make_multisite_df(n_years=10, n_sites=3, seed=7)


@pytest.fixture
def df_2sites_3yr():
    """Two-site DataFrame, 3 years (1095 days). Minimum viable length."""
    return _make_multisite_df(n_years=3, n_sites=2, seed=99)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_params(self):
        gen = MultisitePhaseRandomizationGenerator()
        assert gen.wavelet == "cmor1.5-1.0"
        assert gen.n_scales == 100
        assert gen.win_h_length == 15
        assert gen.transform == "mean_center"
        assert gen.is_preprocessed is False
        assert gen.is_fitted is False
        assert gen.supports_multisite is True
        assert gen.output_frequency == "D"
        assert "D" in gen.supported_frequencies

    def test_custom_params(self):
        gen = MultisitePhaseRandomizationGenerator(
            wavelet="cmor1.5-1.0", n_scales=50, win_h_length=10
        )
        assert gen.n_scales == 50
        assert gen.win_h_length == 10

    def test_transform_normal_score_accepted(self):
        gen = MultisitePhaseRandomizationGenerator(transform="normal_score")
        assert gen.transform == "normal_score"

    def test_invalid_transform_raises(self):
        with pytest.raises(ValueError, match="transform must be one of"):
            MultisitePhaseRandomizationGenerator(transform="bad_option")

    def test_name_stored(self):
        gen = MultisitePhaseRandomizationGenerator(name="my_gen")
        assert "my_gen" in gen.name


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def test_basic_preprocessing(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator()
        gen.preprocessing(df_2sites_10yr)

        assert gen.is_preprocessed is True
        assert hasattr(gen, "Q_obs_df_")
        assert hasattr(gen, "day_index_")
        assert hasattr(gen, "n_years_")
        assert gen.n_years_ == 10

    def test_leap_days_removed(self):
        # Build data that includes Feb 29
        dates = pd.date_range("2000-01-01", "2003-12-31", freq="D")
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "site_1": rng.lognormal(size=len(dates)) * 100 + 50,
                "site_2": rng.lognormal(size=len(dates)) * 100 + 50,
            },
            index=dates,
        )
        gen = MultisitePhaseRandomizationGenerator()
        gen.preprocessing(df)

        assert len(gen.Q_obs_df_) % 365 == 0
        assert not any((gen.Q_obs_index_.month == 2) & (gen.Q_obs_index_.day == 29))

    def test_day_index_range(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator()
        gen.preprocessing(df_2sites_10yr)

        assert gen.day_index_.min() >= 1
        assert gen.day_index_.max() <= 365

    def test_raises_on_too_short(self):
        index = MultisitePhaseRandomizationGenerator._build_noleap_index(365)
        df = pd.DataFrame(
            {"site_1": np.ones(365) * 50, "site_2": np.ones(365) * 60},
            index=index,
        )
        gen = MultisitePhaseRandomizationGenerator()
        with pytest.raises(ValueError, match="730 days"):
            gen.preprocessing(df)

    def test_sites_attribute_set(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator()
        gen.preprocessing(df_2sites_10yr)

        assert gen.n_sites == 2
        assert "site_1" in gen.sites
        assert "site_2" in gen.sites


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_lifecycle(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator()
        gen.fit(df_2sites_10yr)

        assert gen.is_preprocessed is True
        assert gen.is_fitted is True

    def test_fitted_attributes_exist(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator()
        gen.fit(df_2sites_10yr)

        assert hasattr(gen, "par_day_")
        assert hasattr(gen, "cwt_amplitudes_")
        assert hasattr(gen, "norm_")
        assert hasattr(gen, "scales_")
        assert hasattr(gen, "delta_j_")

    def test_cwt_amplitudes_shape(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=50)
        gen.fit(df_2sites_10yr)

        N = 10 * 365
        for site in gen.sites:
            amp = gen.cwt_amplitudes_[site]
            assert amp.shape == (50, N), f"Expected ({50}, {N}), got {amp.shape}"
            assert np.all(amp >= 0)

    def test_kappa_params_structure(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator()
        gen.fit(df_2sites_10yr)

        for site in gen.sites:
            assert site in gen.par_day_
            fitted_days = [d for d, v in gen.par_day_[site].items() if v is not None]
            assert len(fitted_days) > 0
            d0 = fitted_days[0]
            params = gen.par_day_[site][d0]
            for key in ("xi", "alfa", "k", "h"):
                assert key in params

    def test_scales_shape(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=80)
        gen.fit(df_2sites_10yr)

        assert gen.scales_.shape == (80,)
        assert gen.delta_j_ > 0

    def test_fit_without_preprocessing_raises(self):
        gen = MultisitePhaseRandomizationGenerator()
        with pytest.raises(Exception):
            gen.fit()

    def test_fitted_params_metadata(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator()
        gen.fit(df_2sites_10yr)

        fp = gen.fitted_params_
        assert fp.n_sites_ == 2
        assert fp.sample_size_ == 10 * 365

    def test_fit_three_sites(self, df_3sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_3sites_10yr)

        assert gen.n_sites == 3
        assert len(gen.cwt_amplitudes_) == 3

    def test_fit_mean_center_stores_obs_mean(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40, transform="mean_center")
        gen.fit(df_2sites_10yr)

        for site in gen.sites:
            assert site in gen.obs_mean_
            assert abs(gen.obs_mean_[site] - df_2sites_10yr[site].mean()) < 1e-6

    def test_fit_normal_score_transform(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(
            n_scales=40, transform="normal_score"
        )
        gen.fit(df_2sites_10yr)

        assert gen.is_fitted
        assert gen.obs_mean_ == {}
        for site in gen.sites:
            assert site in gen.par_day_
            fitted_days = [d for d, v in gen.par_day_[site].items() if v is not None]
            assert len(fitted_days) > 0

    def test_fit_completes_under_budget_4sites_10yr(self):
        """
        Regression test for the daily-scale hang: fit() on 4 sites x 10 years
        at default settings must complete well under a minute. Historically the
        kappa-fit Nelder-Mead loop would stall >9 minutes before warm-starting,
        maxiter=200, and the smooth infeasibility penalty were added.
        """
        df = _make_multisite_df(n_years=10, n_sites=4, seed=2024)
        gen = MultisitePhaseRandomizationGenerator()

        t0 = time.perf_counter()
        gen.fit(df)
        elapsed = time.perf_counter() - t0

        assert gen.is_fitted
        assert gen.n_sites == 4
        for site in gen.sites:
            fitted_days = sum(1 for v in gen.par_day_[site].values() if v is not None)
            assert fitted_days >= 350, f"site {site}: only {fitted_days}/365 days fit"
        assert (
            elapsed < 30.0
        ), f"fit() took {elapsed:.1f}s; expected <30s on 4 sites x 10 years"


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_ensemble(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)
        result = gen.generate(n_realizations=1, seed=0)

        assert isinstance(result, Ensemble)

    def test_output_shape_default_length(self, df_2sites_10yr):
        """Default output length equals the observed record."""
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)
        result = gen.generate(n_realizations=2, seed=0)

        assert len(result.realization_ids) == 2
        expected_len = 10 * 365
        for r in result.realization_ids:
            df = result.data_by_realization[r]
            assert df.shape == (expected_len, 2), f"Got {df.shape}"

    def test_output_shape_n_years(self, df_2sites_10yr):
        """n_years controls output length correctly."""
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)
        result = gen.generate(n_realizations=2, n_years=5, seed=1)

        expected_len = 5 * 365
        for r in result.realization_ids:
            df = result.data_by_realization[r]
            assert df.shape == (expected_len, 2)

    def test_output_shape_three_sites(self, df_3sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_3sites_10yr)
        result = gen.generate(n_realizations=1, seed=5)

        df = result.data_by_realization[0]
        assert df.shape[1] == 3
        assert list(df.columns) == ["site_1", "site_2", "site_3"]

    def test_non_negativity(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)
        result = gen.generate(n_realizations=5, seed=42)

        for r in result.realization_ids:
            vals = result.data_by_realization[r].values
            assert np.all(vals >= 0), "Negative values found in realization"

    def test_reproducibility(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)

        r1 = gen.generate(n_realizations=1, seed=77)
        r2 = gen.generate(n_realizations=1, seed=77)

        np.testing.assert_array_almost_equal(
            r1.data_by_realization[0].values,
            r2.data_by_realization[0].values,
        )

    def test_different_seeds_differ(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)

        r1 = gen.generate(n_realizations=1, seed=1)
        r2 = gen.generate(n_realizations=1, seed=2)

        assert not np.allclose(
            r1.data_by_realization[0].values,
            r2.data_by_realization[0].values,
        )

    def test_no_leapdays_in_output(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)
        result = gen.generate(n_realizations=1, n_years=8, seed=0)

        idx = result.data_by_realization[0].index
        feb29 = idx[(idx.month == 2) & (idx.day == 29)]
        assert len(feb29) == 0

    def test_no_nans(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)
        result = gen.generate(n_realizations=3, seed=0)

        for r in result.realization_ids:
            assert not result.data_by_realization[r].isna().any().any()

    def test_generate_without_fit_raises(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator()
        gen.preprocessing(df_2sites_10yr)
        with pytest.raises(Exception):
            gen.generate(n_realizations=1)

    def test_n_years_longer_than_obs(self, df_2sites_10yr):
        """n_years greater than observed record triggers chunking."""
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)
        result = gen.generate(n_realizations=1, n_years=25, seed=3)

        df = result.data_by_realization[0]
        assert df.shape[0] == 25 * 365


# ---------------------------------------------------------------------------
# Spatial correlation preservation
# ---------------------------------------------------------------------------


class TestSpatialCorrelation:
    def test_spatial_correlation_preserved(self, df_2sites_10yr):
        """Ensemble-mean pairwise correlation should be close to observed."""
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)

        # Observed correlation
        obs_corr = df_2sites_10yr["site_1"].corr(df_2sites_10yr["site_2"])

        # Synthetic ensemble (many realizations for stable mean)
        result = gen.generate(n_realizations=20, seed=0)
        syn_corrs = []
        for r in result.realization_ids:
            df_r = result.data_by_realization[r]
            syn_corrs.append(df_r["site_1"].corr(df_r["site_2"]))

        mean_syn_corr = np.mean(syn_corrs)
        # Allow up to 0.25 absolute difference
        assert (
            abs(mean_syn_corr - obs_corr) < 0.25
        ), f"Observed corr={obs_corr:.3f}, mean synthetic corr={mean_syn_corr:.3f}"

    def test_spatial_correlation_sign_preserved(self, df_3sites_10yr):
        """Sign of pairwise correlations should be preserved on average."""
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_3sites_10yr)

        result = gen.generate(n_realizations=10, seed=0)

        sites = df_3sites_10yr.columns.tolist()
        for i in range(len(sites)):
            for j in range(i + 1, len(sites)):
                obs_corr = df_3sites_10yr[sites[i]].corr(df_3sites_10yr[sites[j]])
                syn_corrs = [
                    result.data_by_realization[r][sites[i]].corr(
                        result.data_by_realization[r][sites[j]]
                    )
                    for r in result.realization_ids
                ]
                mean_syn = np.mean(syn_corrs)
                assert np.sign(mean_syn) == np.sign(obs_corr) or abs(obs_corr) < 0.05


# ---------------------------------------------------------------------------
# Statistical moment preservation
# ---------------------------------------------------------------------------


class TestStatisticalMoments:
    def test_mean_within_tolerance(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)
        result = gen.generate(n_realizations=20, seed=0)

        for site in gen.sites:
            obs_mean = df_2sites_10yr[site].mean()
            syn_means = [
                result.data_by_realization[r][site].mean()
                for r in result.realization_ids
            ]
            rel_err = abs(np.mean(syn_means) - obs_mean) / obs_mean
            assert rel_err < 0.25, (
                f"Site {site}: observed mean={obs_mean:.1f}, "
                f"synthetic mean={np.mean(syn_means):.1f}"
            )

    def test_std_within_tolerance(self, df_2sites_10yr):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)
        result = gen.generate(n_realizations=20, seed=0)

        for site in gen.sites:
            obs_std = df_2sites_10yr[site].std()
            syn_stds = [
                result.data_by_realization[r][site].std()
                for r in result.realization_ids
            ]
            rel_err = abs(np.mean(syn_stds) - obs_std) / obs_std
            assert rel_err < 0.35, (
                f"Site {site}: observed std={obs_std:.1f}, "
                f"synthetic std={np.mean(syn_stds):.1f}"
            )


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load(self, df_2sites_10yr, tmp_path):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)

        path = tmp_path / "ms_phase_rand.pkl"
        gen.save(str(path))
        loaded = MultisitePhaseRandomizationGenerator.load(str(path))

        assert loaded.is_fitted is True
        assert loaded.is_preprocessed is True
        assert loaded.wavelet == gen.wavelet
        assert loaded.n_scales == gen.n_scales
        assert loaded.n_sites == gen.n_sites

    def test_loaded_generator_generates(self, df_2sites_10yr, tmp_path):
        gen = MultisitePhaseRandomizationGenerator(n_scales=40)
        gen.fit(df_2sites_10yr)

        original = gen.generate(n_realizations=1, seed=0)

        path = tmp_path / "ms_phase_rand.pkl"
        gen.save(str(path))
        loaded = MultisitePhaseRandomizationGenerator.load(str(path))
        from_loaded = loaded.generate(n_realizations=1, seed=0)

        assert (
            original.data_by_realization[0].shape
            == from_loaded.data_by_realization[0].shape
        )


# ---------------------------------------------------------------------------
# Helper method unit tests
# ---------------------------------------------------------------------------


class TestHelperMethods:
    def test_build_noleap_index(self):
        idx = MultisitePhaseRandomizationGenerator._build_noleap_index(365 * 5)
        assert len(idx) == 365 * 5
        feb29 = idx[(idx.month == 2) & (idx.day == 29)]
        assert len(feb29) == 0

    def test_get_window_days_length(self):
        gen = MultisitePhaseRandomizationGenerator(win_h_length=15)
        window = gen._get_window_days(100)
        assert len(window) == 31  # 15 before + 15 after + target

    def test_get_window_days_wraps_at_start(self):
        gen = MultisitePhaseRandomizationGenerator(win_h_length=15)
        window = gen._get_window_days(5)
        assert 5 in window
        assert any(d > 350 for d in window)

    def test_get_window_days_wraps_at_end(self):
        gen = MultisitePhaseRandomizationGenerator(win_h_length=15)
        window = gen._get_window_days(360)
        assert 360 in window
        assert any(d < 10 for d in window)

    def test_lmoments_l1_is_mean(self):
        gen = MultisitePhaseRandomizationGenerator()
        rng = np.random.default_rng(0)
        data = rng.gamma(2.0, 50.0, size=500)
        lmom = gen._compute_lmoments(data)
        np.testing.assert_almost_equal(lmom["l1"], np.mean(data), decimal=5)

    def test_lmoments_insufficient_data(self):
        gen = MultisitePhaseRandomizationGenerator()
        with pytest.raises(ValueError, match="at least 4"):
            gen._compute_lmoments(np.array([1.0, 2.0, 3.0]))

    def test_invF_kappa_monotone(self):
        gen = MultisitePhaseRandomizationGenerator()
        F = np.array([0.1, 0.5, 0.9])
        x = gen._invF_kappa(F, xi=0.0, alfa=1.0, k=0.5, h=0.5)
        assert np.all(np.isfinite(x))
        assert x[0] < x[1] < x[2]

    def test_invF_kappa_gev_case(self):
        gen = MultisitePhaseRandomizationGenerator()
        F = np.array([0.1, 0.5, 0.9])
        x = gen._invF_kappa(F, xi=0.0, alfa=1.0, k=0.5, h=0.0)
        assert np.all(np.isfinite(x))

    def test_rand_kappa_shape(self):
        gen = MultisitePhaseRandomizationGenerator()
        rng = np.random.default_rng(0)
        samples = gen._rand_kappa(200, xi=0.0, alfa=1.0, k=0.5, h=0.5, rng=rng)
        assert samples.shape == (200,)
        assert np.all(np.isfinite(samples))
