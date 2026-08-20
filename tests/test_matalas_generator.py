"""
Tests for MatalasGenerator (Matalas 1967, multi-site MAR(1)).
"""

import pickle
import numpy as np
import pandas as pd
import pytest

from synhydro.methods.generation.parametric.matalas import MatalasGenerator
from synhydro.core.ensemble import Ensemble


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monthly_multisite():
    """30 years of correlated monthly flows at 3 sites."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1990-01-01", periods=360, freq="MS")
    n = len(dates)
    # Seasonal pattern + correlated noise
    seasonal = 200 + 150 * np.sin(2 * np.pi * np.arange(n) / 12)
    data = {}
    base = rng.gamma(shape=3.0, scale=1.0, size=n)
    for i, name in enumerate(["A", "B", "C"]):
        data[name] = np.maximum(
            seasonal * (1 + 0.3 * i) + 80 * base + rng.normal(0, 20, n), 1.0
        )
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def ar1_two_site():
    """150 years of a 2-site lognormal AR(1) process with rho = 0.7."""
    rng = np.random.default_rng(7)
    rho = 0.7
    n = 150 * 12
    L = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
    z = np.zeros((n, 2))
    z[0] = L @ rng.standard_normal(2)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + np.sqrt(1.0 - rho**2) * (L @ rng.standard_normal(2))
    dates = pd.date_range("1850-01-01", periods=n, freq="MS")
    return pd.DataFrame(np.exp(0.5 * z + 3.0), index=dates, columns=["a", "b"])


@pytest.fixture
def monthly_single_site(monthly_multisite):
    """Single site for degenerate (Thomas-Fiering-equivalent) tests."""
    return monthly_multisite[["A"]]


@pytest.fixture
def short_monthly(monthly_multisite):
    """Shorter record for edge-case tests."""
    return monthly_multisite.iloc[:60]  # 5 years


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestMatalasInit:
    def test_default_params(self, monthly_multisite):
        gen = MatalasGenerator()
        assert gen.log_transform is True
        assert gen.is_preprocessed is False
        assert gen.is_fitted is False

    def test_log_transform_false(self, monthly_multisite):
        gen = MatalasGenerator(log_transform=False)
        assert gen.log_transform is False

    def test_stores_algorithm_params(self, monthly_multisite):
        gen = MatalasGenerator()
        assert gen.init_params.algorithm_params["method"] == "Matalas MAR(1)"
        assert gen.init_params.algorithm_params["burn_in"] == 120

    def test_burn_in_default_and_rounding(self):
        assert MatalasGenerator().burn_in == 120
        assert MatalasGenerator(burn_in=0).burn_in == 0
        # Rounded up to whole years so output still starts in January
        assert MatalasGenerator(burn_in=13).burn_in == 24
        assert MatalasGenerator(burn_in=24).burn_in == 24

    def test_negative_burn_in_raises(self):
        with pytest.raises(ValueError, match="burn_in"):
            MatalasGenerator(burn_in=-1)

    def test_accepts_series_via_fit(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite.iloc[:, 0])
        assert gen is not None

    def test_accepts_dataframe_via_fit(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        assert gen is not None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestMatalasPreprocessing:
    def test_preprocessed_flag(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.preprocessing(monthly_multisite)
        assert gen.is_preprocessed is True

    def test_sites_stored(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.preprocessing(monthly_multisite)
        assert gen._sites == ["A", "B", "C"]
        assert gen._n_sites == 3

    def test_single_site(self, monthly_single_site):
        gen = MatalasGenerator()
        gen.preprocessing(monthly_single_site)
        assert gen._n_sites == 1

    def test_site_subset(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.preprocessing(monthly_multisite, sites=["A", "B"])
        assert gen._sites == ["A", "B"]
        assert gen._n_sites == 2

    def test_monthly_index_preserved(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.preprocessing(monthly_multisite)
        assert len(gen.Q_obs_monthly) == 360

    def test_daily_resampled_to_monthly(self):
        dates = pd.date_range("2000-01-01", periods=365 * 10, freq="D")
        Q = pd.DataFrame({"X": np.random.gamma(2, 50, len(dates))}, index=dates)
        gen = MatalasGenerator()
        gen.preprocessing(Q)
        assert gen.Q_obs_monthly.index.freqstr in ("MS", "MS-JAN")

    def test_no_fit_before_preprocessing_raises(self, monthly_multisite):
        gen = MatalasGenerator()
        with pytest.raises(ValueError, match="preprocessing"):
            gen.fit()


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestMatalasFit:
    def test_fitted_flag(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        assert gen.is_fitted is True

    def test_twelve_matrices(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        assert len(gen._A) == 12
        assert len(gen._B) == 12

    def test_matrix_shapes(self, monthly_multisite):
        n = 3
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        for m in range(12):
            assert gen._A[m].shape == (n, n), f"A[{m}] wrong shape"
            assert gen._B[m].shape == (n, n), f"B[{m}] wrong shape"

    def test_b_lower_triangular(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        for m in range(12):
            B = gen._B[m]
            # Lower triangular: upper-right elements should be ~0
            assert np.allclose(np.triu(B, 1), 0, atol=1e-8)

    def test_mu_sigma_shape(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        assert gen._mu.shape == (12, 3)
        assert gen._sigma.shape == (12, 3)
        assert gen._sigma.values.min() > 0

    def test_mu_sigma_index(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        assert list(gen._mu.index) == list(range(1, 13))

    def test_fitted_params_stored(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        assert gen.fitted_params_ is not None
        assert gen.fitted_params_.n_sites_ == 3

    def test_single_site_fit(self, monthly_single_site):
        gen = MatalasGenerator()
        gen.fit(monthly_single_site)
        assert gen.is_fitted
        for m in range(12):
            assert gen._A[m].shape == (1, 1)

    def test_no_generate_before_fit_raises(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.preprocessing(monthly_multisite)
        with pytest.raises(ValueError, match="fit"):
            gen.generate(n_years=5)

    def test_log_transform_false_fits(self, monthly_multisite):
        gen = MatalasGenerator(log_transform=False)
        gen.fit(monthly_multisite)
        assert gen.is_fitted


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class TestMatalasGenerate:
    def test_returns_ensemble(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_years=10, n_realizations=2, seed=0)
        assert isinstance(result, Ensemble)

    def test_n_realizations(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_years=5, n_realizations=7, seed=0)
        assert result.metadata.n_realizations == 7

    def test_ensemble_metadata(self, monthly_multisite):
        """Ensemble carries the metadata needed for pipeline chaining."""
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_years=5, n_realizations=2, seed=0)
        assert result.frequency == "MS"
        assert result.metadata.time_resolution == "MS"
        assert result.metadata.generator_class == "MatalasGenerator"
        assert result.metadata.n_sites == 3
        assert result.metadata.time_period is not None

    def test_shape_multisite(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_years=10, n_realizations=1, seed=0)
        df = result.data_by_realization[0]
        assert df.shape == (120, 3)  # 10*12 months, 3 sites

    def test_site_columns_match(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_years=5, n_realizations=1, seed=0)
        assert list(result.data_by_realization[0].columns) == ["A", "B", "C"]

    def test_datetime_index(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_years=5, n_realizations=1, seed=0)
        df = result.data_by_realization[0]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_non_negative_flows(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_years=20, n_realizations=10, seed=0)
        for i in range(10):
            assert (result.data_by_realization[i].values >= 0).all()

    def test_no_nans(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_years=20, n_realizations=5, seed=0)
        for i in range(5):
            assert not result.data_by_realization[i].isna().any().any()

    def test_seed_reproducibility(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        r1 = gen.generate(n_years=10, n_realizations=3, seed=99)
        r2 = gen.generate(n_years=10, n_realizations=3, seed=99)
        for i in range(3):
            pd.testing.assert_frame_equal(
                r1.data_by_realization[i], r2.data_by_realization[i]
            )

    def test_different_seeds_differ(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        r1 = gen.generate(n_years=10, n_realizations=1, seed=1)
        r2 = gen.generate(n_years=10, n_realizations=1, seed=2)
        assert not r1.data_by_realization[0].equals(r2.data_by_realization[0])

    def test_n_timesteps_override(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_timesteps=36, n_realizations=1, seed=0)
        assert len(result.data_by_realization[0]) == 36

    def test_default_n_years(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_realizations=1, seed=0)
        # Should match historic length in years
        expected = len(gen.Q_obs_monthly) // 12 * 12
        assert len(result.data_by_realization[0]) == expected

    def test_realizations_differ(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_years=10, n_realizations=5, seed=7)
        # Not all realizations should be identical
        frames = [result.data_by_realization[i] for i in range(5)]
        n_unique = sum(1 for j in range(1, 5) if not frames[0].equals(frames[j]))
        assert n_unique >= 3


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------


class TestMatalasStatistics:
    def test_monthly_mean_preserved(self, monthly_multisite):
        """Ensemble mean by month should approximate historical monthly mean."""
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_years=50, n_realizations=20, seed=0)

        # Pool all realizations
        all_dfs = pd.concat([result.data_by_realization[i] for i in range(20)])
        syn_means = all_dfs.groupby(all_dfs.index.month).mean()
        obs_means = gen.Q_obs_monthly.groupby(gen.Q_obs_monthly.index.month).mean()

        # Log-space estimation reproduces real-space means only approximately
        # (no bias correction), so allow 20-25% per month.
        for col in gen._sites:
            ratio = syn_means[col] / obs_means[col]
            assert (ratio > 0.8).all() and (
                ratio < 1.25
            ).all(), f"Site {col}: monthly mean ratio out of range\n{ratio}"

    def test_lag1_and_cross_correlation_reproduced(self, ar1_two_site):
        """Long-run log-space lag-1 and lag-0 correlations match fitted targets."""
        gen = MatalasGenerator()
        gen.fit(ar1_two_site)
        result = gen.generate(n_years=500, n_realizations=4, seed=3)
        log_obs = np.log(gen.Q_obs_monthly + 1.0)

        def standardize(df):
            z = df.copy()
            for m in range(1, 13):
                mask = z.index.month == m
                z.loc[mask] = (z.loc[mask] - gen._mu.loc[m].values) / gen._sigma.loc[
                    m
                ].values
            return z

        def lag1(z):
            return np.array(
                [
                    np.corrcoef(z.iloc[1:, j].values, z.iloc[:-1, j].values)[0, 1]
                    for j in range(z.shape[1])
                ]
            )

        z_obs = standardize(log_obs)
        z_syn = pd.concat(
            [standardize(np.log(result.data_by_realization[i] + 1.0)) for i in range(4)]
        )
        # Lag-1 targets: pooled across all month transitions
        assert np.allclose(lag1(z_syn), lag1(z_obs), atol=0.05), (
            lag1(z_syn),
            lag1(z_obs),
        )
        # Lag-0 cross-site target
        r_obs = z_obs.corr().iloc[0, 1]
        r_syn = z_syn.corr().iloc[0, 1]
        assert abs(r_syn - r_obs) < 0.05, (r_syn, r_obs)

    def test_first_month_cross_correlation_after_burn_in(self, ar1_two_site):
        """With burn-in, the first output month carries the fitted spatial
        correlation instead of the N(0, I) starting state."""
        gen = MatalasGenerator()
        gen.fit(ar1_two_site)
        result = gen.generate(n_years=1, n_realizations=400, seed=11)
        first = np.array(
            [
                np.log(result.data_by_realization[i].iloc[0].values + 1.0)
                for i in range(400)
            ]
        )
        r_syn = np.corrcoef(first[:, 0], first[:, 1])[0, 1]
        log_obs = np.log(gen.Q_obs_monthly + 1.0)
        jan = log_obs[log_obs.index.month == 1]
        r_obs = jan.corr().iloc[0, 1]
        assert abs(r_syn - r_obs) < 0.1, (r_syn, r_obs)

    def test_burn_in_zero_first_month_uncorrelated(self, ar1_two_site):
        """burn_in=0 retains the original N(0, I) start, so the first month
        has near-zero cross-site correlation."""
        gen = MatalasGenerator(burn_in=0)
        gen.fit(ar1_two_site)
        assert gen.burn_in == 0
        result = gen.generate(n_years=1, n_realizations=400, seed=11)
        first = np.array(
            [
                np.log(result.data_by_realization[i].iloc[0].values + 1.0)
                for i in range(400)
            ]
        )
        r_syn = np.corrcoef(first[:, 0], first[:, 1])[0, 1]
        assert abs(r_syn) < 0.15, r_syn
        assert result.data_by_realization[0].shape == (12, 2)
        assert result.data_by_realization[0].index[0].month == 1

    def test_spatial_correlation_sign_preserved(self, monthly_multisite):
        """Contemporaneous cross-site correlation sign should be preserved."""
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)
        result = gen.generate(n_years=50, n_realizations=10, seed=0)

        df_syn = pd.concat([result.data_by_realization[i] for i in range(10)])
        syn_corr = df_syn.corr()
        obs_corr = gen.Q_obs_monthly.corr()

        for i, si in enumerate(gen._sites):
            for j, sj in enumerate(gen._sites):
                if i != j:
                    assert np.sign(syn_corr.loc[si, sj]) == np.sign(
                        obs_corr.loc[si, sj]
                    ), f"Correlation sign mismatch between {si} and {sj}"

    def test_innovation_covariance_not_rescaled(self, ar1_two_site):
        """diag(B B^T) must equal 1 - rho^2 (Matalas Eq. 17), not 1."""
        gen = MatalasGenerator()
        gen.fit(ar1_two_site)
        diag = np.array([np.diag(B @ B.T) for B in gen._B])
        # rho = 0.7 -> innovation variance ~ 0.51 per month; the old
        # unit-diagonal rescaling gave exactly 1.0 for every month.
        assert 0.4 < diag.mean() < 0.6, diag
        assert (diag < 0.75).all(), diag

    def test_monthly_std_preserved(self, ar1_two_site):
        """Synthetic monthly log-std must match observed within 10%."""
        gen = MatalasGenerator()
        gen.fit(ar1_two_site)
        result = gen.generate(n_years=150, n_realizations=5, seed=0)
        syn = pd.concat([result.data_by_realization[i] for i in range(5)])
        log_syn = np.log(syn + 1.0)
        log_obs = np.log(gen.Q_obs_monthly + 1.0)
        ratio = (
            log_syn.groupby(log_syn.index.month).std()
            / log_obs.groupby(log_obs.index.month).std()
        ).mean()
        assert ((ratio > 0.9) & (ratio < 1.1)).all(), ratio


class TestRepairCovarianceMatrix:
    def test_preserves_matrix_when_already_pd(self):
        from synhydro.core.statistics import repair_covariance_matrix

        cov = np.array([[0.5, 0.2], [0.2, 0.4]])
        assert np.allclose(repair_covariance_matrix(cov), cov)

    def test_clips_without_rescaling(self):
        from synhydro.core.statistics import (
            repair_correlation_matrix,
            repair_covariance_matrix,
        )

        # Indefinite matrix with diagonal 0.5
        cov = np.array([[0.5, 0.6], [0.6, 0.5]])
        assert np.linalg.eigvalsh(cov).min() < 0
        rep = repair_covariance_matrix(cov)
        assert np.linalg.eigvalsh(rep).min() > 0
        np.linalg.cholesky(rep)
        assert np.allclose(np.diag(rep), 0.5, atol=0.11)
        # Correlation repair (existing behaviour) rescales to unit diagonal
        assert np.allclose(np.diag(repair_correlation_matrix(cov)), 1.0)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestMatalasSerialization:
    def test_pickle_roundtrip(self, monthly_multisite, tmp_path):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)

        path = tmp_path / "matalas.pkl"
        with open(path, "wb") as f:
            pickle.dump(gen, f)
        with open(path, "rb") as f:
            gen2 = pickle.load(f)

        assert gen2.is_fitted
        assert gen2._n_sites == 3

    def test_generate_after_pickle(self, monthly_multisite, tmp_path):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite)

        path = tmp_path / "matalas.pkl"
        with open(path, "wb") as f:
            pickle.dump(gen, f)
        with open(path, "rb") as f:
            gen2 = pickle.load(f)

        result = gen2.generate(n_years=5, n_realizations=2, seed=0)
        assert result.metadata.n_realizations == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMatalasEdgeCases:
    def test_short_record_warns(self, short_monthly):
        """Short record should fit without crash (may log warnings)."""
        gen = MatalasGenerator()
        gen.fit(short_monthly)
        assert gen.is_fitted

    def test_single_site_equivalent(self, monthly_single_site):
        """Single-site MAR(1) should produce valid output."""
        gen = MatalasGenerator()
        gen.fit(monthly_single_site)
        result = gen.generate(n_years=10, n_realizations=3, seed=0)
        for i in range(3):
            assert result.data_by_realization[i].shape == (120, 1)

    def test_two_sites(self, monthly_multisite):
        gen = MatalasGenerator()
        gen.fit(monthly_multisite[["A", "B"]])
        result = gen.generate(n_years=5, n_realizations=2, seed=0)
        assert result.data_by_realization[0].shape == (60, 2)

    def test_log_transform_false_end_to_end(self, monthly_multisite):
        gen = MatalasGenerator(log_transform=False)
        gen.fit(monthly_multisite)
        result = gen.generate(n_years=10, n_realizations=2, seed=0)
        assert not result.data_by_realization[0].isna().any().any()
