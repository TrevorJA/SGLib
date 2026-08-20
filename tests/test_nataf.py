"""Tests for the Nataf joint distribution model utilities."""

import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist
from scipy.stats import lognorm, norm

from synhydro.core.nataf import (
    cas_acf,
    fit_cas,
    hurst_acf,
    nataf_forward_gh,
    nataf_forward_int,
    nataf_forward_mc,
    nataf_inverse,
    sma_weights_fft,
)


# ---------------------------------------------------------------------------
# Fixtures: ICDF callables
# ---------------------------------------------------------------------------


@pytest.fixture
def gamma_icdf():
    """Gamma(shape=2, scale=1) quantile function."""
    return lambda p: gamma_dist.ppf(p, a=2, scale=1)


@pytest.fixture
def lognorm_icdf():
    """Log-normal(s=0.5, scale=1) quantile function."""
    return lambda p: lognorm.ppf(p, s=0.5, scale=1)


@pytest.fixture
def normal_icdf():
    """Standard normal quantile function."""
    return lambda p: norm.ppf(p)


@pytest.fixture
def heavy_lognorm_icdf():
    """Log-normal(s=1, scale=1) quantile function (heavy-tailed)."""
    return lambda p: lognorm.ppf(p, s=1.0, scale=1)


def lognorm_unit_sigma_closed_form(rho_z):
    """Exact Nataf forward map for two log-normal(0, 1) marginals.

    For X = exp(Z1), Y = exp(Z2) with corr(Z1, Z2) = rho_z,
    corr(X, Y) = (exp(rho_z) - 1) / (e - 1).
    """
    rho_z = np.asarray(rho_z, dtype=float)
    return (np.exp(rho_z) - 1.0) / (np.e - 1.0)


# ---------------------------------------------------------------------------
# Forward Nataf: Gauss-Hermite
# ---------------------------------------------------------------------------


class TestNatafForwardGH:
    def test_zero_correlation_returns_zero(self, gamma_icdf):
        result = nataf_forward_gh(0.0, gamma_icdf, gamma_icdf)
        assert np.isclose(result[0], 0.0, atol=1e-6)

    def test_identity_for_normal(self, normal_icdf):
        """When both marginals are normal, rho_x should equal rho_z."""
        rho_z = np.array([0.3, 0.6, 0.9])
        rho_x = nataf_forward_gh(rho_z, normal_icdf, normal_icdf)
        np.testing.assert_allclose(rho_x, rho_z, atol=0.02)

    def test_monotonically_increasing(self, gamma_icdf):
        rho_z = np.linspace(0.0, 1.0, 11)
        rho_x = nataf_forward_gh(rho_z, gamma_icdf, gamma_icdf)
        diffs = np.diff(rho_x)
        assert np.all(
            diffs >= -1e-6
        ), "Forward Nataf should be monotonically increasing"

    def test_rho_x_leq_rho_z(self, gamma_icdf):
        """Lemma 3: |rho_x| <= |rho_z| for non-normal marginals."""
        rho_z = np.array([0.3, 0.5, 0.7, 0.9])
        rho_x = nataf_forward_gh(rho_z, gamma_icdf, gamma_icdf)
        assert np.all(np.abs(rho_x) <= np.abs(rho_z) + 1e-6)

    def test_unit_correlation_near_one(self, gamma_icdf):
        """rho_z=1 with identical marginals should give rho_x=1."""
        rho_x = nataf_forward_gh(1.0, gamma_icdf, gamma_icdf)
        assert rho_x[0] > 0.95

    def test_negative_correlations(self, gamma_icdf):
        rho_z = np.array([-0.5, -0.3])
        rho_x = nataf_forward_gh(rho_z, gamma_icdf, gamma_icdf)
        assert np.all(rho_x < 0)


# ---------------------------------------------------------------------------
# Forward Nataf: Monte Carlo
# ---------------------------------------------------------------------------


    def test_heavy_tailed_closed_form(self, heavy_lognorm_icdf):
        """GH must not truncate the upper tail (anySim only replaces u == 1).

        Before the fix, u was clipped to 1 - 1e-3, giving errors up to
        0.036 for log-normal(sigma = 1).
        """
        rho_z = np.array([0.3, 0.6, 0.9, -0.9])
        rho_gh = nataf_forward_gh(rho_z, heavy_lognorm_icdf, heavy_lognorm_icdf)
        np.testing.assert_allclose(
            rho_gh, lognorm_unit_sigma_closed_form(rho_z), atol=1e-6
        )

    def test_gamma_half_round_trip(self):
        """Gamma(0.5) has an ICDF singular at u = 0; GH must remain finite."""
        icdf = lambda p: gamma_dist.ppf(p, a=0.5)  # noqa: E731
        targets = np.array([0.3, 0.7, -0.4])
        equiv, _ = nataf_inverse(targets, icdf, icdf, method="GH")
        recovered = nataf_forward_gh(equiv, icdf, icdf)
        assert np.all(np.isfinite(recovered))
        np.testing.assert_allclose(recovered, targets, atol=1e-3)


class TestNatafInverseDegenerate:
    def test_all_zero_targets_return_zeros(self, gamma_icdf):
        """All-zero targets must not reach np.polyfit on a collapsed grid
        (rmin = rmax = 0); Lemma 2 gives rho_z = 0 directly."""
        targets = np.zeros(4)
        equiv, df_nataf = nataf_inverse(targets, gamma_icdf, gamma_icdf)
        np.testing.assert_array_equal(equiv, np.zeros(4))
        assert df_nataf.shape[1] == 2
        assert np.all(np.isfinite(df_nataf))

    def test_scalar_zero_target(self, gamma_icdf):
        equiv, _ = nataf_inverse(0.0, gamma_icdf, gamma_icdf)
        np.testing.assert_array_equal(equiv, np.zeros(1))


class TestNatafForwardMC:
    def test_agreement_with_gh(self, gamma_icdf):
        rho_z = np.array([0.3, 0.6, 0.9])
        rho_gh = nataf_forward_gh(rho_z, gamma_icdf, gamma_icdf)
        rho_mc = nataf_forward_mc(
            rho_z,
            gamma_icdf,
            gamma_icdf,
            n_samples=200_000,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_allclose(rho_mc, rho_gh, atol=0.02)

    def test_zero_returns_zero(self, gamma_icdf):
        rho_mc = nataf_forward_mc(0.0, gamma_icdf, gamma_icdf)
        assert np.isclose(rho_mc[0], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Forward Nataf: numerical integration
# ---------------------------------------------------------------------------


class TestNatafForwardInt:
    def test_heavy_tailed_closed_form(self, heavy_lognorm_icdf):
        """method="Int" matches the log-normal closed form (was all-NaN)."""
        rho_z = np.array([0.6, -0.9])
        rho_int = nataf_forward_int(rho_z, heavy_lognorm_icdf, heavy_lognorm_icdf)
        assert np.all(np.isfinite(rho_int))
        np.testing.assert_allclose(
            rho_int, lognorm_unit_sigma_closed_form(rho_z), atol=1e-4
        )

    def test_zero_returns_zero(self, gamma_icdf):
        rho_int = nataf_forward_int(0.0, gamma_icdf, gamma_icdf)
        assert rho_int[0] == 0.0

    def test_programming_errors_propagate(self):
        """A broken ICDF must raise, not be swallowed into NaN."""

        def bad_icdf(p):
            raise TypeError("broken")

        with pytest.raises(TypeError):
            nataf_forward_int(0.5, bad_icdf, bad_icdf)


# ---------------------------------------------------------------------------
# Nataf Inverse (round-trip)
# ---------------------------------------------------------------------------


class TestNatafInverse:
    def test_round_trip_gamma(self, gamma_icdf):
        """nataf_forward(nataf_inverse(target)) should recover target."""
        targets = np.array([0.2, 0.5, 0.8])
        equiv, _ = nataf_inverse(targets, gamma_icdf, gamma_icdf, method="GH")
        recovered = nataf_forward_gh(equiv, gamma_icdf, gamma_icdf)
        np.testing.assert_allclose(recovered, targets, atol=0.03)

    def test_round_trip_lognorm(self, lognorm_icdf):
        targets = np.array([0.3, 0.6, 0.9])
        equiv, _ = nataf_inverse(targets, lognorm_icdf, lognorm_icdf, method="GH")
        recovered = nataf_forward_gh(equiv, lognorm_icdf, lognorm_icdf)
        np.testing.assert_allclose(recovered, targets, atol=0.03)

    def test_cross_distribution(self, gamma_icdf, lognorm_icdf):
        """Round-trip with different marginals for x and y."""
        targets = np.array([0.3, 0.6])
        equiv, _ = nataf_inverse(targets, gamma_icdf, lognorm_icdf, method="GH")
        recovered = nataf_forward_gh(equiv, gamma_icdf, lognorm_icdf)
        np.testing.assert_allclose(recovered, targets, atol=0.03)

    def test_df_nataf_shape(self, gamma_icdf):
        _, df = nataf_inverse(np.array([0.5]), gamma_icdf, gamma_icdf, n_eval=9)
        assert df.shape == (9, 2)

    def test_negative_target(self, gamma_icdf):
        targets = np.array([-0.3])
        equiv, _ = nataf_inverse(targets, gamma_icdf, gamma_icdf, method="GH")
        assert equiv[0] < 0

    def test_below_lower_bound_warns(self, heavy_lognorm_icdf, caplog):
        """Targets below the Frechet-Hoeffding bound clip to -1 with a warning."""
        # For log-normal(0, 1) the minimum attainable correlation is
        # (exp(-1) - 1) / (e - 1) ~ -0.368.
        with caplog.at_level("WARNING", logger="synhydro.core.nataf"):
            equiv, _ = nataf_inverse(
                np.array([-0.9]), heavy_lognorm_icdf, heavy_lognorm_icdf
            )
        assert equiv[0] == -1.0
        assert any("Frechet-Hoeffding" in r.message for r in caplog.records)

    def test_equiv_greater_than_target(self, gamma_icdf):
        """Lemma 3: |equiv| >= |target| for non-normal marginals."""
        targets = np.array([0.3, 0.5, 0.7])
        equiv, _ = nataf_inverse(targets, gamma_icdf, gamma_icdf, method="GH")
        assert np.all(np.abs(equiv) >= np.abs(targets) - 0.02)


# ---------------------------------------------------------------------------
# CAS autocorrelation
# ---------------------------------------------------------------------------


class TestCASAcf:
    def test_lag_zero_is_one(self):
        acf = cas_acf(kappa=1.0, beta=0.5, max_lag=10)
        assert acf[0] == 1.0

    def test_length(self):
        acf = cas_acf(kappa=1.0, beta=0.5, max_lag=50)
        assert len(acf) == 51

    def test_monotonic_decrease(self):
        acf = cas_acf(kappa=1.0, beta=0.5, max_lag=100)
        assert np.all(np.diff(acf) <= 0)

    def test_srd_case_beta_zero(self):
        """beta=0 should give exponential decay: exp(-kappa * tau)."""
        kappa = 0.5
        acf = cas_acf(kappa=kappa, beta=0.0, max_lag=10)
        expected = np.exp(-kappa * np.arange(0, 11))
        np.testing.assert_allclose(acf, expected, atol=1e-12)

    def test_positive_values(self):
        acf = cas_acf(kappa=0.5, beta=1.5, max_lag=100)
        assert np.all(acf > 0)


# ---------------------------------------------------------------------------
# Hurst ACF
# ---------------------------------------------------------------------------


class TestHurstAcf:
    def test_lag_zero_is_one(self):
        acf = hurst_acf(H=0.7, max_lag=10)
        assert acf[0] == 1.0

    def test_white_noise_h05(self):
        """H=0.5 should give rho(tau>0) = 0."""
        acf = hurst_acf(H=0.5, max_lag=10)
        np.testing.assert_allclose(acf[1:], 0.0, atol=1e-12)

    def test_persistent_positive(self):
        """H>0.5 should give positive autocorrelations."""
        acf = hurst_acf(H=0.8, max_lag=50)
        assert np.all(acf[1:] > 0)


# ---------------------------------------------------------------------------
# Fit CAS
# ---------------------------------------------------------------------------


class TestFitCAS:
    def test_recover_known_parameters(self):
        kappa_true, beta_true = 2.0, 0.8
        acf = cas_acf(kappa_true, beta_true, max_lag=50)
        kappa_fit, beta_fit = fit_cas(acf)
        assert abs(kappa_fit - kappa_true) < 0.3
        assert abs(beta_fit - beta_true) < 0.3

    def test_srd_recovery(self):
        """Recover exponential decay (beta=0)."""
        acf = cas_acf(kappa=1.0, beta=0.0, max_lag=50)
        kappa_fit, beta_fit = fit_cas(acf)
        assert abs(kappa_fit - 1.0) < 0.2
        assert beta_fit < 0.2  # Should be near 0


# ---------------------------------------------------------------------------
# SMA weights
# ---------------------------------------------------------------------------


class TestSMAWeights:
    def test_unit_variance(self):
        """sum(a^2) should be approximately 1."""
        acf = cas_acf(kappa=0.5, beta=0.0, max_lag=64)
        weights = sma_weights_fft(acf)
        assert np.isclose(np.sum(weights**2), 1.0, atol=0.05)

    def test_length(self):
        q = 64
        acf = cas_acf(kappa=0.5, beta=0.0, max_lag=q)
        weights = sma_weights_fft(acf)
        assert len(weights) == 2 * q + 1

    def test_symmetry(self):
        """Weights should be symmetric: a_zeta = a_{-zeta}."""
        acf = cas_acf(kappa=0.5, beta=0.0, max_lag=32)
        weights = sma_weights_fft(acf)
        q = 32
        left = weights[:q]
        right = weights[q + 1 :][::-1]
        np.testing.assert_allclose(left, right, atol=1e-10)

    def test_lrd_acf(self):
        """SMA weights from a Hurst ACF should also have sum(a^2) ~ 1."""
        acf = hurst_acf(H=0.8, max_lag=128)
        weights = sma_weights_fft(acf)
        assert np.isclose(np.sum(weights**2), 1.0, atol=0.05)
