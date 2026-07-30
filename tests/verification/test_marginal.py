"""Analytic ground-truth tests for marginal verification metrics."""

import numpy as np
import pandas as pd
import pytest

from synhydro.verification import (
    mean,
    std,
    cv,
    skewness,
    kurtosis,
    minimum,
    maximum,
    flow_q10,
    flow_q50,
    flow_q90,
    ks_statistic,
)

MU = 3.0
SIGMA = 0.5


@pytest.fixture(scope="module")
def lognormal_series():
    rng = np.random.default_rng(123)
    index = pd.date_range("1950-01-01", periods=20000, freq="D")
    return pd.Series(rng.lognormal(mean=MU, sigma=SIGMA, size=len(index)), index=index)


class TestLognormalMoments:
    def test_mean(self, lognormal_series):
        expected = np.exp(MU + SIGMA**2 / 2)
        assert mean(lognormal_series) == pytest.approx(expected, rel=0.02)

    def test_cv(self, lognormal_series):
        expected = np.sqrt(np.exp(SIGMA**2) - 1)
        assert cv(lognormal_series) == pytest.approx(expected, rel=0.05)

    def test_std(self, lognormal_series):
        expected = np.exp(MU + SIGMA**2 / 2) * np.sqrt(np.exp(SIGMA**2) - 1)
        assert std(lognormal_series) == pytest.approx(expected, rel=0.05)

    def test_skewness(self, lognormal_series):
        c = np.sqrt(np.exp(SIGMA**2) - 1)
        expected = c**3 + 3 * c
        assert skewness(lognormal_series) == pytest.approx(expected, rel=0.15)

    def test_median(self, lognormal_series):
        assert flow_q50(lognormal_series) == pytest.approx(np.exp(MU), rel=0.02)

    def test_quantiles(self, lognormal_series):
        from scipy.stats import norm

        q10 = np.exp(MU + SIGMA * norm.ppf(0.10))
        q90 = np.exp(MU + SIGMA * norm.ppf(0.90))
        assert flow_q10(lognormal_series) == pytest.approx(q10, rel=0.03)
        assert flow_q90(lognormal_series) == pytest.approx(q90, rel=0.03)

    def test_min_max_bracket_quantiles(self, lognormal_series):
        assert minimum(lognormal_series) < flow_q10(lognormal_series)
        assert maximum(lognormal_series) > flow_q90(lognormal_series)


class TestNormalShape:
    def test_normal_skewness_zero(self):
        rng = np.random.default_rng(5)
        x = pd.Series(rng.standard_normal(50000))
        assert skewness(x) == pytest.approx(0.0, abs=0.05)
        assert kurtosis(x) == pytest.approx(0.0, abs=0.1)


class TestKSStatistic:
    def test_identical_samples_zero(self, lognormal_series):
        assert ks_statistic(lognormal_series, lognormal_series) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_shifted_sample_positive(self, lognormal_series):
        shifted = lognormal_series + 10.0
        assert ks_statistic(shifted, lognormal_series) > 0.2

    def test_bounded(self, lognormal_series):
        very_different = lognormal_series * 100.0
        assert ks_statistic(very_different, lognormal_series) <= 1.0
