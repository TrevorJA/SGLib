"""Tests for seasonal (per-calendar-month) verification metrics."""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble
from synhydro.verification import (
    verify,
    monthly_mean,
    monthly_std,
    monthly_skewness,
    monthly_lag1_correlation,
    monthly_ranksum_pvalue,
)

N_YEARS = 40


@pytest.fixture(scope="module")
def sinusoidal_series():
    """Monthly series whose month-m mean is 20 + 10 sin(2 pi m / 12)."""
    rng = np.random.default_rng(31)
    index = pd.date_range("1960-01-01", periods=N_YEARS * 12, freq="MS")
    months = index.month.to_numpy()
    values = (
        20.0 + 10.0 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 0.5, len(index))
    )
    return pd.Series(values, index=index)


class TestMonthlyStatistics:
    def test_monthly_mean_recovers_cycle(self, sinusoidal_series):
        curve = monthly_mean(sinusoidal_series)
        assert list(curve.index) == list(range(1, 13))
        for month in range(1, 13):
            expected = 20.0 + 10.0 * np.sin(2 * np.pi * month / 12)
            assert curve[month] == pytest.approx(expected, abs=0.3)

    def test_monthly_std_recovers_noise_scale(self, sinusoidal_series):
        curve = monthly_std(sinusoidal_series)
        for month in range(1, 13):
            assert curve[month] == pytest.approx(0.5, abs=0.15)

    def test_monthly_skewness_near_zero(self, sinusoidal_series):
        curve = monthly_skewness(sinusoidal_series)
        assert np.all(np.abs(curve.to_numpy()) < 1.0)

    def test_monthly_lag1_correlation_independent_months(self, sinusoidal_series):
        # Independent noise: month-to-month correlation near zero
        curve = monthly_lag1_correlation(sinusoidal_series)
        assert list(curve.index) == list(range(1, 13))
        assert np.all(np.abs(curve.to_numpy()) < 0.5)


class TestMonthlyTestsPerRealization:
    def test_ranksum_identical_distribution_high_pvalues(self, sinusoidal_series):
        rng = np.random.default_rng(37)
        index = sinusoidal_series.index
        months = index.month.to_numpy()
        other = pd.Series(
            20.0
            + 10.0 * np.sin(2 * np.pi * months / 12)
            + rng.normal(0, 0.5, len(index)),
            index=index,
        )
        pvals = monthly_ranksum_pvalue(other, sinusoidal_series)
        assert len(pvals) == 12
        # Under the null most months should not reject at alpha = 0.05
        assert (pvals < 0.05).mean() < 0.3

    def test_reject_rate_near_alpha_under_null(self):
        """Regression test for the pooled-sample p-value artifact.

        With realizations drawn from the observed distribution, the
        per-realization rejection rate stays near alpha regardless of
        ensemble size. The old pooled implementation drove p-values to
        zero as the ensemble grew.
        """
        rng = np.random.default_rng(41)
        index = pd.date_range("1970-01-01", periods=30 * 12, freq="MS")

        def draw():
            return pd.DataFrame(
                {"site_a": rng.lognormal(3.0, 0.5, len(index))}, index=index
            )

        observed = draw()
        ensemble = Ensemble({rid: draw() for rid in range(30)})
        result = verify(ensemble, observed, metrics=["monthly_ranksum_pvalue"])
        summary = result.summary()
        # Mean rejection rate across the 12 months near alpha = 0.05
        assert summary["reject_rate"].mean() < 0.15
        assert summary["obs_percentile"].isna().all()
