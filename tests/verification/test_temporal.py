"""Analytic ground-truth tests for temporal verification metrics."""

import numpy as np
import pandas as pd
import pytest

from synhydro.verification import (
    lag1_autocorrelation,
    lag2_autocorrelation,
    acf,
    hurst,
)
from tests.verification.conftest import ar1_series

PHI = 0.6


@pytest.fixture(scope="module")
def ar1():
    return ar1_series(n=6000, phi=PHI, seed=21)


class TestAutocorrelation:
    def test_lag1(self, ar1):
        assert lag1_autocorrelation(ar1) == pytest.approx(PHI, abs=0.05)

    def test_lag2(self, ar1):
        assert lag2_autocorrelation(ar1) == pytest.approx(PHI**2, abs=0.05)

    def test_acf_decay(self, ar1):
        curve = acf(ar1, acf_lags=6)
        assert list(curve.index) == [1, 2, 3, 4, 5, 6]
        for lag in range(1, 7):
            assert curve[lag] == pytest.approx(PHI**lag, abs=0.07)

    def test_white_noise_near_zero(self):
        x = ar1_series(n=6000, phi=0.0, seed=3)
        assert abs(lag1_autocorrelation(x)) < 0.05

    def test_short_series_nan(self):
        x = pd.Series(
            [1.0, 2.0], index=pd.date_range("2000-01-01", periods=2, freq="MS")
        )
        assert np.isnan(lag1_autocorrelation(x))


class TestHurst:
    def test_white_noise_annual(self):
        # 100 years of independent annual values: H near 0.5
        rng = np.random.default_rng(17)
        index = pd.date_range("1900-01-01", periods=100, freq="YS")
        x = pd.Series(rng.lognormal(3.0, 0.5, size=100), index=index)
        estimate = hurst(x, steps_per_year=1.0)
        assert 0.3 < estimate < 0.75

    def test_monthly_input_aggregated(self):
        # Monthly white noise aggregates to near-independent annual totals
        rng = np.random.default_rng(19)
        index = pd.date_range("1900-01-01", periods=100 * 12, freq="MS")
        x = pd.Series(rng.lognormal(3.0, 0.5, size=len(index)), index=index)
        estimate = hurst(x, steps_per_year=12.0)
        assert 0.3 < estimate < 0.75

    def test_short_record_nan(self):
        rng = np.random.default_rng(23)
        index = pd.date_range("2000-01-01", periods=10, freq="YS")
        x = pd.Series(rng.lognormal(3.0, 0.5, size=10), index=index)
        assert np.isnan(hurst(x, steps_per_year=1.0))
