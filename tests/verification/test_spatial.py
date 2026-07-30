"""Tests for spatial (cross-site) verification metrics."""

import numpy as np
import pandas as pd
import pytest

from synhydro.verification import cross_correlation, cross_correlation_lag1

RHO = 0.7


@pytest.fixture(scope="module")
def correlated_frame():
    rng = np.random.default_rng(53)
    n = 20000
    cov = [[1.0, RHO], [RHO, 1.0]]
    values = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    index = pd.date_range("1900-01-01", periods=n, freq="D")
    return pd.DataFrame(values, columns=["site_a", "site_b"], index=index)


class TestCrossCorrelation:
    def test_recovers_rho(self, correlated_frame):
        result = cross_correlation(correlated_frame)
        assert list(result.index) == ["site_a|site_b"]
        assert result.iloc[0] == pytest.approx(RHO, abs=0.02)

    def test_three_sites_three_pairs(self, correlated_frame):
        frame = correlated_frame.copy()
        frame["site_c"] = frame["site_a"] * 0.5 + frame["site_b"] * 0.5
        result = cross_correlation(frame)
        assert set(result.index) == {
            "site_a|site_b",
            "site_a|site_c",
            "site_b|site_c",
        }


class TestCrossCorrelationLag1:
    def test_directional_pairs(self, correlated_frame):
        result = cross_correlation_lag1(correlated_frame)
        assert set(result.index) == {"site_a->site_b", "site_b->site_a"}

    def test_independent_in_time_near_zero(self, correlated_frame):
        # Rows are iid, so any lagged correlation is near zero
        result = cross_correlation_lag1(correlated_frame)
        assert np.all(np.abs(result.to_numpy()) < 0.05)

    def test_lagged_construction_recovered(self):
        rng = np.random.default_rng(59)
        n = 10000
        upstream = rng.standard_normal(n)
        downstream = np.roll(upstream, 1)  # downstream lags upstream by one step
        index = pd.date_range("1950-01-01", periods=n, freq="D")
        frame = pd.DataFrame({"down": downstream, "up": upstream}, index=index).iloc[1:]
        result = cross_correlation_lag1(frame)
        # down at time t equals up at time t-1
        assert result["down->up"] == pytest.approx(1.0, abs=0.01)
        assert abs(result["up->down"]) < 0.05
