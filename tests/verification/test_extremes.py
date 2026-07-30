"""Tests for extreme flow verification metrics."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genextreme

from synhydro.verification import (
    annual_max_mean,
    annual_max_cv,
    gev_rp10,
    gev_rp50,
    gev_rp100,
    annual_min_mean,
    seven_day_min_mean,
)

GEV_SHAPE_C = -0.1  # scipy convention; heavy upper tail
GEV_LOC = 100.0
GEV_SCALE = 20.0


@pytest.fixture(scope="module")
def gev_annual_series():
    """Annual-frequency series sampled from a known GEV distribution."""
    rng = np.random.default_rng(79)
    n = 500
    values = genextreme.rvs(
        GEV_SHAPE_C, loc=GEV_LOC, scale=GEV_SCALE, size=n, random_state=rng
    )
    index = pd.date_range("1700-01-01", periods=n, freq="YS")
    return pd.Series(values, index=index)


class TestGEVRoundTrip:
    def test_rp10_recovered(self, gev_annual_series):
        expected = genextreme.isf(0.1, GEV_SHAPE_C, loc=GEV_LOC, scale=GEV_SCALE)
        estimate = gev_rp10(gev_annual_series, steps_per_year=1.0)
        assert estimate == pytest.approx(expected, rel=0.10)

    def test_rp100_recovered(self, gev_annual_series):
        expected = genextreme.isf(0.01, GEV_SHAPE_C, loc=GEV_LOC, scale=GEV_SCALE)
        estimate = gev_rp100(gev_annual_series, steps_per_year=1.0)
        assert estimate == pytest.approx(expected, rel=0.15)

    def test_return_levels_ordered(self, gev_annual_series):
        rp10 = gev_rp10(gev_annual_series, steps_per_year=1.0)
        rp50 = gev_rp50(gev_annual_series, steps_per_year=1.0)
        rp100 = gev_rp100(gev_annual_series, steps_per_year=1.0)
        assert rp10 < rp50 < rp100

    def test_import_regression(self):
        """GEV metrics fail loudly or return nan, never raise NameError.

        Regression test for the old implementation, which imported
        genextreme inside a try block and lost all GEV metrics when the
        observed fit raised first.
        """
        x = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.date_range("2000-01-01", periods=3, freq="YS"),
        )
        assert np.isnan(gev_rp10(x, steps_per_year=1.0))


class TestAnnualExtremes:
    def test_annual_max_of_constant_years(self):
        # Monthly series with a known single peak per year
        frames = []
        for year in range(15):
            index = pd.date_range(f"{2000 + year}-01-01", periods=12, freq="MS")
            values = np.full(12, 10.0)
            values[5] = 100.0 + year
            frames.append(pd.Series(values, index=index))
        x = pd.concat(frames)
        expected_mean = np.mean([100.0 + year for year in range(15)])
        assert annual_max_mean(x, steps_per_year=12.0) == pytest.approx(expected_mean)
        assert annual_min_mean(x, steps_per_year=12.0) == pytest.approx(10.0)

    def test_annual_max_cv(self, gev_annual_series):
        expected = gev_annual_series.std(ddof=1) / gev_annual_series.mean()
        assert annual_max_cv(gev_annual_series, steps_per_year=1.0) == pytest.approx(
            float(expected), rel=1e-6
        )


class TestSevenDayLowFlow:
    def test_constant_low_period_recovered(self):
        # Daily series: constant 50, with a 30-day period at 5 each year
        frames = []
        for year in range(12):
            index = pd.date_range(f"{2000 + year}-01-01", periods=365, freq="D")
            values = np.full(365, 50.0)
            values[180:210] = 5.0
            frames.append(pd.Series(values, index=index))
        x = pd.concat(frames)
        assert seven_day_min_mean(x, steps_per_year=365.25) == pytest.approx(5.0)
