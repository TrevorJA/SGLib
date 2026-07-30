"""Tests for annual aggregate verification metrics."""

import numpy as np
import pandas as pd
import pytest

from synhydro.verification import (
    annual_mean,
    annual_sd,
    annual_cv,
    annual_lag1_autocorrelation,
    annual_minimum,
    annual_maximum,
)


@pytest.fixture(scope="module")
def constructed_monthly():
    """Monthly series whose annual totals are exactly 1200 + 100 * year_offset."""
    frames = []
    for year_offset in range(20):
        index = pd.date_range(f"{1990 + year_offset}-01-01", periods=12, freq="MS")
        monthly_value = (1200.0 + 100.0 * year_offset) / 12.0
        frames.append(pd.Series(monthly_value, index=index))
    return pd.concat(frames)


class TestAnnualAggregates:
    def test_annual_mean(self, constructed_monthly):
        totals = 1200.0 + 100.0 * np.arange(20)
        assert annual_mean(constructed_monthly, steps_per_year=12.0) == pytest.approx(
            totals.mean()
        )

    def test_annual_sd(self, constructed_monthly):
        totals = 1200.0 + 100.0 * np.arange(20)
        assert annual_sd(constructed_monthly, steps_per_year=12.0) == pytest.approx(
            totals.std(ddof=1)
        )

    def test_annual_cv(self, constructed_monthly):
        totals = 1200.0 + 100.0 * np.arange(20)
        expected = totals.std(ddof=1) / totals.mean()
        assert annual_cv(constructed_monthly, steps_per_year=12.0) == pytest.approx(
            expected
        )

    def test_annual_min_max(self, constructed_monthly):
        assert annual_minimum(
            constructed_monthly, steps_per_year=12.0
        ) == pytest.approx(1200.0)
        assert annual_maximum(
            constructed_monthly, steps_per_year=12.0
        ) == pytest.approx(1200.0 + 100.0 * 19)

    def test_trending_series_positive_lag1(self, constructed_monthly):
        value = annual_lag1_autocorrelation(constructed_monthly, steps_per_year=12.0)
        assert value > 0.5


class TestIncompleteYears:
    def test_partial_year_excluded(self, constructed_monthly):
        # Append 3 months of a final year: that year must be dropped
        extra = pd.Series(1e6, index=pd.date_range("2010-01-01", periods=3, freq="MS"))
        padded = pd.concat([constructed_monthly, extra])
        assert annual_maximum(padded, steps_per_year=12.0) == pytest.approx(
            1200.0 + 100.0 * 19
        )

    def test_too_few_years_nan(self):
        index = pd.date_range("2000-01-01", periods=24, freq="MS")
        x = pd.Series(1.0, index=index)
        assert np.isnan(annual_mean(x, steps_per_year=12.0))
