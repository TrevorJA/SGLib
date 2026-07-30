"""Tests for flow duration curve verification metrics."""

import numpy as np
import pandas as pd
import pytest

from synhydro.verification import fdc, fdc_log_rmse
from synhydro.verification.metrics.fdc import EXCEEDANCE_GRID


@pytest.fixture(scope="module")
def lognormal_series():
    rng = np.random.default_rng(61)
    index = pd.date_range("1950-01-01", periods=10000, freq="D")
    return pd.Series(rng.lognormal(3.0, 0.5, size=len(index)), index=index)


class TestFDC:
    def test_grid_components(self, lognormal_series):
        curve = fdc(lognormal_series)
        assert list(curve.index) == list(EXCEEDANCE_GRID)

    def test_monotone_decreasing_in_exceedance(self, lognormal_series):
        curve = fdc(lognormal_series)
        values = curve.to_numpy()
        assert np.all(np.diff(values) <= 0)

    def test_median_matches(self, lognormal_series):
        curve = fdc(lognormal_series)
        assert curve[0.50] == pytest.approx(
            float(np.median(lognormal_series)), rel=0.01
        )

    def test_short_series_nan(self):
        x = pd.Series(
            np.arange(5.0), index=pd.date_range("2000-01-01", periods=5, freq="D")
        )
        assert fdc(x).isna().all()


class TestFDCLogRMSE:
    def test_identical_zero(self, lognormal_series):
        assert fdc_log_rmse(lognormal_series, lognormal_series) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_scaled_series_log_offset(self, lognormal_series):
        # Multiplying flows by e shifts the log FDC by exactly 1
        scaled = lognormal_series * np.e
        assert fdc_log_rmse(scaled, lognormal_series) == pytest.approx(1.0, rel=1e-6)
