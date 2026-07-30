"""Analytic ground-truth tests for L-moment verification metrics."""

import numpy as np
import pandas as pd
import pytest

from synhydro.verification import l_cv, l_skewness, l_kurtosis


def make_series(values):
    index = pd.date_range("1950-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=index)


class TestAnalyticLMoments:
    def test_uniform(self):
        # Uniform(0, 1): l1 = 1/2, l2 = 1/6, tau3 = 0, tau4 = 0
        rng = np.random.default_rng(67)
        x = make_series(rng.uniform(0, 1, 50000))
        assert l_cv(x) == pytest.approx((1 / 6) / (1 / 2), abs=0.01)
        assert l_skewness(x) == pytest.approx(0.0, abs=0.01)
        assert l_kurtosis(x) == pytest.approx(0.0, abs=0.01)

    def test_exponential(self):
        # Exponential: tau3 = 1/3, tau4 = 1/6
        rng = np.random.default_rng(71)
        x = make_series(rng.exponential(scale=1.0, size=50000))
        assert l_skewness(x) == pytest.approx(1 / 3, abs=0.02)
        assert l_kurtosis(x) == pytest.approx(1 / 6, abs=0.02)
        # Exponential: l1 = scale, l2 = scale / 2
        assert l_cv(x) == pytest.approx(0.5, abs=0.01)

    def test_normal(self):
        # Normal: tau3 = 0, tau4 = 0.1226
        rng = np.random.default_rng(73)
        x = make_series(rng.normal(10.0, 2.0, 50000))
        assert l_skewness(x) == pytest.approx(0.0, abs=0.01)
        assert l_kurtosis(x) == pytest.approx(0.1226, abs=0.01)

    def test_short_series_nan(self):
        x = make_series(np.arange(5.0))
        assert np.isnan(l_cv(x))
        assert np.isnan(l_skewness(x))
        assert np.isnan(l_kurtosis(x))
