"""Tests for spectral verification metrics."""

import numpy as np
import pandas as pd
import pytest

from synhydro.verification import spectral_density, low_frequency_variance_fraction
from synhydro.verification.metrics.spectral import PERIOD_BANDS
from tests.verification.conftest import ar1_series


class TestSpectralDensity:
    def test_band_fractions_sum_to_one(self):
        x = ar1_series(n=1200, phi=0.5, seed=83)
        curve = spectral_density(x, steps_per_year=12.0)
        assert list(curve.index) == [label for label, _, _ in PERIOD_BANDS]
        assert curve.sum() == pytest.approx(1.0, abs=1e-6)

    def test_persistent_series_shifts_variance_to_long_periods(self):
        white = ar1_series(n=2400, phi=0.0, seed=89)
        persistent = ar1_series(n=2400, phi=0.9, seed=89)
        white_curve = spectral_density(white, steps_per_year=12.0)
        persistent_curve = spectral_density(persistent, steps_per_year=12.0)
        long_bands = [">8y", "4-8y", "2-4y"]
        assert persistent_curve[long_bands].sum() > white_curve[long_bands].sum()

    def test_short_series_nan(self):
        x = ar1_series(n=12, phi=0.5, seed=97)
        assert spectral_density(x, steps_per_year=12.0).isna().all()


class TestLowFrequencyVarianceFraction:
    def test_persistent_exceeds_white_noise(self):
        white = ar1_series(n=2400, phi=0.0, seed=101)
        persistent = ar1_series(n=2400, phi=0.9, seed=101)
        assert low_frequency_variance_fraction(
            persistent, steps_per_year=12.0
        ) > low_frequency_variance_fraction(white, steps_per_year=12.0)

    def test_bounded(self):
        x = ar1_series(n=1200, phi=0.5, seed=103)
        value = low_frequency_variance_fraction(x, steps_per_year=12.0)
        assert 0.0 <= value <= 1.0
