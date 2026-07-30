"""Tests for threshold drought validation metrics."""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble
from synhydro._evaluation import extract_runs
from synhydro.validation import validate


class TestExtractRuns:
    def test_known_events(self):
        # Threshold 10: two events, durations 2 and 1
        values = np.array([20.0, 5.0, 5.0, 20.0, 8.0, 20.0])
        durations, severities = extract_runs(values, 10.0)
        assert durations == [2, 1]
        assert severities[0] == pytest.approx((10 - 5) + (10 - 5))
        assert severities[1] == pytest.approx(10 - 8)

    def test_no_events(self):
        values = np.array([20.0, 15.0, 30.0])
        durations, severities = extract_runs(values, 10.0)
        assert durations == []
        assert severities == []

    def test_open_ended_event(self):
        # Series ends inside an event: the event is still counted
        values = np.array([20.0, 5.0, 5.0])
        durations, severities = extract_runs(values, 10.0)
        assert durations == [2]


class TestThresholdDroughtValues:
    @pytest.fixture(scope="class")
    def square_wave_result(self):
        """Ensemble and observed built from a known square wave.

        Each 12-month year has months 7-9 at flow 2 and the rest at
        flow 20. With threshold 10 there is exactly one 3-month event
        per year with severity 3 * (10 - 2) = 24.
        """
        n_years = 10
        index = pd.date_range("2000-01-01", periods=n_years * 12, freq="MS")
        values = np.tile(np.array([20.0] * 6 + [2.0, 2.0, 2.0] + [20.0] * 3), n_years)
        observed = pd.DataFrame({"site_A": values}, index=index)
        ensemble = Ensemble(
            {rid: pd.DataFrame({"site_A": values}, index=index) for rid in range(3)}
        )
        return validate(
            ensemble,
            observed,
            metrics=["threshold_drought"],
            drought_threshold=10.0,
        )

    def test_durations(self, square_wave_result):
        summary = square_wave_result.summary().set_index("metric")
        assert summary.loc["mean_drought_duration", "observed"] == pytest.approx(3.0)
        assert summary.loc["max_drought_duration", "observed"] == pytest.approx(3.0)

    def test_severities(self, square_wave_result):
        summary = square_wave_result.summary().set_index("metric")
        assert summary.loc["mean_drought_severity", "observed"] == pytest.approx(24.0)
        assert summary.loc["max_drought_severity", "observed"] == pytest.approx(24.0)

    def test_frequency_per_timestep(self, square_wave_result):
        # 10 events over 120 timesteps
        summary = square_wave_result.summary().set_index("metric")
        assert summary.loc["drought_frequency", "observed"] == pytest.approx(10 / 120)

    def test_identical_ensemble_matches_observed(self, square_wave_result):
        summary = square_wave_result.summary()
        assert np.allclose(summary["observed"], summary["syn_median"])


class TestDefaultThreshold:
    def test_site_specific_q20(self, monthly_ensemble, monthly_observed):
        result = validate(
            monthly_ensemble, monthly_observed, metrics=["threshold_drought"]
        )
        summary = result.summary()
        # Both sites evaluated, five metrics each
        assert len(summary) == 10
        assert set(summary["site"]) == {"site_A", "site_B"}
        assert (summary["n_realizations"] == 10).all()
