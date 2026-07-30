"""Tests for SSI drought validation metrics (migrated from the old suite)."""

import numpy as np
import pandas as pd
import pytest

from synhydro.validation import validate
from synhydro.validation.metrics.ssi_drought import _extract_ssi_droughts


class TestExtractSSIDroughts:
    def test_basic_event(self):
        """Simple SSI sequence with one drought event."""
        ssi = np.array([0.5, -0.5, -1.2, -1.5, -0.8, 0.3, 0.5])
        result = _extract_ssi_droughts(ssi, threshold=-1.0)
        # Drought starts at -1.2, ends when returning above 0 at 0.3
        assert len(result["durations"]) == 1
        assert result["durations"][0] >= 2

    def test_no_drought(self):
        """No drought when SSI stays above threshold."""
        ssi = np.array([0.5, 0.2, -0.3, -0.5, 0.1, 0.8])
        result = _extract_ssi_droughts(ssi, threshold=-1.0)
        assert len(result["durations"]) == 0

    def test_severity_positive(self):
        """Severity values are positive (absolute cumulative deficit)."""
        ssi = np.array([0.5, -1.5, -2.0, -1.0, 0.5])
        result = _extract_ssi_droughts(ssi, threshold=-1.0)
        for severity in result["severities"]:
            assert severity > 0

    def test_open_ended_event_counted(self):
        ssi = np.array([0.5, -1.5, -0.5])
        result = _extract_ssi_droughts(ssi, threshold=-1.0)
        assert result["durations"] == [2]


class TestSSIDroughtCategory:
    def test_metrics_computed(self, monthly_ensemble, monthly_observed):
        result = validate(monthly_ensemble, monthly_observed, metrics=["ssi_drought"])
        metrics = set(result.values["metric"])
        assert metrics == {
            "ssi_mean_drought_duration",
            "ssi_max_drought_duration",
            "ssi_mean_drought_severity",
            "ssi_max_drought_severity",
            "ssi_drought_frequency",
        }

    def test_frequency_units_per_year(self, monthly_ensemble, monthly_observed):
        result = validate(monthly_ensemble, monthly_observed, metrics=["ssi_drought"])
        frequency_rows = result.values[
            result.values["metric"] == "ssi_drought_frequency"
        ]
        assert (frequency_rows["units"] == "per_year").all()

    def test_short_observed_skipped(self, monthly_ensemble):
        short_obs = pd.DataFrame(
            {"site_A": np.random.default_rng(42).lognormal(5, 0.3, 12)},
            index=pd.date_range("2000-01-01", periods=12, freq="MS"),
        )
        result = validate(monthly_ensemble, short_obs, metrics=["ssi_drought"])
        assert result.values.empty
        assert len(result.skipped) == 1
        assert "36" in result.skipped.iloc[0]["reason"]
