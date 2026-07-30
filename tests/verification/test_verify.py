"""Tests for the verify() orchestrator."""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble
from synhydro.verification import verify, list_metrics, VerificationResult
from synhydro._evaluation import VALUE_COLUMNS


class TestMetricSelection:
    def test_all(self, monthly_ensemble, monthly_observed):
        result = verify(monthly_ensemble, monthly_observed, metrics="all")
        assert isinstance(result, VerificationResult)
        categories = set(result.values["category"])
        assert categories >= {
            "marginal",
            "temporal",
            "seasonal",
            "annual",
            "spatial",
            "fdc",
            "lmoments",
            "extremes",
            "spectral",
        }

    def test_none_raises(self, monthly_ensemble, monthly_observed):
        with pytest.raises(ValueError, match="No metrics selected"):
            verify(monthly_ensemble, monthly_observed, metrics=None)

    def test_by_category(self, monthly_ensemble, monthly_observed):
        result = verify(monthly_ensemble, monthly_observed, metrics=["marginal"])
        assert set(result.values["category"]) == {"marginal"}

    def test_by_name(self, monthly_ensemble, monthly_observed):
        result = verify(monthly_ensemble, monthly_observed, metrics=["mean", "std"])
        assert set(result.values["metric"]) == {"mean", "std"}

    def test_custom_callable(self, monthly_ensemble, monthly_observed):
        def q25(x):
            """25th percentile of flows."""
            return float(x.quantile(0.25))

        result = verify(monthly_ensemble, monthly_observed, metrics=[q25])
        assert set(result.values["metric"]) == {"q25"}
        assert set(result.values["category"]) == {"custom"}

    def test_unknown_metric_raises(self, monthly_ensemble, monthly_observed):
        with pytest.raises(ValueError, match="Unknown metric"):
            verify(monthly_ensemble, monthly_observed, metrics=["bogus"])


class TestResultContent:
    def test_tidy_schema(self, monthly_ensemble, monthly_observed):
        result = verify(monthly_ensemble, monthly_observed, metrics=["mean"])
        assert list(result.values.columns) == VALUE_COLUMNS
        n_realizations = len(monthly_ensemble.data_by_realization)
        assert len(result.values) == n_realizations * 2

    def test_metadata(self, monthly_ensemble, monthly_observed):
        result = verify(monthly_ensemble, monthly_observed, metrics=["mean"])
        assert result.metadata["suite"] == "verification"
        assert result.metadata["base_frequency"] == "monthly"
        assert result.metadata["steps_per_year"] == 12.0
        assert result.metadata["n_realizations"] == 10
        assert set(result.metadata["obs_site_median_flow"]) == {"site_a", "site_b"}

    def test_sites_subset(self, monthly_ensemble, monthly_observed):
        result = verify(
            monthly_ensemble, monthly_observed, metrics=["mean"], sites=["site_a"]
        )
        assert set(result.values["site"]) == {"site_a"}

    def test_no_shared_sites_raises(self, monthly_ensemble, monthly_index):
        other = pd.DataFrame(
            {"elsewhere": np.ones(len(monthly_index))}, index=monthly_index
        )
        with pytest.raises(ValueError, match="No shared sites"):
            verify(monthly_ensemble, other, metrics=["mean"])


class TestFrequencyBehavior:
    def test_daily_computes_seven_day_min(self, daily_ensemble, daily_observed):
        result = verify(daily_ensemble, daily_observed, metrics=["seven_day_min_mean"])
        assert not result.values.empty
        assert result.skipped.empty

    def test_monthly_skips_seven_day_min(self, monthly_ensemble, monthly_observed):
        result = verify(
            monthly_ensemble, monthly_observed, metrics=["seven_day_min_mean"]
        )
        assert result.values.empty
        assert len(result.skipped) == 1
        assert "daily" in result.skipped.iloc[0]["reason"]

    def test_observed_frequency_mismatch_raises(self, monthly_ensemble, daily_observed):
        with pytest.raises(ValueError, match="does not match"):
            verify(monthly_ensemble, daily_observed, metrics=["mean"])

    def test_acf_lags_default_by_frequency(self, daily_ensemble, daily_observed):
        result = verify(daily_ensemble, daily_observed, metrics=["acf"])
        assert result.metadata["options"]["acf_lags"] == 30
        assert result.values["component"].max() == 30


class TestListMetrics:
    def test_inventory(self):
        inventory = list_metrics()
        assert len(inventory) > 40
        assert {"name", "category", "kind", "units", "citation"} <= set(
            inventory.columns
        )
        assert "mean" in list(inventory["name"])
