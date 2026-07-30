"""Tests for the validate() orchestrator."""

import numpy as np
import pandas as pd
import pytest

from synhydro.validation import validate, list_metrics, ValidationResult
from synhydro._evaluation import VALUE_COLUMNS


class TestSelection:
    def test_all(self, monthly_ensemble, monthly_observed):
        result = validate(monthly_ensemble, monthly_observed, metrics="all")
        assert isinstance(result, ValidationResult)
        assert set(result.values["category"]) == {
            "threshold_drought",
            "ssi_drought",
        }

    def test_none_raises(self, monthly_ensemble, monthly_observed):
        with pytest.raises(ValueError, match="No metrics selected"):
            validate(monthly_ensemble, monthly_observed, metrics=None)

    def test_single_category(self, monthly_ensemble, monthly_observed):
        result = validate(
            monthly_ensemble, monthly_observed, metrics=["threshold_drought"]
        )
        assert set(result.values["category"]) == {"threshold_drought"}

    def test_unknown_raises(self, monthly_ensemble, monthly_observed):
        with pytest.raises(ValueError, match="Unknown validation category"):
            validate(monthly_ensemble, monthly_observed, metrics=["bogus"])


class TestResultContent:
    def test_tidy_schema(self, monthly_ensemble, monthly_observed):
        result = validate(monthly_ensemble, monthly_observed, metrics="all")
        assert list(result.values.columns) == VALUE_COLUMNS

    def test_metadata(self, monthly_ensemble, monthly_observed):
        result = validate(monthly_ensemble, monthly_observed, metrics="all")
        assert result.metadata["suite"] == "validation"
        assert result.metadata["base_frequency"] == "monthly"
        assert result.metadata["options"]["ssi_timescale"] == 12

    def test_summary_ranks(self, monthly_ensemble, monthly_observed):
        result = validate(
            monthly_ensemble, monthly_observed, metrics=["threshold_drought"]
        )
        summary = result.summary()
        finite = summary["obs_percentile"].dropna()
        assert len(finite) > 0
        assert ((finite >= 0) & (finite <= 1)).all()

    def test_no_shared_sites_raises(self, monthly_ensemble, monthly_observed):
        other = pd.DataFrame(
            {"elsewhere": np.ones(len(monthly_observed))},
            index=monthly_observed.index,
        )
        with pytest.raises(ValueError, match="No shared sites"):
            validate(monthly_ensemble, other, metrics="all")


class TestListMetrics:
    def test_inventory(self):
        inventory = list_metrics()
        assert len(inventory) == 10
        assert set(inventory["category"]) == {"threshold_drought", "ssi_drought"}
        assert {"name", "units", "citation", "description"} <= set(inventory.columns)
