"""Tests for evaluation frequency normalization and inference."""

import logging

import numpy as np
import pandas as pd
import pytest

from synhydro._evaluation._frequency import (
    normalize_frequency,
    infer_frequency,
    resolve_frequency,
    check_observed_frequency,
)


class TestNormalizeFrequency:
    @pytest.mark.parametrize(
        "value,expected_base",
        [
            ("D", "daily"),
            ("d", "daily"),
            ("daily", "daily"),
            ("W", "weekly"),
            ("W-SUN", "weekly"),
            ("weekly", "weekly"),
            ("M", "monthly"),
            ("MS", "monthly"),
            ("ME", "monthly"),
            ("monthly", "monthly"),
            ("Monthly", "monthly"),
            ("Y", "annual"),
            ("YS", "annual"),
            ("A", "annual"),
            ("AS", "annual"),
            ("A-DEC", "annual"),
            ("annual", "annual"),
            ("yearly", "annual"),
        ],
    )
    def test_recognized(self, value, expected_base):
        info = normalize_frequency(value)
        assert info is not None
        assert info.base == expected_base

    @pytest.mark.parametrize("value", [None, "", "2D", "3MS", "fortnightly", "H"])
    def test_unrecognized(self, value):
        assert normalize_frequency(value) is None

    def test_steps_per_year(self):
        assert normalize_frequency("MS").steps_per_year == 12.0
        assert normalize_frequency("D").steps_per_year == 365.25
        assert normalize_frequency("YS").steps_per_year == 1.0


class TestInferFrequency:
    def test_monthly(self):
        index = pd.date_range("2000-01-01", periods=120, freq="MS")
        assert infer_frequency(index).base == "monthly"

    def test_daily(self):
        index = pd.date_range("2000-01-01", periods=365, freq="D")
        assert infer_frequency(index).base == "daily"

    def test_weekly(self):
        index = pd.date_range("2000-01-02", periods=104, freq="W-SUN")
        assert infer_frequency(index).base == "weekly"

    def test_annual(self):
        index = pd.date_range("2000-01-01", periods=30, freq="YS")
        assert infer_frequency(index).base == "annual"

    def test_too_short(self):
        index = pd.DatetimeIndex(["2000-01-01", "2000-02-01"])
        with pytest.raises(ValueError, match="at least 3"):
            infer_frequency(index)

    def test_unsupported_spacing(self):
        index = pd.date_range("2000-01-01", periods=50, freq="4h")
        with pytest.raises(ValueError, match="Cannot infer"):
            infer_frequency(index)


class TestResolveFrequency:
    def setup_method(self):
        self.monthly_index = pd.date_range("2000-01-01", periods=120, freq="MS")

    def test_explicit_wins(self):
        info = resolve_frequency("daily", "MS", self.monthly_index)
        assert info.base == "daily"

    def test_explicit_invalid_raises(self):
        with pytest.raises(ValueError, match="Unrecognized frequency"):
            resolve_frequency("bogus", None, self.monthly_index)

    def test_metadata_used(self):
        info = resolve_frequency(None, "MS", self.monthly_index)
        assert info.base == "monthly"

    def test_metadata_literal_used(self):
        info = resolve_frequency(None, "monthly", self.monthly_index)
        assert info.base == "monthly"

    def test_metadata_disagreement_prefers_inferred(self, caplog):
        with caplog.at_level(logging.WARNING):
            info = resolve_frequency(None, "D", self.monthly_index)
        assert info.base == "monthly"
        assert any("disagrees" in rec.message for rec in caplog.records)

    def test_inference_fallback(self):
        info = resolve_frequency(None, None, self.monthly_index)
        assert info.base == "monthly"

    def test_no_source_raises(self):
        index = pd.DatetimeIndex(["2000-01-01", "2000-02-01"])
        with pytest.raises(ValueError, match="Could not determine"):
            resolve_frequency(None, None, index)


class TestCheckObservedFrequency:
    def test_match_passes(self):
        index = pd.date_range("2000-01-01", periods=120, freq="MS")
        ensemble_info = normalize_frequency("MS")
        check_observed_frequency(index, ensemble_info)

    def test_mismatch_raises(self):
        index = pd.date_range("2000-01-01", periods=365, freq="D")
        ensemble_info = normalize_frequency("MS")
        with pytest.raises(ValueError, match="does not match"):
            check_observed_frequency(index, ensemble_info)
