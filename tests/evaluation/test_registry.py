"""Tests for the evaluation metric registry."""

import numpy as np
import pandas as pd
import pytest

from synhydro._evaluation._registry import MetricRegistry


@pytest.fixture
def registry():
    reg = MetricRegistry(suite="test")

    @reg.register(category="marginal", kind="scalar", units="flow")
    def mean(x):
        """Mean flow."""
        return float(np.mean(x))

    @reg.register(category="marginal", kind="scalar", units="flow")
    def std(x):
        """Flow standard deviation."""
        return float(np.std(x, ddof=1))

    @reg.register(category="temporal", kind="scalar")
    def lag1(x):
        """Lag-1 autocorrelation."""
        return float(x.autocorr(lag=1))

    return reg


class TestRegister:
    def test_duplicate_name_raises(self, registry):
        with pytest.raises(ValueError, match="already registered"):

            @registry.register(category="marginal")
            def mean(x):
                return 0.0

    def test_invalid_kind_raises(self, registry):
        with pytest.raises(ValueError, match="Invalid metric kind"):

            @registry.register(kind="tensor")
            def new_metric(x):
                return 0.0

    def test_invalid_summary_mode_raises(self, registry):
        with pytest.raises(ValueError, match="Invalid summary_mode"):

            @registry.register(summary_mode="average")
            def new_metric(x):
                return 0.0

    def test_description_from_docstring(self, registry):
        assert registry.get("mean").description == "Mean flow."

    def test_plain_call_registration(self, registry):
        def custom(x):
            return 1.0

        registry.register(custom, category="custom")
        assert "custom" in registry.names()


class TestSelect:
    def test_all(self, registry):
        specs = registry.select("all")
        assert [s.name for s in specs] == ["mean", "std", "lag1"]

    def test_by_name(self, registry):
        specs = registry.select(["mean", "lag1"])
        assert [s.name for s in specs] == ["mean", "lag1"]

    def test_by_category(self, registry):
        specs = registry.select(["marginal"])
        assert [s.name for s in specs] == ["mean", "std"]

    def test_single_string_name(self, registry):
        specs = registry.select("mean")
        assert [s.name for s in specs] == ["mean"]

    def test_mixed_with_callable(self, registry):
        def my_metric(x):
            """A custom metric."""
            return 0.0

        specs = registry.select(["mean", my_metric])
        assert [s.name for s in specs] == ["mean", "my_metric"]
        custom = specs[1]
        assert custom.category == "custom"
        assert custom.kind == "scalar"
        assert custom.description == "A custom metric."

    def test_deduplicates(self, registry):
        specs = registry.select(["mean", "marginal"])
        assert [s.name for s in specs] == ["mean", "std"]

    def test_unknown_raises(self, registry):
        with pytest.raises(ValueError, match="Unknown metric or category"):
            registry.select(["bogus"])

    def test_none_raises(self, registry):
        with pytest.raises(ValueError, match="No metrics selected"):
            registry.select(None)


class TestInventory:
    def test_names_and_categories(self, registry):
        assert registry.names() == ["mean", "std", "lag1"]
        assert registry.categories() == ["marginal", "temporal"]

    def test_to_frame(self, registry):
        frame = registry.to_frame()
        assert list(frame["name"]) == ["mean", "std", "lag1"]
        assert set(frame.columns) >= {
            "name",
            "category",
            "kind",
            "units",
            "frequencies",
            "citation",
            "description",
        }
        assert frame.loc[0, "frequencies"] == "any"

    def test_get_unknown_raises(self, registry):
        with pytest.raises(KeyError, match="Unknown metric"):
            registry.get("bogus")
