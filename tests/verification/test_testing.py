"""Tests for bootstrap CI and generator comparison tools."""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble
from synhydro.verification import verify, bootstrap_metric_ci, compare_methods


def make_ensemble(n_realizations, sigma, seed, n_months=240):
    rng = np.random.default_rng(seed)
    index = pd.date_range("1990-01-01", periods=n_months, freq="MS")
    return Ensemble(
        {
            rid: pd.DataFrame(
                {"site_a": rng.lognormal(3.0, sigma, n_months)}, index=index
            )
            for rid in range(n_realizations)
        }
    )


@pytest.fixture(scope="module")
def observed():
    rng = np.random.default_rng(1)
    index = pd.date_range("1990-01-01", periods=240, freq="MS")
    return pd.DataFrame({"site_a": rng.lognormal(3.0, 0.5, 240)}, index=index)


@pytest.fixture(scope="module")
def result_small(observed):
    return verify(make_ensemble(10, 0.5, 2), observed, metrics=["marginal"])


@pytest.fixture(scope="module")
def result_large(observed):
    return verify(make_ensemble(40, 0.5, 3), observed, metrics=["marginal"])


class TestBootstrapMetricCI:
    def test_output_schema(self, result_small):
        table = bootstrap_metric_ci(result_small, seed=0)
        assert {
            "category",
            "metric",
            "site",
            "component",
            "observed",
            "estimate",
            "ci_lower",
            "ci_upper",
            "relative_diff",
            "rd_ci_lower",
            "rd_ci_upper",
            "n_realizations",
        } <= set(table.columns)
        assert (table["ci_lower"] <= table["ci_upper"]).all()

    def test_seed_reproducibility(self, result_small):
        first = bootstrap_metric_ci(result_small, seed=42)
        second = bootstrap_metric_ci(result_small, seed=42)
        pd.testing.assert_frame_equal(first, second)

    def test_ci_narrows_with_more_realizations(self, result_small, result_large):
        small = bootstrap_metric_ci(result_small, seed=0).set_index("metric")
        large = bootstrap_metric_ci(result_large, seed=0).set_index("metric")
        width_small = small.loc["mean", "ci_upper"] - small.loc["mean", "ci_lower"]
        width_large = large.loc["mean", "ci_upper"] - large.loc["mean", "ci_lower"]
        assert width_large < width_small

    def test_accepts_tidy_frame(self, result_small):
        table = bootstrap_metric_ci(result_small.to_dataframe(), seed=0)
        assert len(table) > 0

    def test_comparison_metric_no_relative(self, result_small):
        table = bootstrap_metric_ci(result_small, seed=0).set_index("metric")
        assert np.isnan(table.loc["ks_statistic", "relative_diff"])
        assert np.isfinite(table.loc["ks_statistic", "estimate"])

    def test_invalid_statistic_raises(self, result_small):
        with pytest.raises(ValueError, match="Unknown statistic"):
            bootstrap_metric_ci(result_small, statistic="mode")

    def test_invalid_input_raises(self):
        with pytest.raises(TypeError, match="Expected an EvaluationResult"):
            bootstrap_metric_ci([1, 2, 3])


class TestCompareMethods:
    def test_identical_generators_not_significant(self, observed):
        result_a = verify(make_ensemble(20, 0.5, 5), observed, metrics=["mean"])
        result_b = verify(make_ensemble(20, 0.5, 6), observed, metrics=["mean"])
        table = compare_methods(result_a, result_b, seed=0)
        assert len(table) == 1
        row = table.iloc[0]
        assert bool(row["paired"]) is True
        assert row["better_method"] == "none" or not row["significant"]

    def test_biased_generator_detected(self, observed):
        result_good = verify(make_ensemble(20, 0.5, 7), observed, metrics=["mean"])
        biased = make_ensemble(20, 0.5, 8)
        biased_data = {
            rid: frame * 3.0 for rid, frame in biased.data_by_realization.items()
        }
        result_bad = verify(Ensemble(biased_data), observed, metrics=["mean"])
        table = compare_methods(result_good, result_bad, seed=0)
        row = table.iloc[0]
        assert bool(row["significant"]) is True
        assert row["better_method"] == "a"
        assert row["method_a_mae"] < row["method_b_mae"]

    def test_unequal_sizes_unpaired(self, observed, caplog):
        result_a = verify(make_ensemble(10, 0.5, 9), observed, metrics=["mean"])
        result_b = verify(make_ensemble(15, 0.5, 10), observed, metrics=["mean"])
        import logging

        with caplog.at_level(logging.WARNING):
            table = compare_methods(result_a, result_b, seed=0)
        assert (~table["paired"]).all()
        assert any("independent" in rec.message for rec in caplog.records)

    def test_seed_reproducibility(self, observed):
        result_a = verify(make_ensemble(10, 0.5, 11), observed, metrics=["mean"])
        result_b = verify(make_ensemble(10, 0.5, 12), observed, metrics=["mean"])
        first = compare_methods(result_a, result_b, seed=3)
        second = compare_methods(result_a, result_b, seed=3)
        pd.testing.assert_frame_equal(first, second)
