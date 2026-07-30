"""Tests for the EvaluationResult container."""

import numpy as np
import pandas as pd
import pytest

from synhydro._evaluation._result import EvaluationResult
from synhydro._evaluation._runner import VALUE_COLUMNS, SKIP_COLUMNS


def make_values(rows):
    return pd.DataFrame(rows, columns=VALUE_COLUMNS)


def scalar_rows(metric, site, values, observed, units="flow", category="marginal"):
    return [
        (category, metric, "scalar", site, pd.NA, rid, value, observed, units)
        for rid, value in enumerate(values)
    ]


@pytest.fixture
def simple_result():
    values = make_values(scalar_rows("mean", "site_a", list(range(1, 21)), 10.5))
    return EvaluationResult(
        values=values,
        skipped=pd.DataFrame(columns=SKIP_COLUMNS),
        metadata={
            "n_realizations": 20,
            "n_sites": 1,
            "base_frequency": "monthly",
            "obs_site_median_flow": {"site_a": 10.0},
        },
    )


class TestSummary:
    def test_obs_percentile_at_median(self, simple_result):
        summary = simple_result.summary()
        assert len(summary) == 1
        row = summary.iloc[0]
        # observed 10.5 among values 1..20: 10 values below, none equal
        assert row["obs_percentile"] == pytest.approx((10 + 0.5) / 21)
        assert bool(row["in_90_band"]) is True

    def test_quantile_columns(self, simple_result):
        row = simple_result.summary().iloc[0]
        vals = np.arange(1, 21)
        assert row["syn_median"] == pytest.approx(np.median(vals))
        assert row["syn_mean"] == pytest.approx(np.mean(vals))
        assert row["syn_q05"] == pytest.approx(np.percentile(vals, 5))
        assert row["syn_q95"] == pytest.approx(np.percentile(vals, 95))
        assert row["n_realizations"] == 20

    def test_relative_diff(self, simple_result):
        row = simple_result.summary().iloc[0]
        expected = (np.median(np.arange(1, 21)) - 10.5) / 10.5
        assert row["relative_diff"] == pytest.approx(expected)

    def test_obs_in_tail(self):
        values = make_values(scalar_rows("mean", "site_a", list(range(1, 21)), 100.0))
        result = EvaluationResult(
            values=values, skipped=pd.DataFrame(columns=SKIP_COLUMNS)
        )
        row = result.summary().iloc[0]
        assert row["obs_percentile"] == pytest.approx((20 + 0.5) / 21)
        assert bool(row["in_90_band"]) is False

    def test_near_zero_observed_flow_guard(self):
        # Observed value below 1 percent of site median flow: no relative diff
        values = make_values(scalar_rows("minimum", "site_a", [0.01, 0.02, 0.03], 0.05))
        result = EvaluationResult(
            values=values,
            skipped=pd.DataFrame(columns=SKIP_COLUMNS),
            metadata={"obs_site_median_flow": {"site_a": 100.0}},
        )
        row = result.summary().iloc[0]
        assert np.isnan(row["relative_diff"])
        # Rank reporting still works
        assert np.isfinite(row["obs_percentile"])

    def test_reject_rate_metric(self):
        pvals = [0.01, 0.2, 0.03, 0.5, 0.04]
        values = make_values(
            scalar_rows(
                "ranksum_pvalue",
                "site_a",
                pvals,
                np.nan,
                units="pvalue",
                category="seasonal",
            )
        )
        result = EvaluationResult(
            values=values,
            skipped=pd.DataFrame(columns=SKIP_COLUMNS),
            metadata={"reject_rate_metrics": ["ranksum_pvalue"]},
        )
        row = result.summary().iloc[0]
        assert row["reject_rate"] == pytest.approx(3 / 5)
        assert np.isnan(row["obs_percentile"])
        assert np.isnan(row["relative_diff"])

    def test_comparison_metric_no_rank(self):
        values = make_values(
            [
                (
                    "marginal",
                    "ks_statistic",
                    "comparison",
                    "site_a",
                    pd.NA,
                    rid,
                    val,
                    np.nan,
                    "dimensionless",
                )
                for rid, val in enumerate([0.1, 0.2, 0.15])
            ]
        )
        result = EvaluationResult(
            values=values, skipped=pd.DataFrame(columns=SKIP_COLUMNS)
        )
        row = result.summary().iloc[0]
        assert np.isnan(row["observed"])
        assert np.isnan(row["obs_percentile"])
        assert row["syn_median"] == pytest.approx(0.15)

    def test_curve_components_grouped(self):
        rows = []
        for month in [1, 2]:
            for rid in range(3):
                rows.append(
                    (
                        "seasonal",
                        "monthly_mean",
                        "curve",
                        "site_a",
                        month,
                        rid,
                        float(month * 10 + rid),
                        float(month * 10 + 1),
                        "flow",
                    )
                )
        result = EvaluationResult(
            values=make_values(rows), skipped=pd.DataFrame(columns=SKIP_COLUMNS)
        )
        summary = result.summary()
        assert len(summary) == 2
        assert sorted(summary["component"]) == [1, 2]

    def test_empty_result(self):
        result = EvaluationResult(
            values=pd.DataFrame(columns=VALUE_COLUMNS),
            skipped=pd.DataFrame(columns=SKIP_COLUMNS),
        )
        assert result.summary().empty
        assert result.category_summary().empty


class TestCategorySummary:
    def test_rollup(self):
        rows = (
            scalar_rows("mean", "site_a", [9.0, 10.0, 11.0], 10.0)
            + scalar_rows("std", "site_a", [2.0, 2.5, 3.0], 2.5)
            + scalar_rows(
                "lag1",
                "site_a",
                [0.4, 0.5, 0.6],
                0.9,
                units="dimensionless",
                category="temporal",
            )
        )
        result = EvaluationResult(
            values=make_values(rows), skipped=pd.DataFrame(columns=SKIP_COLUMNS)
        )
        rollup = result.category_summary()
        assert len(rollup) == 2
        marginal = rollup[rollup["category"] == "marginal"].iloc[0]
        assert marginal["n_metrics"] == 2
        assert marginal["n_in_90_band"] == 2
        temporal = rollup[rollup["category"] == "temporal"].iloc[0]
        # observed 0.9 lies above all synthetic values
        assert temporal["n_in_90_band"] == 0
        assert temporal["median_obs_percentile_distance"] > 0.3


class TestDisplay:
    def test_repr(self, simple_result):
        text = repr(simple_result)
        assert "EvaluationResult" in text
        assert "metrics=1" in text

    def test_repr_html_smoke(self, simple_result):
        html = simple_result._repr_html_()
        assert "<table" in html

    def test_to_dataframe_is_copy(self, simple_result):
        frame = simple_result.to_dataframe()
        frame.loc[0, "value"] = -999.0
        assert simple_result.values.loc[0, "value"] != -999.0
