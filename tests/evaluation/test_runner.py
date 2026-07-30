"""Tests for the generic evaluation runner."""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble
from synhydro._evaluation._context import MetricContext
from synhydro._evaluation._frequency import normalize_frequency
from synhydro._evaluation._registry import MetricRegistry
from synhydro._evaluation._runner import (
    run_metrics,
    resolve_sites,
    VALUE_COLUMNS,
    SKIP_COLUMNS,
    pair_label,
)

N_YEARS = 10
N_REALIZATIONS = 3
SITES = ["site_a", "site_b"]


@pytest.fixture
def monthly_index():
    return pd.date_range("2000-01-01", periods=N_YEARS * 12, freq="MS")


@pytest.fixture
def observed(monthly_index):
    rng = np.random.default_rng(42)
    data = {
        site: rng.lognormal(mean=3.0, sigma=0.5, size=len(monthly_index))
        for site in SITES
    }
    return pd.DataFrame(data, index=monthly_index)


@pytest.fixture
def ensemble(monthly_index):
    rng = np.random.default_rng(7)
    data = {}
    for rid in range(N_REALIZATIONS):
        data[rid] = pd.DataFrame(
            {
                site: rng.lognormal(mean=3.0, sigma=0.5, size=len(monthly_index))
                for site in SITES
            },
            index=monthly_index,
        )
    return Ensemble(data)


@pytest.fixture
def registry():
    reg = MetricRegistry(suite="test")

    @reg.register(category="marginal", kind="scalar", units="flow")
    def mean(x):
        """Mean flow."""
        return float(np.mean(x))

    @reg.register(category="seasonal", kind="curve", units="flow")
    def monthly_mean(x):
        """Mean flow per calendar month."""
        return x.groupby(x.index.month).mean()

    @reg.register(category="spatial", kind="matrix")
    def cross_correlation(frame):
        """Lag-0 cross-correlation per site pair."""
        corr = frame.corr()
        out = {}
        cols = list(frame.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                out[pair_label(a, b)] = corr.loc[a, b]
        return pd.Series(out)

    @reg.register(category="marginal", kind="comparison")
    def mean_diff(x, reference):
        """Difference of means from observed."""
        return float(np.mean(x) - np.mean(reference))

    @reg.register(category="extremes", kind="scalar", frequencies=("daily",))
    def daily_only(x):
        """A daily-only metric."""
        return 0.0

    @reg.register(category="temporal", kind="scalar", min_years=50)
    def long_record_only(x):
        """Requires 50 years of observed data."""
        return 0.0

    return reg


@pytest.fixture
def context():
    return MetricContext(frequency=normalize_frequency("MS"))


class TestResolveSites:
    def test_shared_sites(self, ensemble, observed):
        assert resolve_sites(ensemble, observed) == SITES

    def test_subset(self, ensemble, observed):
        assert resolve_sites(ensemble, observed, sites=["site_b"]) == ["site_b"]

    def test_missing_requested_site_raises(self, ensemble, observed):
        with pytest.raises(ValueError, match="not present"):
            resolve_sites(ensemble, observed, sites=["site_c"])

    def test_no_shared_sites_raises(self, ensemble, monthly_index):
        other = pd.DataFrame(
            {"elsewhere": np.ones(len(monthly_index))}, index=monthly_index
        )
        with pytest.raises(ValueError, match="No shared sites"):
            resolve_sites(ensemble, other)


class TestRunMetrics:
    def test_tidy_schema(self, ensemble, observed, registry, context):
        specs = registry.select(["mean"])
        values, skipped = run_metrics(ensemble, observed, specs, context, SITES)
        assert list(values.columns) == VALUE_COLUMNS
        assert list(skipped.columns) == SKIP_COLUMNS
        assert len(values) == N_REALIZATIONS * len(SITES)

    def test_scalar_observed_column(self, ensemble, observed, registry, context):
        specs = registry.select(["mean"])
        values, _ = run_metrics(ensemble, observed, specs, context, SITES)
        site_a = values[values["site"] == "site_a"]
        expected = float(observed["site_a"].mean())
        assert np.allclose(site_a["observed"], expected)
        assert site_a["component"].isna().all()

    def test_curve_components(self, ensemble, observed, registry, context):
        specs = registry.select(["monthly_mean"])
        values, _ = run_metrics(ensemble, observed, specs, context, SITES)
        assert len(values) == N_REALIZATIONS * len(SITES) * 12
        components = sorted(values["component"].unique())
        assert components == list(range(1, 13))
        january = values[(values["site"] == "site_a") & (values["component"] == 1)]
        obs_jan = float(observed["site_a"][observed.index.month == 1].mean())
        assert np.allclose(january["observed"], obs_jan)

    def test_matrix_pair_labels(self, ensemble, observed, registry, context):
        specs = registry.select(["cross_correlation"])
        values, _ = run_metrics(ensemble, observed, specs, context, SITES)
        assert set(values["site"]) == {pair_label("site_a", "site_b")}
        assert len(values) == N_REALIZATIONS
        expected = float(observed["site_a"].corr(observed["site_b"]))
        assert np.allclose(values["observed"], expected)

    def test_matrix_single_site_skipped(self, ensemble, observed, registry, context):
        specs = registry.select(["cross_correlation"])
        values, skipped = run_metrics(ensemble, observed, specs, context, ["site_a"])
        assert values.empty
        assert skipped.iloc[0]["reason"] == "requires at least 2 sites"

    def test_comparison_observed_nan(self, ensemble, observed, registry, context):
        specs = registry.select(["mean_diff"])
        values, _ = run_metrics(ensemble, observed, specs, context, SITES)
        assert values["observed"].isna().all()
        assert len(values) == N_REALIZATIONS * len(SITES)

    def test_frequency_gating(self, ensemble, observed, registry, context):
        specs = registry.select(["daily_only"])
        values, skipped = run_metrics(ensemble, observed, specs, context, SITES)
        assert values.empty
        assert len(skipped) == 1
        assert "daily" in skipped.iloc[0]["reason"]

    def test_min_years_gating(self, ensemble, observed, registry, context):
        specs = registry.select(["long_record_only"])
        values, skipped = run_metrics(ensemble, observed, specs, context, SITES)
        assert values.empty
        assert "50" in skipped.iloc[0]["reason"]
