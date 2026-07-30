"""
Tests for plot_verification_panel and result-based metric plots.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from synhydro.plotting import (
    plot_verification_panel,
    plot_metric_distributions,
    plot_metric_curve,
)
from synhydro.verification import verify

logger = logging.getLogger(__name__)


class TestVerificationPanel:
    def test_default(self, small_ensemble):
        fig, axes = plot_verification_panel(small_ensemble)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, list)
        assert len(axes) == 5

    def test_with_observed(self, small_ensemble, observed_series):
        fig, axes = plot_verification_panel(small_ensemble, observed=observed_series)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 5

    def test_log_space(self, small_ensemble, observed_series):
        fig, axes = plot_verification_panel(
            small_ensemble, observed=observed_series, log_space=True
        )
        assert len(axes) == 5

    def test_weekly(self, small_ensemble, observed_series):
        fig, axes = plot_verification_panel(
            small_ensemble, observed=observed_series, timestep="weekly"
        )
        assert len(axes) == 5

    def test_invalid_timestep(self, small_ensemble):
        with pytest.raises(ValueError, match="timestep"):
            plot_verification_panel(small_ensemble, timestep="annual")

    def test_seed_determinism(self, small_ensemble, observed_series):
        fig_a, axes_a = plot_verification_panel(
            small_ensemble, observed=observed_series, seed=7
        )
        fig_b, axes_b = plot_verification_panel(
            small_ensemble, observed=observed_series, seed=7
        )
        # Same seed reproduces identical observed-resample boxplot vertices
        paths_a = [p.get_path().vertices for p in axes_a[1].patches]
        paths_b = [p.get_path().vertices for p in axes_b[1].patches]
        assert len(paths_a) == len(paths_b)
        for va, vb in zip(paths_a, paths_b):
            assert np.allclose(va, vb)


@pytest.fixture(scope="module")
def verification_result(module_monthly_ensemble_and_observed):
    ensemble, observed = module_monthly_ensemble_and_observed
    return verify(ensemble, observed, metrics=["marginal", "seasonal", "fdc"])


@pytest.fixture(scope="module")
def module_monthly_ensemble_and_observed():
    import pandas as pd

    from synhydro.core.ensemble import Ensemble

    rng = np.random.default_rng(3)
    index = pd.date_range("1995-01-01", periods=240, freq="MS")
    observed = pd.DataFrame({"site_a": rng.lognormal(3.0, 0.5, 240)}, index=index)
    ensemble = Ensemble(
        {
            rid: pd.DataFrame({"site_a": rng.lognormal(3.0, 0.5, 240)}, index=index)
            for rid in range(6)
        }
    )
    return ensemble, observed


class TestMetricDistributions:
    def test_default_panels(self, verification_result):
        fig, axes = plot_metric_distributions(verification_result)
        assert isinstance(fig, plt.Figure)

    def test_metric_subset(self, verification_result):
        fig, axes = plot_metric_distributions(
            verification_result, metrics=["mean", "std"]
        )
        visible = [ax for row in axes for ax in row if ax.get_visible()]
        assert len(visible) == 2

    def test_accepts_tidy_frame(self, verification_result):
        fig, axes = plot_metric_distributions(
            verification_result.to_dataframe(), metrics=["mean"]
        )
        assert isinstance(fig, plt.Figure)

    def test_no_match_raises(self, verification_result):
        with pytest.raises(ValueError, match="No scalar"):
            plot_metric_distributions(verification_result, metrics=["bogus"])


class TestMetricCurve:
    def test_fdc_log_default(self, verification_result):
        fig, ax = plot_metric_curve(verification_result, "fdc", "site_a")
        assert ax.get_yscale() == "log"

    def test_monthly_mean(self, verification_result):
        fig, ax = plot_metric_curve(verification_result, "monthly_mean", "site_a")
        assert ax.get_yscale() == "linear"

    def test_missing_metric_raises(self, verification_result):
        with pytest.raises(ValueError, match="No values found"):
            plot_metric_curve(verification_result, "acf", "site_a")

    def test_scalar_metric_raises(self, verification_result):
        with pytest.raises(ValueError, match="requires a curve metric"):
            plot_metric_curve(verification_result, "mean", "site_a")
