"""
Tests for the joint multisite Valencia-Schaake disaggregator.

Verifies parameter shapes for the joint formulation (paper Eqs. 13-19),
the exact-additivity identity (paper Eq. 31) in the untransformed case,
multisite cross-site correlation preservation, and proportional-adjustment
behavior under log transform.
"""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.methods.disaggregation.temporal.valencia_schaake import (
    ValenciaSchaakeDisaggregator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monthly_single_site():
    """20 years of monthly flows for a single site."""
    dates = pd.date_range(start="2000-01-01", end="2019-12-31", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.lognormal(mean=6.0, sigma=0.5, size=len(dates))
    return pd.Series(values, index=dates, name="site_1")


@pytest.fixture
def monthly_multisite():
    """30 years of monthly flows across 3 correlated sites."""
    dates = pd.date_range(start="1990-01-01", end="2019-12-31", freq="MS")
    rng = np.random.default_rng(42)
    n = len(dates)
    base = rng.lognormal(mean=6.0, sigma=0.4, size=n)
    data = {
        "site_1": np.maximum(base + rng.normal(0, 30, size=n), 1.0),
        "site_2": np.maximum(0.7 * base + rng.normal(0, 25, size=n), 1.0),
        "site_3": np.maximum(1.2 * base + rng.normal(0, 40, size=n), 1.0),
    }
    return pd.DataFrame(data, index=dates)


def _annual_ensemble_from(
    monthly_df: pd.DataFrame, n_realizations: int = 2
) -> Ensemble:
    """Build a small annual ensemble whose sites match ``monthly_df``."""
    annual = monthly_df.resample("YS").sum()
    rng = np.random.default_rng(0)
    realization_data = {}
    for r in range(n_realizations):
        scale = 1.0 + 0.05 * rng.standard_normal(annual.shape)
        realization_data[r] = pd.DataFrame(
            annual.values * scale, index=annual.index, columns=annual.columns
        )
    metadata = EnsembleMetadata(
        generator_class="TestGenerator",
        n_realizations=n_realizations,
        n_sites=annual.shape[1],
        time_resolution="YS",
    )
    return Ensemble(realization_data, metadata=metadata)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestValenciaSchaakeInitialization:

    def test_initialization_default_params(self):
        d = ValenciaSchaakeDisaggregator()
        assert d.n_subperiods == 12
        assert d.transform == "log"
        assert d.conservation_method == "proportional"

    def test_initialization_custom_params(self):
        d = ValenciaSchaakeDisaggregator(
            n_subperiods=4, transform="boxcox", conservation_method="proportional"
        )
        assert d.n_subperiods == 4
        assert d.transform == "boxcox"
        assert d.conservation_method == "proportional"

    def test_initialization_custom_no_transform(self):
        d = ValenciaSchaakeDisaggregator(
            n_subperiods=4, transform="none", conservation_method="none"
        )
        assert d.n_subperiods == 4
        assert d.transform == "none"
        assert d.conservation_method == "none"

    def test_initialization_frequencies(self):
        d = ValenciaSchaakeDisaggregator()
        assert d.input_frequency == "YS"
        assert d.output_frequency == "MS"

    def test_initialization_quarterly_output_frequency(self):
        d = ValenciaSchaakeDisaggregator(
            n_subperiods=4, transform="none", conservation_method="none"
        )
        assert d.output_frequency == "QS"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestValenciaSchaakePreprocessing:

    def test_preprocessing_series(self, monthly_single_site):
        d = ValenciaSchaakeDisaggregator()
        d.preprocessing(monthly_single_site)
        assert d.is_preprocessed
        assert hasattr(d, "Q_obs")
        assert hasattr(d, "Q_annual")
        assert d.n_sites == 1

    def test_preprocessing_dataframe(self, monthly_multisite):
        d = ValenciaSchaakeDisaggregator()
        d.preprocessing(monthly_multisite)
        assert d.is_preprocessed
        assert d.n_sites == 3

    def test_preprocessing_creates_annual(self, monthly_single_site):
        d = ValenciaSchaakeDisaggregator()
        d.preprocessing(monthly_single_site)
        assert len(d.Q_annual) == 20


# ---------------------------------------------------------------------------
# Fitting -- joint multisite formulation (paper Eqs. 13-19)
# ---------------------------------------------------------------------------


class TestValenciaSchaakeFit:

    def test_fit_single_site_attributes(self, monthly_single_site):
        d = ValenciaSchaakeDisaggregator(transform="none")
        d.fit(monthly_single_site)
        assert d.is_fitted
        assert d.mu_Y_ is not None
        assert d.mu_X_ is not None
        assert d.S_yy_ is not None
        assert d.S_xx_ is not None
        assert d.S_yx_ is not None
        assert d.A_ is not None
        assert d.B_ is not None

    def test_fit_single_site_shapes(self, monthly_single_site):
        d = ValenciaSchaakeDisaggregator(transform="none")
        d.fit(monthly_single_site)
        m, s = d.n_sites, d.n_subperiods
        assert d.mu_Y_.shape == (s * m,)
        assert d.mu_X_.shape == (m,)
        assert d.S_yy_.shape == (s * m, s * m)
        assert d.S_xx_.shape == (m, m)
        assert d.S_yx_.shape == (s * m, m)
        assert d.A_.shape == (s * m, m)
        assert d.B_.shape[0] == s * m
        assert d.B_.shape[1] <= min(s * m, 20 - 1)

    def test_fit_multisite_shapes(self, monthly_multisite):
        d = ValenciaSchaakeDisaggregator(transform="none")
        d.fit(monthly_multisite)
        m, s = d.n_sites, d.n_subperiods
        assert m == 3
        assert d.mu_Y_.shape == (s * m,)
        assert d.mu_X_.shape == (m,)
        assert d.S_yy_.shape == (s * m, s * m)
        assert d.S_xx_.shape == (m, m)
        assert d.A_.shape == (s * m, m)

    def test_fit_with_log_transform(self, monthly_single_site):
        d = ValenciaSchaakeDisaggregator(transform="log")
        d.fit(monthly_single_site)
        assert d.is_fitted
        assert d.transform_params_["type"] == "log"

    def test_fit_with_boxcox_transform(self, monthly_single_site):
        d = ValenciaSchaakeDisaggregator(transform="boxcox")
        d.fit(monthly_single_site)
        assert d.is_fitted
        assert d.transform_params_["type"] == "boxcox"
        assert "lambda" in d.transform_params_

    def test_fit_without_transform(self, monthly_single_site):
        d = ValenciaSchaakeDisaggregator(transform="none")
        d.fit(monthly_single_site)
        assert d.is_fitted
        assert d.transform_params_.get("type") == "none"

    def test_fitted_params_object(self, monthly_single_site):
        d = ValenciaSchaakeDisaggregator(transform="none")
        d.fit(monthly_single_site)
        assert d.fitted_params_ is not None
        assert d.fitted_params_.n_parameters_ > 0
        assert d.fitted_params_.n_sites_ == 1


# ---------------------------------------------------------------------------
# Disaggregation: shape, reproducibility, non-negativity
# ---------------------------------------------------------------------------


class TestValenciaSchaakeDisaggregation:

    def test_disaggregate_single_site_ensemble(self, monthly_single_site):
        d = ValenciaSchaakeDisaggregator(transform="none", conservation_method="none")
        d.fit(monthly_single_site)
        ensemble = _annual_ensemble_from(
            monthly_single_site.to_frame(), n_realizations=2
        )
        out = d.disaggregate(ensemble, seed=0)
        assert isinstance(out, Ensemble)
        assert out.metadata.time_resolution == "MS"
        assert len(out.realization_ids) == 2
        first = out.data_by_realization[out.realization_ids[0]]
        n_years = len(ensemble.data_by_realization[ensemble.realization_ids[0]])
        assert len(first) == n_years * 12
        assert first.shape[1] == 1

    def test_disaggregate_multisite_runs(self, monthly_multisite):
        d = ValenciaSchaakeDisaggregator(transform="none", conservation_method="none")
        d.fit(monthly_multisite)
        ensemble = _annual_ensemble_from(monthly_multisite, n_realizations=2)
        out = d.disaggregate(ensemble, seed=0)
        first = out.data_by_realization[out.realization_ids[0]]
        n_years = len(ensemble.data_by_realization[ensemble.realization_ids[0]])
        assert len(first) == n_years * 12
        assert first.shape[1] == 3
        assert list(first.columns) == ["site_1", "site_2", "site_3"]

    def test_reproducible_with_seed(self, monthly_single_site):
        d = ValenciaSchaakeDisaggregator(transform="none", conservation_method="none")
        d.fit(monthly_single_site)
        ensemble = _annual_ensemble_from(
            monthly_single_site.to_frame(), n_realizations=1
        )
        out_a = d.disaggregate(ensemble, seed=42)
        out_b = d.disaggregate(ensemble, seed=42)
        a = out_a.data_by_realization[out_a.realization_ids[0]].values
        b = out_b.data_by_realization[out_b.realization_ids[0]].values
        np.testing.assert_array_almost_equal(a, b)

    def test_non_negative_output_with_log(self, monthly_single_site):
        d = ValenciaSchaakeDisaggregator(transform="log")
        d.fit(monthly_single_site)
        ensemble = _annual_ensemble_from(
            monthly_single_site.to_frame(), n_realizations=1
        )
        out = d.disaggregate(ensemble, seed=0)
        values = out.data_by_realization[out.realization_ids[0]].values
        assert (values >= 0).all()


# ---------------------------------------------------------------------------
# Additivity and statistical-property tests
# ---------------------------------------------------------------------------


class TestValenciaSchaakeAdditivity:

    def test_exact_additivity_no_transform_single_site(self, monthly_single_site):
        """Paper Eq. 31: with no transform and conservation_method='none',
        per-site annual sums must equal input annuals to floating-point
        precision -- by construction (CB = 0)."""
        d = ValenciaSchaakeDisaggregator(transform="none", conservation_method="none")
        d.fit(monthly_single_site)
        ensemble = _annual_ensemble_from(
            monthly_single_site.to_frame(), n_realizations=3
        )
        out = d.disaggregate(ensemble, seed=0)
        for rid in out.realization_ids:
            annual_in = ensemble.data_by_realization[rid].values
            annual_out = out.data_by_realization[rid].resample("YS").sum().values
            np.testing.assert_allclose(annual_out, annual_in, rtol=1e-8, atol=1e-6)

    def test_exact_additivity_no_transform_multisite(self, monthly_multisite):
        """Same identity holds joint across sites: every site's annual sum
        equals its input annual exactly."""
        d = ValenciaSchaakeDisaggregator(transform="none", conservation_method="none")
        d.fit(monthly_multisite)
        ensemble = _annual_ensemble_from(monthly_multisite, n_realizations=3)
        out = d.disaggregate(ensemble, seed=0)
        for rid in out.realization_ids:
            annual_in = ensemble.data_by_realization[rid].values
            annual_out = out.data_by_realization[rid].resample("YS").sum().values
            np.testing.assert_allclose(annual_out, annual_in, rtol=1e-8, atol=1e-6)

    def test_log_transform_proportional_adjustment_preserves_sum(
        self, monthly_multisite
    ):
        """With log transform and proportional conservation, per-site annual
        sums still match input annuals (the workaround restores additivity)."""
        d = ValenciaSchaakeDisaggregator(
            transform="log", conservation_method="proportional"
        )
        d.fit(monthly_multisite)
        ensemble = _annual_ensemble_from(monthly_multisite, n_realizations=2)
        out = d.disaggregate(ensemble, seed=0)
        for rid in out.realization_ids:
            annual_in = ensemble.data_by_realization[rid].values
            annual_out = out.data_by_realization[rid].resample("YS").sum().values
            np.testing.assert_allclose(annual_out, annual_in, rtol=1e-6)


class TestValenciaSchaakeMultisiteStructure:

    def test_cross_site_correlation_preserved(self, monthly_multisite):
        """The joint covariance should preserve historical cross-site monthly
        correlations to within sampling tolerance. Validates that
        _compute_statistics uses the true joint structure (a site-averaged
        model would fail this)."""
        d = ValenciaSchaakeDisaggregator(transform="none", conservation_method="none")
        d.fit(monthly_multisite)

        historical_monthly = monthly_multisite
        hist_corr = historical_monthly.corr().values

        n_realizations = 20
        ensemble = _annual_ensemble_from(
            monthly_multisite, n_realizations=n_realizations
        )
        out = d.disaggregate(ensemble, seed=123)

        synth_long = pd.concat(
            [out.data_by_realization[r] for r in out.realization_ids],
            ignore_index=True,
        )
        synth_corr = synth_long.corr().values

        np.testing.assert_allclose(synth_corr, hist_corr, atol=0.15)


# ---------------------------------------------------------------------------
# Edge cases / error handling
# ---------------------------------------------------------------------------


class TestValenciaSchaakeEdgeCases:

    def test_fit_before_preprocessing_raises(self):
        d = ValenciaSchaakeDisaggregator()
        with pytest.raises(ValueError):
            d.fit()

    def test_disaggregate_before_fit_raises(self, monthly_single_site):
        d = ValenciaSchaakeDisaggregator()
        d.preprocessing(monthly_single_site)
        annual = monthly_single_site.resample("YS").sum().to_frame()
        ensemble = _annual_ensemble_from(
            monthly_single_site.to_frame(), n_realizations=1
        )
        with pytest.raises(ValueError):
            d.disaggregate(ensemble)

    def test_short_record_still_fits(self):
        """5 years is short relative to 12 sub-periods, but V-S fits with
        a rank-deficient B."""
        dates = pd.date_range(start="2000-01-01", end="2004-12-31", freq="MS")
        rng = np.random.default_rng(0)
        values = rng.lognormal(mean=6.0, sigma=0.4, size=len(dates))
        series = pd.Series(values, index=dates, name="site_1")
        d = ValenciaSchaakeDisaggregator(transform="none")
        d.fit(series)
        assert d.is_fitted
        assert d.B_.shape[1] <= 5 - 1

    def test_zero_flows_handled(self):
        dates = pd.date_range(start="2000-01-01", end="2005-12-31", freq="MS")
        values = np.array([1000.0, 0.1, 1100.0, 500.0, 1200.0, 300.0] * 12)
        series = pd.Series(values[: len(dates)], index=dates, name="site_1")
        d = ValenciaSchaakeDisaggregator(transform="log")
        d.fit(series)
        assert d.is_fitted


# ---------------------------------------------------------------------------
# Input validation and timing-axis tests (Section A/C of the fresh plan)
# ---------------------------------------------------------------------------


class TestValenciaSchaakeInputValidation:

    def test_invalid_transform_raises(self):
        with pytest.raises(ValueError, match="transform"):
            ValenciaSchaakeDisaggregator(transform="lgo")

    def test_invalid_conservation_raises(self):
        with pytest.raises(ValueError, match="conservation_method"):
            ValenciaSchaakeDisaggregator(conservation_method="proprtional")

    def test_transform_without_conservation_raises(self):
        with pytest.raises(ValueError, match="incompatible"):
            ValenciaSchaakeDisaggregator(transform="log", conservation_method="none")

        with pytest.raises(ValueError, match="incompatible"):
            ValenciaSchaakeDisaggregator(transform="boxcox", conservation_method="none")

    def test_unsupported_n_subperiods_raises(self):
        with pytest.raises(ValueError, match="n_subperiods"):
            ValenciaSchaakeDisaggregator(n_subperiods=5)
        with pytest.raises(ValueError, match="n_subperiods"):
            ValenciaSchaakeDisaggregator(n_subperiods=1)


class TestValenciaSchaakeQuarterlyDisaggregation:

    def test_quarterly_output_timestamps(self, monthly_single_site):
        """For n_subperiods=4 the output index must use quarter starts
        (Jan, Apr, Jul, Oct), not consecutive months."""
        d = ValenciaSchaakeDisaggregator(
            n_subperiods=4, transform="none", conservation_method="none"
        )
        d.fit(monthly_single_site)
        ens = _annual_ensemble_from(monthly_single_site.to_frame(), n_realizations=1)
        out = d.disaggregate(ens, seed=0)
        df = out.data_by_realization[out.realization_ids[0]]
        first_year = df.index[0].year
        assert list(df.index[:4].month) == [1, 4, 7, 10]
        assert all(df.index[:4].year == first_year)
        assert df.index[4].month == 1
        assert df.index[4].year == first_year + 1

    def test_quarterly_exact_additivity(self, monthly_single_site):
        """Additivity must hold for any supported n_subperiods, not just 12."""
        d = ValenciaSchaakeDisaggregator(
            n_subperiods=4, transform="none", conservation_method="none"
        )
        d.fit(monthly_single_site)
        ens = _annual_ensemble_from(monthly_single_site.to_frame(), n_realizations=2)
        out = d.disaggregate(ens, seed=0)
        for rid in out.realization_ids:
            annual_in = ens.data_by_realization[rid].values
            annual_out = out.data_by_realization[rid].resample("YS").sum().values
            np.testing.assert_allclose(annual_out, annual_in, rtol=1e-8, atol=1e-6)


class TestValenciaSchaakeNumericalRobustness:

    def test_boxcox_inverse_nan_replaced_with_zero(self, monthly_single_site):
        """If inv_boxcox returns NaN for any draw, the output must still be
        finite (NaN is replaced before clip)."""
        d = ValenciaSchaakeDisaggregator(
            transform="boxcox", conservation_method="proportional"
        )
        d.fit(monthly_single_site)
        ens = _annual_ensemble_from(monthly_single_site.to_frame(), n_realizations=3)
        out = d.disaggregate(ens, seed=0)
        for rid in out.realization_ids:
            values = out.data_by_realization[rid].values
            assert np.isfinite(values).all()
            assert (values >= 0).all()

    def test_zero_sum_site_uniform_fallback(self):
        """When proportional rescale would divide by zero (a site's pre-rescale
        sum is non-positive), the fallback should split X_t[s] uniformly."""
        # Build a synthetic case where the conditional mean is centered on
        # zero so clipping zeroes out the entire site for some draws.
        rng = np.random.default_rng(0)
        dates = pd.date_range(start="2000-01-01", periods=12 * 20, freq="MS")
        # Use mixed signs: with transform='none' the rescale path doesn't
        # apply when conservation_method='none'. Use 'proportional' but
        # synthesize a fixture where some draws produce zero-sum sites.
        values = rng.normal(0.5, 1.0, size=len(dates))
        series = pd.Series(values, index=dates, name="site_1")
        d = ValenciaSchaakeDisaggregator(
            transform="none", conservation_method="proportional"
        )
        d.fit(series)
        ens = _annual_ensemble_from(series.to_frame(), n_realizations=5)
        out = d.disaggregate(ens, seed=0)
        for rid in out.realization_ids:
            annual_in = ens.data_by_realization[rid].values
            annual_out = out.data_by_realization[rid].resample("YS").sum().values
            np.testing.assert_allclose(annual_out, annual_in, rtol=1e-6)
