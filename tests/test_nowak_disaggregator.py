"""
Tests for the Nowak temporal disaggregator across timescale pairs.
"""

import numpy as np
import pandas as pd
import pytest

from synhydro.core.ensemble import Ensemble, EnsembleMetadata
from synhydro.methods.disaggregation.temporal.nowak import NowakDisaggregator

# (input_timestep, output_timestep, obs fixture name)
SCALE_CASES = [
    ("monthly", "daily", "sample_daily_dataframe"),
    ("weekly", "daily", "sample_daily_dataframe"),
    ("monthly", "weekly", "sample_weekly_dataframe"),
    ("annual", "monthly", "sample_monthly_dataframe"),
    ("annual", "weekly", "sample_weekly_dataframe"),
    ("annual", "daily", "sample_daily_dataframe"),
]


def _make_ensemble(coarse_df, freq, n_realizations=2):
    """Build a small ensemble by scaling one coarse DataFrame."""
    data = {r: coarse_df * (1.0 + 0.1 * r) for r in range(n_realizations)}
    metadata = EnsembleMetadata(
        n_realizations=n_realizations,
        n_sites=coarse_df.shape[1],
        time_resolution=freq,
        time_period=(
            str(coarse_df.index[0].date()),
            str(coarse_df.index[-1].date()),
        ),
    )
    return Ensemble(data, metadata=metadata)


def _make_coarse(input_timestep, output_timestep, obs):
    """Build a synthetic coarse DataFrame appropriate for the scale pair."""
    if input_timestep == "monthly":
        return obs.resample("MS").sum().iloc[:12]
    if input_timestep == "weekly":
        weekly = obs.resample("W-SUN").sum()
        index = pd.DatetimeIndex(
            [pd.Timestamp.fromisocalendar(2016, w, 7) for w in range(1, 53)]
        )
        return pd.DataFrame(
            weekly.iloc[:52].to_numpy(), index=index, columns=obs.columns
        )
    # annual: use interior full years to get representative magnitudes
    annual = obs.resample("YS").sum()
    return annual.iloc[1:3]


class TestNowakDisaggregatorInitialization:
    """Tests for NowakDisaggregator initialization."""

    def test_initialization_default_params(self):
        """Default configuration is monthly to daily with shift 7."""
        disagg = NowakDisaggregator()
        assert disagg.input_timestep == "monthly"
        assert disagg.output_timestep == "daily"
        assert disagg.input_frequency == "MS"
        assert disagg.output_frequency == "D"
        assert disagg.n_neighbors == 5
        assert disagg.max_knn_pool_shift_timesteps == 7
        assert disagg.boundary_blend_timesteps == 0

    def test_initialization_custom_params(self):
        """Custom parameters are stored."""
        disagg = NowakDisaggregator(
            n_neighbors=10,
            max_knn_pool_shift_timesteps=10,
            boundary_blend_timesteps=2,
        )
        assert disagg.n_neighbors == 10
        assert disagg.max_knn_pool_shift_timesteps == 10
        assert disagg.boundary_blend_timesteps == 2

    @pytest.mark.parametrize(
        "input_timestep,output_timestep", [(i, o) for i, o, _ in SCALE_CASES]
    )
    def test_initialization_supported_pairs(self, input_timestep, output_timestep):
        """All six supported scale pairs construct successfully."""
        disagg = NowakDisaggregator(
            input_timestep=input_timestep, output_timestep=output_timestep
        )
        assert disagg.input_timestep == input_timestep
        assert disagg.output_timestep == output_timestep

    @pytest.mark.parametrize(
        "input_timestep,output_timestep",
        [
            ("weekly", "monthly"),
            ("monthly", "monthly"),
            ("weekly", "weekly"),
            ("daily", "daily"),
            ("daily", "monthly"),
            ("monthly", "annual"),
            ("hourly", "daily"),
        ],
    )
    def test_initialization_invalid_pairs_raise(self, input_timestep, output_timestep):
        """Invalid or non-refining timestep pairs raise ValueError."""
        with pytest.raises(ValueError):
            NowakDisaggregator(
                input_timestep=input_timestep, output_timestep=output_timestep
            )

    def test_per_pair_default_shift(self):
        """Per-pair defaults resolve when shift is not given."""
        assert NowakDisaggregator().max_knn_pool_shift_timesteps == 7
        assert (
            NowakDisaggregator(
                input_timestep="annual", output_timestep="monthly"
            ).max_knn_pool_shift_timesteps
            == 0
        )
        assert (
            NowakDisaggregator(
                input_timestep="weekly", output_timestep="daily"
            ).max_knn_pool_shift_timesteps
            == 2
        )


class TestNowakDisaggregatorPreprocessing:
    """Tests for NowakDisaggregator preprocessing."""

    def test_preprocessing_daily_series(self, sample_daily_series):
        """Preprocessing accepts a daily Series."""
        disagg = NowakDisaggregator()
        disagg.preprocessing(sample_daily_series)
        assert disagg.is_preprocessed is True

    def test_preprocessing_daily_dataframe(self, sample_daily_dataframe):
        """Preprocessing accepts a daily DataFrame."""
        disagg = NowakDisaggregator()
        disagg.preprocessing(sample_daily_dataframe)
        assert disagg.is_preprocessed is True

    def test_preprocessing_frequency_mismatch_raises(self, sample_daily_dataframe):
        """Observed data must be at the configured output timestep."""
        disagg = NowakDisaggregator(input_timestep="annual", output_timestep="monthly")
        with pytest.raises(ValueError, match="output_timestep"):
            disagg.preprocessing(sample_daily_dataframe)


class TestNowakDisaggregatorFit:
    """Tests for NowakDisaggregator fitting."""

    def test_fit_single_site(self, sample_daily_series):
        """Fit builds pools and KNN models for a single site."""
        disagg = NowakDisaggregator()
        disagg.preprocessing(sample_daily_series)
        disagg.fit()

        assert disagg.is_fitted is True
        assert hasattr(disagg, "knn_models")
        assert hasattr(disagg, "coarse_totals")
        assert hasattr(disagg, "flow_profiles")
        assert len(disagg.knn_models) == 12  # One model per month
        assert disagg.is_multisite is False

    def test_fit_multiple_sites(self, sample_daily_dataframe):
        """Fit detects multisite configuration."""
        disagg = NowakDisaggregator()
        disagg.preprocessing(sample_daily_dataframe)
        disagg.fit()

        assert disagg.is_fitted is True
        assert disagg.is_multisite is True
        assert disagg.n_sites == 3
        assert len(disagg.site_names) == 3
        assert len(disagg.knn_models) == 12

    def test_fit_creates_knn_models(self, sample_daily_series):
        """Fit creates one trained KNN model per month."""
        disagg = NowakDisaggregator()
        disagg.preprocessing(sample_daily_series)
        disagg.fit()

        for month in range(1, 13):
            assert month in disagg.knn_models
            assert hasattr(disagg.knn_models[month], "n_samples_fit_")

    def test_fit_creates_historic_profiles(self, sample_daily_series):
        """Fit creates one pool of totals and profiles per month."""
        disagg = NowakDisaggregator()
        disagg.preprocessing(sample_daily_series)
        disagg.fit()

        assert len(disagg.coarse_totals) == 12
        assert len(disagg.flow_profiles) == 12

        for month in range(1, 13):
            assert month in disagg.coarse_totals
            assert month in disagg.flow_profiles

    def test_pool_sizes(self, sample_daily_series):
        """Pool size equals n_years * (2 * max_shift + 1)."""
        disagg = NowakDisaggregator(max_knn_pool_shift_timesteps=3)
        disagg.fit(sample_daily_series)
        expected = disagg.n_historic_years * (2 * 3 + 1)
        assert len(disagg.coarse_totals[1]) == expected
        assert disagg.flow_profiles[1].shape == (expected, 31)


class TestNowakDisaggregatorKNNSearch:
    """Tests for KNN search functionality."""

    def test_find_knn_indices_single_site(self, sample_daily_series):
        """KNN index search returns (n_samples, n_neighbors) arrays."""
        disagg = NowakDisaggregator(n_neighbors=5)
        disagg.preprocessing(sample_daily_series)
        disagg.fit()

        flow_array = np.array([100.0])
        distances, indices = disagg.find_knn_indices(flow_array, label=1)

        assert isinstance(indices, np.ndarray)
        assert isinstance(distances, np.ndarray)
        assert indices.shape == (1, 5)
        assert distances.shape == (1, 5)

    def test_find_knn_indices_multiple_periods(self, sample_daily_series):
        """KNN index search handles several query values at once."""
        disagg = NowakDisaggregator(n_neighbors=5)
        disagg.preprocessing(sample_daily_series)
        disagg.fit()

        flow_array = np.array([100.0, 120.0, 90.0])
        distances, indices = disagg.find_knn_indices(flow_array, label=1)

        assert indices.shape == (3, 5)
        assert distances.shape == (3, 5)

    def test_sample_knn_flows(self, sample_daily_series):
        """Sampling returns in-range pool indices."""
        disagg = NowakDisaggregator(n_neighbors=5)
        disagg.preprocessing(sample_daily_series)
        disagg.fit()

        flow_array = np.array([100.0])
        sampled_idx = disagg.sample_knn_flows(
            flow_array, label=1, rng=np.random.default_rng(0)
        )

        assert isinstance(sampled_idx, np.ndarray)
        assert len(sampled_idx) == 1
        assert sampled_idx[0] < len(disagg.coarse_totals[1])


class TestNowakDisaggregatorMonthlyToDaily:
    """Behavioral tests for the monthly-to-daily path."""

    def _disaggregate(self, disagg, coarse, seed=0, **kwargs):
        return disagg._disaggregate_single_realization(
            coarse, rng=np.random.default_rng(seed), **kwargs
        )

    def test_disaggregate_single_month_single_site(self, sample_daily_series):
        """One January disaggregates to 31 days preserving the total."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)

        synthetic_monthly = pd.Series(
            [3000.0], index=pd.DatetimeIndex(["2020-01-01"], freq="MS")
        )
        daily = self._disaggregate(disagg, synthetic_monthly)

        assert isinstance(daily, pd.DataFrame)
        assert len(daily) == 31
        assert np.abs(daily.iloc[:, 0].sum() - 3000.0) < 1e-6

    def test_disaggregate_full_year_single_site(self, sample_daily_series):
        """A leap year disaggregates to 366 days with monthly mass balance."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)

        rng = np.random.default_rng(1)
        synthetic_monthly = pd.Series(
            rng.gamma(2.0, 100.0, 12),
            index=pd.date_range("2020-01-01", periods=12, freq="MS"),
        )
        daily = self._disaggregate(disagg, synthetic_monthly)

        assert len(daily) == 366  # 2020 is a leap year

        for month in range(1, 13):
            month_mask = daily.index.month == month
            monthly_sum = daily.loc[month_mask].iloc[:, 0].sum()
            expected = synthetic_monthly.iloc[month - 1]
            assert np.abs(monthly_sum - expected) < 1e-6

    def test_disaggregate_multiple_sites(self, sample_daily_dataframe):
        """Multisite disaggregation preserves per-site monthly totals."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_dataframe)

        rng = np.random.default_rng(2)
        synthetic_monthly = pd.DataFrame(
            rng.gamma(2.0, 100.0, (12, 3)),
            index=pd.date_range("2020-01-01", periods=12, freq="MS"),
            columns=sample_daily_dataframe.columns,
        )
        daily = self._disaggregate(disagg, synthetic_monthly)

        assert isinstance(daily, pd.DataFrame)
        assert daily.shape[1] == 3
        assert len(daily) == 366
        assert daily.columns.tolist() == sample_daily_dataframe.columns.tolist()

        for site in daily.columns:
            for month in range(1, 13):
                month_mask = daily.index.month == month
                monthly_sum = daily.loc[month_mask, site].sum()
                expected = synthetic_monthly.iloc[month - 1][site]
                assert np.abs(monthly_sum - expected) < 1e-6

    def test_disaggregate_leap_year_february(self, sample_daily_series):
        """February in a leap year gets 29 days and an exact total."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)

        synthetic_monthly = pd.Series(
            [3000.0], index=pd.DatetimeIndex(["2020-02-01"], freq="MS")
        )
        daily = self._disaggregate(disagg, synthetic_monthly)

        assert len(daily) == 29
        assert np.abs(daily.iloc[:, 0].sum() - 3000.0) < 1e-6

    def test_disaggregate_non_leap_february(self, sample_daily_series):
        """February in a non-leap year gets 28 days with an exact total.

        A 29-day candidate profile truncated to 28 days is renormalized so
        the monthly volume is conserved (renormalize_truncated).
        """
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)

        synthetic_monthly = pd.Series(
            [2800.0], index=pd.DatetimeIndex(["2019-02-01"], freq="MS")
        )
        daily = self._disaggregate(disagg, synthetic_monthly)

        assert len(daily) == 28
        assert np.abs(daily.iloc[:, 0].sum() - 2800.0) < 1e-6

    def test_disaggregate_different_sample_methods(self, sample_daily_series):
        """Both neighbor sampling methods produce valid output."""
        synthetic_monthly = pd.Series(
            [3000.0], index=pd.DatetimeIndex(["2020-01-01"], freq="MS")
        )

        for method in ("distance_weighted", "lall_and_sharma_1996"):
            disagg = NowakDisaggregator()
            disagg.fit(sample_daily_series)
            daily = self._disaggregate(disagg, synthetic_monthly, sample_method=method)
            assert len(daily) == 31

    def test_disaggregate_invalid_sample_method_raises(self, sample_daily_series):
        """Unknown sampling methods raise ValueError."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)
        synthetic_monthly = pd.Series(
            [3000.0], index=pd.DatetimeIndex(["2020-01-01"], freq="MS")
        )
        with pytest.raises(ValueError, match="sample method"):
            self._disaggregate(disagg, synthetic_monthly, sample_method="not_a_method")

    def test_disaggregate_with_different_n_neighbors(self, sample_daily_series):
        """n_neighbors sweep produces valid output."""
        synthetic_monthly = pd.Series(
            [3000.0], index=pd.DatetimeIndex(["2020-01-01"], freq="MS")
        )

        for n in [3, 5, 10]:
            disagg = NowakDisaggregator(n_neighbors=n)
            disagg.fit(sample_daily_series)
            daily = self._disaggregate(disagg, synthetic_monthly)
            assert len(daily) == 31

    def test_disaggregate_with_different_pool_shifts(self, sample_daily_series):
        """Pool shift sweep produces valid output."""
        synthetic_monthly = pd.Series(
            [3000.0], index=pd.DatetimeIndex(["2020-01-01"], freq="MS")
        )

        for shift in [3, 7, 14]:
            disagg = NowakDisaggregator(max_knn_pool_shift_timesteps=shift)
            disagg.fit(sample_daily_series)
            daily = self._disaggregate(disagg, synthetic_monthly)
            assert len(daily) == 31


class TestNowakDisaggregatorScalePairs:
    """Shared invariants across all supported timescale pairs."""

    @pytest.mark.parametrize("input_timestep,output_timestep,fixture", SCALE_CASES)
    def test_fit_and_mass_balance(
        self, input_timestep, output_timestep, fixture, request
    ):
        """Fit succeeds and per-period totals match the coarse flows."""
        obs = request.getfixturevalue(fixture)
        disagg = NowakDisaggregator(
            input_timestep=input_timestep, output_timestep=output_timestep
        )
        disagg.fit(obs)

        coarse = _make_coarse(input_timestep, output_timestep, obs)
        ensemble = _make_ensemble(coarse, disagg.input_frequency)
        fine = disagg.disaggregate(ensemble, seed=42)

        assert fine.metadata.time_resolution == disagg.output_frequency

        for rid, fine_df in fine.data_by_realization.items():
            coarse_df = ensemble.data_by_realization[rid]
            assert not fine_df.isnull().any().any()
            assert (fine_df.to_numpy() >= 0).all()
            for ts in coarse_df.index:
                start, end, expected_steps = disagg._period_window(ts)
                window = fine_df.loc[start:end]
                assert len(window) == expected_steps
                for site in coarse_df.columns:
                    target = coarse_df.loc[ts, site]
                    rel_err = abs(window[site].sum() - target) / max(target, 1e-12)
                    assert rel_err < 1e-8

    @pytest.mark.parametrize("input_timestep,output_timestep,fixture", SCALE_CASES)
    def test_same_seed_reproducible(
        self, input_timestep, output_timestep, fixture, request
    ):
        """Two runs with the same seed produce identical output."""
        obs = request.getfixturevalue(fixture)
        disagg = NowakDisaggregator(
            input_timestep=input_timestep, output_timestep=output_timestep
        )
        disagg.fit(obs)

        coarse = _make_coarse(input_timestep, output_timestep, obs)
        ensemble = _make_ensemble(coarse, disagg.input_frequency)
        fine_a = disagg.disaggregate(ensemble, seed=7)
        fine_b = disagg.disaggregate(ensemble, seed=7)

        for rid in fine_a.data_by_realization:
            pd.testing.assert_frame_equal(
                fine_a.data_by_realization[rid], fine_b.data_by_realization[rid]
            )

    @pytest.mark.parametrize(
        "input_timestep,output_timestep,fixture",
        [c for c in SCALE_CASES if c[0] in ("monthly", "weekly")],
    )
    def test_different_seed_differs(
        self, input_timestep, output_timestep, fixture, request
    ):
        """Different seeds produce different output (many-pool pairs)."""
        obs = request.getfixturevalue(fixture)
        disagg = NowakDisaggregator(
            input_timestep=input_timestep, output_timestep=output_timestep
        )
        disagg.fit(obs)

        coarse = _make_coarse(input_timestep, output_timestep, obs)
        ensemble = _make_ensemble(coarse, disagg.input_frequency)
        fine_a = disagg.disaggregate(ensemble, seed=7)
        fine_b = disagg.disaggregate(ensemble, seed=8)

        any_diff = any(
            not fine_a.data_by_realization[rid].equals(fine_b.data_by_realization[rid])
            for rid in fine_a.data_by_realization
        )
        assert any_diff

    def test_monthly_to_weekly_excludes_partial_first_year(self):
        """A weekly record missing a year's first Sunday excludes that year.

        Regression test: the monthly-to-weekly complete-year rule must
        require the year's first and last Sunday anchors, not merely one
        anchor in each month, or shifted pool windows poke past the wrap
        padding and fit raises.
        """
        # Sundays of 2015-2017, minus the first two anchors of 2015
        dates = pd.date_range("2015-01-04", "2017-12-31", freq="W-SUN")[2:]
        rng = np.random.default_rng(5)
        obs = pd.DataFrame({"site_1": rng.gamma(2.0, 25.0, len(dates))}, index=dates)

        disagg = NowakDisaggregator(input_timestep="monthly", output_timestep="weekly")
        disagg.fit(obs)

        assert 2015 not in disagg.historic_years
        assert set(disagg.historic_years) == {2016, 2017}

    def test_weekly_output_is_sunday_anchored(self, sample_weekly_dataframe):
        """Annual-to-weekly output has 52 Sunday anchors per year."""
        disagg = NowakDisaggregator(input_timestep="annual", output_timestep="weekly")
        disagg.fit(sample_weekly_dataframe)

        coarse = _make_coarse("annual", "weekly", sample_weekly_dataframe)
        ensemble = _make_ensemble(coarse, disagg.input_frequency)
        fine = disagg.disaggregate(ensemble, seed=3)

        fine_df = fine.data_by_realization[0]
        assert len(fine_df) == 52 * len(coarse)
        assert (fine_df.index.dayofweek == 6).all()
        # anchors follow the ISO calendar year by year (no drift)
        iso = fine_df.index.isocalendar()
        assert (iso.week.to_numpy() == np.tile(np.arange(1, 53), len(coarse))).all()


class TestNowakDisaggregatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_disaggregate_without_fit_raises(self, sample_daily_series):
        """Disaggregating before fit raises ValueError."""
        disagg = NowakDisaggregator()

        coarse = pd.DataFrame(
            {"site_1": [3000.0]}, index=pd.DatetimeIndex(["2020-01-01"], freq="MS")
        )
        ensemble = _make_ensemble(coarse, "MS")

        with pytest.raises(ValueError):
            disagg.disaggregate(ensemble)

    def test_disaggregate_frequency_mismatch_raises(self, sample_daily_series):
        """An ensemble at the wrong frequency raises ValueError."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_series)

        coarse = pd.DataFrame(
            {"site_1": [3000.0]},
            index=pd.DatetimeIndex(["2020-01-03"]),
        )
        ensemble = _make_ensemble(coarse, "W-SUN")

        with pytest.raises(ValueError, match="frequency"):
            disagg.disaggregate(ensemble)

    def test_weekly_alias_frequency_accepted(self, sample_daily_series):
        """Weekly frequency aliases normalize when validating ensembles."""
        disagg = NowakDisaggregator(input_timestep="weekly", output_timestep="daily")
        disagg.fit(sample_daily_series)

        index = pd.DatetimeIndex(
            [pd.Timestamp.fromisocalendar(2016, w, 7) for w in range(1, 5)]
        )
        coarse = pd.DataFrame({"site_1": [700.0, 650.0, 720.0, 800.0]}, index=index)
        ensemble = _make_ensemble(coarse, "W")  # generic weekly alias

        fine = disagg.disaggregate(ensemble, seed=1)
        assert len(fine.data_by_realization[0]) == 28


class TestAnchoredDailyIndex:
    """The daily index follows the anchored synthetic years, leap days included."""

    def test_daily_index_follows_kirsch_start_year(self, sample_daily_dataframe):
        from synhydro.methods.generation.hybrid.kirsch import KirschGenerator

        gen = KirschGenerator()
        gen.fit(sample_daily_dataframe)
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_dataframe)

        monthly = gen.generate(n_realizations=1, n_years=2, seed=0, start_year=1947)
        daily = disagg.disaggregate(monthly, seed=0)

        idx = daily.data_by_realization[0].index
        assert idx[0] == pd.Timestamp("1947-01-01")
        assert idx[-1] == pd.Timestamp("1948-12-31")
        assert len(idx) == 365 + 366  # 1948 is a leap year
        assert int(((idx.month == 2) & (idx.year == 1948)).sum()) == 29


class TestFittedPoolImmutability:
    """Disaggregation must never mutate the fitted proportion pools.

    Regression tests for a view-aliasing bug: the sampled profile slice was a
    view into ``flow_profiles``, and the leap-February length fix wrote through
    it, persisting the fix into the fitted pool. That coupled realizations
    (output depended on the batch/partition order, breaking the global-index
    determinism contract) and progressively violated February mass balance.
    """

    @staticmethod
    def _monthly_ensemble(columns, start, n_years, n_realizations=3):
        """Monthly ensemble spanning ``n_years`` from ``start`` (keys 0..R-1)."""
        rng = np.random.default_rng(11)
        index = pd.date_range(start, periods=12 * n_years, freq="MS")
        data = {
            r: pd.DataFrame(
                rng.gamma(2.0, 100.0, (len(index), len(columns))),
                index=index,
                columns=columns,
            )
            for r in range(n_realizations)
        }
        metadata = EnsembleMetadata(
            n_realizations=n_realizations,
            n_sites=len(columns),
            time_resolution="MS",
            time_period=(str(index[0].date()), str(index[-1].date())),
        )
        return Ensemble(data, metadata=metadata)

    def test_flow_profiles_unchanged_across_leap_year(self, sample_daily_dataframe):
        """The pools are bit-identical after disaggregating leap-year content."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_dataframe)
        snapshot = {k: v.copy() for k, v in disagg.flow_profiles.items()}

        # 2019-2021 includes Feb 2020 (29-day target, fires the length fix)
        # and Feb 2019/2021 (28-day targets, fire the truncation path).
        ensemble = self._monthly_ensemble(
            sample_daily_dataframe.columns, "2019-01-01", 3
        )
        disagg.disaggregate(ensemble, seed=3)

        assert set(disagg.flow_profiles) == set(snapshot)
        for label, arr in snapshot.items():
            np.testing.assert_array_equal(disagg.flow_profiles[label], arr)

    def test_batch_vs_isolated_realization_identical(self, sample_daily_dataframe):
        """A realization's daily output is invariant to its batch, across a leap year."""
        disagg = NowakDisaggregator()
        disagg.fit(sample_daily_dataframe)

        full = self._monthly_ensemble(
            sample_daily_dataframe.columns, "2019-01-01", 3, n_realizations=4
        )
        fine_batch = disagg.disaggregate(full, seed=5)

        target = 3  # last key: most exposed to any pool mutation by earlier keys
        alone = Ensemble(
            {target: full.data_by_realization[target]},
            metadata=EnsembleMetadata(
                n_realizations=1,
                n_sites=len(sample_daily_dataframe.columns),
                time_resolution="MS",
                time_period=full.metadata.time_period,
            ),
        )
        fine_alone = disagg.disaggregate(alone, seed=5)

        pd.testing.assert_frame_equal(
            fine_batch.data_by_realization[target],
            fine_alone.data_by_realization[target],
        )
