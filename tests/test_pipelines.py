"""
Tests for prebuilt pipelines: KirschNowakPipeline and ThomasFieringNowakPipeline.

End-to-end tests covering the documented usage patterns from
docs/tutorials/03_pipeline.md. These guard against frequency-handoff
bugs between the generator and disaggregator steps.
"""

import pytest
import pandas as pd

from synhydro.pipelines import KirschNowakPipeline, ThomasFieringNowakPipeline
from synhydro.core.ensemble import Ensemble


@pytest.fixture
def daily_dataframe():
    """Daily multi-site DataFrame spanning multiple full years (no freq set)."""
    dates = pd.date_range(start="2000-01-01", end="2009-12-31", freq="D")
    import numpy as np

    rng = np.random.default_rng(0)
    data = {
        "site_1": rng.gamma(shape=2.0, scale=50.0, size=len(dates)),
        "site_2": rng.gamma(shape=2.5, scale=40.0, size=len(dates)),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.freq = None
    return df


class TestKirschNowakPipeline:
    """End-to-end tests for the multi-site Kirsch-Nowak pipeline."""

    def test_fit_and_generate_multisite(self, daily_dataframe):
        """fit(daily) then generate() returns a daily ensemble with both sites."""
        pipe = KirschNowakPipeline()
        pipe.fit(daily_dataframe)
        ensemble = pipe.generate(n_realizations=2, n_years=3, seed=1)

        assert isinstance(ensemble, Ensemble)
        assert ensemble.frequency == "D"
        first = ensemble.data_by_realization[ensemble.realization_ids[0]]
        assert list(first.columns) == ["site_1", "site_2"]
        assert (first.values >= 0).all()

    def test_output_length_matches_n_years(self, daily_dataframe):
        """Daily ensemble length is consistent with the requested n_years."""
        pipe = KirschNowakPipeline()
        pipe.fit(daily_dataframe)
        ensemble = pipe.generate(n_realizations=1, n_years=2, seed=1)
        first = ensemble.data_by_realization[ensemble.realization_ids[0]]
        # 2 years of daily flows; tolerate +/- leap-year handling
        assert 720 <= len(first) <= 732


class TestThomasFieringNowakPipeline:
    """End-to-end tests for the single-site Thomas-Fiering-Nowak pipeline."""

    def test_fit_and_generate_single_site(self, daily_dataframe):
        """fit(single-site daily) then generate() returns a daily ensemble.

        Regression test: previously the pipeline raised
        ``NowakDisaggregator expects input frequency 'MS', but got 'None'``
        because ThomasFieringGenerator was producing an ensemble without
        time_resolution metadata when the input arrived at daily frequency.
        """
        pipe = ThomasFieringNowakPipeline()
        pipe.fit(daily_dataframe.iloc[:, [0]])
        ensemble = pipe.generate(n_realizations=2, n_years=3, seed=1)

        assert isinstance(ensemble, Ensemble)
        assert ensemble.frequency == "D"
        first = ensemble.data_by_realization[ensemble.realization_ids[0]]
        assert first.shape[1] == 1
        assert (first.values >= 0).all()

    def test_rejects_multisite_input(self, daily_dataframe):
        """Pipeline rejects multi-site input with a clear ValueError."""
        pipe = ThomasFieringNowakPipeline()
        with pytest.raises(ValueError, match="univariate"):
            pipe.fit(daily_dataframe)
