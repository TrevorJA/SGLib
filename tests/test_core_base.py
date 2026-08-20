"""
Tests for synhydro.core.base module (Generator base class and GeneratorState).
Updated to match the current API.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import pickle

from synhydro.core.base import Generator, GeneratorState, make_output_index
from synhydro.core.ensemble import Ensemble


class MockGenerator(Generator):
    """Concrete implementation of Generator for testing."""

    def __init__(self, *, debug=False):
        super().__init__(debug=debug)
        self.preprocessing_called = False
        self.fit_called = False
        self.generate_called = False

    @property
    def output_frequency(self) -> str:
        """Return output frequency for this generator."""
        return "D"  # Daily output for testing

    def preprocessing(self, Q_obs, *, sites=None, **kwargs):
        """Mock preprocessing implementation."""
        Q_validated = self.validate_input_data(Q_obs)
        self._Q_obs = Q_validated
        self._sites = Q_validated.columns.tolist()
        self.preprocessing_called = True
        self.update_state(preprocessed=True)

    def _compute_fitted_params(self):
        """Mock implementation of required abstract method."""
        from synhydro.core.base import FittedParams

        # Return minimal FittedParams for testing
        return FittedParams()

    def fit(self, Q_obs=None, *, sites=None, **kwargs):
        """Mock fit implementation."""
        if Q_obs is not None:
            self.preprocessing(Q_obs, sites=sites)
        self.validate_preprocessing()
        self.fit_called = True
        self.fitted_params_ = self._compute_fitted_params()
        self.update_state(fitted=True)

    def generate(
        self,
        n_realizations=1,
        start_date="2020-01-01",
        end_date="2020-12-31",
        freq="D",
        **kwargs,
    ):
        """Mock generate implementation - returns Ensemble."""
        from synhydro.core.ensemble import Ensemble

        self.validate_fit()
        self.generate_called = True

        # Create synthetic data
        index = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_timesteps = len(index)

        # Generate multiple realizations as dict
        realization_dict = {}
        for r in range(n_realizations):
            data = np.random.randn(n_timesteps, len(self._sites))
            realization_dict[r] = pd.DataFrame(data, index=index, columns=self._sites)

        # Return as Ensemble
        return Ensemble(realization_dict)


class TestGeneratorState:
    """Tests for GeneratorState dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = GeneratorState()
        assert state.is_preprocessed is False
        assert state.is_fitted is False
        assert state.fit_timestamp is None

    def test_state_modification(self):
        """Test state can be modified."""
        state = GeneratorState()
        state.is_preprocessed = True
        state.is_fitted = True
        state.fit_timestamp = datetime.now().isoformat()

        assert state.is_preprocessed is True
        assert state.is_fitted is True
        assert state.fit_timestamp is not None


class TestGenerator:
    """Tests for Generator abstract base class."""

    def test_instantiation(self):
        """Test that MockGenerator can be instantiated."""
        gen = MockGenerator()
        assert gen.state.is_preprocessed is False
        assert gen.state.is_fitted is False
        assert gen.debug is False

    def test_debug_mode(self):
        """Test debug mode initialization."""
        gen = MockGenerator(debug=True)
        assert gen.debug is True

    def test_validate_input_data_series(self, sample_daily_series):
        """Test input validation with pandas Series."""
        gen = MockGenerator()
        result = gen.validate_input_data(sample_daily_series)
        assert isinstance(result, pd.DataFrame)

    def test_validate_input_data_dataframe(self, sample_daily_dataframe):
        """Test input validation with pandas DataFrame."""
        gen = MockGenerator()
        result = gen.validate_input_data(sample_daily_dataframe)
        assert isinstance(result, pd.DataFrame)

    def test_validate_input_data_invalid_type(self):
        """Test input validation with invalid type."""
        gen = MockGenerator()
        with pytest.raises(TypeError):
            gen.validate_input_data([1, 2, 3, 4, 5])

    def test_validate_input_data_no_datetime_index(self):
        """Test input validation converts non-DatetimeIndex."""
        gen = MockGenerator()
        # New behavior: tries to convert index to DatetimeIndex
        data = pd.Series([1, 2, 3, 4, 5])
        result = gen.validate_input_data(data)
        # Should convert to DatetimeIndex automatically
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_preprocessing_workflow(self, sample_daily_series):
        """Test preprocessing workflow."""
        gen = MockGenerator()
        assert gen.state.is_preprocessed is False

        gen.preprocessing(sample_daily_series)

        assert gen.preprocessing_called is True
        assert gen.state.is_preprocessed is True

    def test_validate_preprocessing_passes_when_preprocessed(self, sample_daily_series):
        """Test validate_preprocessing passes after preprocessing."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        gen.validate_preprocessing()  # Should not raise

    def test_validate_preprocessing_fails_when_not_preprocessed(self):
        """Test validate_preprocessing fails before preprocessing."""
        gen = MockGenerator()
        with pytest.raises(ValueError, match="preprocessing"):
            gen.validate_preprocessing()

    def test_fit_workflow(self, sample_daily_series):
        """Test fit workflow."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)

        assert gen.state.is_fitted is False
        gen.fit()

        assert gen.fit_called is True
        assert gen.state.is_fitted is True
        assert gen.state.fit_timestamp is not None

    def test_fit_fails_without_preprocessing(self):
        """Test fit fails without preprocessing."""
        gen = MockGenerator()
        with pytest.raises(ValueError, match="preprocessing"):
            gen.fit()

    def test_validate_fit_passes_when_fitted(self, sample_daily_series):
        """Test validate_fit passes after fitting."""
        gen = MockGenerator()
        gen.fit(sample_daily_series)
        gen.validate_fit()  # Should not raise

    def test_validate_fit_fails_when_not_fitted(self, sample_daily_series):
        """Test validate_fit fails before fitting."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        with pytest.raises(ValueError, match="fit"):
            gen.validate_fit()

    def test_generate_workflow(self, sample_daily_series):
        """Test generate workflow."""
        gen = MockGenerator()
        gen.fit(sample_daily_series)

        result = gen.generate(n_realizations=1)

        assert gen.generate_called is True
        assert isinstance(result, Ensemble)

    def test_generate_fails_without_fit(self, sample_daily_series):
        """Test generate fails without fitting."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        with pytest.raises(ValueError, match="fit"):
            gen.generate()

    def test_n_sites_property_series(self, sample_daily_series):
        """Test n_sites property with Series input."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        assert gen.n_sites == 1

    def test_n_sites_property_dataframe(self, sample_daily_dataframe):
        """Test n_sites property with DataFrame input."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_dataframe)
        assert gen.n_sites == 3

    def test_sites_property_series(self, sample_daily_series):
        """Test sites property with Series input."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)
        expected_name = sample_daily_series.name if sample_daily_series.name else "flow"
        assert gen.sites == [expected_name]

    def test_sites_property_dataframe(self, sample_daily_dataframe):
        """Test sites property with DataFrame input."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_dataframe)
        assert gen.sites == sample_daily_dataframe.columns.tolist()

    def test_format_output_single_realization_series(self, sample_daily_series):
        """Test output format with single realization from Series."""
        gen = MockGenerator()
        gen.fit(sample_daily_series)

        result = gen.generate(
            n_realizations=1, start_date="2020-01-01", end_date="2020-01-31"
        )

        assert isinstance(result, Ensemble)
        assert len(result.data_by_realization[0]) == 31

    def test_format_output_multiple_realizations_series(self, sample_daily_series):
        """Test output format with multiple realizations from Series."""
        gen = MockGenerator()
        gen.fit(sample_daily_series)

        result = gen.generate(
            n_realizations=3, start_date="2020-01-01", end_date="2020-01-31"
        )

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 3
        assert len(result.data_by_realization[0]) == 31

    def test_format_output_single_realization_dataframe(self, sample_daily_dataframe):
        """Test output format with single realization from DataFrame."""
        gen = MockGenerator()
        gen.fit(sample_daily_dataframe)

        result = gen.generate(
            n_realizations=1, start_date="2020-01-01", end_date="2020-01-31"
        )

        assert isinstance(result, Ensemble)
        assert result.data_by_realization[0].shape[1] == 3
        assert len(result.data_by_realization[0]) == 31
        assert (
            result.data_by_realization[0].columns.tolist()
            == sample_daily_dataframe.columns.tolist()
        )

    def test_format_output_multiple_realizations_dataframe(
        self, sample_daily_dataframe
    ):
        """Test output format with multiple realizations from DataFrame."""
        gen = MockGenerator()
        gen.fit(sample_daily_dataframe)

        result = gen.generate(
            n_realizations=2, start_date="2020-01-01", end_date="2020-01-31"
        )

        assert isinstance(result, Ensemble)
        assert result.metadata.n_realizations == 2
        assert all(
            isinstance(df, pd.DataFrame) for df in result.data_by_realization.values()
        )

    def test_get_params(self, sample_daily_series):
        """Test get_params method."""
        gen = MockGenerator()
        gen.fit(sample_daily_series)

        params = gen.get_params()
        assert isinstance(params, dict)

    def test_save_and_load(self, sample_daily_series, tmp_path):
        """Test save and load functionality."""
        gen = MockGenerator()
        gen.fit(sample_daily_series)

        # Save
        save_path = tmp_path / "test_generator.pkl"
        gen.save(str(save_path))
        assert save_path.exists()

        # Load
        loaded_gen = MockGenerator.load(str(save_path))
        assert loaded_gen.state.is_preprocessed is True
        assert loaded_gen.state.is_fitted is True
        assert loaded_gen.preprocessing_called is True
        assert loaded_gen.fit_called is True

    def test_update_state(self, sample_daily_series):
        """Test update_state method."""
        gen = MockGenerator()
        gen.preprocessing(sample_daily_series)

        # Update state
        gen.update_state(fitted=True)

        assert gen.state.is_fitted is True
        assert gen.state.fit_timestamp is not None


class TestMakeOutputIndex:
    """Tests for the shared second-resolution output index helper."""

    @pytest.mark.parametrize("freq", ["YS", "MS", "W-SUN", "W", "D"])
    @pytest.mark.parametrize(
        "start", ["1950-01-01", "1950-03-15", "2000-12-31", pd.Timestamp("1987-06-01")]
    )
    def test_matches_date_range_values(self, freq, start):
        idx = make_output_index(start, 25, freq)
        ref = pd.date_range(start, periods=25, freq=freq)
        assert idx.dtype == np.dtype("datetime64[s]")
        assert (idx.values == ref.values).all()

    @pytest.mark.parametrize("freq", ["YS", "MS"])
    def test_keeps_freq_attribute(self, freq):
        idx = make_output_index("2000-01-01", 12, freq)
        assert idx.freq is not None
        assert idx.freq == pd.tseries.frequencies.to_offset(freq)

    def test_annual_beyond_nanosecond_range(self):
        idx = make_output_index("1940-01-01", 5000, "YS")
        assert len(idx) == 5000
        assert idx[-1].year == 1940 + 4999
        assert idx.dtype == np.dtype("datetime64[s]")

    def test_monthly_beyond_nanosecond_range(self):
        idx = make_output_index("1940-01-01", 12 * 5000, "MS")
        assert idx[-1].year == 1940 + 4999
        assert idx[-1].month == 12

    def test_daily_beyond_nanosecond_range(self):
        idx = make_output_index("1940-01-01", 365 * 1000, "D")
        assert idx[-1].year > 2262
        assert idx.dtype == np.dtype("datetime64[s]")

    def test_zero_periods(self):
        idx = make_output_index("2000-01-01", 0, "YS")
        assert len(idx) == 0
        assert idx.dtype == np.dtype("datetime64[s]")

    def test_negative_periods_raises(self):
        with pytest.raises(ValueError):
            make_output_index("2000-01-01", -1, "YS")

    def test_create_output_index_uses_second_resolution(self):
        gen = MockGenerator()
        idx = gen._create_output_index(10, freq="D", start_date=pd.Timestamp("2000-01-01"))
        assert idx.dtype == np.dtype("datetime64[s]")
        assert idx[0] == pd.Timestamp("2000-01-01")
        assert len(idx) == 10
