"""
Tests for synhydro.utils and synhydro.core modules (data loading, Ensemble).
"""

import pytest
import numpy as np
import pandas as pd
import h5py

from synhydro.utils import load_example_data
from synhydro.core import Ensemble


class TestLoadExampleData:
    """Tests for load_example_data function."""

    def test_load_example_data(self):
        """Test loading example data."""
        try:
            data = load_example_data()
            assert isinstance(data, (pd.Series, pd.DataFrame))
            assert isinstance(data.index, pd.DatetimeIndex)
            assert len(data) > 0
        except FileNotFoundError:
            pytest.skip("Example data file not found - skip test")


class TestEnsembleManager:
    """Tests for Ensemble class."""

    def test_initialization_from_realizations(self, sample_ensemble_data):
        """Test initialization with realization-based data."""
        ensemble = Ensemble(sample_ensemble_data)
        assert ensemble.realization_ids == [0, 1, 2]
        assert len(ensemble.data_by_realization) == 3

    def test_data_by_site_structure(self, sample_ensemble_data):
        """Test that data_by_site structure is created automatically."""
        ensemble = Ensemble(sample_ensemble_data)

        assert hasattr(ensemble, "data_by_site")
        assert isinstance(ensemble.data_by_site, dict)
        assert "site_1" in ensemble.data_by_site
        assert "site_2" in ensemble.data_by_site

        # Check that each site has all realizations as columns
        for site, df in ensemble.data_by_site.items():
            assert isinstance(df, pd.DataFrame)
            assert df.shape[1] == 3  # 3 realizations

    def test_dual_representation(self, sample_ensemble_data):
        """Test that both representations are maintained automatically."""
        ensemble = Ensemble(sample_ensemble_data)

        # Both structures should exist
        assert hasattr(ensemble, "data_by_realization")
        assert hasattr(ensemble, "data_by_site")
        assert len(ensemble.data_by_realization) == 3
        assert len(ensemble.data_by_site) == 2  # site_1 and site_2

        # Check that each realization has all sites as columns
        for r_id, df in ensemble.data_by_realization.items():
            assert isinstance(df, pd.DataFrame)
            assert df.shape[1] == 2  # 2 sites

    def test_initialization_from_sites(self, sample_ensemble_data):
        """Test initialization from site-based dict."""
        ensemble = Ensemble(sample_ensemble_data)

        # Create new ensemble from site-based data
        ensemble2 = Ensemble(ensemble.data_by_site)

        assert len(ensemble2.realization_ids) == 3
        assert len(ensemble2.site_names) == 2
        assert "site_1" in ensemble2.site_names
        assert "site_2" in ensemble2.site_names

    def test_site_data_with_string_realization_labels(self):
        """String realization column labels are coerced to int keys.

        Regression test: site-keyed data with string realization labels
        (e.g. "1".."3" from CSV headers) previously produced string keys in
        data_by_realization. A downstream Ensemble built from that dict then
        inferred a site-keyed structure, swapping data_by_realization and
        data_by_site.
        """
        index = pd.date_range("2000-01-01", periods=10, freq="YS")
        site_data = {
            site: pd.DataFrame(
                np.random.rand(10, 3), index=index, columns=["1", "2", "3"]
            )
            for site in ["site_A", "site_B"]
        }
        ensemble = Ensemble(site_data)

        assert sorted(ensemble.realization_ids) == [1, 2, 3]
        assert all(isinstance(k, int) for k in ensemble.data_by_realization)
        assert sorted(ensemble.site_names) == ["site_A", "site_B"]

        # Rebuilding from the realization dict must preserve orientation
        ensemble2 = Ensemble(dict(ensemble.data_by_realization))
        assert sorted(ensemble2.realization_ids) == [1, 2, 3]
        assert sorted(ensemble2.site_names) == ["site_A", "site_B"]

    def test_site_data_with_duplicate_realization_labels_raises(self):
        """Labels that coerce to the same int key raise a ValueError."""
        index = pd.date_range("2000-01-01", periods=10, freq="YS")
        site_data = {
            "site_A": pd.DataFrame(np.random.rand(10, 2), index=index, columns=[1, "1"])
        }
        with pytest.raises(ValueError, match="Duplicate realization key"):
            Ensemble(site_data)


class TestEnsembleHDF5IO:
    """Tests for Ensemble HDF5 save/load functionality."""

    def test_save_and_load_ensemble(self, sample_ensemble_data, temp_hdf5_file):
        """Test saving and loading ensemble to/from HDF5."""
        ensemble = Ensemble(sample_ensemble_data)
        ensemble.to_hdf5(str(temp_hdf5_file))

        assert temp_hdf5_file.exists()

        # Check file structure
        with h5py.File(temp_hdf5_file, "r") as f:
            assert "site_1" in f.keys() or "site_2" in f.keys()

        # Load back
        loaded_ensemble = Ensemble.from_hdf5(str(temp_hdf5_file))

        assert isinstance(loaded_ensemble, Ensemble)
        assert len(loaded_ensemble.realization_ids) == 3

    def test_roundtrip_hdf5_preserves_data(self, sample_ensemble_data, temp_hdf5_file):
        """Test that save/load roundtrip preserves data."""
        original_ensemble = Ensemble(sample_ensemble_data)

        # Save
        original_ensemble.to_hdf5(str(temp_hdf5_file))

        # Load
        loaded_ensemble = Ensemble.from_hdf5(str(temp_hdf5_file))

        # Compare
        assert len(loaded_ensemble.realization_ids) == len(
            original_ensemble.realization_ids
        )

        # Check data for all realizations (default dtype is float32, so
        # values match to ~7 significant digits)
        for r_id in original_ensemble.realization_ids:
            original_df = original_ensemble.data_by_realization[r_id]
            loaded_df = loaded_ensemble.data_by_realization[r_id]

            assert original_df.shape == loaded_df.shape
            assert np.allclose(original_df.values, loaded_df.values, rtol=1e-6)

    def test_save_with_compression(self, sample_ensemble_data, temp_hdf5_file):
        """Test saving with different compression options."""
        ensemble = Ensemble(sample_ensemble_data)

        # Save with gzip compression
        ensemble.to_hdf5(str(temp_hdf5_file), compression="gzip")
        assert temp_hdf5_file.exists()

        # Load and verify
        loaded = Ensemble.from_hdf5(str(temp_hdf5_file))
        assert len(loaded.realization_ids) == 3

    def test_load_subset_of_realizations(self, sample_ensemble_data, temp_hdf5_file):
        """Test loading only a subset of realizations."""
        ensemble = Ensemble(sample_ensemble_data)
        ensemble.to_hdf5(str(temp_hdf5_file))

        # Load only first 2 realizations
        loaded = Ensemble.from_hdf5(str(temp_hdf5_file), realization_subset=[0, 1])

        assert len(loaded.realization_ids) == 2
        assert 0 in loaded.realization_ids
        assert 1 in loaded.realization_ids

    def test_save_with_custom_site_names(self, sample_ensemble_data, temp_hdf5_file):
        """Test saving ensemble with custom site names."""
        # Rename sites to look like nodes
        renamed_ensemble = {}
        for r_id, df in sample_ensemble_data.items():
            renamed_df = df.rename(columns={"site_1": "node_A", "site_2": "node_B"})
            renamed_ensemble[r_id] = renamed_df

        ensemble = Ensemble(renamed_ensemble)
        ensemble.to_hdf5(str(temp_hdf5_file))

        # Load and check
        loaded = Ensemble.from_hdf5(str(temp_hdf5_file))
        assert "node_A" in loaded.site_names
        assert "node_B" in loaded.site_names

    def test_empty_ensemble_raises(self):
        """Test that empty ensemble raises error."""
        empty_ensemble = {}

        with pytest.raises((ValueError, TypeError)):
            Ensemble(empty_ensemble)

    def test_single_realization_ensemble(self, temp_hdf5_file):
        """Test ensemble with single realization."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        single_ensemble = {
            0: pd.DataFrame(
                {"site_1": np.random.randn(100), "site_2": np.random.randn(100)},
                index=dates,
            )
        }

        ensemble = Ensemble(single_ensemble)
        ensemble.to_hdf5(str(temp_hdf5_file))
        loaded = Ensemble.from_hdf5(str(temp_hdf5_file))

        assert len(loaded.realization_ids) == 1
        assert 0 in loaded.realization_ids

    def test_roundtrip_preserves_index_and_columns(
        self, sample_ensemble_data, temp_hdf5_file
    ):
        """Roundtrip preserves the DatetimeIndex and site column names."""
        original = Ensemble(sample_ensemble_data)
        original.to_hdf5(str(temp_hdf5_file))
        loaded = Ensemble.from_hdf5(str(temp_hdf5_file))

        for r_id in original.realization_ids:
            original_df = original.data_by_realization[r_id]
            loaded_df = loaded.data_by_realization[r_id]
            assert (loaded_df.index == original_df.index).all()
            assert sorted(loaded_df.columns) == sorted(original_df.columns)

    def test_hdf5_layout_contract(self, sample_ensemble_data, temp_hdf5_file):
        """The on-disk layout consumed by pywrdrb is preserved.

        Every site group must contain a 'date' dataset plus one dataset per
        realization, with a column_labels attr of str realization IDs. The
        'date' dataset is hard-linked so it is stored only once.
        """
        ensemble = Ensemble(sample_ensemble_data)
        ensemble.to_hdf5(str(temp_hdf5_file))

        with h5py.File(temp_hdf5_file, "r") as f:
            assert sorted(f.keys()) == ["site_1", "site_2"]
            for site in f.keys():
                grp = f[site]
                labels = list(grp.attrs["column_labels"])
                assert labels == ["0", "1", "2"]
                assert "date" in grp
                for label in labels:
                    assert label in grp
                    assert len(grp[label]) == len(grp["date"])

            # 'date' datasets are hard links to one physical dataset
            addrs = {h5py.h5o.get_info(f[site]["date"].id).addr for site in f.keys()}
            assert len(addrs) == 1

    def test_pywrdrb_date_read_path(self, sample_ensemble_data, temp_hdf5_file):
        """Dates survive pywrdrb's extract_realization_from_hdf5 code path."""
        ensemble = Ensemble(sample_ensemble_data)
        ensemble.to_hdf5(str(temp_hdf5_file))

        with h5py.File(temp_hdf5_file, "r") as f:
            node = list(f.keys())[0]
            node_data = f[node]
            data = {"flow": node_data["0"][:]}
            dates = node_data["date"][:].tolist()

        # Mirrors pywrdrb.utils.hdf5.extract_realization_from_hdf5
        df = pd.DataFrame(data, index=dates)
        df.index = pd.to_datetime(df.index.astype(str))

        expected = ensemble.data_by_realization[0].index
        assert (df.index == expected).all()

    def test_save_without_compression(self, sample_ensemble_data, temp_hdf5_file):
        """compression=None writes and reads back correctly."""
        ensemble = Ensemble(sample_ensemble_data)
        ensemble.to_hdf5(str(temp_hdf5_file), compression=None)

        loaded = Ensemble.from_hdf5(str(temp_hdf5_file))
        assert len(loaded.realization_ids) == 3
        for r_id in ensemble.realization_ids:
            assert np.allclose(
                ensemble.data_by_realization[r_id].values,
                loaded.data_by_realization[r_id].values,
                rtol=1e-6,
            )

    def test_save_with_lzf_compression(self, sample_ensemble_data, temp_hdf5_file):
        """compression='lzf' writes and reads back correctly."""
        ensemble = Ensemble(sample_ensemble_data)
        ensemble.to_hdf5(str(temp_hdf5_file), compression="lzf")

        loaded = Ensemble.from_hdf5(str(temp_hdf5_file))
        assert len(loaded.realization_ids) == 3
        for r_id in ensemble.realization_ids:
            assert np.allclose(
                ensemble.data_by_realization[r_id].values,
                loaded.data_by_realization[r_id].values,
                rtol=1e-6,
            )

    def test_save_with_float64_dtype(
        self, sample_ensemble_data, temp_hdf5_file, tmp_path
    ):
        """dtype='float64' roundtrips exactly; the float32 default is smaller."""
        ensemble = Ensemble(sample_ensemble_data)

        f64_path = tmp_path / "f64.h5"
        ensemble.to_hdf5(str(f64_path), dtype="float64")
        ensemble.to_hdf5(str(temp_hdf5_file))  # default float32

        assert temp_hdf5_file.stat().st_size < f64_path.stat().st_size

        loaded = Ensemble.from_hdf5(str(f64_path))
        for r_id in ensemble.realization_ids:
            assert np.array_equal(
                ensemble.data_by_realization[r_id].values,
                loaded.data_by_realization[r_id].values,
            )

    def test_new_format_smaller_than_old(self, sample_ensemble_data, tmp_path):
        """The new writer produces smaller files than the pre-optimization format."""
        ensemble = Ensemble(sample_ensemble_data)

        new_path = tmp_path / "new.h5"
        ensemble.to_hdf5(str(new_path))

        # Reproduce the old format: variable-length string dates duplicated
        # in every site group, no shuffle filter
        old_path = tmp_path / "old.h5"
        with h5py.File(old_path, "w", libver="latest") as f:
            for site, site_df in ensemble.data_by_site.items():
                grp = f.create_group(str(site))
                grp.attrs["column_labels"] = [str(c) for c in site_df.columns]
                dates_list = site_df.index.strftime("%Y-%m-%d").tolist()
                grp.create_dataset("date", data=dates_list, compression="gzip")
                for col in site_df.columns:
                    grp.create_dataset(
                        str(col), data=site_df[col].values, compression="gzip"
                    )

        assert new_path.stat().st_size < old_path.stat().st_size

        # Files written by the old format must still load correctly
        loaded_old = Ensemble.from_hdf5(str(old_path))
        for r_id in ensemble.realization_ids:
            original_df = ensemble.data_by_realization[r_id]
            loaded_df = loaded_old.data_by_realization[r_id]
            assert (loaded_df.index == original_df.index).all()
            assert np.allclose(original_df.values, loaded_df.values, rtol=1e-10)

    def test_stored_by_realization_roundtrip(
        self, sample_ensemble_data, temp_hdf5_file
    ):
        """stored_by_node=False roundtrip preserves data."""
        ensemble = Ensemble(sample_ensemble_data)
        ensemble.to_hdf5(str(temp_hdf5_file), stored_by_node=False)

        with h5py.File(temp_hdf5_file, "r") as f:
            assert sorted(f.keys()) == ["0", "1", "2"]

        loaded = Ensemble.from_hdf5(str(temp_hdf5_file), stored_by_node=False)
        assert len(loaded.realization_ids) == 3
        for r_id in ensemble.realization_ids:
            original_df = ensemble.data_by_realization[r_id]
            loaded_df = loaded.data_by_realization[r_id]
            assert (loaded_df.index == original_df.index).all()
            assert np.allclose(original_df.values, loaded_df.values, rtol=1e-6)


class TestEnsembleStatistics:
    """Tests for Ensemble statistical methods."""

    def test_summary_by_site(self, sample_ensemble_data):
        """Test summary statistics by site."""
        ensemble = Ensemble(sample_ensemble_data)
        stats = ensemble.summary(by="site")

        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 2  # 2 sites
        assert "mean" in stats.columns
        assert "std" in stats.columns
        assert "min" in stats.columns
        assert "max" in stats.columns

    def test_summary_by_realization(self, sample_ensemble_data):
        """Test summary statistics by realization."""
        ensemble = Ensemble(sample_ensemble_data)
        stats = ensemble.summary(by="realization")

        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 3  # 3 realizations

    def test_percentile_by_site(self, sample_ensemble_data):
        """Test percentile calculation by site."""
        ensemble = Ensemble(sample_ensemble_data)
        percentiles = ensemble.percentile([10, 50, 90], by="site")

        assert isinstance(percentiles, dict)
        assert "site_1" in percentiles
        assert "site_2" in percentiles

        for site, df in percentiles.items():
            assert isinstance(df, pd.DataFrame)
            assert "p10" in df.columns
            assert "p50" in df.columns
            assert "p90" in df.columns

    def test_percentile_single_value(self, sample_ensemble_data):
        """Test percentile with single value."""
        ensemble = Ensemble(sample_ensemble_data)
        percentiles = ensemble.percentile(50, by="site")

        assert isinstance(percentiles, dict)
        for site, df in percentiles.items():
            assert "p50" in df.columns


class TestEnsembleSubsetAndResample:
    """Tests for Ensemble subsetting and resampling."""

    def test_subset_by_sites(self, sample_ensemble_data):
        """Test subsetting by sites."""
        ensemble = Ensemble(sample_ensemble_data)
        subset = ensemble.subset(sites=["site_1"])

        assert len(subset.site_names) == 1
        assert "site_1" in subset.site_names
        assert len(subset.realization_ids) == 3

    def test_subset_by_realizations(self, sample_ensemble_data):
        """Test subsetting by realizations."""
        ensemble = Ensemble(sample_ensemble_data)
        subset = ensemble.subset(realizations=[0, 1])

        assert len(subset.realization_ids) == 2
        assert 0 in subset.realization_ids
        assert 1 in subset.realization_ids
        assert len(subset.site_names) == 2

    def test_subset_by_time_period(self, sample_ensemble_data):
        """Test subsetting by time period."""
        ensemble = Ensemble(sample_ensemble_data)
        subset = ensemble.subset(start_date="2010-06-01", end_date="2011-12-31")

        for r_id, df in subset.data_by_realization.items():
            assert df.index.min() >= pd.to_datetime("2010-06-01")
            assert df.index.max() <= pd.to_datetime("2011-12-31")

    def test_subset_combined(self, sample_ensemble_data):
        """Test subsetting by multiple criteria."""
        ensemble = Ensemble(sample_ensemble_data)
        subset = ensemble.subset(
            sites=["site_1"], realizations=[0, 1], start_date="2011-01-01"
        )

        assert len(subset.site_names) == 1
        assert len(subset.realization_ids) == 2

    def test_resample_to_monthly(self, sample_ensemble_data):
        """Test resampling to monthly frequency."""
        ensemble = Ensemble(sample_ensemble_data)
        monthly = ensemble.resample("MS")

        assert isinstance(monthly, Ensemble)
        assert len(monthly.realization_ids) == 3

        # Check that data is monthly
        for r_id, df in monthly.data_by_realization.items():
            assert df.index.freq == "MS"


class TestEnsembleEdgeCases:
    """Tests for edge cases in Ensemble class."""

    def test_single_site_ensemble(self):
        """Test ensemble with single site."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        single_site_ensemble = {
            0: pd.DataFrame({"site_1": np.random.randn(100)}, index=dates),
            1: pd.DataFrame({"site_1": np.random.randn(100)}, index=dates),
        }

        ensemble = Ensemble(single_site_ensemble)
        assert len(ensemble.realization_ids) == 2
        assert ensemble.data_by_realization[0].shape[1] == 1

    def test_large_number_of_realizations(self):
        """Test ensemble with many realizations."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        large_ensemble = {}
        for i in range(50):
            large_ensemble[i] = pd.DataFrame(
                {"site_1": np.random.randn(50), "site_2": np.random.randn(50)},
                index=dates,
            )

        ensemble = Ensemble(large_ensemble)
        assert len(ensemble.realization_ids) == 50

    def test_different_date_ranges_same_length(self):
        """Test ensemble with different date ranges but same length."""
        dates1 = pd.date_range("2020-01-01", periods=100, freq="D")
        dates2 = pd.date_range("2021-01-01", periods=100, freq="D")

        ensemble_data = {
            0: pd.DataFrame({"site_1": np.random.randn(100)}, index=dates1),
            1: pd.DataFrame({"site_1": np.random.randn(100)}, index=dates2),
        }

        ensemble = Ensemble(ensemble_data)
        # Should handle this, though dates are different
        assert len(ensemble.realization_ids) == 2

    def test_ensemble_string_representation(self, sample_ensemble_data):
        """Test ensemble string representation methods."""
        ensemble = Ensemble(sample_ensemble_data)

        repr_str = repr(ensemble)
        assert "Ensemble" in repr_str
        assert "n_realizations=3" in repr_str
        assert "n_sites=2" in repr_str

        str_output = str(ensemble)
        assert "Ensemble Summary" in str_output
        assert "Realizations: 3" in str_output
        assert "Sites: 2" in str_output


class TestEnsembleMetadata:
    """Tests for EnsembleMetadata functionality."""

    def test_metadata_initialization(self, sample_ensemble_data):
        """Test that metadata is created automatically."""
        ensemble = Ensemble(sample_ensemble_data)

        assert hasattr(ensemble, "metadata")
        assert ensemble.metadata.n_realizations == 3
        assert ensemble.metadata.n_sites == 2
        assert ensemble.metadata.creation_timestamp is not None

    def test_metadata_time_period(self, sample_ensemble_data):
        """Test that time period is inferred correctly."""
        ensemble = Ensemble(sample_ensemble_data)

        assert ensemble.metadata.time_period is not None
        start, end = ensemble.metadata.time_period
        assert isinstance(start, str)
        assert isinstance(end, str)
