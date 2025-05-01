import h5py
import pandas as pd
import numpy as np

from .directories import example_data_dir, data_dir

def load_drb_reconstruction(gage_flow=True):
    """
    Load the DRB reconstruction data.

    Returns:
        pd.DataFrame: DataFrame containing the DRB reconstruction data.
    """
    if gage_flow:
        fname = 'gage_flow_obs_pub_nhmv10_BC_ObsScaled_median.csv'
    else:
        fname = 'catchment_inflow_obs_pub_nhmv10_BC_ObsScaled_median.csv'
    
    Q = pd.read_csv(f'{data_dir}/{fname}')
    Q.drop(columns=['datetime'], inplace=True)  # Drop the first column if it's an index
    
    datetime = pd.date_range(start='1945-01-01', 
                             periods=Q.shape[0], 
                             freq='D')
    
    Q.index = datetime
    Q = Q.replace(0, np.nan)  # Replace zeros with NaN
    Q = Q.dropna(axis=1, how='any')
    
    return Q


# Load example data
def load_example_data():
    """
    Loads example data from the package.

    Returns:
        pd.DataFrame: Example data.
    """
    data = pd.read_csv(f'{example_data_dir}/usgs_daily_streamflow_cms.csv', 
                       index_col=0, parse_dates=True)
    return data



class HDF5:
    """
    Class for saving and loading synthetic time series data to/from HDF5 files.
    Handles both numpy array format and dictionary of DataFrames format.
    """

    @staticmethod
    def save_to_hdf5(file_path, data, datetime_index=None, site_names=None):
        """
        Save data to HDF5.

        Parameters:
            file_path (str): Path to save the file.
            data (np.ndarray or dict): Data array (n_realizations, n_sites, n_time) or dict {realization_id: DataFrame}.
            datetime_index (pd.DatetimeIndex, optional): Time index to store.
            site_names (list, optional): Names of the sites.
        """
        with h5py.File(file_path, 'w') as f:
            if isinstance(data, np.ndarray):
                f.create_dataset('data', data=data)
                if datetime_index is not None:
                    f.create_dataset('time', data=np.array(datetime_index.strftime('%Y-%m-%d')).astype('S'))

                if site_names is not None:
                    f.create_dataset('site_names', data=np.array(site_names, dtype='S'))

            elif isinstance(data, dict):
                grp = f.create_group('realizations')
                for key, df in data.items():
                    grp.create_dataset(str(key), data=df.values)
                if site_names is not None:
                    f.create_dataset('site_names', data=np.array(site_names, dtype='S'))
                if datetime_index is not None:
                    f.create_dataset('time', data=np.array(datetime_index.strftime('%Y-%m-%d')).astype('S'))
            else:
                raise TypeError("Data must be a numpy array or a dictionary of DataFrames.")

    @staticmethod
    def load_from_hdf5(file_path, as_dict=False):
        """
        Load data from HDF5.

        Parameters:
            file_path (str): Path to the file.
            as_dict (bool): If True, load as dict {realization_id: DataFrame}.

        Returns:
            data (np.ndarray or dict): Loaded data.
            datetime_index (pd.DatetimeIndex): Loaded time index.
            site_names (list): List of site names.
        """
        with h5py.File(file_path, 'r') as f:
            site_names = f['site_names'][:].astype(str).tolist()
            datetime_index = pd.to_datetime(f['time'][:].astype(str))

            if 'data' in f:
                raw_data = f['data'][:]
                if as_dict:
                    data = {}
                    n_realizations, n_sites, n_time = raw_data.shape
                    for r in range(n_realizations):
                        df = pd.DataFrame(
                            raw_data[r].T,
                            index=datetime_index[:n_time],
                            columns=site_names
                        )
                        data[r] = df
                else:
                    data = raw_data

            elif 'realizations' in f:
                temp_dict = {}
                for key in f['realizations']:
                    df = pd.DataFrame(
                        f['realizations'][key][:],
                        index=datetime_index,
                        columns=site_names
                    )
                    temp_dict[int(key)] = df
                if as_dict:
                    data = temp_dict
                else:
                    realizations = sorted(temp_dict.keys())
                    n_sites = len(site_names)
                    n_time = temp_dict[realizations[0]].shape[0]
                    data_array = np.empty((len(realizations), n_sites, n_time))
                    for idx, r in enumerate(realizations):
                        data_array[idx] = temp_dict[r].T.values
                    data = data_array

            else:
                raise KeyError("HDF5 file does not contain recognizable datasets.")

        return data, datetime_index, site_names
