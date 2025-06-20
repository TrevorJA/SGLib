import h5py
import pandas as pd

from .directories import example_data_dir, data_dir
from .ensemble_manager import Ensemble

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
    # Q = Q.replace(0, np.nan)  # Replace zeros with NaN
    # Q = Q.dropna(axis=1, how='any')
    
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



class HDF5Manager:
    """
    Class for saving and loading synthetic time series data to/from HDF5 files.
    Handles both numpy array format and dictionary of DataFrames format.
    """
    def __init__(self):
        return
    
    def export_ensemble_to_hdf5(self,
                                dict, 
                                output_file):
        """
        Export a dictionary of ensemble data to an HDF5 file.
        Data is stored in the dictionary as {realization number (int): pd.DataFrame}.
        
        Args:
            dict (dict): A dictionary of ensemble data.
            output_file (str): Full output file path & name to write HDF5.
            
        Returns:
            None    
        """
        
        dict_keys = list(dict.keys())
        N = len(dict)
        T, M = dict[dict_keys[0]].shape
        column_labels = dict[dict_keys[0]].columns.to_list()
        
        with h5py.File(output_file, 'w') as f:
            for key in dict_keys:
                data = dict[key]
                datetime = data.index.astype(str).tolist() #.strftime('%Y-%m-%d').tolist()
                
                grp = f.create_group(key)
                        
                # Store column labels as an attribute
                grp.attrs['column_labels'] = column_labels

                # Create dataset for dates
                grp.create_dataset('date', data=datetime)
                
                # Create datasets for each array subset from the group
                for j in range(M):
                    dataset = grp.create_dataset(column_labels[j], 
                                                data=data[column_labels[j]].to_list())
        return


    def get_hdf5_realization_numbers(self, 
                                     filename):
        """
        Checks the contents of an hdf5 file, and returns a list 
        of the realization ID numbers contained.
        Realizations have key 'realization_i' in the HDF5.

        Args:
            filename (str): The HDF5 file of interest

        Returns:
            list: Containing realizations ID numbers; realizations have key 'realization_i' in the HDF5.
        """
        realization_numbers = []
        with h5py.File(filename, 'r') as file:
            # Get the keys in the HDF5 file
            keys = list(file.keys())

            # Get the df using a specific node key
            node_data = file[keys[0]]
            column_labels = node_data.attrs['column_labels']
            
            # Iterate over the columns and extract the realization numbers
            for col in column_labels:
                
                # handle different types of column labels
                if type(col) == str:
                    if col.startswith('realization_'):
                        # Extract the realization number from the key
                        realization_numbers.append(int(col.split('_')[1]))
                    else:
                        realization_numbers.append(col)
                elif type(col) == int:
                    realization_numbers.append(col)
                else:
                    err_msg = f'Unexpected type {type(col)} for column label {col}.'
                    err_msg +=  f'in HDF5 file {filename}'
                    raise ValueError(err_msg)
        
        return realization_numbers


    def extract_realization_from_hdf5(self, 
                                      hdf5_file, 
                                      realization, 
                                      stored_by_node=True):
        """
        Pull a single inflow realization from an HDF5 file of inflows.

        Parameters
        ----------
        hdf5_file : str
                Path to the HDF5 file.
        realization : str or int
                The realization number or name to extract.
        stored_by_node : bool, optional
                    If True, assumes that the data keys are node names. If False, keys are realizations. Default is False.
                    
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the extracted realization data, with datetime as the index.
        """

        with h5py.File(hdf5_file, "r") as f:
            if stored_by_node:
                # Extract timeseries data from realization for each node
                data = {}

                # Get keys in the HDF5 file
                keys = list(f.keys())

                for node in keys:
                    node_data = f[node]
                    column_labels = node_data.attrs["column_labels"]

                    err_msg = f"The specified realization {realization} is not available in the HDF file."
                    assert realization in column_labels, (
                        err_msg + f" Realizations available: {column_labels}"
                    )
                    data[node] = node_data[realization][:]

                dates = node_data["date"][:].tolist()

            else:
                realization_group = f[realization]

                # Extract column labels
                column_labels = realization_group.attrs["column_labels"]
                # Extract timeseries data for each location
                data = {}
                for label in column_labels:
                    dataset = realization_group[label]
                    data[label] = dataset[:]

                # Get date indices
                dates = realization_group["date"][:].tolist()
            data["datetime"] = dates

        # Combine into dataframe
        df = pd.DataFrame(data, index=dates)
        df.index = pd.to_datetime(df.index.astype(str))
        return df

    def load_ensemble(self, 
                      hdf5_file, 
                      realization_subset=None):
        """
        Load an ensemble of realizations from an HDF5 file.
        
        Args:
            hdf5_file (str): The filename for the hdf5 file
            realization (int, optional): If specified, load only this realization.
        
        Returns:
            dict: A dictionary of DataFrames with site names as keys and DataFrames as values.
        """
        
        realization_ids = self.get_hdf5_realization_numbers(hdf5_file)
        if realization_subset is not None:
            for r in realization_subset:
                if r not in realization_ids:
                    msg = f'Realization {r} from realization_subset not found in HDF5 file {hdf5_file}.'
                    raise ValueError(msg)
            realization_ids = realization_subset
        
        ensemble_dict = {}
        for i, ri in enumerate(realization_ids):
            if i%100 == 0:
                print(f'Loading realization {i+1}/{len(realization_ids)} from {hdf5_file}')
            
            df = self.extract_realization_from_hdf5(hdf5_file, ri)
            ensemble_dict[i] = df
        
        ensemble = Ensemble(ensemble_dict)
        return ensemble
        


    # def combine_hdf5_files(self,
    #                        files, 
    #                        output_file):
    #     """
    #     This function reads multiple hdf5 files and 
    #     combines all data into a single file with the same structure.

    #     The function assumes that all files have the same structure.
    #     """    
    #     assert(type(files) == list), 'Input must be a list of file paths'
        
    #     output_file = output_file + '.hdf5' if ('.hdf' not in output_file) else output_file
        
    #     # Extract all
    #     results_dict ={}
    #     for i, f in enumerate(files):
    #         if '.' not in f:
    #             f = f + '.hdf5'
    #         assert(f[-5:] == '.hdf5'), f'Filename {f} must end in .hdf5'
    #         results_dict[i] = extract_loo_results_from_hdf5(f)

    #     # Combine all data
    #     # Each item in the results_dict has site_number as key and dataframe as values.
    #     # We want to combine all dataframes into a single dataframe with columns corresponding to realization numbers
    #     combined_results = {}
    #     combined_column_names = []
    #     for i in results_dict:
    #         for site in results_dict[i]:
    #             if site in combined_results:
    #                 combined_results[site] = pd.concat([combined_results[site], results_dict[i][site]], axis=1)
    #             else:
    #                 combined_results[site] = results_dict[i][site]

    #     # Reset the column names so that there are no duplicates
    #     all_sites = list(combined_results.keys())
    #     n_realizations = len(combined_results[all_sites[0]].columns)
    #     combined_column_names = [str(i) for i in range(n_realizations)]
    #     for site in all_sites:
    #         assert len(combined_results[site].columns) == n_realizations, f'Number of realizations is not consistent for site {site}'
    #         combined_results[site].columns = combined_column_names
        
    #     # Write to file
    #     self.export_ensemble_to_hdf5(combined_results, output_file)        
    #     return


    # def save_to_hdf5(self, 
    #                  file_path, 
    #                  data, 
    #                  datetime_index=None, 
    #                  site_names=None):
    #     """
    #     Save data to HDF5.

    #     Parameters:
    #         file_path (str): Path to save the file.
    #         data (np.ndarray or dict): Data array (n_realizations, n_sites, n_time) or dict {realization_id: DataFrame}.
    #         datetime_index (pd.DatetimeIndex, optional): Time index to store.
    #         site_names (list, optional): Names of the sites.
    #     """
    #     with h5py.File(file_path, 'w') as f:
    #         if isinstance(data, np.ndarray):
    #             f.create_dataset('data', data=data)
    #             if datetime_index is not None:
    #                 f.create_dataset('datetime', data=np.array(datetime_index.strftime('%Y-%m-%d')).astype('S'))

    #             if site_names is not None:
    #                 f.create_dataset('column_labels', data=np.array(site_names, dtype='S'))

    #         elif isinstance(data, dict):
    #             grp = f.create_group('realizations')
    #             for key, df in data.items():
    #                 grp.create_dataset(str(key), data=df.values)
    #             if site_names is not None:
    #                 f.create_dataset('column_labels', data=np.array(site_names, dtype='S'))
    #             if datetime_index is not None:
    #                 f.create_dataset('datetime', data=np.array(datetime_index.strftime('%Y-%m-%d')).astype('S'))
    #         else:
    #             raise TypeError("Data must be a numpy array or a dictionary of DataFrames.")


    # def load_from_hdf5(self,
    #                    file_path, 
    #                    as_dict=False):
    #     """
    #     Load data from HDF5.

    #     Parameters:
    #         file_path (str): Path to the file.
    #         as_dict (bool): If True, load as dict {realization_id: DataFrame}.

    #     Returns:
    #         data (np.ndarray or dict): Loaded data.
    #         datetime_index (pd.DatetimeIndex): Loaded time index.
    #         site_names (list): List of site names.
    #     """
    #     with h5py.File(file_path, 'r') as f:
    #         site_names = f['column_labels'][:].astype(str).tolist()
    #         datetime_index = pd.to_datetime(f['datetime'][:].astype(str))

    #         if 'data' in f:
    #             raw_data = f['data'][:]
    #             if as_dict:
    #                 data = {}
    #                 n_realizations, n_sites, n_time = raw_data.shape
    #                 for r in range(n_realizations):
    #                     df = pd.DataFrame(
    #                         raw_data[r].T,
    #                         index=datetime_index[:n_time],
    #                         columns=site_names
    #                     )
    #                     data[r] = df
    #             else:
    #                 data = raw_data

    #         elif 'realizations' in f:
    #             temp_dict = {}
    #             for key in f['realizations']:
    #                 df = pd.DataFrame(
    #                     f['realizations'][key][:],
    #                     index=datetime_index,
    #                     columns=site_names
    #                 )
    #                 temp_dict[int(key)] = df
    #             if as_dict:
    #                 data = temp_dict
    #             else:
    #                 realizations = sorted(temp_dict.keys())
    #                 n_sites = len(site_names)
    #                 n_time = temp_dict[realizations[0]].shape[0]
    #                 data_array = np.empty((len(realizations), n_sites, n_time))
    #                 for idx, r in enumerate(realizations):
    #                     data_array[idx] = temp_dict[r].T.values
    #                 data = data_array

    #         else:
    #             raise KeyError("HDF5 file does not contain recognizable datasets.")

    #     return data, datetime_index, site_names
