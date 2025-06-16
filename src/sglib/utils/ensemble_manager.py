"""
Used to manage, format, and manipulate ensembles of synthetic timeseries data. 
"""
import pandas as pd
from typing import Dict


class Ensemble:
    def __init__(self, data):
        """
        Initializes the Ensemble class with data.

        Parameters
        ----------
        data : dict
            A dictionary containing ensemble data.
        """
        assert isinstance(data, dict), "Data must be a dictionary."
        
        if self.infer_data_structure(data) == 'realizations':
            self.data_by_site = self.transform_realizations_to_sites(data)
            self.data_by_realization = data
        elif self.infer_data_structure(data) == 'sites':
            self.data_by_site = data
            self.data_by_realization = self.transform_sites_to_realizations(data)
        else:
            raise ValueError("Unknown data structure type. Expected 'realizations' or 'sites'.")

        
        self.realization_ids = list(self.data_by_realization.keys())




    def infer_data_structure(self, data):
        """
        Checks which of the two structures the data is in:
        {
            realization: DataFrame with datetime index and columns representing different sites
        }
        
        or 
        {
            site: DataFrame with datetime index and columns representing different realizations
        }
        
        Parameters
        ----------
        data : dict
            A dictionary containing ensemble data.
        
        Returns
        -------
        str
            A string indicating the data structure type: 'realizations' or 'sites'.
        """
        
        assert isinstance(data, dict), "Data must be a dictionary."
        
        # Check the first key to determine the structure
        first_key = next(iter(data))
        
        # If the first key is an integer, it's likely a realization structure
        if isinstance(first_key, int):
            return 'realizations'
        
        # If the first key is a string, it's likely a site structure
        elif isinstance(first_key, str):
            return 'sites'
        
        else:
            raise ValueError("Unknown data structure type.")

    def transform_realizations_to_sites(self, 
                                                  data_dict: Dict[int, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Optimized version using pandas concat for better performance with large datasets.
        
        Parameters:
        -----------
        data_dict : Dict[int, pd.DataFrame]
            Dictionary where keys are realization numbers and values are DataFrames
            with datetime index and columns representing different sites
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary where keys are site names and values are DataFrames
            with datetime index and columns representing different realizations
        """
        if not data_dict:
            return {}
        
        # Get all unique sites
        all_sites = set()
        for df in data_dict.values():
            all_sites.update(df.columns)
        
        result = {}
        
        for site in all_sites:
            # Collect all series for this site across realizations
            site_series = []
            for realization, df in data_dict.items():
                if site in df.columns:
                    series = df[site].rename(realization)
                    site_series.append(series)
            
            # Concatenate all series for this site
            if site_series:
                site_df = pd.concat(site_series, axis=1, sort=True)
                result[site] = site_df
        
        return result
    
    def transform_sites_to_realizations(site_dict: Dict[str, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """
        Transform data structure from {site: DataFrame} to {realization: DataFrame}.
        This is the inverse of transform_realizations_to_sites().
        
        Parameters:
        -----------
        site_dict : Dict[str, pd.DataFrame]
            Dictionary where keys are site names and values are DataFrames
            with datetime index and columns representing different realizations
        
        Returns:
        --------
        Dict[int, pd.DataFrame]
            Dictionary where keys are realization numbers and values are DataFrames
            with datetime index and columns representing different sites
        """
        if not site_dict:
            return {}
        
        # Get all unique realizations across all sites
        all_realizations = set()
        for df in site_dict.values():
            all_realizations.update(df.columns)
        
        result = {}
        
        for realization in all_realizations:
            # Collect all series for this realization across sites
            realization_series = []
            for site, df in site_dict.items():
                if realization in df.columns:
                    series = df[realization].rename(site)
                    realization_series.append(series)
            
            # Concatenate all series for this realization
            if realization_series:
                realization_df = pd.concat(realization_series, axis=1, sort=True)
                result[realization] = realization_df
        
        return result