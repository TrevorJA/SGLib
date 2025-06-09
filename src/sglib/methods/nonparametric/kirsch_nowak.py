"""
This implements the combined Kirsch generation + Nowak disaggregation 
for daily streamflow generation using multisite disaggregation.

This is built on top of the KirschGenerator, 
then uses the multisite NowakDisaggregator to disaggregate all sites simultaneously.
"""
import numpy as np
import pandas as pd
import warnings

from sglib.methods.nonparametric.kirsch import KirschGenerator
from sglib.methods.temporal_disaggregation.nowak import NowakDisaggregator

class KirschNowakGenerator(KirschGenerator):
    """
    This class implements the combined Kirsch + Nowak generation method for daily streamflows. 

    The KirschGenerator is used to generate monthly streamflows, 
    and the multisite NowakDisaggregator is used to disaggregate all sites simultaneously
    from monthly to daily streamflows using KNN while preserving spatial correlations.
    
    
    Example usage:
    >>> import pandas as pd
    >>> from sglib.methods.nonparametric.kirsch_nowak import KirschNowakGenerator
    >>> # Load your daily streamflow data into a DataFrame Q
    >>> Q = pd.read_csv('daily_streamflow.csv', index_col=0, parse_dates=True)
    >>> # Initialize the generator
    >>> generator = KirschNowakGenerator(Q)
    >>> # Preprocess the data & fit the model
    >>> generator.preprocessing()
    >>> generator.fit()
    >>> # Generate daily streamflows
    >>> daily_flows = generator.generate(n_realizations=10, n_years=5)
    """
    def __init__(self, Q: pd.DataFrame, **kwargs):
        """Initialize the KirschNowakGenerator.
        
        Parameters
        ----------
        Q : pd.DataFrame
            The daily streamflow data used for generation. 
            It should be a DataFrame with datetime index and sites as columns.
        **kwargs : keyword arguments
            Additional keyword arguments for the KirschGenerator and NowakDisaggregator.
            
            Nowak-specific parameters:
            - n_neighbors : int (default: 5)
                Number of nearest neighbors for KNN disaggregation
            - max_month_shift : int (default: 7)
                Maximum number of days to shift around each month center
        """
        # Extract nowak-specific parameters before calling parent
        nowak_params = {
            'n_neighbors': kwargs.pop('n_neighbors', 5),
            'max_month_shift': kwargs.pop('max_month_shift', 7)
        }
        
        # Call the parent KirschGenerator
        super().__init__(Q, **kwargs)
        
        # Create a single multisite NowakDisaggregator for all sites
        self.nowak_disaggregator = NowakDisaggregator(
            Qh_daily=Q,  # Pass the entire multisite DataFrame
            **nowak_params
        )
        
    def preprocessing(self):
        """
        This method preprocesses the data for the KirschGenerator and NowakDisaggregator.
        """
        # Preprocess the KirschGenerator
        super().preprocessing()
        
        # Preprocess the multisite NowakDisaggregator
        self.nowak_disaggregator.preprocessing()
    
    def fit(self):
        """
        This method fits the KirschGenerator and NowakDisaggregator to the data.
        """
        # Fit the KirschGenerator
        super().fit()
        
        print("Fitting multisite NowakDisaggregator")
        # Fit the multisite NowakDisaggregator
        self.nowak_disaggregator.fit()
    
    def generate(self, 
                 n_realizations: int = 1,
                 n_years: int = 1,
                 as_array: bool = False):
        """
        Generate ensembles of daily streamflows using the Kirsch + Nowak method.
        
        First, generate ensemble of monthly flows using the KirschGenerator.
        Then, disaggregate all sites simultaneously using the multisite NowakDisaggregator.
        
        Parameters
        ----------
        n_realizations : int, optional
            The number of realizations to generate. Default is 1.
        n_years : int, optional
            The number of years to generate. Default is 1.
        as_array : bool, optional
            If True, return the generated data as a numpy array. 
            If False, return as a pandas DataFrame. Default is False.
            
        Returns
        -------
        dict
            Dictionary with realization indices as keys and DataFrames as values.
            Each DataFrame contains daily streamflows with sites as columns.
        """
        
        if as_array:
            warnings.warn("as_array=True is not yet implemented. Returning DataFrame instead.")
            as_array = False
        
        ### Generate monthly flows using the KirschGenerator
        # Qse_monthly is dict with {real_id: DataFrame}
        # where each DataFrame has monthly datetime index and columns for each site
        Qse_monthly = super().generate(n_realizations=n_realizations, 
                                      n_years=n_years, 
                                      as_array=False)
        
        # Dict of daily flows, matching Qse_monthly format
        Qse_daily = {}
        
        # For each realization, disaggregate all sites simultaneously
        for real_id in Qse_monthly.keys():
            if self.params.get('debug', False):
                print(f"Disaggregating realization {real_id + 1}/{n_realizations}")
            
            # Get the multisite monthly flows for this realization
            # This should be a pd.DataFrame with sites as columns
            Qs_monthly_multisite = Qse_monthly[real_id]
            
            # Verify the input format
            if not isinstance(Qs_monthly_multisite, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for realization {real_id}, got {type(Qs_monthly_multisite)}")
            
            if not all(col in self.site_names for col in Qs_monthly_multisite.columns):
                raise ValueError(f"Monthly data columns {Qs_monthly_multisite.columns.tolist()} "
                               f"do not match expected sites {self.site_names}")
            
            # Disaggregate all sites simultaneously using multisite disaggregator
            # Output will be pd.DataFrame of daily flows with sites as columns
            Qs_daily_multisite = self.nowak_disaggregator.disaggregate_monthly_flows(
                Qs_monthly=Qs_monthly_multisite
            )
            
            # Verify the output format
            if not isinstance(Qs_daily_multisite, pd.DataFrame):
                raise ValueError(f"Expected DataFrame output from disaggregator, got {type(Qs_daily_multisite)}")
            
            if not all(col in self.site_names for col in Qs_daily_multisite.columns):
                raise ValueError(f"Daily data columns {Qs_daily_multisite.columns.tolist()} "
                               f"do not match expected sites {self.site_names}")
            
            # Verify temporal consistency
            expected_start = Qs_monthly_multisite.index[0]
            expected_end = Qs_monthly_multisite.index[-1] + pd.offsets.MonthEnd(0)
            
            if (Qs_daily_multisite.index[0] != expected_start or 
                Qs_daily_multisite.index[-1] != expected_end):
                raise ValueError(f"Disaggregated daily flows temporal range does not match expected range.\n"
                               f"Expected: {expected_start} to {expected_end}\n"
                               f"Got: {Qs_daily_multisite.index[0]} to {Qs_daily_multisite.index[-1]}")
            
            # Store in the results dictionary
            Qse_daily[real_id] = Qs_daily_multisite
            
            if self.params.get('debug', False):
                print(f"Successfully disaggregated realization {real_id + 1}: "
                      f"Shape {Qs_daily_multisite.shape}, "
                      f"Date range {Qs_daily_multisite.index[0]} to {Qs_daily_multisite.index[-1]}")
        
        return Qse_daily