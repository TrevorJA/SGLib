import numpy as np
import pandas as pd
import scipy.stats as scs
import spei as si

def get_drought_metrics(ssi):
    ## Get historic drought metrics
    drought_data = {}
    drought_counter = 0
    in_critical_drought = False
    drought_days = []

    for ind in range(len(ssi)):
        if ssi.values[ind] < 0:
            drought_days.append(ind)
            
            if ssi.values[ind] <= -1:
                in_critical_drought = True
        else:
            # Record drought info once it ends
            if in_critical_drought:
                drought_counter += 1
                drought_data[drought_counter] = {
                    'start': ssi.index[drought_days[0]],
                    'end': ssi.index[drought_days[-1]],
                    'duration': len(drought_days),
                    'magnitude': sum(ssi.values[drought_days]),
                    'severity': min(ssi.values[drought_days])
                }
                
            in_critical_drought = False
            drought_days = [] 

    drought_metrics = pd.DataFrame(drought_data).transpose()
    return drought_metrics

class SSIDroughtMetrics:
    """
    Class to calculate and store drought metrics based on the Standardized Streamflow Index (SSI).
    
    Attributes:
        ssi (pd.Series): Series of SSI values.
        drought_metrics (pd.DataFrame): DataFrame containing drought metrics.
    """
    
    def __init__(self,
                 timescale: str = 'M',
                 window: int = 12,
                 data = None,):
        """
        Initialize the SSIDroughtMetrics class.

        Parameters:
        """
        assert isinstance(timescale, str), "Timescale must be a string."
        assert isinstance(window, int), "Window must be an integer."
        assert (timescale in ['D', 'M']), "Timescale must be either 'daily' or 'monthly'."
        
        
        self.timescale = timescale
        self.window = window
        
        if data is not None:
            self._set_data(data)
        
        
    def _set_data(self, data):
        """
        Set the data for the class.

        Parameters:
            data (pd.DataFrame or pd.Series): Data to be set.
        """
        # data can be a array, series or dataframe
        # either way, convert to Series
        if isinstance(data, pd.DataFrame):
            data = data.squeeze()
            
            # check if the first column is datetime
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.iloc[:, 0]
            else:
                # set datetime index to the first column of the data
                data.index = pd.date_range(start='1945-01-01', 
                                           periods=data.shape[0], freq='D')
            
        elif isinstance(data, np.ndarray):
            data = pd.Series(data)
            # set datetime index to the first column of the data
            freq = 'D' if self.timescale == 'D' else 'MS'
            data.index = pd.date_range(start='2000-01-01', 
                                       periods=data.shape[0], freq=freq)
        
        elif not isinstance(data, pd.Series):
            raise TypeError("Data must be a pandas DataFrame, Series, or numpy array.")
        
        self.data = data.copy()
        
    def calculate_ssi(self, data=None):
        
        if data is not None:
            self._set_data(data)
        
        elif not hasattr(self, 'data'):
            raise ValueError("Data not set. Please set data before calculating SSI.")
        
        # Get rolling sum
        data_rs = self.data.rolling(self.window, min_periods=self.window).sum().dropna()
        
        # Calculate the Standardized Streamflow Index (SSI)
        ssi = si.ssfi(series = data_rs, 
                     dist=scs.gamma
                     )
        
        return ssi
    
    def calculate_drought_metrics(self, ssi=None):
            
            if ssi is None:
                ssi = self.calculate_ssi()
                
            # Get drought metrics
            drought_metrics = get_drought_metrics(ssi)
            
            return drought_metrics
        
    