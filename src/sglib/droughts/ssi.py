import numpy as np
import pandas as pd
import scipy.stats as scs
import spei as si
import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pandas import DataFrame, Grouper, Series, Timedelta, Timestamp
from scipy.stats import norm

from spei import SI  # Import the original SI class
from spei._typing import ContinuousDist
from spei.utils import get_data_series, group_yearly_df, validate_series


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
    
    def fit(self, series, dist=scs.gamma):
        
        self.SI_fitted = si.SI(series=series,
                               dist=dist,)
        
    
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
    

@dataclass
class SSI:
    """
    Independent SSI calculator that separates training and scoring phases.
    
    Uses the original spei.SI class internally for distribution fitting,
    then applies those fitted distributions to new data.
    """
    
    # Training parameters
    dist: ContinuousDist = field(default=scs.gamma)
    timescale: int = 12
    fit_freq: str | None = field(default=None)
    fit_window: int = field(default=0)
    prob_zero: bool = field(default=False)
    normal_scores_transform: bool = field(default=False)
    agg_func: Literal["sum", "mean"] = "sum"
    
    # Internal state
    _fitted_si: SI = field(init=False, repr=False, compare=False)
    _is_fitted: bool = field(default=False, init=False, repr=False, compare=False)
    _training_series: Series = field(init=False, repr=False, compare=False)
    
    
    def fit(self, training_series: Series) -> 'SSI':
        """
        Fit distributions using training data.
        
        Parameters
        ----------
        training_series : Series
            Time series data for fitting distributions
            
        Returns
        -------
        TrainableSSI
            Self for method chaining
        """
        self._training_series = training_series.copy()
        
        # Apply rolling aggregation if timescale is set
        if self.timescale > 0:
            training_series = (
                training_series.rolling(self.timescale, min_periods=self.timescale)
                .agg(self.agg_func)
                .dropna()
                .copy()
            )
        
        
        # Create and fit the original SI object
        self._fitted_si = SI(
            series=training_series,
            dist=self.dist,
            timescale=self.timescale,
            fit_freq=self.fit_freq,
            fit_window=self.fit_window,
            prob_zero=self.prob_zero,
            normal_scores_transform=self.normal_scores_transform,
            agg_func=self.agg_func,
        )
        
        self._fitted_si.fit_distribution()
        self._is_fitted = True
        
        return self
    
    def transform(self, new_series: Series) -> Series:
        """
        Calculate SSI values for new data using fitted distributions.
        
        Parameters
        ----------
        new_series : Series
            New time series data to transform
            
        Returns
        -------
        Series
            SSI values for the new series
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform()")
        
        # Preprocess new series the same way as training data
        new_series = validate_series(new_series)
        
        if self.timescale > 0:
            new_series = (
                new_series.rolling(self.timescale, min_periods=self.timescale)
                .agg(self.agg_func)
                .dropna()
                .copy()
            )
        
        # Calculate CDF for new data using fitted distributions
        cdf = self._calculate_cdf_for_new_data(new_series)
        
        # Convert to standard normal (SSI values)
        ssi = Series(
            norm.ppf(cdf.values, loc=0, scale=1),
            index=new_series.index,
            dtype=float
        )
        
        return ssi
    
    def fit_transform(self, 
                      training_series: Series, 
                      new_series: Series = None) -> Series:
        """
        Fit on training data and transform new data in one step.
        
        Parameters
        ----------
        training_series : Series
            Data for fitting distributions
        new_series : Series, optional
            Data to transform. If None, transforms training_series.
            
        Returns
        -------
        Series
            SSI values
        """
        self.fit(training_series)
        
        if new_series is not None:
            return self.transform(new_series)
        else:
            # Transform training data using fitted distributions
            return self._fitted_si.norm_ppf()
    
    def _calculate_cdf_for_new_data(self, new_series: Series) -> Series:
        """
        Calculate CDF for new data using the fitted distributions.
        
        Parameters
        ----------
        new_series : Series
            New time series data
            
        Returns
        -------
        Series
            CDF values for new data
        """
        if self.normal_scores_transform:
            return self._calculate_cdf_nsf(new_series)
        else:
            return self._calculate_cdf_parametric(new_series)
    
    def _calculate_cdf_nsf(self, new_series: Series) -> Series:
        """Calculate CDF using Normal Scores Transform approach."""
        cdf = Series(np.nan, index=new_series.index, dtype=float)
        new_grouped = group_yearly_df(series=new_series)
        
        for date, grval in new_grouped.groupby(Grouper(freq=str(self._fitted_si.fit_freq))):
            new_data = get_data_series(grval)
            
            # Find corresponding training data for this period
            if date in self._fitted_si._dist_dict:
                training_data = self._fitted_si._dist_dict[date].data.sort_values()
                n_train = len(training_data)
                
                # For each new data point, find its rank in training data
                for idx, val in new_data.items():
                    # Count how many training values are less than this new value
                    rank = (training_data < val).sum()
                    # Use Weibull plotting position
                    cdf.loc[idx] = (rank + 0.5) / n_train
        
        return cdf
    
    def _calculate_cdf_parametric(self, new_series: Series) -> Series:
        """Calculate CDF using fitted parametric distributions."""
        cdf = Series(np.nan, index=new_series.index, dtype=float)
        new_grouped = group_yearly_df(series=new_series)
        
        for date, grval in new_grouped.groupby(Grouper(freq=str(self._fitted_si.fit_freq))):
            new_data = get_data_series(grval)
            
            if date in self._fitted_si._dist_dict:
                fitted_dist = self._fitted_si._dist_dict[date]
                
                # Construct parameters tuple: (shape_params..., loc, scale)
                if fitted_dist.pars is not None:
                    params = (*fitted_dist.pars, fitted_dist.loc, fitted_dist.scale)
                else:
                    params = (fitted_dist.loc, fitted_dist.scale)
                
                # Handle zero probability case
                if self.prob_zero and fitted_dist.p0 > 0:
                    zero_prob = fitted_dist.p0
                    
                    for idx, val in new_data.items():
                        if val == 0:
                            cdf.loc[idx] = zero_prob
                        else:
                            # Get parametric CDF for non-zero values
                            dist_cdf = fitted_dist.dist.cdf(val, *params)
                            cdf.loc[idx] = zero_prob + (1 - zero_prob) * dist_cdf
                else:
                    # No zero handling needed
                    for idx, val in new_data.items():
                        cdf.loc[idx] = fitted_dist.dist.cdf(val, *params)
        
        return cdf
    
    def get_training_ssi(self) -> Series:
        """
        Get SSI values for the training data.
        
        Returns
        -------
        Series
            SSI values for training data
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before get_training_ssi()")
        
        return self._fitted_si.norm_ppf()
    
    @property
    def fitted_distributions(self) -> dict:
        """Access to fitted distributions (read-only)."""
        if not self._is_fitted:
            raise ValueError("Must call fit() before accessing fitted_distributions")
        return self._fitted_si._dist_dict.copy()

