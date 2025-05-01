"""Contains functions for detecting drought periods in a time series.
"""

import numpy as np
import pandas as pd
import spei as si
import scipy.stats as scs


def aggregate_and_rolling_sum(Q_i,
                              window, 
                              aggregate = None):
    if aggregate is not None:
        Q_i = Q_i.resample(aggregate).sum()
    Q_i = Q_i.rolling(window).sum().iloc[window:, :].dropna()
    Q_i = Q_i.iloc[:-window, :]
    return Q_i
    

def calculate_ssi_values(df, 
                         window, 
                         aggregate = None):
    """Calculate the Standardized Streamflow Index (SSI) for a given time series."""
    df = aggregate_and_rolling_sum(df, window, aggregate)
    
    ssi = pd.DataFrame(index = df.index, columns = df.columns)
    for col in df.columns:
        ssi[col] = si.ssfi(df[col], dist= scs.gamma)
    return ssi



def get_drought_metrics(df, window=12, aggregate = None):
    """Get drought start and end dates, magnitude, severity, and duration.

    Args:
        ssi (pd.Series): Array of SSI values.  

    Returns:
        pd.DataFrame: DataFrame containing all drought metrics for each drought period.
    """
    if len(np.shape(df)) == 1:
        df = pd.DataFrame(df)

    realizations = df.columns
    ssi = calculate_ssi_values(df, window = window, aggregate = aggregate)
    drought_data = {}
    drought_counter = 0
    
    for realization in realizations:
        ssi_i = ssi[realization]
        # Reset counters
        in_critical_drought = False
        drought_days = [] 
        start_drought = None
        end_drought = None
        for ind in range(len(ssi_i)):
        
            if ssi_i.values[ind] < 0:
                drought_days.append(ind)
                if ssi_i.values[ind] <= -1:
                    in_critical_drought = True
                    start_drought = ssi_i.index[ind]
                    
            else:
                # Record drought info once it ends
                if in_critical_drought:
                    end_drought = ssi_i.index[ind - 1]
                    drought_counter += 1
                    drought_data[drought_counter] = {
                        'realization': realization,
                        'duration': len(drought_days),
                        'magnitude': sum(ssi_i.values[drought_days]),
                        'severity': min(ssi_i.values[drought_days]),
                        'start': start_drought,
                        'end': end_drought
                    }

                # Reset counters
                in_critical_drought = False
                drought_days = [] 
                start_drought = None
                end_drought = None

    drought_metrics = pd.DataFrame(drought_data).transpose()
    return drought_metrics