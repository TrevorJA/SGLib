import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from sglib.core.base import Generator

def calculate_lower_bound(df):
    """
    Calculate the lower bound of a time series.
    """
    qmax = df.max()
    qmin = df.min()
    qmedian = df.median()
    tau = (qmax*qmin - qmedian**2) / (qmax + qmin - 2*qmedian)
    tau = tau if tau > 0 else 0
    return tau


def stedinger_normalization(df):
    """
    Normalize a time series using normalization technique in 
    Stedinger and Taylor (1982).
    """
    # Get lower bound; \hat{\tau} for each month
    norm_df = df.copy()
    tau_monthly = df.groupby(df.index.month).apply(calculate_lower_bound)
    for i in range(1, 13):
        tau = tau_monthly[i]
        
        # Normalize
        norm_df[df.index.month == i] = np.log(df[df.index.month == i] - tau)
    return norm_df, tau_monthly

def inverse_stedinger_normalization(df, tau_monthly):
    """
    Inverse normalize a time series using normalization technique in 
    Stedinger and Taylor (1982).
    """
    # Get lower bound; \hat{\tau} for each month
    norm_df = df.copy()
    for i in range(1, 13):
        tau = tau_monthly[i]
        
        # Normalize
        norm_df[df.index.month == i] = np.exp(df[df.index.month == i]) + tau
    
    return norm_df



class ThomasFiering(Generator):
    """
    From Thomas and Fiering, 1962.    
    Also described in Steinder and Taylor (1982).
    
    Usage:
    tf = ThomasFieringGenerator(Q_obs_monthly.iloc[:,2])
    Q_syn = tf.generate(n_years=10, n_realizations=10)
    """
    def __init__(self, Q, **kwargs):
        # Q should be pandas df or series with monthly index
        if Q.index.freq not in ['MS', 'M']:
            if Q.index.freq in ['D', 'W']:
                Q = Q.resample('MS').sum()
        
        self.Q_obs_monthly = Q 
        
        self.is_fit = False
        self.is_preprocessed = False
    
    def preprocessing(self, **kwargs):
        self.Q_norm, self.tau_monthly = stedinger_normalization(self.Q_obs_monthly)
        self.is_preprocessed = True
        return
        
    def fit(self, **kwargs):
        
        # monthly mean and std
        self.mu_monthly = self.Q_norm.groupby(self.Q_norm.index.month).mean()
        self.sigma_monthly = self.Q_norm.groupby(self.Q_norm.index.month).std()
        
        # monthly correlation between month m and m+1
        self.rho_monthly = self.mu_monthly.copy()
        for i in range(1, 13):
            first_month = i
            second_month = i+1 if i < 12 else 1
            first_month_flows = self.Q_norm[self.Q_norm.index.month == first_month]
            second_month_flows = self.Q_norm[self.Q_norm.index.month == second_month]
            if len(first_month_flows) > len(second_month_flows):
                first_month_flows = first_month_flows.iloc[1:]
            elif len(first_month_flows) < len(second_month_flows):
                second_month_flows = second_month_flows.iloc[:-1]
            lag1_r = pearsonr(first_month_flows.values, 
                              second_month_flows.values)
            
            self.rho_monthly[i] = lag1_r[0]
            
        self.is_fit = True
    
    def _generate(self, n_years, **kwargs):
            
        # Generate synthetic sequences
        self.x_syn = np.zeros((n_years*12))
        for i in range(n_years):
            for m in range(12):
                prev_month = m if m > 0 else 12
                month = m + 1
                
                if (i==0) and (m==0):
                    self.x_syn[0] = self.mu_monthly[month] + np.random.normal(0, 1)*self.sigma_monthly[month]

                else:
                    
                    e_rand = np.random.normal(0, 1)
                    
                    self.x_syn[i*12+m] = self.mu_monthly[month] + \
                                        self.rho_monthly[month]*(self.sigma_monthly[month]/self.sigma_monthly[prev_month])*\
                                            (self.x_syn[i*12+m-1] - self.mu_monthly[prev_month]) + \
                                                np.sqrt(1-self.rho_monthly[month]**2)*self.sigma_monthly[month]*e_rand
                        
        # convert to df
        syn_start_year = self.Q_obs_monthly.index[0].year
        syn_start_date = f'{syn_start_year}-01-01'
        self.x_syn = pd.DataFrame(self.x_syn, 
                                index=pd.date_range(start=syn_start_date, 
                                                    periods=len(self.x_syn), freq='MS'))
        self.x_syn[self.x_syn < 0] = self.Q_norm.min()
        
        self.Q_syn = inverse_stedinger_normalization(self.x_syn, self.tau_monthly)
        self.Q_syn[self.Q_syn < 0] = self.Q_obs_monthly.min()
        return self.Q_syn
    
    def generate(self, n_years, n_realizations = 1, **kwargs):
        
        if not self.is_preprocessed:
            self.preprocessing()
        if not self.is_fit:
            self.fit()
        
        # Generate synthetic sequences
        for i in range(n_realizations):
            Q_syn = self._generate(n_years)
            if i == 0:
                self.Q_syn_df = Q_syn
            else:
                self.Q_syn_df = pd.concat([self.Q_syn_df, Q_syn], 
                                          axis=1, ignore_index=True)
    
        return self.Q_syn_df
    
    def plot(self, kind='ts', **kwargs):
        if not self.is_preprocessed:
            self.preprocessing()
        if not self.is_fit:
            self.fit()
        if not hasattr(self, 'x_syn'):
            self.generate(n_years=5)

        fig, ax = plt.subplots(figsize=(10,5))
        
        if kind == 'ts':
            self.Q_obs_monthly.plot(label='Observed', ax=ax, 
                                    zorder=5)
            
            self.Q_syn_df.plot(ax=ax, 
                               color = 'orange', alpha=0.5, 
                               zorder=1,
                               legend=False)
        elif kind == 'hist':
            self.Q_obs_monthly.hist(label='Observed')
            self.Q_syn_df.hist(legend=False, alpha=0.5)
        elif kind == 'kde':
            self.Q_obs_monthly.plot(ax=ax, kind='kde', 
                                    label='Observed', zorder=5)
            self.Q_syn_df.plot(ax=ax, color='orange', 
                               kind='kde', legend=False, alpha=0.5, zorder=1)

        plt.show()
        return
    