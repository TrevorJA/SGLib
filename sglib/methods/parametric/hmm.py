"""
Trevor Amestoy

Contains a Hidden Markov Model generator.
"""

from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as ss
import statsmodels.api as sm

from sglib.core.base import Generator
from sglib.utils import common_default_kwargs, hmm_default_kwargs
from sglib.utils.assertions import set_model_kwargs, update_model_kwargs
from sglib.plotting.hmm_plots import plotDistribution, plotTimeSeries
epsilon = 1e-6

def deseasonalize_data(x, timestep='M',
                       standardize=True):
    # if timestep in ['M', 'MS']:
    #     group_idx = x.index.month
    #     n_annual_timesteps = 12
    # elif timestep in ['W', 'WS']:
    #     group_idx = x.index.week
    #     n_annual_timesteps = 52
    # # center on zero
    # if standardize:
    #     timestep_means = x.groupby(group_idx).mean()
    #     timestep_std = x.groupby(group_idx).std()
    #     for t in range(1, n_annual_timesteps + 1):
    #         x.loc[group_idx == t] = (x.loc[group_idx == t] - timestep_means[t]) / timestep_std[t]
    # else:
    
    result = sm.tsa.seasonal_decompose(x, model='multiplicative')
    x = x / result.seasonal

    # fill na with near
    return x, result


def reseasonalize_data(x, seasonal_result, 
                       timestep='M'):
    # if timestep in ['M', 'MS']:
    #     group_idx = x.index.month
    #     n_annual_timesteps = 12
    # elif timestep in ['W', 'WS']:
    #     group_idx = x.index.week
    #     n_annual_timesteps = 52
    # for m in range(1, n_annual_timesteps + 1):
    #     x.loc[group_idx == m] = x.loc[group_idx == m] * timestep_std[m] + timestep_means[m]
    x = x * seasonal_result.seasonal    
    return x


class HMM():
    """
    Hidden Markov Model for generating synthetic timeseries data.
    
    Methods:
        fit: Fit the model to the data.
        generate: Generate synthetic data from the fitted model.
        plotDistribution: Plot the distribution of the fitted model.
        plotTimeSeries: Plot the time series of the fitted model.
    """
    def __init__(self, Q, **kwargs):
        
        self.Q_obs = Q.copy()
        self.Q_train = Q.copy()
        
        ## Kwargs
        set_model_kwargs(self, hmm_default_kwargs, **kwargs)
        
        
    def preprocessing(self, **kwargs):
        """
        Preprocesses the input time series data. 
        """

        if self.deseasonalize:
            self.Q_train, self.seasonal_result = deseasonalize_data(self.Q_train, 
                                                                    timestep=self.timestep)
        self.Q_train = self.Q_train.dropna()
        self.Q_train = self.Q_train.loc[self.Q_train > 0]
        
        if self.log_transform:
            self.Q_train = np.log(self.Q_train)
        self._is_preprocessed = True
        

    
    def fit(self, **kwargs):
        if not self._is_preprocessed:
            print('Preprocessing data.')
            self.preprocessing()
        Q = self.Q_train.copy()
        if type(Q) == pd.core.frame.DataFrame:
            Q = Q.values.flatten()
        elif type(Q) == pd.core.series.Series:
            Q = Q.values
        elif type(Q) == np.ndarray:
            Q = Q.flatten()
            
        # fit Gaussian HMM to Q
        model = GaussianHMM(n_components=self.n_hidden_states, 
                            n_iter=self.max_iter, 
                            tol=self.tolerance).fit(np.reshape(Q,[len(Q),1]))
        self._is_fit = True
        
        # classify each observation as state 0 or 1
        hidden_states = model.predict(np.reshape(Q,[len(Q),1]))
    
        # find parameters of Gaussian HMM
        mus = np.array(model.means_)
        sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[i]) for i in range(self.n_hidden_states)])))
        P = np.array(model.transmat_)
    
        # find log-likelihood of Gaussian HMM
        logProb = model.score(np.reshape(Q,[len(Q),1]))
    
        # re-organize mus, sigmas and P so that first row is lower mean (if not already)
        if mus[0] > mus[1]:
            mus = np.flipud(mus)
            sigmas = np.flipud(sigmas)
            P = np.fliplr(np.flipud(P))
            hidden_states = 1 - hidden_states
        
        self.hidden_states = hidden_states
        self.mus = mus
        self.P = P
        self.sigmas = sigmas
        self.logProb = logProb
        self.model = model
        return
    
    def generate(self, **kwargs):
        if not self._is_preprocessed:
            print('Preprocessing data.')
            self.preprocessing()
        
        if not self._is_fit:
            print('Fitting model to historic data.')
            self.fit()   
        
        # Set kwargs if provided
        # update_model_kwargs(self, hmm_default_kwargs, **kwargs)
        
        # Generate n_realizations from Gaussian HMM
        X_syn = np.zeros((self.n_timesteps, self.n_realizations))
        X_syn_states = np.zeros((self.n_timesteps, self.n_realizations))
        for i in range(self.n_realizations):
            synthetic_timeseries, synthetic_states = self.model.sample(self.n_timesteps)
            X_syn[:,i] = synthetic_timeseries.flatten()
            X_syn_states[:,i] = synthetic_states.flatten()
            
        # Arrange in DF
        start_year = self.Q_obs.index.year[0]
        start_date = f'{start_year}-01-01'
        syn_datetime_index = pd.date_range(start=start_date, 
                                       periods=self.n_timesteps, 
                                       freq=self.timestep)
        X_syn = pd.DataFrame(X_syn, index=syn_datetime_index,
                             columns=[f'realization_{i}' for i in range(self.n_realizations)])
            
        self.X_syn = X_syn.copy()
        self.X_syn_states = X_syn_states
        
        # Transform back to original scale
        if self.log_transform:
            X_syn = np.exp(X_syn)
        if self.deseasonalize:
            X_syn = reseasonalize_data(X_syn, self.seasonal_result,
                                       timestep=self.timestep)

            



        self.Q_syn = X_syn
        return self.Q_syn
    
    
    def plot(self, kind='line', **kwargs):
        if not self._is_fit:
            print('Fitting model to historic data.')
            self.fit()   
        
        # Plot 
        if kind == 'line':
            plotTimeSeries(self, **kwargs)
        elif kind == 'dist':
            plotDistribution(self, **kwargs)
        return
        