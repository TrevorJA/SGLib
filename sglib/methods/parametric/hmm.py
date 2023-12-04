"""
Trevor Amestoy

Contains a Hidden Markov Model generator.
"""

from hmmlearn.hmm import GaussianHMM
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as ss

from sglib.core.base import Generator
from sglib.utils import common_default_kwargs, hmm_default_kwargs
from sglib.utils.assertions import set_model_kwargs, update_model_kwargs
from sglib.plotting.hmm_plots import plotDistribution, plotTimeSeries

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
        self.Q = Q
            
        ## Kwargs
        set_model_kwargs(self, hmm_default_kwargs, **kwargs)
        
        
    def preprocessing(self, **kwargs):
        """
        Preprocesses the input time series data. Currently a placeholder, can be expanded for specific preprocessing needs.
        """
        
        if self.log_transform:
            self.Q = np.log(self.Q)
        pass
    
             
    
    def fit(self, **kwargs):
        Q = self.Q
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
        if not self._is_fit:
            print('Fitting model to historic data.')
            self.fit()   
        
        # Set kwargs if provided
        update_model_kwargs(self, hmm_default_kwargs, **kwargs)
        
        # Generate n_realizations from Gaussian HMM
        Q_syn = np.zeros((self.n_timesteps, self.n_realizations))
        Q_syn_states = np.zeros((self.n_timesteps, self.n_realizations))
        for i in range(self.n_realizations):
            synthetic_timeseries, synthetic_states = self.model.sample(self.n_timesteps)
            Q_syn[:,i] = synthetic_timeseries.flatten()
            Q_syn_states[:,i] = synthetic_states.flatten()
        self.Q_syn = Q_syn
        self.Q_syn_states = Q_syn_states
        return Q_syn
    
    
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
        