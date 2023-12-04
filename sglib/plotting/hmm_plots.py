import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as ss

from sglib.utils.kwargs import default_plot_kwargs


def plotDistribution(self, n_bins = 100):
    
    # calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(self.P))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])

    x_0 = np.linspace(self.mus[0]-4*self.sigmas[0], self.mus[0]+4*self.sigmas[0], 10000)
    fx_0 = pi[0]*ss.norm.pdf(x_0,self.mus[0],self.sigmas[0])

    x_1 = np.linspace(self.mus[1]-4*self.sigmas[1], self.mus[1]+4*self.sigmas[1], 10000)
    fx_1 = pi[1]*ss.norm.pdf(x_1,self.mus[1],self.sigmas[1])

    x = np.linspace(self.mus[0]-4*self.sigmas[0], self.mus[1]+4*self.sigmas[1], 10000)
    fx = pi[0]*ss.norm.pdf(x,self.mus[0],self.sigmas[0]) + \
        pi[1]*ss.norm.pdf(x,self.mus[1],self.sigmas[1])

    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.hist(self.Q, color='k', alpha=0.5, density=True, bins= n_bins)
    
    for i in range(self.n_hidden_states):
        if i == 0:
            c = 'peru'
        elif i == 1:
            c = 'royalblue'
        else:
            c = 'green'
        x = np.linspace(self.mus[i]-4*self.sigmas[i], self.mus[i]+4*self.sigmas[i], 10000)
        fx = pi[i]*ss.norm.pdf(x, self.mus[i], self.sigmas[i])
        
        ax.plot(x, fx, linewidth=2, label=f'State {i+1} Dist.', color = c)
    

    low_state_index = np.argmin(self.mus)
    high_state_index = np.argmax(self.mus)
    
    x_combined = np.linspace(self.mus[low_state_index]-4*self.sigmas[low_state_index], 
                                self.mus[high_state_index]+4*self.sigmas[high_state_index], 10000)
    
    for i in range(self.n_hidden_states):
        if i == 0:
            fx_combined = pi[0]*ss.norm.pdf(x_combined,self.mus[0],self.sigmas[0])
        else:
            fx_combined = fx_combined + pi[i]*ss.norm.pdf(x_combined,self.mus[i],self.sigmas[i])
    
    ax.plot(x_combined, fx_combined, c='k', linewidth=2, label='Combined State Distn')

    fig.subplots_adjust(bottom=0.15)
    handles, labels = plt.gca().get_legend_handles_labels()
    matplotlib.rc('legend', fontsize = 16)
    plt.legend() 
    
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    
    plt.show()
    plt.close()
    
    return


def plotTimeSeries(self, ylabel = 'Flow', n_bins = 100, start_year = None):

    sns.set_theme(style='white')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(13, 6.5), dpi = 200, 
                            sharey=True, gridspec_kw={'width_ratios':[5,1]})

    xs = np.arange(len(self.Q))+start_year if start_year else np.arange(len(self.Q))
    
    for i in range(self.n_hidden_states):
        if i == 0:
            c = 'peru'
        elif i == 1:
            c = 'royalblue'
        else:
            c = 'green'
        
        masks = self.hidden_states == i
                    
        # Plot distribution in second plot
        ax[1].hist(self.Q[masks], bins = n_bins, 
                    color=c, label = f'State {i+1}', 
                    orientation = 'horizontal', density = True, 
                    alpha = 0.5)
        
        # Plot scatter of hidden states
        ax[0].scatter(xs[masks], self.Q[masks], color=c, label=f'State {i+1}')
    
    ax[0].plot(xs, self.Q, c='k', linewidth = 0.75, label = 'Observed Flow')
    ax[0].set_xlabel('Year',fontsize=16)
    ax[0].set_ylabel(ylabel,fontsize=16)
    ax[0].legend(loc = 'upper left', fontsize =16, framealpha = 1)
    
    ax[1].set_xlabel('Modeled Hidden States')
    ax[1].legend(loc = 'lower center', fontsize = 16)
    
    fig.subplots_adjust(bottom=0.2)

    # matplotlib.rc('legend', fontsize = 16)
    plt.legend() 

    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)

    plt.show()
    plt.close()

    return None
