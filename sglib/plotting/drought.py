import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def drought_metric_scatter_plot(obs_drought_metrics, syn_drought_metrics=None, window=12):
    fig, ax = plt.subplots(figsize = (7,6))
    
    p = ax.scatter(obs_drought_metrics['severity'], -obs_drought_metrics['magnitude'],
            c= obs_drought_metrics['duration'], cmap = 'viridis_r', s=100, 
            edgecolor='k', lw=1.5, label='Observed', 
            zorder=5, alpha=1)
    if syn_drought_metrics is not None:
        ax.scatter(syn_drought_metrics['severity'], -syn_drought_metrics['magnitude'],
                c= syn_drought_metrics['duration'], 
                cmap = 'viridis_r', s=100, edgecolor='none', label='Synthetic',
                zorder=1, alpha=0.5)
    
    plt.colorbar(p).set_label(label = 'Drought Duration (days)',size=15)
    plt.xlabel(r'Severity ($Min.\, SSI$)', fontsize = 15)
    plt.ylabel(r'Magnitude (Acc. Deficit)', fontsize = 15)
    plt.title(f'Drought Characteristics', fontsize = 16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
    return