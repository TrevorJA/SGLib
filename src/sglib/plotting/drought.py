import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def drought_metric_scatter_plot(obs_drought_metrics, 
                                syn_drought_metrics=None, 
                                x_char = 'magnitude',
                                y_char = 'duration',
                                color_char = 'severity',
                                fname=None):
    fig, ax = plt.subplots(figsize = (7,6))
    
    max_color_val = obs_drought_metrics[color_char].max()
    if syn_drought_metrics is not None:
        max_color_val = max(max_color_val, syn_drought_metrics[color_char].max())
    
    p = ax.scatter(obs_drought_metrics[x_char], 
                   -obs_drought_metrics[y_char],
                   c= obs_drought_metrics[color_char], 
                   cmap = 'viridis_r', s=100, 
                   vmin = 0, vmax = max_color_val,
                   edgecolor='k', lw=1.5, label='Observed', 
                   zorder=5, alpha=1)
    
    if syn_drought_metrics is not None:
        ax.scatter(syn_drought_metrics[x_char], 
                   -syn_drought_metrics[y_char],
                   c= syn_drought_metrics[color_char], 
                   cmap = 'viridis_r', s=100,
                   vmin = 0, vmax = max_color_val, 
                   edgecolor='none', 
                   label='Synthetic',
                   zorder=1, alpha=0.5)
    
    plt.colorbar(p).set_label(label = 'Drought Duration (days)',size=15)
    plt.title(f'Drought Characteristics', fontsize = 16)
    plt.legend(fontsize=14)
    if x_char == 'severity':
        plt.xlim(-3.5, -1.0)
    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {fname}")
    
    plt.show()
    return