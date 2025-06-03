
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def plot_flow_ranges(Qh, Qs, 
                     timestep = 'daily',
                     units = 'cms', y_scale = 'log',
                     savefig = False, fname = None,
                     figsize = (7,5), colors = ['black', 'orange'],
                     title_addon = ""):
    """Plots the range of flow for historic and syntehtic streamflows for a specific timestep scale.
     
    Args:
        Qh (pd.Series): Historic daily streamflow timeseries. Index must be pd.DatetimeIndex. 
        Qs (pd.DataFrame): Synthetic daily streamflow timeseries realizations. Each column is a unique realization. Index must be pd.DatetimeIndex.
        timestep (str, optional): The timestep which data should be aggregated over. Defaults to 'daily'. Options are 'daily', 'weekly', or 'monthly'.
        units (str, optional): Streamflow units, for axis label. Defaults to 'cms'.
        y_scale (str, optional): Scale of the y-axis. Defaults to 'log'.
        savefig (bool, optional): Allows for png to be saved to fname. Defaults to False.
        fname (str, optional): Location of saved figure output. Defaults to '.' (working directory).
        figsize (tuple, optional): The figure size. Defaults to (4,4).
        colors (list, optional): List of two colors for historic and synthetic data respectively. Defaults to ['black', 'orange'].
        title_addon (str, optional): Text to be added to the end of the title. Defaults to "".
    """
 
    # Assert formatting matches expected
    assert(type(Qh.index) == pd.DatetimeIndex), 'Historic streamflow (Qh) should have pd.DatatimeIndex.'
    assert(type(Qs.index) == pd.DatetimeIndex), 'Synthetic streamflow (Qh) should have pd.DatatimeIndex.'
 
    # Handle conditional datetime formatting
    if timestep == 'daily':
        h_grouper = Qh.index.dayofyear
        s_grouper = Qs.index.dayofyear
        x_lab = 'Day of the Year (Jan-Dec)'
    elif timestep == 'monthly':
        h_grouper = Qh.index.month
        s_grouper = Qs.index.month
        x_lab = 'Month of the Year (Jan-Dec)'
    elif timestep == 'weekly':
        h_grouper = pd.Index(Qh.index.isocalendar().week, dtype = int)
        s_grouper = pd.Index(Qs.index.isocalendar().week, dtype = int)
        x_lab = 'Week of the Year (Jan-Dec)'
    else:
        print('Invalid timestep input. Options: "daily", "monthly", "weekly".')
        return
 
    # Find flow ranges
    s_max = Qs.groupby(s_grouper).max().max(axis=1)
    s_min = Qs.groupby(s_grouper).min().min(axis=1)
    s_median = Qs.groupby(s_grouper).median().median(axis=1)
    h_max = Qh.groupby(h_grouper).max()
    h_min = Qh.groupby(h_grouper).min()
    h_median = Qh.groupby(h_grouper).median()
   
    ## Plotting  
    fig, ax = plt.subplots(figsize = figsize, dpi=150)
    xs = h_max.index
    # print(f'xs: {xs} \n\n s_min: {s_min} \n\n')
    ax.fill_between(xs, s_min, s_max, color = colors[1], label = 'Synthetic Range', alpha = 0.5)
    ax.plot(xs, s_median, color = colors[1], label = 'Synthetic Median')
    ax.fill_between(xs, h_min, h_max, color = colors[0], label = 'Historic Range', alpha = 0.3)
    ax.plot(xs, h_median, color = colors[0], label = 'Historic Median')
     
    ax.set_yscale(y_scale)
    ax.set_ylabel(f'{timestep.capitalize()} Flow ({units})', fontsize=12)
    ax.set_xlabel(x_lab, fontsize=12)
    ax.legend(ncols = 2, fontsize = 10, bbox_to_anchor = (0, -.5, 1.0, 0.2), loc = 'upper center')    
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_title(f'{timestep.capitalize()} Streamflow Ranges\nHistoric & Synthetic Timeseries at One Location\n{title_addon}')
    plt.tight_layout()
     
    if savefig:
        plt.savefig(fname, dpi = 150)
    return


######################################################################################
######################################################################################


def plot_fdc_ranges(Qh, Qs, 
                    units = 'cms', y_scale = 'log',
                    savefig = False, fname = None,
                    figsize = (5,5), colors = ['black', 'orange'],                   
                    title_addon = ""):
    """Plots the range and aggregate flow duration curves for historic and synthetic streamflows.
     
    Args:
        Qh (pd.Series): Historic daily streamflow timeseries. Index must be pd.DatetimeIndex. 
        Qs (pd.DataFrame): Synthetic daily streamflow timeseries realizations. Each column is a unique realization. Index must be pd.DatetimeIndex.
        units (str, optional): Streamflow units, for axis label. Defaults to 'cms'.
        y_scale (str, optional): Scale of the y-axis. Defaults to 'log'.
        savefig (bool, optional): Allows for png to be saved to fname. Defaults to False.
        fname (str, optional): Location of saved figure output. Defaults to '.' (working directory).
        figsize (tuple, optional): The figure size. Defaults to (4,4).
        colors (list, optional): List of two colors for historic and synthetic data respectively. Defaults to ['black', 'orange'].
        title_addon (str, optional): Text to be added to the end of the title. Defaults to "".
    """
     
    ## Assertions
    assert(type(Qs) == pd.DataFrame), 'Synthetic streamflow should be type pd.DataFrame.'
    assert(type(Qh.index) == pd.DatetimeIndex), 'Historic streamflow (Qh) should have pd.DatatimeIndex.'
    assert(type(Qs.index) == pd.DatetimeIndex), 'Synthetic streamflow (Qh) should have pd.DatatimeIndex.'
 
     
    # Calculate FDCs for total period and each realization
    nonexceedance = np.linspace(0.0001, 0.9999, 50)
    s_total_fdc = np.quantile(Qs.values.flatten(), nonexceedance)
    h_total_fdc = np.quantile(Qh.values.flatten(), nonexceedance) 
     
    s_fdc_max = np.zeros_like(nonexceedance)
    s_fdc_min = np.zeros_like(nonexceedance)
    h_fdc_max = np.zeros_like(nonexceedance)
    h_fdc_min = np.zeros_like(nonexceedance)
 
    annual_synthetics = Qs.groupby(Qs.index.year)
    annual_historic = Qh.groupby(Qh.index.year)
 
    for i, quant in enumerate(nonexceedance):
            s_fdc_max[i] = annual_synthetics.quantile(quant).max().max()
            s_fdc_min[i] = annual_synthetics.quantile(quant).min().min()
            h_fdc_max[i] = annual_historic.quantile(quant).max()
            h_fdc_min[i] = annual_historic.quantile(quant).min()
     
    ## Plotting
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
     
    #for quant in syn_fdc_quants:
    ax.fill_between(nonexceedance, s_fdc_min, s_fdc_max, color = colors[1], label = 'Synthetic Annual FDC Range', alpha = 0.5)
    ax.fill_between(nonexceedance, h_fdc_min, h_fdc_max, color = colors[0], label = 'Historic Annual FDC Range', alpha = 0.3)
 
    ax.plot(nonexceedance, s_total_fdc, color = colors[1], label = 'Synthetic Total FDC', alpha = 1)
    ax.plot(nonexceedance, h_total_fdc, color = colors[0], label = 'Historic Total FDC', alpha = 1, linewidth = 2)
 
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_yscale(y_scale)
    ax.set_ylabel(f'Flow ({units})')
    ax.set_xlabel('Non-Exceedance Probability')
    ax.legend(fontsize= 10)
    ax.grid(True, linestyle='--', linewidth=0.5)
     
    plt.title(f'Flow Duration Curve Ranges\nHistoric & Synthetic Streamflow\n{title_addon}')
    if savefig:
        assert(fname is not None), 'If savefig is True, fname must be provided.'
        plt.savefig(fname, dpi=200)
    plt.show()
    return


######################################################################################
######################################################################################

def plot_autocorrelation(Qh, Qs, lag_range, 
                         timestep = 'daily',
                         savefig = False, fname = None,
                         figsize=(7, 5), colors = ['black', 'orange'],
                         alpha = 0.3):
    """Plot autocorrelation of historic and synthetic flow over some range of lags.
 
    Args:
        Qh (pd.Series): Historic daily streamflow timeseries. Index must be pd.DatetimeIndex. 
        Qs (pd.DataFrame): Synthetic daily streamflow timeseries realizations. Each column is a unique realization. Index must be pd.DatetimeIndex.
        lag_range (iterable): A list or range of lag values to be used for autocorrelation calculation.
         
        timestep (str, optional): The timestep which data should be aggregated over. Defaults to 'daily'. Options are 'daily', 'weekly', or 'monthly'.
        savefig (bool, optional): Allows for png to be saved to fname. Defaults to False.
        fname (str, optional): Location of saved figure output. Defaults to '.' (working directory).
        figsize (tuple, optional): The figure size. Defaults to (4,4).
        colors (list, optional): List of two colors for historic and synthetic data respectively. Defaults to ['black', 'orange'].
        alpha (float, optional): The opacity of synthetic data. Defaults to 0.3.
    """
     
    ## Assertions
    assert(type(Qs) == pd.DataFrame), 'Synthetic streamflow should be type pd.DataFrame.'
    assert(type(Qh.index) == pd.DatetimeIndex), 'Historic streamflow (Qh) should have pd.DatatimeIndex.'
    assert(type(Qs.index) == pd.DatetimeIndex), 'Synthetic streamflow (Qh) should have pd.DatatimeIndex.'
 
    if timestep == 'monthly':
        Qh = Qh.resample('MS').sum()
        Qs = Qs.resample('MS').sum()
        time_label = 'months'
    elif timestep == 'weekly':
        Qh = Qh.resample('W-SUN').sum()
        Qs = Qs.resample('W-SUN').sum()
        time_label = f'weeks'
    elif timestep == 'daily':
        time_label = f'days'
         
    # Calculate autocorrelations
    autocorr_h = np.zeros(len(lag_range))
    confidence_autocorr_h = np.zeros((2, len(lag_range)))
    for i, lag in enumerate(lag_range):
        h_corr = pearsonr(Qh.values[:-lag], Qh.values[lag:])
        autocorr_h[i] = h_corr[0]
        confidence_autocorr_h[0,i] = h_corr.confidence_interval().low
        confidence_autocorr_h[1,i] = h_corr.confidence_interval().high
        
    autocorr_s = np.zeros((Qs.shape[1], len(lag_range)))
    
    for i, realization in enumerate(Qs.columns):
        autocorr_s[i] = [pearsonr(Qs[realization].values[:-lag], Qs[realization].values[lag:])[0] for lag in lag_range]
 
    ## Plotting
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
 
    # Plot autocorrelation for each synthetic timeseries
    for i, realization in enumerate(Qs.columns):
        if i == 0:
            ax.plot(lag_range, autocorr_s[i], alpha=1, color = colors[1], label=f'Synthetic Realization')
        else:
            ax.plot(lag_range, autocorr_s[i], color = colors[1], alpha=alpha, zorder = 1)
        ax.scatter(lag_range, autocorr_s[i], alpha=alpha, color = 'orange', zorder = 2)
 
    # Plot autocorrelation for the historic timeseries
    ax.plot(lag_range, autocorr_h, color=colors[0], linewidth=2, label='Historic', zorder = 3)
    ax.scatter(lag_range, autocorr_h, color=colors[0], zorder = 4)
    
 
    # Set labels and title
    ax.set_xlabel(f'Lag ({time_label})', fontsize=12)
    ax.set_ylabel('Autocorrelation (Pearson)', fontsize=12)
    ax.set_title('Autocorrelation Comparison\nHistoric Timeseries vs. Synthetic Ensemble', fontsize=14)
 
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92)
 
    if savefig:
        assert(fname is not None), 'If savefig is True, fname must be provided.'
        plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.show()
    return

def plot_correlation(Qh, Qs_i,
                        timestep = 'daily',
                        savefig = False, fname = None,
                        figsize = (8,4), color_map = 'BuPu',
                        cbar_on = True):
    """Plots the side-by-side heatmaps of flow correlation between sites for historic and synthetic multi-site data.
     
    Args:
        Qh (pd.DataFrame): Historic daily streamflow timeseries at many different locations. The columns of Qh and Qs_i must match, and index must be pd.DatetimeIndex. 
        Qs (pd.DataFrame): Synthetic daily streamflow timeseries realizations. Each column is a unique realization. The columns of Qh and Qs_i must match, and index must be pd.DatetimeIndex.
        timestep (str, optional): The timestep which data should be aggregated over. Defaults to 'daily'. Options are 'daily', 'weekly', or 'monthly'.
        savefig (bool, optional): Allows for png to be saved to fname. Defaults to False.
        fname (str, optional): Location of saved figure output. Defaults to '.' (working directory).
        figsize (tuple, optional): The figure size. Defaults to (4,4).
        color_map (str, optional): The colormap used for the heatmaps. Defaults to 'BuPu.
        cbar_on (bool, optional): Indictor if the colorbar should be shown or not. Defaults to True.
    """
     
    ## Assertions
    assert(type(Qh) == type(Qs_i) and (type(Qh) == pd.DataFrame)), 'Both inputs Qh and Qs_i should be pd.DataFrames.'
    assert(np.mean(Qh.columns == Qs_i.columns) == 1.0), 'Historic and synthetic data should have same columns.'
     
    if timestep == 'monthly':
        Qh = Qh.resample('MS').sum()
        Qs_i = Qs_i.resample('MS').sum()
    elif timestep == 'weekly':
        Qh = Qh.resample('W-SUN').sum()
        Qs_i = Qs_i.resample('W-SUN').sum()
     
    ## Plotting
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi = 250)
     
    # Heatmap of historic correlation
    h_correlation = np.corrcoef(Qh.values.transpose())
    sns.heatmap(h_correlation, ax = ax1, 
                square = True, annot = False,
                xticklabels = 5, yticklabels = 5, 
                cmap = color_map, cbar = False, vmin = 0, vmax = 1)
 
    # Synthetic correlation
    s_correlation = np.corrcoef(Qs_i.values.transpose())
    sns.heatmap(s_correlation, ax = ax2,
                square = True, annot = False,
                xticklabels = 5, yticklabels = 5, 
                cmap = color_map, cbar = False, vmin = 0, vmax = 1)
     
    for axi in (ax1, ax2):
        axi.set_xticklabels([])
        axi.set_yticklabels([])
    ax1.set(xlabel = 'Site i', ylabel = 'Site j', title = 'Historic Streamflow')
    ax2.set(xlabel = 'Site i', ylabel = 'Site j', title = 'Synthetic Realization')
    title_string = f'Pearson correlation across different streamflow locations\n{timestep.capitalize()} timescale\n'
         
    # Adjust the layout to make room for the colorbar
    if cbar_on:
        plt.tight_layout(rect=[0, 0, 0.9, 0.85])
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.6])  # [left, bottom, width, height]
        cbar = fig.colorbar(ax1.collections[0], cax=cbar_ax)
        cbar.set_label("Pearson Correlation")
    plt.suptitle(title_string, fontsize = 12)
     
    if savefig:
        assert(fname is not None), 'If savefig is True, fname must be provided.'
        plt.savefig(fname, dpi = 250)
    plt.show()
    return