import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def init_plotting():
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (14, 14)
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.labelsize'] = 1.1 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.1 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color, linestyle='solid')
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color='k')

def boxplots(syn, hist, xticks=True, legend=True, loc='upper right'):
    bpl = plt.boxplot(syn, positions=np.arange(1, len(syn[0]) + 1) - 0.15, sym='', widths=0.25, patch_artist=True)
    bpr = plt.boxplot(hist, positions=np.arange(1, len(hist[0]) + 1) + 0.15, sym='', widths=0.25, patch_artist=True)
    set_box_color(bpl, 'lightskyblue')
    set_box_color(bpr, 'lightcoral')
    plt.plot([], c='lightskyblue', label='Synthetic')
    plt.plot([], c='lightcoral', label='Historical')
    plt.gca().xaxis.grid(False)
    if xticks:
        points = range(1, len(syn[0]) + 1)
        plt.gca().set_xticks(points)
        plt.gca().set_xticklabels(points)
    else:
        plt.gca().set_xticks([])
    if legend:
        plt.legend(ncol=2, loc=loc)

def plot_validation(H_df, S_df, 
                    scale='weekly', 
                    sitename='Site'):
    if not isinstance(H_df, pd.DataFrame) or not isinstance(S_df, pd.DataFrame):
        raise TypeError("H_df and S_df must be pandas DataFrames")

    init_plotting()
    assure_path_exists('./figures/')
    sitefile = re.sub(r'\W+', '_', sitename)


    ### Data prep and reorganization
    # Based on the timescale, we want to aggregate the data
    # Then, reformat each so that it is in shape: (n_realizations, n_years, n_periods)
    # For historic data, we will resample the data n_realizations times to match the number of synthetic realizations
    if scale == 'monthly':
        H = H_df.resample('MS').sum()
        S = S_df.resample('MS').sum()
        
        # Reorganize to be shape (n_years, n_periods)
        H = H.pivot_table(index=H.index.year, columns=H.index.month, values=H.columns[0]).values
        
        # S has shape (n_years * 12, n_realizations)
        # we want it in shape (n_realizations, n_years, 12)
        S = (S.T
                .groupby(S.index.year, axis=1)
                .apply(lambda x: x.values)
                .values)
        S = np.stack(list(S), axis=1) 
        
        
    # For historic data, 
    # we want to resample so that it has the same number of 
    # realizations as the synthetic data
    n_realizations_S = S.shape[0]
    n_years_H = H.shape[0]
    idx = np.random.choice(n_years_H, size=(n_realizations_S, n_years_H), 
                           replace=True)
    # Make H_resamp so that it is shape (n_realizations, n_years, n_periods)
    H_resamp = H[idx]
    
        
    for logspace in [False, True]:
        H_proc = np.log(np.clip(H, a_min=1e-6, a_max=None)) if logspace else H
        S_proc = np.log(np.clip(S, a_min=1e-6, a_max=None)) if logspace else S

        # Print shapes
        print(f"Historic data shape: {H_proc.shape}")
        print(f"Historic resampled data shape: {H_resamp.shape}")
        print(f"Synthetic data shape: {S_proc.shape}")

        time_dim = H_proc.shape[1]

        fig = plt.figure()

        ax = fig.add_subplot(5, 1, 1)
        boxplots(S_proc.reshape((np.shape(S_proc)[0]*np.shape(S_proc)[1], 12)), H_proc, xticks=False, legend=True)
        ax.set_ylabel('Log(Q)' if logspace else 'Q')

        ax = fig.add_subplot(5, 1, 2)
        boxplots(S_proc.mean(axis=0), H_resamp.mean(axis=0), xticks=False, legend=False)
        ax.set_ylabel('$\hat{\mu}_Q$')

        ax = fig.add_subplot(5, 1, 3)
        boxplots(S_proc.std(axis=0), H_resamp.std(axis=0), xticks=False, legend=False)
        ax.set_ylabel('$\hat{\sigma}_Q$')

        stat_pvals = np.zeros((2, time_dim))
        for i in range(time_dim):
            stat_pvals[0, i] = stats.ranksums(H_proc[:, i], S_proc.reshape((np.shape(S_proc)[0]*np.shape(S_proc)[1], 12))[:, i])[1]
            stat_pvals[1, i] = stats.levene(H_proc[:, i], S_proc.reshape((np.shape(S_proc)[0]*np.shape(S_proc)[1], 12))[:, i])[1]

        ax = fig.add_subplot(5, 1, 4)
        ax.bar(np.arange(1, time_dim + 1), stat_pvals[0], facecolor='0.7', edgecolor='None')
        ax.plot([0, time_dim + 1], [0.05, 0.05], color='k')
        ax.set_xlim([0, time_dim + 1])
        ax.set_ylabel('Wilcoxon $p$')

        ax = fig.add_subplot(5, 1, 5)
        ax.bar(np.arange(1, time_dim + 1), stat_pvals[1], facecolor='0.7', edgecolor='None')
        ax.plot([0, time_dim + 1], [0.05, 0.05], color='k')
        ax.set_xlim([0, time_dim + 1])
        ax.set_ylabel('Levene $p$')

        fig.suptitle(('Log space' if logspace else 'Real space') + f' ({sitename})')
        fig.tight_layout()
        fig.savefig(f'figures/{scale}_moments_pvalues_{"log" if logspace else "real"}_{sitefile}.pdf')
        plt.close(fig)



# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import ranksums
# from scipy.stats import levene

# def plot_statistical_flow_test(Qh, Qs, timescale='monthly', fname=None):
#     """Creates a multi-panel plot comparing historic and synthetic flow statistics. 
    
#     The five panels include, for each timescale:
#     1. Total flow distributions (boxplots of all flow values)
#     2. Distribution of yearly mean flows (boxplots of yearly means)
#     3. Distribution of yearly std deviations (boxplots of yearly stds)
#     4. Willcoxon rank-sum test p-values comparing yearly means (barplots)
#     5. Levene's test p-values comparing yearly standard deviations (barplots)

#     Parameters
#     ----------
#     Qh : pd.Series or pd.DataFrame
#         Historic flow data with datetime index. If DataFrame, should have single column.
#         Expected shape: (n_historic_years * 365, 1)
#     Qs : pd.DataFrame
#         Synthetic flow data with datetime index and multiple columns representing
#         different realizations of stochastic timeseries.
#         Expected shape: (n_synthetic_years * 365, n_realizations)
#     timescale : str, optional
#         Timescale for analysis, by default 'monthly'. Options are 'monthly', or 'weekly'.
#     fname : str, optional
#         Filename to save the plot. If None, plot is not saved.
#     """
    
#     # Input validation and conversion
#     if isinstance(Qh, pd.Series):
#         Qh = Qh.to_frame(name='flow')
#     elif isinstance(Qh, pd.DataFrame):
#         if Qh.shape[1] != 1:
#             raise ValueError("Qh DataFrame must have exactly one column.")
#         Qh.columns = ['flow']  # Standardize column name
#     else:
#         raise ValueError("Qh must be a pandas Series or single-column DataFrame with datetime index.")
    
#     if not isinstance(Qs, pd.DataFrame):
#         raise ValueError("Qs must be a pandas DataFrame with datetime index and multiple columns.")
    
#     # Check datetime indices
#     if not isinstance(Qh.index, pd.DatetimeIndex):
#         raise ValueError("Qh must have a datetime index.")
#     if not isinstance(Qs.index, pd.DatetimeIndex):
#         raise ValueError("Qs must have a datetime index.")

#     # Aggregate the data based on the specified timescale
#     if timescale == 'monthly':
#         Qh_agg = Qh.resample('MS').sum()
#         Qs_agg = Qs.resample('MS').sum()
#         n_periods = 12
#         period_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
#                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
#     elif timescale == 'weekly':
#         Qh_agg = Qh.resample('WS').sum()
#         Qs_agg = Qs.resample('WS').sum()
#         n_periods = 52
#         period_labels = [str(x) if x % 4 == 1 else '' for x in range(1, 53)]
#     else:
#         raise ValueError("Invalid timescale. Choose from 'monthly' or 'weekly'.")
    
#     # Add period and year columns
#     if timescale == 'monthly':
#         Qh_agg['period'] = Qh_agg.index.month
#         Qs_agg['period'] = Qs_agg.index.month
#     elif timescale == 'weekly':
#         Qh_agg['period'] = Qh_agg.index.isocalendar().week
#         Qs_agg['period'] = Qs_agg.index.isocalendar().week
    
#     Qh_agg['year'] = Qh_agg.index.year
#     Qs_agg['year'] = Qs_agg.index.year
    
#     # Reshape data into arrays
#     # Qh_array: shape (n_periods, n_historic_years)
#     Qh_pivot = Qh_agg.pivot(index='period', columns='year', values='flow')
#     Qh_array = Qh_pivot.reindex(range(1, n_periods + 1)).values
    
#     # Qs_array: shape (n_periods, n_synthetic_years * n_realizations)
#     synthetic_cols = [col for col in Qs_agg.columns if col not in ['period', 'year']]
#     Qs_arrays = []
    
#     for col in synthetic_cols:
#         Qs_pivot = Qs_agg.pivot(index='period', columns='year', values=col)
#         Qs_pivot_reindexed = Qs_pivot.reindex(range(1, n_periods + 1)).values
#         Qs_arrays.append(Qs_pivot_reindexed)
    
#     # Concatenate all realizations horizontally
#     Qs_array = np.concatenate(Qs_arrays, axis=1)
    
#     # Prepare data for boxplots
#     # for the historic flows, we need to resample with replacement to match
#     # the number of synthetic realizations
#     n_resamples = Qs_array.shape[1] // Qh_array.shape[1]
#     resampled_indices = np.random.randint(0, Qh_array.shape[1], 
#                                           size=(n_resamples, Qh_array.shape[1]))
    
#     # For raw flows (subplot 1)
#     Qh_flows = [Qh_array[i, :][~np.isnan(Qh_array[i, :])] for i in range(n_periods)]
#     Qs_flows = [Qs_array[i, :][~np.isnan(Qs_array[i, :])] for i in range(n_periods)]
    
#     # Make resampled 
    
#     # For means (subplot 2) - mean along years/realizations axis
#     # shape should be (n_periods, n_samples)
#     # where n_samples is the number of synthetic realizations and/or historic resampled years
#     Qh_means = np.zeros((n_periods, n_resamples))
#     for i in range(n_resamples)
    
    
#     # For standard deviations (subplot 3) - std along years/realizations axis  
#     Qh_stds = [np.nanstd(Qh_array[i, :]) for i in range(n_periods)]
#     Qs_stds = [np.nanstd(Qs_array[i, :]) for i in range(n_periods)]

#     ### Plotting
#     fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)
#     xs = np.arange(1, n_periods + 1)
    
#     ## Plot 1: Box plots of total flows
#     axs[0].boxplot(Qh_flows, 
#                    positions=xs - 0.2, widths=0.4, 
#                    patch_artist=True, sym='',
#                    boxprops=dict(facecolor='lightblue', color='blue'), 
#                    medianprops=dict(color='blue'))
#     axs[0].boxplot(Qs_flows,
#                    positions=xs + 0.2, widths=0.4, 
#                    patch_artist=True, sym='',
#                    boxprops=dict(facecolor='lightgreen', color='green'), 
#                    medianprops=dict(color='green'))
#     axs[0].set_ylabel('Flow')
#     axs[0].set_title('Flow Distributions')
#     axs[0].legend(['Historic', 'Synthetic'], loc='upper right')
    
#     ## Plot 2: Bar plots of means
#     axs[1].bar(xs - 0.2, Qh_means, width=0.4, color='lightblue', 
#                edgecolor='blue', label='Historic')
#     axs[1].bar(xs + 0.2, Qs_means, width=0.4, color='lightgreen', 
#                edgecolor='green', label='Synthetic')
#     axs[1].set_ylabel('Mean Flow')
#     axs[1].set_title('Mean Flow by Period')
#     axs[1].legend()

#     ## Plot 3: Bar plots of standard deviations
#     axs[2].bar(xs - 0.2, Qh_stds, width=0.4, color='lightblue', 
#                edgecolor='blue', label='Historic')
#     axs[2].bar(xs + 0.2, Qs_stds, width=0.4, color='lightgreen', 
#                edgecolor='green', label='Synthetic')
#     axs[2].set_ylabel('Std Deviation of Flow')
#     axs[2].set_title('Standard Deviation by Period')
#     axs[2].legend()
    
#     ## Plot 4: Wilcoxon rank-sum test p-values (comparing flow distributions)
#     wilcoxon_p_values = []
#     for i in range(n_periods):
#         if len(Qh_flows[i]) > 0 and len(Qs_flows[i]) > 0:
#             try:
#                 p_val = ranksums(Qh_flows[i], Qs_flows[i]).pvalue
#                 wilcoxon_p_values.append(p_val)
#             except:
#                 wilcoxon_p_values.append(np.nan)
#         else:
#             wilcoxon_p_values.append(np.nan)
    
#     axs[3].bar(xs, wilcoxon_p_values, width=0.6, color='blue', alpha=0.7)
#     axs[3].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
#     axs[3].set_ylim(0, 1)
#     axs[3].set_ylabel('Wilcoxon p-value')
#     axs[3].set_title('Wilcoxon Rank-Sum Test p-values (Flow Distributions)')
#     axs[3].legend()
    
#     ## Plot 5: Levene's test p-values (comparing flow variances)
#     levene_p_values = []
#     for i in range(n_periods):
#         if len(Qh_flows[i]) > 0 and len(Qs_flows[i]) > 0:
#             try:
#                 p_val = levene(Qh_flows[i], Qs_flows[i]).pvalue
#                 levene_p_values.append(p_val)
#             except:
#                 levene_p_values.append(np.nan)
#         else:
#             levene_p_values.append(np.nan)
    
#     axs[4].bar(xs, levene_p_values, width=0.6, color='green', alpha=0.7)
#     axs[4].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
#     axs[4].set_ylim(0, 1)
#     axs[4].set_ylabel("Levene's p-value")
#     axs[4].set_title("Levene's Test p-values (Flow Variances)")
#     axs[4].legend()
    
#     # Set x-ticks and labels
#     axs[4].set_xticks(xs)
#     axs[4].set_xticklabels(period_labels)
    
#     if timescale == 'weekly':
#         axs[4].set_xlabel('Week of Year')
    
#     # Cleanup
#     plt.tight_layout() 

#     if fname is not None:
#         plt.savefig(fname, dpi=300, bbox_inches='tight')
    
#     plt.show()
#     return fig, axs