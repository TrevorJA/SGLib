

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pywrdrb.utils.hdf5 import extract_realization_from_hdf5
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers

from sglib.utils.load import load_drb_reconstruction
from sglib.plotting.plot import plot_autocorrelation, plot_fdc_ranges, plot_flow_ranges
from sglib.plotting.plot import plot_correlation
from sglib.plotting.drought import drought_metric_scatter_plot
from sglib.droughts.ssi import SSIDroughtMetrics
from config import gage_flow_ensemble_fname, catchment_inflow_ensemble_fname
from config import FIG_DIR

### Loading data
## Historic reconstruction data
# Total flow
Q = load_drb_reconstruction()
Q.replace(0, np.nan, inplace=True)
Q.drop(columns=['delTrenton'], inplace=True)  # Remove Trenton gage as it is not used in the ensemble

# Catchment inflows
Q_inflows = load_drb_reconstruction(gage_flow=False)
Q_inflows.replace(0, np.nan, inplace=True)
Q_inflows.drop(columns=['delTrenton'], inplace=True)  # Remove Trenton gage as it is not used in the ensemble

print(f"Loaded reconstruction data with {Q.shape[0]// 365} years of daily data for {Q.shape[1]} sites.")

Q_monthly = Q.resample('MS').sum()


## Synthetic ensemble
realization_ids = get_hdf5_realization_numbers(gage_flow_ensemble_fname)
n_realizations = len(realization_ids)

syn_ensemble = {}
inflow_ensemble = {}
for i in realization_ids:
    # Extract realization data
    df = extract_realization_from_hdf5(
        gage_flow_ensemble_fname, i, 
        stored_by_node=True)
    df.drop(columns=['datetime', 'delTrenton'], inplace=True)
    syn_ensemble[i] = df
    
    df = extract_realization_from_hdf5(
        catchment_inflow_ensemble_fname, i,
        stored_by_node=True)
    df.drop(columns=['datetime', 'delTrenton'], inplace=True)
    inflow_ensemble[i] = df
    
# Make a second copy of each with sites as keys
Q_syn = {}
Qs_inflows = {}
syn_datetime = syn_ensemble[realization_ids[0]].index
for site in syn_ensemble[realization_ids[0]].columns:
    if site == 'datetime':
        continue
    
    Q_syn[site] = np.zeros((len(syn_datetime), n_realizations),
                 dtype=float)
    Qs_inflows[site] = np.zeros((len(syn_datetime), n_realizations),
                    dtype=float)
    
    
    for i in realization_ids:
        # print(f"Processing realization {i} for site {site}")
        # print(f"Columns in syn_ensemble[{i}]: {syn_ensemble[i].columns}")
        
        Q_syn[site][:, int(i)] = syn_ensemble[i][site].values 
        Qs_inflows[site][:, int(i)] = inflow_ensemble[i][site].values
            
    # Convert to DataFrame
    Q_syn[site] = pd.DataFrame(Q_syn[site], 
                               index=syn_datetime, 
                               columns=[i for i in range(n_realizations)])
    Qs_inflows[site] = pd.DataFrame(Qs_inflows[site],
                                    index=syn_datetime, 
                                    columns=[i for i in range(n_realizations)])



### SSI Drought Metrics
SSI = SSIDroughtMetrics(timescale='M', window=12)
ssi_obs = SSI.calculate_ssi(data=Q_monthly.loc[:,'delMontague'])
obs_droughts = SSI.calculate_drought_metrics(ssi_obs)



SSI = SSIDroughtMetrics(timescale='M', window=12)
syn_ssi = pd.DataFrame(index=syn_ensemble[realization_ids[0]].index,
                       columns=np.arange(0, n_realizations))
for i in realization_ids:
    syn_ssi.loc[:,int(i)] = SSI.calculate_ssi(data=syn_ensemble[i].loc[:, 'delMontague'])
    if i == realization_ids[0]:
        syn_droughts = SSI.calculate_drought_metrics(syn_ssi.loc[:,int(i)])
    else:
        syn_droughts = pd.concat([syn_droughts, SSI.calculate_drought_metrics(syn_ssi.loc[:,int(i)])], axis=0)


drought_metric_scatter_plot(obs_droughts, syn_drought_metrics=syn_droughts, 
                            x_char='severity', y_char='magnitude', color_char='duration')



# print(f"Historic flow columns: {Q.columns.tolist()}")
# print(f"Ensemble flow columns: {syn_ensemble[realization_ids[0]].columns.tolist()}")


# ### Plotting
# plot_correlation(Q, syn_ensemble[realization_ids[0]],
#                  savefig=True,
#                  fname=f"{FIG_DIR}gage_correlation_syn.png")

# plot_correlation(Q_inflows, inflow_ensemble[realization_ids[0]].loc[:, Q_inflows.columns],
#                     savefig=True,
#                     fname=f"{FIG_DIR}inflow_correlation_syn.png")

# for plot_site in ['delLordville', 'delMontague', 'delDRCanal']:

#     plot_autocorrelation(Q.loc[:, plot_site], 
#                         Q_syn[plot_site], 
#                         lag_range=np.arange(1,60, 5), timestep='daily',
#                         savefig=True,
#                         fname=f"{FIG_DIR}gage_autocorr_{plot_site}_syn.png",)

#     plot_flow_ranges(Q.loc[:,plot_site], 
#                     Q_syn[plot_site], 
#                     timestep='daily',
#                     savefig=True,
#                     fname=f"{FIG_DIR}gage_flow_ranges_{plot_site}_syn.png",)

#     plot_fdc_ranges(Q_inflows.loc[:,plot_site], 
#                     Qs_inflows[plot_site],
#                     savefig=True,
#                     fname=f"{FIG_DIR}inflow_fdc_{plot_site}.png",)

#     plot_fdc_ranges(Q.loc[:,plot_site],
#                     Q_syn[plot_site],
#                     savefig=True,
#                     fname=f"{FIG_DIR}gage_flow_fdc_{plot_site}_syn.png",)