import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sglib.utils.load import HDF5Manager
from sglib.utils.load import load_drb_reconstruction
from sglib.plotting.plot import plot_fdc_ranges
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

## Synthetic ensemble data
hdf_manager = HDF5Manager()
Qs_ensemble = hdf_manager.load_ensemble(gage_flow_ensemble_fname)
Qs_inflow_ensemble = hdf_manager.load_ensemble(catchment_inflow_ensemble_fname)

Q_syn = Qs_ensemble.data_by_site
Qs_inflow_syn = Qs_inflow_ensemble.data_by_site
realization_ids = Qs_ensemble.realization_ids
n_realizations = len(realization_ids)


### Plot gridded FDCs
# Settings
sites = list(Q.columns)
ncols = 5
nrows = int(np.ceil(len(sites) / ncols))

# Plot
plot_combinations = [
    [Q, Q_syn, FIG_DIR + '/fdc_gage_flow_grid.png'], 
    [Q_inflows, Qs_inflow_syn, FIG_DIR + '/fdc_catchment_inflows_grid.png']
 ]


# Loop through gage flow and inflows
for combination in plot_combinations:
    Qh, Qs, fname = combination
    
    # Create the plot
    fig, axs = plt.subplots(figsize=(ncols*3, nrows*3), 
                        nrows=nrows, ncols=ncols, 
                        sharex=True, sharey=True)

    for i, site in enumerate(sites):
        
        # use plot_fdc_ranges to plot the FDC
        ax = axs[i // ncols, i % ncols]
        
        plot_fdc_ranges(Qh.loc[:, site],
                        Qs[site].replace(0, np.nan),
                        legend=False,
                        ax=ax,
                        title=site,
                        units='MGD',
                        xylabels=False)

        if i % ncols == 0:
            ax.set_ylabel('Flow (MGD)')
        if i // ncols == nrows - 1:
            ax.set_xlabel('Nonexceedance')    

    plt.savefig(fname, dpi=200)
    plt.show()