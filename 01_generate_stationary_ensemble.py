

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../')

from sglib.utils.load import load_drb_reconstruction
from sglib.methods.nonparametric.kirsch_nowak import KirschNowakGenerator
from sglib.utils.load import HDF5Manager
    
if __name__ == "__main__":
    ### Loading data
    Q = load_drb_reconstruction()
    Q.replace(0, np.nan, inplace=True)
    Q.dropna(axis=1, how='any', inplace=True)

    Q_inflows = load_drb_reconstruction(gage_flow=False)
    print(f"Loaded reconstruction data with {Q.shape[0]// 365} years of daily data for {Q.shape[1]} sites.")

    Q_monthly = Q.resample('MS').sum()

    ### Generation

    # Initialize the generator
    kn_gen = KirschNowakGenerator(Q, debug=False)

    # Preprocess the data
    kn_gen.preprocessing()

    # Fit the model
    print("Fitting the model...")
    kn_gen.fit()

    # Generate 10 years
    print("Generating synthetic ensemble...")
    n_years = 50
    n_realizations = 10
    syn_ensemble = kn_gen.generate(n_realizations=n_realizations,
                                    n_years=n_years, 
                                    as_array=False)


    ### Postprocessing
    ### Create marginal catchment inflows by subtracting upstream inflows
    from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows

    inflow_ensemble = {}
    for real in syn_ensemble:
        syn_ensemble[real]['delTrenton'] = 0.0
        
        flows_i = syn_ensemble[real].copy()
        
        # change the datetime index to be 2000-01-01 to 2010-01-01
        flows_i.index = pd.date_range(start='1970-01-01', 
                                      periods=len(flows_i), freq='D')
        
        inflow_ensemble[real] = _subtract_upstream_catchment_inflows(flows_i)
    
    # rearrange so that the node name is the dict key and realizations 
    # are the columns of the pd.DataFrame    
    Q_syn = {}
    Qs_inflows = {}
    syn_datetime = inflow_ensemble[0].index
    for site in inflow_ensemble[0].columns:
        Q_syn[site] = np.zeros((len(syn_datetime), n_realizations),
                    dtype=float)
        Qs_inflows[site] = np.zeros((len(syn_datetime), n_realizations),
                        dtype=float)
        
        
        for i in range(n_realizations):
            Q_syn[site][:, i] = syn_ensemble[i][site].values 
            Qs_inflows[site][:, i] = inflow_ensemble[i][site].values
                
        # Convert to DataFrame
        Q_syn[site] = pd.DataFrame(Q_syn[site], 
                                index=syn_datetime, 
                                columns=[str(i) for i in range(n_realizations)])
        Qs_inflows[site] = pd.DataFrame(Qs_inflows[site],
                                        index=syn_datetime, 
                                        columns=[str(i) for i in range(n_realizations)])

    ### Save
    hdf_manager = HDF5Manager()
    
    fname = "./pywrdrb/inputs/stationary_ensemble/gage_flow_mgd.hdf5"
    hdf_manager.export_ensemble_to_hdf5(Q_syn, fname)
    
    fname = "./pywrdrb/inputs/stationary_ensemble/catchment_inflow_mgd.hdf5"
    hdf_manager.export_ensemble_to_hdf5(Qs_inflows, fname)
