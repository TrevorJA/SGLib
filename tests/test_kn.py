from sglib.methods.non_parametric.kirsch_nowak import KirschGenerator
from sglib.utils.load import HDF5
from sglib.droughts.ssi import SSIDroughtMetrics

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Load data
    fname = "./data/gage_flow_obs_pub_nhmv10_BC_ObsScaled_median.csv"
    Q = pd.read_csv(fname)
    Q.drop(columns=['datetime'], inplace=True)  # Drop the first column if it's an index
    
    datetime = pd.date_range(start='1945-01-01', 
                             periods=Q.shape[0], 
                             freq='D')
    
    Q.index = datetime
    Q = Q.replace(0, np.nan)  # Replace zeros with NaN
    Q = Q.dropna(axis=1, how='any')
    
    # Q = Q.iloc[:, 1:10]
    
    print(f"Historic data shape: {Q.shape}") #(28854, 2)
    
    # Initialize the generator
    kn_gen = KirschGenerator(Q)

    # Preprocess the data
    kn_gen.preprocessing()

    # Fit the model
    kn_gen.fit()

    # Generate synthetic series for 10 years
    synthetic_series = kn_gen.generate_single_series(n_years=10)
    print(synthetic_series.shape) # (10, 24)

    # Convert to series
    synthetic_series = pd.DataFrame(synthetic_series.T, 
                                     columns=Q.columns.tolist(), 
                                     index=pd.date_range(start='1945-01-01', 
                                                           periods=synthetic_series.shape[1], 
                                                           freq='M'))
    
    SSI = SSIDroughtMetrics(timescale='M', window=12)
    ssi = SSI.calculate_ssi(data=synthetic_series.iloc[:, 0])
    print(f'SSI: {ssi}')
    droughts = SSI.calculate_drought_metrics(ssi)
    print(f'Droughts: {droughts}')
    
    # ensemble = kn_gen.generate(n_realizations=20, n_years=10)
    # print(ensemble.shape)
    
    # hdf = HDF5()
    # syn_datetime = pd.date_range(start='1945-01-01', 
    #                          periods=ensemble.shape[1], 
    #                          freq='D')
    
    # hdf.save_to_hdf5(data = ensemble, 
    #                  file_path="./outputs/kn_ensemble.h5", 
    #                  site_names=Q.columns.tolist(), 
    #                  datetime_index=syn_datetime)
    