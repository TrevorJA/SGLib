import pywrdrb
import matplotlib.pyplot as plt
import numpy as np

inflow_type = 'stationary_ensemble'
output_fname = f"./pywrdrb/outputs/{inflow_type}.hdf5"
results_sets = ['major_flow', 'res_storage']

data = pywrdrb.Data(results_sets=results_sets, 
                    print_status=True)
data.load_output(output_filenames=[output_fname])

realization_ids = list(data.major_flow[inflow_type].keys())

nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

# Plot the aggregate NYC reservoir storage 
# for each realization and for each year
fig, ax = plt.subplots(figsize=(10, 6))
for realization_id in realization_ids:
    df_storage = data.res_storage[inflow_type][realization_id]
    
    # calculate nyc total storge
    df_storage['nyc_agg'] = df_storage[nyc_reservoirs].sum(axis=1)

    years = df_storage.index.year.unique()
    
    # For each year, plot the annual storage series
    # make x axis from 1-365
    for y in years:
        df_year = df_storage[df_storage.index.year == y]
        xs = np.arange(1, len(df_year) + 1)
        ax.plot(xs, df_year['nyc_agg'], 
                color='darkorange', 
                alpha=0.5)
        
plt.show()