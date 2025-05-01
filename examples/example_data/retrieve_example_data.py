"""
Retrieves USGS gauge data using hyriver suite.
"""

import numpy as np
import pandas as pd
import pygeohydro as gh
from pygeohydro import NWIS


# Specify bounded box region of interest
bbox = (-77, 42, -76.0, 43)

# Specify time period of interest
dates = ('1900-01-01', '2022-12-31')

# Make query and search for gauges
nwis = NWIS()
query_request = {"bBox": ",".join(f"{b:.06f}" for b in bbox),
                    "hasDataTypeCd": "dv",
                    "outputDataTypeCd": "dv"}
query_result = nwis.get_info(query_request, expanded= False, nhd_info= False)

# Filter non-streamflow stations
query_result = query_result.query("site_tp_cd in ('ST','ST-TS')")
query_result = query_result[query_result.parm_cd == '00060']  # https://help.waterdata.usgs.gov/parameter_cd?group_cd=PHY
query_result = query_result.reset_index(drop = True)
stations = list(set(query_result.site_no.tolist()))
stations.append('01434000') # a known gauge with 100 years of data

# Retrieve data for the gauges
Q = nwis.get_streamflow(stations, dates)
Q.index = pd.to_datetime(Q.index.date)

# remove dates with no data
Q = Q.dropna(axis=0, how='all')

# remove gauges which have no data prior to 1915
drop_gauges = []
for gauge in Q.columns:
    if Q[gauge].dropna().index[0].year > 1915:
        drop_gauges.append(gauge)
Q = Q.drop(drop_gauges, axis=1)

# Transform to monthly
Q_monthly = Q.resample('M').sum()

# Export
Q.to_csv(f'./usgs_daily_streamflow_cms.csv', sep=',')
Q_monthly.to_csv(f'./usgs_monthly_streamflow_cms.csv', sep=',')

print(f"Gage data gathered, {Q.shape[1]} USGS streamflow gauges found in date range.")

