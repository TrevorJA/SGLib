import pandas as pd
import numpy as np

from .directories import example_data_dir

# Load example data
def load_example_data():
    """
    Loads example data from the package.

    Returns:
        pd.DataFrame: Example data.
    """
    data = pd.read_csv(f'{example_data_dir}/usgs_daily_streamflow_cms.csv', 
                       index_col=0, parse_dates=True)
    return data