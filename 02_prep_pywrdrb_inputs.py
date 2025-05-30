import os
from pprint import pprint

import pywrdrb
from piep import PredictedInflowEnsemblePreprocessor


ensemble_folder = "./pywrdrb/inputs/stationary_ensemble/" 
ensemble_fname = f"{ensemble_folder}catchment_inflow_mgd.hdf5"

# Setup pathnavigator
pn_config = pywrdrb.get_pn_config()
pn_config["flows/stationary_ensemble"] = os.path.abspath(ensemble_folder)
pywrdrb.load_pn_config(pn_config)

# Initialize the preprocessor
preprocessor = PredictedInflowEnsemblePreprocessor(
    flow_type="stationary_ensemble",
    ensemble_hdf5_file=ensemble_fname,
    realization_ids=[str(i) for i in range(5)],  
    start_date=None,  # Use full range
    end_date=None,
    modes=('regression_disagg',),
    use_log=True,
    remove_zeros=False,
    use_const=False,
    use_mpi=False
)

preprocessor.load()
preprocessor.process()
preprocessor.save()