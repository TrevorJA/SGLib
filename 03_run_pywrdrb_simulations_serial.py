import os
import pywrdrb
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers

from config import N_YEARS, N_REALIZATIONS 
from config import START_DATE, END_DATE
from config import OUTPUT_DIR

inflow_type = 'stationary_ensemble'
ensemble_folder = "./pywrdrb/inputs/stationary_ensemble/" 
ensemble_fname = f"{ensemble_folder}catchment_inflow_mgd.hdf5"

model_fname = "./pywrdrb/inputs/stationary_ensemble/model.json"
output_filename = "./pywrdrb/outputs/stationary_ensemble.hdf5"

# Setup pathnavigator
pn_config = pywrdrb.get_pn_config()
pn_config["flows/stationary_ensemble"] = os.path.abspath(ensemble_folder)

pywrdrb.load_pn_config(pn_config)



if __name__ == "__main__":

                
    ### Setup simulation batches 
    # Get the IDs for the realizations
    realization_ids = get_hdf5_realization_numbers(ensemble_fname)

    print(f"Found {len(realization_ids)} realizations in the ensemble file.")

    # make the model
    model_options = {
        "inflow_ensemble_indices" : [int(i) for i in realization_ids],
    }

    mb = pywrdrb.ModelBuilder(
        inflow_type='stationary_ensemble',
        start_date=START_DATE,
        end_date=END_DATE,
        options=model_options,
        )


    mb.make_model()
    mb.write_model(model_fname)
    print("You secussessfully created a model with custom inflow type")

    # load model
    model = pywrdrb.Model.load(model_fname)

    # attached the output recorder
    recorder = pywrdrb.OutputRecorder(
        model = model,
        output_filename=output_filename)

    print('Running simulation...')
    # run 
    model.run()

    print("Successfully ran the simulation with custom inflow type")