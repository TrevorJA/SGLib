import os
import glob
import math
import numpy as np
import pandas as pd
from mpi4py import MPI

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

# mpi
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# parallel batch size
batch_size = 2


if __name__ == "__main__":

    ## Clear old batched output files if they exist
    if rank == 0:
        batched_filenames = glob.glob(
            f"{OUTPUT_DIR}{inflow_type}_rank*_batch*.hdf5"
        )
        batched_modelnames = glob.glob(
            f"{OUTPUT_DIR}../models/{inflow_type}_rank*_batch*.hdf5"
        )


        ### Combine batched output files
        if len(batched_filenames) > 0:
            for file in batched_filenames:
                os.remove(file)
        if len(batched_modelnames) > 0:
            for file in batched_modelnames:
                os.remove(file)
                
    ### Setup simulation batches 
    # Get the IDs for the realizations
    if rank == 0:
        realization_ids = get_hdf5_realization_numbers(ensemble_fname)
        
    else:
        realization_ids = None
    realization_ids = comm.bcast(realization_ids, root=0)

    # Split the realizations into batches
    rank_realization_ids = np.array_split(realization_ids, size)[rank]
    n_rank_realizations = len(rank_realization_ids)

    # Split the realizations into batches
    n_batches = math.ceil(n_rank_realizations / batch_size)
    batched_indices = {
        i: rank_realization_ids[
            (i * batch_size) : min((i * batch_size + batch_size), n_rank_realizations)
        ]
        for i in range(n_batches)
    }
    batched_filenames = []

    # Run individual batches
    for batch, indices in batched_indices.items():
        
        # make the model
        model_options = {
            "inflow_ensemble_indices" : indices,
        }

        mb = pywrdrb.ModelBuilder(
            inflow_type='stationary_ensemble',
            start_date=START_DATE,
            end_date=END_DATE,
            options=model_options,
            )

        model_fname = f"{OUTPUT_DIR}../models/{inflow_type}_rank{rank}_batch{batch}.json"
        mb.make_model()
        mb.write_model(model_fname)
        print("You secussessfully created a model with custom inflow type")

        # load model
        model = pywrdrb.Model.load(model_fname)

        # attached the output recorder
        output_filename = f"{OUTPUT_DIR}{inflow_type}_rank{rank}_batch{batch}.hdf5"
        recorder = pywrdrb.OutputRecorder(
            model = model,
            output_filename=output_filename)

        # run 
        model.run()

    print("Successfully ran the simulation with custom inflow type")