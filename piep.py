
from pywrdrb.pre import PredictedInflowPreprocessor

import h5py
import numpy as np
import pandas as pd
from pywrdrb.utils.hdf5 import extract_realization_from_hdf5



class PredictedInflowEnsemblePreprocessor(PredictedInflowPreprocessor):
    """
    Generates ensemble predictions for inflows at Montague and Trenton using MPI parallelization.
    
    Processes multiple realization members from an ensemble HDF5 file and saves predictions
    in HDF5 format compatible with PredictionEnsemble parameter.
    """
    
    def __init__(self,
                 flow_type,
                 ensemble_hdf5_file,
                 realization_ids=None,
                 start_date=None,
                 end_date=None,
                 modes=('regression_disagg',),
                 use_log=True,
                 remove_zeros=False,
                 use_const=False,
                 use_mpi=False):
        """
        Initialize the PredictedInflowEnsemblePreprocessor.
        
        Args:
            flow_type: Label for the dataset.
            ensemble_hdf5_file: Path to HDF5 file containing ensemble inflow data.
            realization_ids: List of realization IDs to process. If None, uses all available.
            start_date: Start date for predictions.
            end_date: End date for predictions.
            modes: Prediction modes to use.
            use_log: Whether to use log transformation.
            remove_zeros: Whether to remove zero values.
            use_const: Whether to use constant in regression.
        """
        super().__init__(flow_type, start_date, end_date, modes, use_log, remove_zeros, use_const)
        
        self.ensemble_hdf5_file = ensemble_hdf5_file
        self.realization_ids = realization_ids
        
        self.use_mpi = use_mpi
        if self.use_mpi:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        
        # Update output path for ensemble predictions
        self.output_dirs = {
            "predicted_inflows_mgd.hdf5": self.pn.sc.get(f"flows/{self.flow_type}") / "predicted_inflows_mgd.hdf5",
        }
        
        # Storage for ensemble results
        self.ensemble_predictions = {}

    def load(self):
        """Load available realization IDs and catchment water consumption data."""
        # Load water consumption data (same for all realizations)
        fname = self.input_dirs["sw_avg_wateruse_pywrdrb_catchments_mgd.csv"]
        wc = pd.read_csv(fname)
        wc.index = wc["node"]
        self.catchment_wc = wc
        
        # Get available realization IDs if not specified
        if self.realization_ids is None:
            with h5py.File(self.ensemble_hdf5_file, 'r') as f:
                self.realization_ids = [key for key in f.keys()]
        
        if self.rank == 0:
            print(f"Processing {len(self.realization_ids)} realizations across {self.size} processes")

    def process(self):
        """Process ensemble predictions using MPI parallelization."""
        if not hasattr(self, 'realization_ids'):
            self.load()
        
        # Distribute realizations across MPI processes
        realizations_per_rank = np.array_split(self.realization_ids, self.size)
        my_realizations = realizations_per_rank[self.rank]
        
        local_predictions = {}
        
        # Process assigned realizations
        for realization_id in my_realizations:
            if self.rank == 0:
                print(f"Processing realization {realization_id}")
            
            # Extract realization data
            self.timeseries_data = extract_realization_from_hdf5(
                self.ensemble_hdf5_file, 
                realization_id, 
                stored_by_node=True
            )
            
            # Train regressions and make predictions for this realization
            regressions = self.train_regressions()
            realization_predictions = self.make_predictions(regressions)
            
            local_predictions[str(realization_id)] = realization_predictions
        
        # Gather all predictions to rank 0
        if self.use_mpi:
            all_predictions = self.comm.gather(local_predictions, root=0)
        else:
            all_predictions = [local_predictions]
        
        if self.rank == 0:
            # Combine predictions from all processes
            for predictions_dict in all_predictions:
                self.ensemble_predictions.update(predictions_dict)

    def save(self):
        """Save ensemble predictions to HDF5 format."""
        if self.rank == 0:
            if not self.ensemble_predictions:
                raise ValueError("No ensemble predictions to save. Run process() first.")
            
            fname = self.output_dirs["predicted_inflows_mgd.hdf5"]
            
            with h5py.File(fname, 'w') as hf:
                for realization_id, predictions_df in self.ensemble_predictions.items():
                    # Create group for this realization
                    realization_group = hf.create_group(realization_id)
                    
                    # Store datetime
                    datetime_strings = predictions_df['datetime'].astype(str).values
                    realization_group.create_dataset('datetime', data=datetime_strings)
                    
                    # Store prediction columns
                    for col in predictions_df.columns:
                        if col != 'datetime':
                            realization_group.create_dataset(col, data=predictions_df[col].values)
            
            print(f"Saved ensemble predictions to {fname}")
        
        # Ensure all processes wait for save to complete
        if self.use_mpi:
            self.comm.barrier()