#!/bin/bash
#SBATCH --account=pmr82_0001
#SBATCH --partition=normal
#SBATCH --job-name=synhydro_diag
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err
#
# Run all 14 generator diagnostics in parallel on Hopper via MPI.
# One MPI rank per generator, one core per rank.
#
# Submit:  sbatch run_all.sh
# Status:  squeue -u $USER
# Logs:    tail -f logs/diag_<jobid>.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs

module purge
module load python/3.11.5
module load gnu9/9.3.0
module load openmpi4/4.0.5

source ../../venv/bin/activate

mpirun -np "$SLURM_NTASKS" python -u run_all_mpi.py --n_realizations 3 --n_years 30
