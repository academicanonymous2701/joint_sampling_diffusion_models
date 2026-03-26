#!/bin/bash
#SBATCH -J lightning_gen                 # Job name
#SBATCH -o slurmlogs/lightning_gen.%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/lightning_gen.%j          # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p h100                        # Queue (partition) name
#SBATCH -N 2                          # Total number of nodes requested
#SBATCH --ntasks-per-node=4                            # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
#SBATCH -A ASC25005                            # Allocation name
#SBATCH --array=0   # 20 jobs

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

# source $WORK/miniconda3/bin/activate flow
# source $SCRATCH/conda/flow/bin/activate
source $SCRATCH/miniconda3/bin/activate stamp
module load gcc/13 cuda/12.8 
export HYDRA_FULL_ERROR=1
export TRITON_CACHE_DIR="/tmp"
export PYTHONFAULTHANDLER=1
export SLURM_NTASKS_PER_NODE=4 
export SLURM_NTASKS=4 
export HF_HOME="$SCRATCH/.cache"


srun --kill-on-bad-exit=1 python -u -m main +machine=stampede +data_generations=uncond128 seed=100 
