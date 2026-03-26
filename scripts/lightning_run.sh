#!/bin/bash
#SBATCH -J joint_train                 # Job name
#SBATCH -o slurmlogs/joint_train.%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/joint_train.%j          # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p h100                        # Queue (partition) name
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --ntasks-per-node=4                            # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
#SBATCH -A ASC25005                            # Allocation name

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

# moved conda using conda-pack : https://conda.github.io/conda-pack/
# source $WORK/miniconda3/bin/activate flow
# source $SCRATCH/conda/flow/bin/activate
source $SCRATCH/miniconda3/bin/activate stamp
module load gcc/13 cuda/12.8 
export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export HF_HOME="$SCRATCH/.cache"
export SLURM_NTASKS_PER_NODE=4
export SLURM_NTASKS=4
export TMPDIR="$SCRATCH/tmp"
export TMP=$TMPDIR
export TEMP=$TMPDIR
export TRITON_CACHE_DIR=$TMPDIR


# export CUDA_VISIBLE_DEVICES=0

TIMESTEP=$(date +"%m.%d.%H%M%S")

# EXPT=base
EXPT=metamath_instruct

srun --kill-on-bad-exit=1 python -u -m main timestep=$TIMESTEP +machine=stampede +setting=kl_random +expt=$EXPT # val_check_interval_global_step=1 training.num_gen_sample=1
