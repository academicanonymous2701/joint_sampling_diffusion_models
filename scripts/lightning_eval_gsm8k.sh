#!/bin/bash
#SBATCH -J joint_eval                 # Job name
#SBATCH -o slurmlogs/joint_eval.%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/joint_eval.%j          # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p 100                        # Queue (partition) name
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --ntasks-per-node=4                            # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
#SBATCH -A ASC25005                            # Allocation name

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

source $SCRATCH/miniconda3/bin/activate stamp
module load gcc/13 cuda/12.8 
export HYDRA_FULL_ERROR=1
export TRITON_CACHE_DIR="/tmp"
export PYTHONFAULTHANDLER=1
export SLURM_NTASKS_PER_NODE=1
export SLURM_NTASKS=1
export HF_ALLOW_CODE_EVAL=1
export HF_HOME="$SCRATCH/.cache"

CKPT_PATH='<path_to_trained_model>' # "$SCRATCH/diffusion/eagle/runs/kl_randomsampled_True_one_entropy_history_20K_lowtemp_metamath_256_$TIMESTEP/checkpoints/last.ckpt"

SEED=0
GPUID=0
MARGINAL_SAMPLING=true
TASK="gsm8k_instruct"
CUDA_VISIBLE_DEVICES=$GPUID python -u -m main +machine=stampede +setting=kl_random +lm_evals=$TASK sampling.decoding_strategy=fixed lm_eval.gpu_id=$GPUID  drafting_params.speculation_len=$SPEC_LEN timestep=$TIMESTEP seed=$SEED lm_eval.ckpt_path=$CKPT_PATH sampling.use_marginals=$MARGINAL_SAMPLING
