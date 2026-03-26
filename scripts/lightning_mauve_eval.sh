#!/bin/bash
#SBATCH -J joint_eval                 # Job name
#SBATCH -o slurmlogs/joint_eval.%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/joint_eval.%j          # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p h100                        # Queue (partition) name
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --ntasks-per-node=4                            # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
#SBATCH -A ASC25005                            # Allocation name

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
export HF_ALLOW_CODE_EVAL=1
export HF_HOME="$SCRATCH/.cache"

MODE='ppl_eval'

TIMESTEP="09.06.082429" # 3 spec

CKPT_PATH= '<path_to_trained_model>' # "$SCRATCH/diffusion/eagle/runs/kl_randomsampled_True_one_entropy_history_100K_uncond_128_$TIMESTEP/checkpoints/last.ckpt"
NUM_SAMPLES=100

MAUVE_FILE='<path_to_generations_with_one_token_at_a_time>' # "$SCRATCH/diffusion/eagle/lm_eval/09.06.082429/gen_ppl_eval/0908.134543/sts_temp1_128toks.json"

SEED=0
SPEC_LEN=2
MARGINAL_SAMPLING=false
srun --kill-on-bad-exit=1 python -u -m main +machine=stampede +setting=kl_random drafting_params.speculation_len=$SPEC_LEN lm_eval.mauve_ref_path=$MAUVE_FILE sampling.decoding_strategy=fixed sampling.confidence_thres=0.0 sampling.temperature=1.0 timestep=$TIMESTEP seed=$SEED  mode=$MODE lm_eval.ckpt_path=$CKPT_PATH sampling.use_marginals=$MARGINAL_SAMPLING  +uncond_generation.num_samples=$NUM_SAMPLES
