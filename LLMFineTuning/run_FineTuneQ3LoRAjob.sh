#!/bin/bash
#SBATCH --job-name=FineTuneLoRAQ3
#SBATCH --partition=gpu_h100        # request H100 partition
#SBATCH --gres=gpu:3                # request 3 GPUs
#SBATCH --time=3:00:00              # runtime
#SBATCH --output=results/%x_%j.log  # log file
#SBATCH --error=error/%x_%j.txt     # error file


# load environemtn
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source env.sh || { echo "env.sh not found: submit this job from the repository root"; exit 1; }

export HF_ENABLE_PARALLEL_LOADING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# run python file
python LLMFineTuning/FineTuneQ3LoRA.py