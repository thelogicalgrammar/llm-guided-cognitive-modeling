#!/bin/bash
#SBATCH --job-name=LLMInference
#SBATCH --partition=gpu_h100        # request H100 partition
#SBATCH --gres=gpu:3                # request 3 GPUs
#SBATCH --time=3:00:00              # runtime
#SBATCH --output=results/%x_%j.log  # log file
#SBATCH --error=error/%x_%j.txt     # error file


# load environment
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source env.sh || { echo "env.sh not found: submit this job from the repository root"; exit 1; }

export HF_ENABLE_PARALLEL_LOADING=1

# run the inference script, use 1 for fine-tuned model (merged), 0 for base model
python  LLMInference/LLMPredict.py "1"