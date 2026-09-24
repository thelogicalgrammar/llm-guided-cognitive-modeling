#!/bin/bash
#SBATCH --job-name=qwen3_merge
#SBATCH --partition=genoa                  # Your specified partition
#SBATCH --nodes=1                         
#SBATCH --exclusive                      # CRITICAL: Gives you the entire node and its full memory
#SBATCH --time=03:00:00
#SBATCH --output=result/merge_output_%j.log
#SBATCH --error=error/%x_%j.txt


# load environment
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source env.sh || { echo "env.sh not found: submit this job from the repository root"; exit 1; }

# run python file
python LLMFineTuning/mergeLoRA.py
