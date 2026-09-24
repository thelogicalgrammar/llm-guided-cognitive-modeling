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

# the LoRA needs peft 0.18.1, which can conflict with the versions vllm wants in FT/;
# MERGE_VENV=<path> runs the merge in its own environment instead
[ -n "${MERGE_VENV:-}" ] && { source "$MERGE_VENV/bin/activate" || exit 1; }
python -c "import peft, transformers; print('peft', peft.__version__, '| transformers', transformers.__version__)"

# run python file
python LLMFineTuning/mergeLoRA.py
