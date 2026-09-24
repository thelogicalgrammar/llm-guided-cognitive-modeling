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

# The LoRA needs peft 0.18.1 and transformers 4.57 (the version the base model was saved
# with): newer transformers store the MoE experts differently, and peft then silently skips
# the expert adapters. $PROJECT/venvs/merge is used when it exists, otherwise FT/ from env.sh.
# Create it with:
#   python -m venv $PROJECT/venvs/merge && source $PROJECT/venvs/merge/bin/activate
#   pip install torch accelerate "peft==0.18.1" "transformers==4.57.*"
MERGE_VENV="${MERGE_VENV:-$PROJECT/venvs/merge}"
[ -d "$MERGE_VENV" ] && { echo "using $MERGE_VENV"; source "$MERGE_VENV/bin/activate" || exit 1; }
python -c "import peft, transformers; print('peft', peft.__version__, '| transformers', transformers.__version__)"

# run python file
python LLMFineTuning/mergeLoRA.py
