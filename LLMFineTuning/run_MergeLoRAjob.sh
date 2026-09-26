#!/bin/bash
#SBATCH --job-name=qwen3_merge
#SBATCH --account=gusr58621
#SBATCH --partition=genoa                  # Your specified partition
#SBATCH --nodes=1                         
#SBATCH --exclusive                      # CRITICAL: Gives you the entire node and its full memory
#SBATCH --time=03:00:00
#SBATCH --output=result/merge_output_%j.log
#SBATCH --error=error/%x_%j.txt


# load environment
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source env.sh || { echo "env.sh not found: submit this job from the repository root"; exit 1; }

# Merge in the environment the adapter was trained in. An adapter that targets the fused MoE
# parameters through target_parameters is only merged correctly by the peft that wrote it: 0.18.1
# uses a different layout and does not merge those parameters at all, so merging a 0.21 adapter with
# it silently drops or scrambles every expert delta. venvs/finetune has peft 0.21 and the
# transformers that loaded this base model during training.
#   $PROJECT/venvs/merge (peft 0.18.1) is only right for the originally published adapter.
MERGE_VENV="${MERGE_VENV:-$PROJECT/venvs/finetune}"
[ -d "$MERGE_VENV" ] && { echo "using $MERGE_VENV"; source "$MERGE_VENV/bin/activate" || exit 1; }
python -c "import peft, transformers; print('peft', peft.__version__, '| transformers', transformers.__version__)"

mkdir -p result error

# run python file
python LLMFineTuning/mergeLoRA.py
