#!/bin/bash
#SBATCH --job-name=FineTuneLoRAQ3
#SBATCH --account=gusr58621
#SBATCH --partition=gpu_h100        # request H100 partition
#SBATCH --gres=gpu:3                # request 3 GPUs
#SBATCH --time=3:00:00              # runtime
#SBATCH --output=results/%x_%j.log  # log file
#SBATCH --error=error/%x_%j.txt     # error file
#SBATCH --export=ALL                # COGMOD_SMOKE_STEPS and the env.sh paths reach the job


# load environemtn
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source env.sh || { echo "env.sh not found: submit this job from the repository root"; exit 1; }

# Training the MoE experts needs peft >= 0.19.1, where the target_parameters layout was fixed;
# the adapter published with this project was trained with 0.18.1 and its expert adapters never
# left their zero initialisation. $PROJECT/venvs/finetune is used when it exists.
#   python -m venv $PROJECT/venvs/finetune && source $PROJECT/venvs/finetune/bin/activate
#   pip install torch transformers trl "peft>=0.19.1" accelerate liger-kernel bitsandbytes
FINETUNE_VENV="${FINETUNE_VENV:-$PROJECT/venvs/finetune}"
[ -d "$FINETUNE_VENV" ] && { echo "using $FINETUNE_VENV"; source "$FINETUNE_VENV/bin/activate" || exit 1; }
python -c "import peft, transformers, trl; print('peft', peft.__version__, '| transformers', transformers.__version__, '| trl', trl.__version__)"

mkdir -p "${COGMOD_FINETUNE_OUT:-$PROJECT/FineTuneNew}" results error

# A short run that only checks whether the expert adapters receive gradients:
#   COGMOD_SMOKE_STEPS=8 sbatch LLMFineTuning/run_FineTuneQ3LoRAjob.sh
[ -n "${COGMOD_SMOKE_STEPS:-}" ] && echo "smoke run: ${COGMOD_SMOKE_STEPS} optimizer steps, no baseline eval or checkpoints"

export HF_ENABLE_PARALLEL_LOADING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# run python file
python LLMFineTuning/FineTuneQ3LoRA.py