# Settings for working on this project on Snellius:  source env.sh
# (the `cogmod` alias in SERVER_SETUP.txt step 1.0 does this for you)
#
# Project-local on purpose: OMP_NUM_THREADS=1 and HF_HOME would be wrong
# defaults for other work, so these are not set in ~/.bashrc.

export PROJECT=/gpfs/work5/0/prjs2269          # project space: models, data, results
export ACCOUNT=gusr58621                       # SLURM account holding the budget
export SCRATCH=/scratch-shared/$USER           # temporary files only, cleaned after 14 days
export REPO=$HOME/llm-guided-cognitive-modeling

export MODELS=$PROJECT/Models                  # Qwen3-Coder-Next and the merged fine-tuned model
export COGMOD_DATA_PATH=$PROJECT/Data          # read by OpenEvolve/evaluator.py
export HF_HOME=$PROJECT/hf                     # Hugging Face downloads, kept off the home quota
export TMPDIR=$SCRATCH/tmp
export COGMOD_RESULTS_PATH=$PROJECT/Results/EvolvedCogModels   # test-set results of fitted programs

# inputs and output of LLMFineTuning/mergeLoRA.py
export COGMOD_BASE_MODEL=$MODELS/Qwen3-Coder-Next
# The adapter trained on 2026-09-26: 3 epochs, choice-only loss, and all 96 expert lora_B tensors
# non-zero. The published one ($PROJECT/FineTune/checkpoint-40/FineTuneResults/checkpoint-40) had
# untrained experts and was fitted on the whole transcript rather than the choices, so merging it
# reproduces neither the intended model nor a useful comparison.
export COGMOD_LORA=$COGMOD_DATA_PATH/FineTune/Final_LoRA
export COGMOD_MERGED_MODEL=$MODELS/Qwen3-Coder-Next-Merged

# where FineTuneQ3LoRA.py writes; the same default the job script uses, so the training run and the
# scripts that read it agree without having to be told twice
export COGMOD_FINETUNE_OUT=$PROJECT/FineTuneNew
export MYQUOTA_PROJECTSPACES=$PROJECT          # so `myquota` reports the project space

# one thread per process: OpenEvolve runs many evaluations in parallel
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

module load 2025 Python/3.13.1-GCCcore-14.2.0 NVHPC/25.3-CUDA-12.8.0
source $REPO/FT/bin/activate
