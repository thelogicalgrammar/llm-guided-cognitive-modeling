# Settings for working on this project on Snellius:  source env.sh
# (the `cogmod` alias in SERVER_SETUP.txt step 1.0 does this for you)
#
# Project-local on purpose: OMP_NUM_THREADS=1 and HF_HOME would be wrong
# defaults for other work, so these are not set in ~/.bashrc.

export PROJECT=/gpfs/work1/0/prjs2269          # project space: models, data, results
export ACCOUNT=gusr58621                       # SLURM account holding the budget
export SCRATCH=/scratch-shared/$USER           # temporary files only, cleaned after 14 days
export REPO=$HOME/llm-guided-cognitive-modeling

export MODELS=$PROJECT/Models                  # Qwen3-Coder-Next and the merged fine-tuned model
export COGMOD_DATA_PATH=$PROJECT/Data          # read by OpenEvolve/evaluator.py
export HF_HOME=$PROJECT/hf                     # Hugging Face downloads, kept off the home quota
export TMPDIR=$SCRATCH/tmp
export MYQUOTA_PROJECTSPACES=$PROJECT          # so `myquota` reports the project space

# one thread per process: OpenEvolve runs many evaluations in parallel
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

module load 2025 Python/3.13.1-GCCcore-14.2.0 NVHPC/25.3-CUDA-12.8.0
source $REPO/FT/bin/activate
