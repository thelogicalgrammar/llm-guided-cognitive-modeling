#!/bin/bash
# Run the search: vLLM serves the generator model, OpenEvolve evolves and scores programs.
#
# SEARCH_MODEL and SEARCH_OUTPUT are required and have no defaults, because a default for either
# produces a complete, plausible run of a condition you did not ask for:
#   fine-tuned: SEARCH_MODEL=$COGMOD_MERGED_MODEL SEARCH_OUTPUT=$PROJECT/Results/OpenEvolveFT_v2 \
#                 sbatch OpenEvolve/run_OpEvProg6Job.sh
#   base:       SEARCH_MODEL=$COGMOD_BASE_MODEL   SEARCH_OUTPUT=$PROJECT/Results/OpenEvolveBase_v2 \
#                 sbatch OpenEvolve/run_OpEvProg6Job.sh
# ITERATIONS is optional (config.yaml decides otherwise); the run records what it used in
# $SEARCH_OUTPUT/run_config.txt.
#SBATCH --job-name=OpenEvolve
#SBATCH --account=gusr58621
#SBATCH --partition=gpu_h100        # request H100 GPU
#SBATCH --gres=gpu:2                # request 2 GPU's
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # request 32 cpus per task (only one call to openevolve, which utilises multiple process pools)
#SBATCH --time=24:00:00             # maximum wall time; the search checkpoints every 30 iterations
#SBATCH --output=results_%x_%j.log  # log file
#SBATCH --error=errors_%x_%j.txt     # error file
#SBATCH --export=ALL

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source env.sh || { echo "env.sh not found: submit this job from the repository root"; exit 1; }

# mandatory fake API key
export OPENAI_API_KEY="sk-no-key"

# Which model to evolve with and where results go. Both required: see need() in env.sh.
need SEARCH_MODEL SEARCH_OUTPUT || exit 1
export MODEL_DIR="$SEARCH_MODEL"
OUTPUT_DIR="$SEARCH_OUTPUT"
[ -d "$MODEL_DIR" ] || { echo "ERROR: SEARCH_MODEL is not a directory: $MODEL_DIR"; exit 1; }

# Anything that is not the base model is a fine-tuned run, and is only the *current* fine-tune if
# the merge postdates the adapter. Checked before the 20-minute model load, not after.
if [ "$(readlink -f "$MODEL_DIR")" != "$(readlink -f "$COGMOD_BASE_MODEL")" ]; then
  refuse_stale_merge "$MODEL_DIR" "$COGMOD_LORA" || exit 1
fi

ITERATION_ARG=""
[ -n "${ITERATIONS:-}" ] && ITERATION_ARG="--iterations $ITERATIONS"

mkdir -p "$PROJECT/logs" "$OUTPUT_DIR"
export LOG_FILE="$PROJECT/logs/vllm_${SLURM_JOB_ID}.log"
echo "model:   $MODEL_DIR"
echo "output:  $OUTPUT_DIR"
echo "vllm log: $LOG_FILE"

# Record which model produced these results, so the directory name is never the only claim about it
provenance "$OUTPUT_DIR" \
  model="$MODEL_DIR" base="$COGMOD_BASE_MODEL" adapter="$COGMOD_LORA" \
  data="$COGMOD_DATA_PATH" score_split="${COGMOD_SCORE_SPLIT:-Val}" \
  iterations="${ITERATIONS:-from config.yaml}" vllm_log="$LOG_FILE"

# the merged model does not necessarily ship a chat template
CHAT_TEMPLATE_ARG=""
[ -f "$MODEL_DIR/chat_template.jinja" ] && CHAT_TEMPLATE_ARG="--chat-template $MODEL_DIR/chat_template.jinja"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# host a local vllm server
vllm serve $MODEL_DIR \
  --port 11434 \
  $CHAT_TEMPLATE_ARG \
  --quantization fp8 \
  --max-model-len ${MAX_MODEL_LEN:-65536} \
  --tensor-parallel-size 2 \
  > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait until server is ready (loading takes ~20 minutes), and stop if it dies
echo "Waiting for vLLM server to be ready..."
until curl -s localhost:11434/v1/models | grep -q "data"; do
  sleep 15
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "vLLM server died, last lines of $LOG_FILE:"; tail -30 "$LOG_FILE"; exit 1
  fi
done
echo "vLLM server ready!"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run the openevolve search; add --checkpoint <dir> to resume
# --primary-model must match what vllm serves, or every request fails with "model not found"
openevolve-run OpenEvolve/initial_program.py OpenEvolve/evaluator.py --config OpenEvolve/config.yaml \
  --primary-model "$MODEL_DIR" --api-base "http://localhost:11434/v1" $ITERATION_ARG --output "$OUTPUT_DIR"

# close server
kill $SERVER_PID
