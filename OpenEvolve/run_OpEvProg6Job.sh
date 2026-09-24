#!/bin/bash
# Run the search: vLLM serves the generator model, OpenEvolve evolves and scores programs.
# Paths come from env.sh; override the model, output or length with environment variables:
#   SEARCH_MODEL=$MODELS/Qwen3-Coder-Next-Merged ITERATIONS=600 sbatch OpenEvolve/run_OpEvProg6Job.sh
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

# model to evolve with (base by default), where results go, and how many iterations
export MODEL_DIR="${SEARCH_MODEL:-$MODELS/Qwen3-Coder-Next}"
OUTPUT_DIR="${SEARCH_OUTPUT:-$PROJECT/Results/OpenEvolve}"
ITERATION_ARG=""
[ -n "${ITERATIONS:-}" ] && ITERATION_ARG="--iterations $ITERATIONS"

mkdir -p "$PROJECT/logs" "$OUTPUT_DIR"
export LOG_FILE="$PROJECT/logs/vllm_${SLURM_JOB_ID}.log"
echo "model:   $MODEL_DIR"
echo "output:  $OUTPUT_DIR"
echo "vllm log: $LOG_FILE"

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
  --max-model-len 32768 \
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
