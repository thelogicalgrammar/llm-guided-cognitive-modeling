#!/bin/bash
#SBATCH --job-name=OpenEvolve
#SBATCH --partition=gpu_h100        # request H100 GPU
#SBATCH --gres=gpu:2                # request 2 GPU's
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # request 32 cpus per task (only one call to openevolve, which utilises multiple process pools)
#SBATCH --time=3:35:00              # maximum wall time
#SBATCH --output=results_%x_%j.log  # log file
#SBATCH --error=errors_%x_%j.txt     # error file
#SBATCH --export=ALL

# load necessary modules

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source env.sh || { echo "env.sh not found: submit this job from the repository root"; exit 1; }

# mandatory fake API key
export OPENAI_API_KEY="sk-no-key"

export MODEL_DIR="/path/to/your/Models/Qwen3-Coder-Next-Merged"

# create log file for vllm logs
export LOG_FILE="/path/to/your/Logs/vllm_${SLURM_JOB_ID}.log"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# host a local vllm server
vllm serve $MODEL_DIR \
  --port 11434 \
  --chat-template "$MODEL_DIR/chat_template.jinja" \
  --quantization fp8 \
  --max-model-len 131072 \
  --download-dir $MODEL_DIR \
  --tensor-parallel-size 2 \
  > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait until server is ready
echo "Waiting for vLLM server to be ready..."
until curl -s localhost:11434/v1/models | grep -q "Qwen"; do
  sleep 15
done
echo "vLLM server ready!"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run the openevolve test process
openevolve-run OpenEvolve/initial_program.py OpenEvolve/evaluator.py --config OpenEvolve/config.yaml --output /path/to/your/Results/OpenEvolve/

# close server
kill $SERVER_PID