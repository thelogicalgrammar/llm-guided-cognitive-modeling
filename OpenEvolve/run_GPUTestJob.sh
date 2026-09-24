#!/bin/bash
# Smoke test before spending a pilot: GPUs visible, vLLM serves the model, one request answered,
# and the evaluator fits a program. Costs ~2 GPU-hours of budget at most (40 min on 2 H100s).
#SBATCH --job-name=gputest
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:40:00
#SBATCH --output=result/gputest_%j.log
#SBATCH --error=error/gputest_%j.txt
#SBATCH --export=ALL

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source env.sh                     # paths, account, modules, venv

echo "=== 1. GPUs as seen by SLURM ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

echo "=== 2. torch / jax ==="
python -u -c "
import torch, jax
print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| GPUs', torch.cuda.device_count())
print('devices:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print('jax devices:', jax.devices())   # CPU only by design: fitting runs on CPU
"

echo "=== 3. evaluator fits the initial program (CPU) ==="
python -u -c "
import importlib.util, time
spec = importlib.util.spec_from_file_location('e', 'OpenEvolve/evaluator.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
t0 = time.time(); r = e.evaluate_stage2('OpenEvolve/initial_program.py')
print('metrics:', r.metrics, '| seconds', round(time.time() - t0, 1))
print('expected: combined_score 0.45, nll 0.8318, jax_fit 1.0')
"

echo "=== 4. vLLM serves the model ==="
MODEL_DIR=${TEST_MODEL_DIR:-$MODELS/Qwen3-Coder-Next}
echo "serving $MODEL_DIR"
CHAT_TEMPLATE_ARG=""
[ -f "$MODEL_DIR/chat_template.jinja" ] && CHAT_TEMPLATE_ARG="--chat-template $MODEL_DIR/chat_template.jinja"
export OPENAI_API_KEY="sk-no-key"

vllm serve "$MODEL_DIR" \
  --port 11434 \
  $CHAT_TEMPLATE_ARG \
  --quantization fp8 \
  --max-model-len 32768 \
  --tensor-parallel-size 2 \
  > "$PROJECT/logs/vllm_gputest_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!

echo "waiting for the server (up to 25 minutes) ..."
for i in $(seq 1 100); do
    sleep 15
    if curl -s localhost:11434/v1/models | grep -q "data"; then echo "server ready after $((i*15))s"; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "SERVER DIED - last lines of its log:"; tail -30 "$PROJECT/logs/vllm_gputest_${SLURM_JOB_ID}.log"; exit 1
    fi
done

echo "=== 5. one request ==="
curl -s localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "'"$MODEL_DIR"'",
  "messages": [{"role": "user", "content": "Reply with exactly: GPU OK"}],
  "max_tokens": 20
}' | python -u -c "import json,sys; d=json.load(sys.stdin); print('response:', d['choices'][0]['message']['content']); print('usage:', d.get('usage'))"

echo "=== 6. server throughput counters ==="
curl -s localhost:11434/metrics | grep -E "num_requests_running|num_requests_waiting|gpu_cache_usage" | head -5

kill $SERVER_PID
echo "=== done ==="
