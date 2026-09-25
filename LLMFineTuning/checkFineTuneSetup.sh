#!/bin/bash
# Check the fine-tuning setup without a GPU: the SFTConfig arguments this trl version accepts, the
# tokeniser, every transcript, and the optional dependencies the run needs. Takes about a minute on
# a login node, and saves discovering the same problems after a ten-minute model load on a reserved
# GPU. Run from the repository root:
#
#   bash LLMFineTuning/checkFineTuneSetup.sh
#
set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh || { echo "env.sh not found: run this from the repository root"; exit 1; }

FINETUNE_VENV="${FINETUNE_VENV:-$PROJECT/venvs/finetune}"
if [ -d "$FINETUNE_VENV" ]; then
  echo "using $FINETUNE_VENV"
  source "$FINETUNE_VENV/bin/activate" || exit 1
else
  echo "no venv at $FINETUNE_VENV; using whatever python is active"
fi

python -c "import trl" 2>/dev/null || {
  echo "trl is not importable in this environment. The fine-tuning venv is the one with it:"
  echo "  source $FINETUNE_VENV/bin/activate"
  exit 1
}

COGMOD_CONFIG_CHECK=1 COGMOD_SMOKE_STEPS="${COGMOD_SMOKE_STEPS:-8}" \
  python LLMFineTuning/FineTuneQ3LoRA.py
