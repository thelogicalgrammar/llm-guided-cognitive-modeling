# Settings for working on this project on Snellius:  source env.sh
# (the `cogmod` alias in SERVER_SETUP.txt step 1.0 does this for you)
#
# Project-local on purpose: OMP_NUM_THREADS=1 and HF_HOME would be wrong
# defaults for other work, so these are not set in ~/.bashrc.

# Every value is a default: ${VAR:-...} leaves a variable alone when it is already set, so
#   COGMOD_MERGED_MODEL=... sbatch job.sh
# survives the job sourcing this file. Plain assignment would overwrite the override, silently
# sending the job at the wrong path.
export PROJECT=${PROJECT:-/gpfs/work5/0/prjs2269}   # project space: models, data, results
export ACCOUNT=${ACCOUNT:-gusr58621}                # SLURM account holding the budget
export SCRATCH=${SCRATCH:-/scratch-shared/$USER}    # temporary files only, cleaned after 14 days
export REPO=${REPO:-$HOME/llm-guided-cognitive-modeling}

export MODELS=${MODELS:-$PROJECT/Models}            # Qwen3-Coder-Next and the merged fine-tuned model
export COGMOD_DATA_PATH=${COGMOD_DATA_PATH:-$PROJECT/Data}   # read by OpenEvolve/evaluator.py
export HF_HOME=${HF_HOME:-$PROJECT/hf}              # Hugging Face downloads, kept off the home quota
export TMPDIR=${TMPDIR:-$SCRATCH/tmp}
export COGMOD_RESULTS_PATH=${COGMOD_RESULTS_PATH:-$PROJECT/Results/EvolvedCogModels}  # test-set results

# inputs and output of LLMFineTuning/mergeLoRA.py
export COGMOD_BASE_MODEL=${COGMOD_BASE_MODEL:-$MODELS/Qwen3-Coder-Next}
# The adapter trained on 2026-09-26: 3 epochs, choice-only loss, and all 96 expert lora_B tensors
# non-zero. The published one ($PROJECT/FineTune/checkpoint-40/FineTuneResults/checkpoint-40) had
# untrained experts and was fitted on the whole transcript rather than the choices, so merging it
# reproduces neither the intended model nor a useful comparison.
export COGMOD_LORA=${COGMOD_LORA:-$COGMOD_DATA_PATH/FineTune/Final_LoRA}
export COGMOD_MERGED_MODEL=${COGMOD_MERGED_MODEL:-$MODELS/Qwen3-Coder-Next-Merged}

# where FineTuneQ3LoRA.py writes; the same default the job script uses, so the training run and the
# scripts that read it agree without having to be told twice
export COGMOD_FINETUNE_OUT=${COGMOD_FINETUNE_OUT:-$PROJECT/FineTuneNew}
# so `myquota` reports the project space. It wants the project NAME (prjs2269), not the path:
# given a path it warns and silently reports only home and scratch, which is how 447 GB of model
# directories accumulated in $PROJECT without ever appearing in a quota check.
export MYQUOTA_PROJECTSPACES=${MYQUOTA_PROJECTSPACES:-$(basename "$PROJECT")}

# one thread per process: OpenEvolve runs many evaluations in parallel
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Guards.
#
# Everything above is a default, which is convenient right up until a variable
# that was never set - or was set to an empty string by an expansion that named
# a variable which does not exist - resolves to something plausible instead of
# failing. On 2026-09-26 a search submitted as the fine-tuned condition served
# the base model for an hour: SEARCH_MODEL expanded to empty, and
# ${SEARCH_MODEL:-$MODELS/Qwen3-Coder-Next} treats empty exactly like unset.
# Nothing in the output said so; the results directory was named FT.
#
# The rule these helpers enforce: a value that decides *which condition is
# running* - which model, which adapter, where results go - is never defaulted
# and is always recorded. Values that only tune a run (threads, workers,
# iterations) keep their defaults, because getting one wrong is visible.
# ---------------------------------------------------------------------------

# need VAR [VAR ...]  -> 1 if any is unset or empty, with an explanation
need() {
    local name value missing=0
    for name in "$@"; do
        value="${!name-}"
        [ -n "$value" ] && continue
        echo "ERROR: $name is unset or empty, and this script will not guess it." >&2
        missing=1
    done
    [ "$missing" -eq 0 ] && return 0
    echo "  Variables that decide which condition runs have no defaults on purpose: a default" >&2
    echo "  here yields a plausible run of the wrong condition rather than an error. Name it:" >&2
    echo '    SEARCH_MODEL=$COGMOD_MERGED_MODEL SEARCH_OUTPUT=$PROJECT/Results/... sbatch ...' >&2
    echo '  (note $COGMOD_MERGED_MODEL, not $MERGED - the latter is only a label printed below)' >&2
    return 1
}

# newest_epoch PATH -> mtime in seconds of the newest *.safetensors inside PATH,
# falling back to PATH's own mtime, or 0 when it does not exist
newest_epoch() {
    local path="$1" newest=""
    [ -e "$path" ] || { echo 0; return; }
    [ -d "$path" ] && newest=$(find "$path" -maxdepth 1 -name '*.safetensors' -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
    [ -n "$newest" ] || newest=$(stat -c '%Y' "$path" 2>/dev/null || echo 0)
    echo "${newest%%.*}"
}

# refuse_stale_merge MERGED_DIR ADAPTER_DIR
# A merged model is a snapshot of an adapter, but nothing inside it records which one, so an
# adapter retrained after the merge leaves a directory that still answers to the same name and
# silently measures the previous fine-tune. The mtimes are the only available evidence.
refuse_stale_merge() {
    local merged="$1" adapter="$2" merged_at adapter_at
    merged_at=$(newest_epoch "$merged")
    adapter_at=$(newest_epoch "$adapter")
    if [ "$merged_at" -eq 0 ]; then
        echo "ERROR: no weights found in $merged" >&2
        return 1
    fi
    [ "$merged_at" -ge "$adapter_at" ] && return 0
    echo "ERROR: the merged model is older than the adapter it is supposed to contain." >&2
    echo "  merged:  $merged ($(date -d "@$merged_at" '+%F %R'))" >&2
    echo "  adapter: $adapter ($(date -d "@$adapter_at" '+%F %R'))" >&2
    echo "  Searching with it would measure the previous adapter. Re-merge first:" >&2
    echo "    COGMOD_OVERWRITE=1 sbatch LLMFineTuning/run_MergeLoRAjob.sh" >&2
    return 1
}

# provenance OUTPUT_DIR [label=value ...]
# Writes run_config.txt into OUTPUT_DIR. A directory named after a condition is a claim; this is
# the evidence. Arguments that are paths also get the mtime of their newest weights, so a stale
# input is visible in the output it produced rather than only by comparing directories by hand.
provenance() {
    local out_dir="$1"; shift
    local file="$out_dir/run_config.txt" pair label value repo="${SLURM_SUBMIT_DIR:-$PWD}"
    mkdir -p "$out_dir" 2>/dev/null || { echo "provenance: cannot write to $out_dir" >&2; return 1; }
    {
        echo "# What this directory actually holds. Written at launch by provenance() in env.sh."
        printf '%-13s %s\n' "written:" "$(date -Is)"
        printf '%-13s %s\n' "job:" "${SLURM_JOB_NAME:-interactive} ${SLURM_JOB_ID:-}"
        printf '%-13s %s\n' "host:" "$(hostname)"
        printf '%-13s %s\n' "repo_commit:" "$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
        printf '%-13s %s\n' "repo_dirty:" "$(git -C "$repo" status --porcelain 2>/dev/null | wc -l) uncommitted files"
        printf '%-13s %s\n' "python:" "$(command -v python 2>/dev/null)"
        # The environment that WROTE THE STAMP. Inside a job that is the job's environment, which
        # is what you want. Stamping something after the fact from a login shell records that
        # shell instead: a retroactive stamp on the merged model reported peft 0.18.1 (the FT
        # venv) although the merge itself ran in venvs/finetune with peft 0.21 - and 0.18.1 is
        # exactly the version documented as unable to merge this adapter. Named so it cannot be
        # read as a property of the artifact; pass artifact facts explicitly as label=value.
        python -c 'import peft, transformers; print(f"stamped_in:   peft {peft.__version__} | transformers {transformers.__version__}")' 2>/dev/null \
            || printf '%-13s %s\n' "stamped_in:" "peft/transformers not importable where the stamp was written"
        for pair in "$@"; do
            label="${pair%%=*}"; value="${pair#*=}"
            if [ -e "$value" ]; then
                printf '%-13s %s  (weights %s)\n' "$label:" "$value" "$(date -d "@$(newest_epoch "$value")" '+%F %R')"
            else
                printf '%-13s %s\n' "$label:" "$value"
            fi
        done
    } > "$file"
    echo "provenance: wrote $file"
}

# Print what the paths actually resolved to. Because every value above is a default, a variable
# exported earlier in the shell wins, which is what makes overrides work for jobs but also means a
# stale value from a previous session is used silently. Seeing them is the cheap way to notice.
# The labels are the full variable names: an abbreviated label reads like a variable that can be
# used on an sbatch line, and $MERGED (which does not exist) expanding to empty is what sent a
# base-model search into a fine-tuned results directory.
echo "cogmod: COGMOD_LORA=$COGMOD_LORA"
echo "        COGMOD_MERGED_MODEL=$COGMOD_MERGED_MODEL"
echo "        COGMOD_FINETUNE_OUT=$COGMOD_FINETUNE_OUT"
echo "        COGMOD_DATA_PATH=$COGMOD_DATA_PATH"

module load 2025 Python/3.13.1-GCCcore-14.2.0 NVHPC/25.3-CUDA-12.8.0
source $REPO/FT/bin/activate
