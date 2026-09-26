import json
import os
import shutil
from pathlib import Path

import peft
import torch
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


optimized_threads = 96
os.environ["OMP_NUM_THREADS"] = str(optimized_threads)
os.environ["MKL_NUM_THREADS"] = str(optimized_threads)
torch.set_num_threads(optimized_threads)
torch.set_num_interop_threads(2)

# ---------- PATHS ----------
# edit these, or set the environment variables (env.sh sets $MODELS and $PROJECT)
base_model_path = os.environ.get("COGMOD_BASE_MODEL", "/path/to/your/Models/Qwen3-Coder-Next")
lora_adapter_path = os.environ.get("COGMOD_LORA", "/path/to/your/Data/FineTune/checkpoint-40/")
merged_output_path = os.environ.get("COGMOD_MERGED_MODEL", "/path/to/your/Models/Qwen3-Coder-Next-Merged")

# ---------- CHECK THE INPUTS FIRST ----------
# loading the base model takes ~40 minutes, so fail immediately on a wrong path
for description, path, required_file, hint in [
    ("base model", base_model_path, "config.json", ""),
    ("LoRA adapter", lora_adapter_path, "adapter_config.json",
     "\n  note: the LoRA repository nests its checkpoints, e.g. <download>/FineTuneResults/checkpoint-40/"),
]:
    if not os.path.isfile(os.path.join(path, required_file)):
        raise SystemExit(f"No {required_file} in the {description} directory: {path}\n"
                         f"  contents: {sorted(os.listdir(path))[:10] if os.path.isdir(path) else 'directory does not exist'}{hint}")
os.makedirs(os.path.dirname(merged_output_path.rstrip("/")) or ".", exist_ok=True)
print(f"base model:  {base_model_path}\nLoRA:        {lora_adapter_path}\nmerged into: {merged_output_path}", flush=True)
print(f"peft {peft.__version__} | transformers {transformers.__version__}", flush=True)

# An adapter targeting the fused MoE parameters is merged correctly only by peft 0.19.1 or newer:
# 0.18 uses a different layout for them and its merge leaves them untouched, so the merged model
# would differ from the base only in attention and the shared MLPs, exactly the defect this
# retraining was meant to remove. Nothing would report it.
adapter_config = json.load(open(os.path.join(lora_adapter_path, "adapter_config.json")))
if adapter_config.get("target_parameters"):
    print(f"the adapter targets parameters directly: {adapter_config['target_parameters']}")
    if Version(peft.__version__) < Version("0.19.1"):
        raise SystemExit(
            f"peft {peft.__version__} cannot merge an adapter that uses target_parameters: it would\n"
            f"  silently drop every expert delta. Merge in the environment the adapter was trained\n"
            f"  in: MERGE_VENV=$PROJECT/venvs/finetune sbatch LLMFineTuning/run_MergeLoRAjob.sh"
        )

# 150 GB of output: say what is being replaced rather than overwrite it without a word
existing = os.path.isdir(merged_output_path) and os.listdir(merged_output_path)
if existing and not os.environ.get("COGMOD_OVERWRITE"):
    raise SystemExit(
        f"{merged_output_path} already holds {len(existing)} files, and merging would replace the\n"
        f"  model the previous search used. Set COGMOD_OVERWRITE=1 to replace it, or point\n"
        f"  COGMOD_MERGED_MODEL somewhere else to keep both."
    )

# transformers renamed torch_dtype to dtype in v5
dtype_kwarg = {"dtype": torch.bfloat16} if Version(transformers.__version__) >= Version("5.0.0") \
    else {"torch_dtype": torch.bfloat16}


# ---------- LOAD TOKENIZER ----------
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True,
)

# ---------- LOAD BASE MODEL ----------
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    low_cpu_mem_usage=True,
    device_map="cpu",   # safer for merge
    local_files_only=True,
    **dtype_kwarg,
)

def adapter_with_ordered_patterns(source):
    """A copy of the adapter whose rank_pattern is ordered specific-first, or the adapter unchanged

    peft records a rank in the saved config that it did not use. Training this adapter with
    target_modules gate_proj and up_proj made peft fuse them, drop them from target_modules, and add
    `.*\\.gate_up_proj: 32` to rank_pattern (16 + 16) and 64 to alpha_pattern. That key also matches
    `...mlp.experts.gate_up_proj`, and peft.utils.other.get_pattern_key returns the *first* matching
    key, so on reload the expert parameter is rebuilt at rank 32 while the file holds rank 1:

        size mismatch for ...mlp.experts.base_layer.lora_A.default.weight

    During training the config held only the two specific keys, so rank 1 was used and the weights
    are correct. Ordering the patterns longest-first makes the specific keys win, which reproduces
    the training ranks. Verified against peft 0.21.0.
    """
    config_path = Path(source) / "adapter_config.json"
    config = json.load(open(config_path))
    patterns = {attr: config.get(attr) or {} for attr in ("rank_pattern", "alpha_pattern")}
    if not any(patterns.values()):
        return source

    ordered = {attr: dict(sorted(value.items(), key=lambda kv: -len(kv[0])))
               for attr, value in patterns.items()}
    if all(list(ordered[attr]) == list(patterns[attr]) for attr in patterns):
        print("pattern keys are already ordered specific-first")
        return source

    # a directory of links, so the weights are not copied and the original is untouched
    work = Path(os.environ.get("TMPDIR", "/tmp")) / "cogmod_adapter_reordered"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for entry in Path(source).iterdir():
        if entry.name != "adapter_config.json":
            (work / entry.name).symlink_to(entry.resolve())
    for attr, value in ordered.items():
        if config.get(attr) is not None:
            config[attr] = value
    json.dump(config, open(work / "adapter_config.json", "w"), indent=2)
    print(f"reordered pattern keys so the specific ones match first:\n  {ordered['rank_pattern']}")
    return str(work)


# ---------- LOAD LORA ----------
model = PeftModel.from_pretrained(
    base_model,
    adapter_with_ordered_patterns(lora_adapter_path),
    **dtype_kwarg,
)

print("Loaded PEFT adapter.")

# ---------- MERGE LORA INTO BASE MODEL ----------
model = model.merge_and_unload()

print("Merged adapter into base model.")

# ---------- SAVE MERGED MODEL ----------
model.save_pretrained(
    merged_output_path,
    safe_serialization=True,
    max_shard_size="10GB"  # Optimal for Snellius Lustre/Staging filesystems
)

tokenizer.save_pretrained(merged_output_path)

print(f"Merged model saved to: {merged_output_path}")