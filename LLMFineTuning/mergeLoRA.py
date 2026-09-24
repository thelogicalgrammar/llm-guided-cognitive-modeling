import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os


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


# ---------- LOAD TOKENIZER ----------
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True,
)

# ---------- LOAD BASE MODEL ----------
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu",   # safer for merge
    local_files_only=True,
)

# ---------- LOAD LORA ----------
model = PeftModel.from_pretrained(
    base_model,
    lora_adapter_path,
    torch_dtype=torch.bfloat16,
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