"""
Check that merging the LoRA actually changed the weights it was supposed to change.

PEFT warns "Unsupported layer type 'Qwen3NextExperts'" on this model, which means the
expert adapters (target_parameters: gate_up_proj, down_proj) can be silently dropped,
leaving a model where only the attention and dense projections are fine-tuned. This
reads a few tensors from each model and reports which ones differ.

    python LLMFineTuning/checkMerge.py

Paths come from COGMOD_BASE_MODEL, COGMOD_MERGED_MODEL and COGMOD_LORA (env.sh sets them).
"""
import json
import os
import sys
from safetensors import safe_open

# torch reads bfloat16 weights; numpy is enough for float32 checkpoints and for testing
try:
    import torch
    FRAMEWORK = "pt"
except ImportError:
    FRAMEWORK = "np"


def weight_index(model_dir):
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_file):
        raise SystemExit(f"No model.safetensors.index.json in {model_dir}")
    return json.load(open(index_file))["weight_map"]

def tensor(model_dir, weight_map, name):
    shard = weight_map.get(name)
    if shard is None:
        return None
    with safe_open(os.path.join(model_dir, shard), framework=FRAMEWORK) as f:
        values = f.get_tensor(name)
    return values.float() if FRAMEWORK == "pt" else values.astype("float32")

def adapter_targets(lora_dir):
    """Which module names and parameter names the adapter claims to change"""
    config = json.load(open(os.path.join(lora_dir, "adapter_config.json")))
    return config.get("target_modules", []), config.get("target_parameters", [])


if __name__ == "__main__":
    base_dir = os.environ.get("COGMOD_BASE_MODEL")
    merged_dir = os.environ.get("COGMOD_MERGED_MODEL")
    lora_dir = os.environ.get("COGMOD_LORA")
    if not base_dir or not merged_dir:
        raise SystemExit("Set COGMOD_BASE_MODEL and COGMOD_MERGED_MODEL (env.sh does)")

    base_map, merged_map = weight_index(base_dir), weight_index(merged_dir)
    print(f"base:   {base_dir}\nmerged: {merged_dir}")
    print(f"tensors: {len(base_map)} in base, {len(merged_map)} in merged\n")

    if lora_dir and os.path.isfile(os.path.join(lora_dir, "adapter_config.json")):
        modules, parameters = adapter_targets(lora_dir)
        print(f"adapter targets modules {sorted(modules)}\n                 parameters {parameters}\n")

    # one tensor per kind: a plain LoRA target, and the MoE expert parameters
    names = [name for name in ["model.layers.0.self_attn.q_proj.weight",
                               "model.layers.0.linear_attn.in_proj_qkvz.weight",
                               "model.layers.0.mlp.experts.gate_up_proj",
                               "model.layers.0.mlp.experts.down_proj"] if name in base_map]
    if not names:
        # fall back to whatever the model calls its layers
        names = [n for n in list(base_map)[:6]]

    changed, unchanged, missing = [], [], []
    for name in names:
        base_tensor, merged_tensor = tensor(base_dir, base_map, name), tensor(merged_dir, merged_map, name)
        if base_tensor is None or merged_tensor is None:
            missing.append(name)
            print(f"  {name}: missing from one of the models")
            continue
        if base_tensor.shape != merged_tensor.shape:
            print(f"  {name}: SHAPES DIFFER {tuple(base_tensor.shape)} vs {tuple(merged_tensor.shape)}")
            changed.append(name)
            continue
        difference = float(abs(base_tensor - merged_tensor).max())
        print(f"  {name}: max abs diff {difference:.6f}   {'changed' if difference > 0 else 'IDENTICAL'}")
        (changed if difference > 0 else unchanged).append(name)

    print()
    experts_unchanged = [n for n in unchanged if "experts" in n]
    if experts_unchanged:
        print("The expert weights are identical to the base model: the adapters for")
        print("target_parameters (gate_up_proj, down_proj) were NOT merged, so the merged")
        print("model is only partly fine-tuned. Try the transformers/peft versions the")
        print("adapter was trained with, in a separate venv (see SERVER_SETUP.txt step 3.2).")
        sys.exit(1)
    if not changed:
        print("Nothing changed: this is the base model, not a merged one.")
        sys.exit(1)
    print("Merged model differs from the base model in the expected places.")
