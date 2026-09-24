"""
Check that merging the LoRA actually changed the weights it was supposed to change.

PEFT warns about an unsupported layer type (Qwen3NextExperts) on this model, which means
the expert adapters (target_parameters: gate_up_proj, down_proj) can be dropped without an
error, leaving a model where only the attention and dense projections are fine-tuned.

The tensor names to compare are taken from the adapter config and the model's own weight
index, so nothing is assumed about how this model names its layers. Only a slice of each
tensor is read, so multi-GB expert weights cost nothing.

    python LLMFineTuning/checkMerge.py [samples_per_target]

Paths come from COGMOD_BASE_MODEL, COGMOD_MERGED_MODEL and COGMOD_LORA (env.sh sets them).
Exit code 0: every kind of targeted weight changed. 1: some kind did not change at all.
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

def first_slice(model_dir, weight_map, name, rows=1):
    """A few rows of a tensor, without reading the whole thing"""
    with safe_open(os.path.join(model_dir, weight_map[name]), framework=FRAMEWORK) as f:
        values = f.get_slice(name)[:rows]
    return values.float() if FRAMEWORK == "pt" else values.astype("float32")

def max_difference(base_dir, base_map, merged_dir, merged_map, name):
    base_values, merged_values = first_slice(base_dir, base_map, name), first_slice(merged_dir, merged_map, name)
    if base_values.shape != merged_values.shape:
        return None
    return float(abs(base_values - merged_values).max())

def targets(lora_dir):
    """The module names and parameter names the adapter claims to change"""
    if not lora_dir or not os.path.isfile(os.path.join(lora_dir, "adapter_config.json")):
        return ["q_proj", "o_proj"], ["gate_up_proj", "down_proj"]
    config = json.load(open(os.path.join(lora_dir, "adapter_config.json")))
    return config.get("target_modules", []), config.get("target_parameters", [])


if __name__ == "__main__":
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    base_dir, merged_dir = os.environ.get("COGMOD_BASE_MODEL"), os.environ.get("COGMOD_MERGED_MODEL")
    lora_dir = os.environ.get("COGMOD_LORA")
    if not base_dir or not merged_dir:
        raise SystemExit("Set COGMOD_BASE_MODEL and COGMOD_MERGED_MODEL (env.sh does)")

    base_map, merged_map = weight_index(base_dir), weight_index(merged_dir)
    print(f"base:   {base_dir}\nmerged: {merged_dir}")
    print(f"tensors: {len(base_map)} in base, {len(merged_map)} in merged")

    target_modules, target_parameters = targets(lora_dir)
    print(f"\nadapter targets modules {sorted(target_modules)}\n                 parameters {target_parameters}")

    # find the real tensor names of each target, spread over the model rather than all in layer 0
    def matching(target, expert_only):
        names = [n for n in base_map if f".{target}" in n and n in merged_map
                 and (".experts." in n) == expert_only]
        return names[::max(1, len(names) // samples)][:samples]

    groups = [(f"module {t}", matching(t, False)) for t in sorted(set(target_modules))]
    groups += [(f"expert parameter {t}", matching(t, True)) for t in sorted(set(target_parameters))]

    failures, checked = [], 0
    for label, names in groups:
        if not names:
            print(f"\n{label}: no tensors found with this name")
            continue
        print(f"\n{label}:")
        differences = []
        for name in names:
            difference = max_difference(base_dir, base_map, merged_dir, merged_map, name)
            checked += 1
            if difference is None:
                print(f"  {name}: shapes differ")
                differences.append(float("inf"))
            else:
                print(f"  {name}: max abs diff {difference:.6f}   {'changed' if difference > 0 else 'IDENTICAL'}")
                differences.append(difference)
        if differences and max(differences) == 0:
            failures.append(label)

    print(f"\n{checked} tensors compared")
    if failures:
        print("These targets are identical to the base model, so their adapters were not merged:")
        for label in failures:
            print(f"  - {label}")
        print("\nIf the expert parameters are among them, the merged model is only partly\n"
              "fine-tuned. Try the transformers/peft versions the adapter was trained with,\n"
              "in a separate venv (SERVER_SETUP.txt step 3.2).")
        sys.exit(1)
    print("Every targeted weight changed: the merge looks complete.")
