"""
Print an adapter's config and the shapes of its tensors, grouped by kind.

    python LLMFineTuning/inspectAdapter.py [adapter_dir]

adapter_dir defaults to $COGMOD_LORA. Needed because an adapter that targets the fused MoE
parameters through `target_parameters` records its ranks in `rank_pattern`, and a mismatch between
the rank used when training and the rank peft reconstructs when loading shows up only as a size
mismatch deep inside load_state_dict.
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COGMOD_LORA", ""))
    if not (path / "adapter_config.json").is_file():
        raise SystemExit(f"no adapter_config.json in {path} (pass the directory, or set COGMOD_LORA)")

    config = json.load(open(path / "adapter_config.json"))
    print(f"adapter: {path}")
    for key in ("peft_type", "r", "lora_alpha", "target_parameters", "rank_pattern",
                "alpha_pattern", "target_modules"):
        value = config.get(key)
        if isinstance(value, list):
            value = sorted(value)
        print(f"  {key}: {value}")

    # group tensors by the shape pattern of their name, so 48 identical layers print once
    groups = defaultdict(list)
    with safe_open(path / "adapter_model.safetensors", framework="pt") as f:
        for name in f.keys():
            shape = tuple(f.get_slice(name).get_shape())
            kind = re.sub(r"\.layers\.\d+\.", ".layers.N.", name)
            groups[(kind, shape)].append(name)

    print(f"\n{len(groups)} distinct (name pattern, shape) combinations:")
    for (kind, shape), names in sorted(groups.items()):
        print(f"  {len(names):4d} x {shape}   {kind}")

    # the expert ranks implied by the stored shapes, which is what the loader has to reproduce
    experts = [(k, s) for (k, s), _ in groups.items() if ".mlp.experts" in k]
    if experts:
        print("\nexpert tensors, and the rank each implies:")
        num_experts = config.get("num_experts")
        for kind, shape in sorted(experts):
            note = ""
            if "lora_A" in kind and num_experts:
                note = f"  -> rank {shape[0] / num_experts:g} per expert"
            print(f"  {shape}  {kind}{note}")
        print("\nIf lora_A has as many rows as there are experts, the trained rank was 1 and\n"
              "rank_pattern was honoured. Many times that, and it was not.")


if __name__ == "__main__":
    main()
