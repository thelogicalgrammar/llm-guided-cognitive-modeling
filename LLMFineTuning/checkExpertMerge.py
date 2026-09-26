"""
Compare the expert weights of a base model and a merged model, whatever they are called.

    python LLMFineTuning/checkExpertMerge.py [experts_to_sample]

Paths come from COGMOD_BASE_MODEL and COGMOD_MERGED_MODEL. Unlike checkMerge.py, no tensor name is
constructed: the names are read from each model's own weight index and intersected, so a fused
(experts, in, out) parameter and a per-expert layout are both handled, and a model saved by a
different transformers version is compared only on the names it actually shares with the base.

Reports, per expert tensor, how many sampled experts changed and by how much. Deltas are rank 1 per
expert, so an expert that was never routed during training legitimately shows no change; a tensor
where *no* expert changed means the merge did not apply that parameter at all.
"""
import json
import os
import re
import sys
from collections import defaultdict

import torch
from safetensors import safe_open

EXPERTISH = re.compile(r"experts?\b.*\.(gate_up_proj|gate_proj|up_proj|down_proj)")


def weight_map(model_dir):
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index):
        raise SystemExit(f"no model.safetensors.index.json in {model_dir}")
    return json.load(open(index))["weight_map"]


def slice_of(model_dir, weights, name, expert):
    """One expert's slice, for a fused (experts, ...) tensor or a whole 2D per-expert tensor"""
    with safe_open(os.path.join(model_dir, weights[name]), framework="pt") as f:
        handle = f.get_slice(name)
        shape = handle.get_shape()
        if len(shape) == 3:
            if expert >= shape[0]:
                return None
            return handle[expert:expert + 1].float()
        return handle[:2].float() if expert == 0 else None


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    base_dir = os.environ.get("COGMOD_BASE_MODEL")
    merged_dir = os.environ.get("COGMOD_MERGED_MODEL")
    if not base_dir or not merged_dir:
        raise SystemExit("set COGMOD_BASE_MODEL and COGMOD_MERGED_MODEL (env.sh does)")

    base_weights, merged_weights = weight_map(base_dir), weight_map(merged_dir)
    base_expert = {n for n in base_weights if EXPERTISH.search(n)}
    merged_expert = {n for n in merged_weights if EXPERTISH.search(n)}

    print(f"base:   {len(base_weights)} tensors, {len(base_expert)} expert-like")
    print(f"merged: {len(merged_weights)} tensors, {len(merged_expert)} expert-like")
    for label, names in (("base", base_expert), ("merged", merged_expert)):
        example = sorted(names)[:2]
        print(f"  {label} examples: {example}")

    # Which module paths the adapter actually covers. peft may drop entries from target_modules and
    # handle them as fused parameters instead, which is what happened here: the shared-expert MLP was
    # never adapted, so its weights are identical to the base by design, not by a failed merge.
    adapted_prefixes = set()
    adapter_dir = os.environ.get("COGMOD_LORA")
    if adapter_dir and os.path.isfile(os.path.join(adapter_dir, "adapter_model.safetensors")):
        with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), framework="pt") as f:
            for key in f.keys():
                path = re.sub(r"^base_model\.model\.", "", key)
                path = re.split(r"\.(?:base_layer\.)?lora_[AB]\b", path)[0]
                adapted_prefixes.add(path)
        print(f"\nthe adapter covers {len(adapted_prefixes)} module paths, e.g. "
              f"{sorted(adapted_prefixes)[:2]}")
    else:
        print("\nCOGMOD_LORA not readable: reporting every expert tensor, targeted or not")

    def is_adapted(name):
        return not adapted_prefixes or any(name.startswith(p + ".") for p in adapted_prefixes)

    shared = sorted(base_expert & merged_expert)
    if not shared:
        raise SystemExit(
            "the two models share no expert tensor names, so they cannot be compared directly.\n"
            "  That usually means they were saved by different transformers versions with different\n"
            "  layouts, which also means the merged model is not a drop-in replacement for the base."
        )
    print(f"\n{len(shared)} expert tensors in common; sampling {samples} experts of each kind")

    # group by name pattern so the 48 layers of one parameter report together
    groups = defaultdict(list)
    for name in shared:
        groups[re.sub(r"\.\d+\.", ".N.", name)].append(name)

    failures = []
    for pattern, names in sorted(groups.items()):
        targeted = is_adapted(names[0])
        changed = identical = 0
        worst = 0.0
        # spread the samples over the group, since a per-expert layout puts each expert in its own
        # tensor and most experts of a 512-expert layer are never routed in a short run
        for name in names[:: max(1, len(names) // (samples * 2))][:samples * 2]:
            for expert in range(samples):
                left = slice_of(base_dir, base_weights, name, expert)
                if left is None:
                    break
                right = slice_of(merged_dir, merged_weights, name, expert)
                if torch.equal(left, right):
                    identical += 1
                else:
                    changed += 1
                    denominator = left.norm().item() or 1.0
                    worst = max(worst, (right - left).norm().item() / denominator)
        if changed:
            verdict = "CHANGED"
        elif targeted:
            verdict = "FAILED"
            failures.append(pattern)
        else:
            verdict = "not targeted"
        print(f"  {verdict:12s} {changed:3d} changed, {identical:3d} identical, "
              f"largest |delta|/|W| {worst:.4f}   {pattern}")

    if failures:
        print("\nthese parameters are covered by the adapter but identical to the base everywhere\n"
              "sampled, so the merge did not apply them:")
        for pattern in failures:
            print(f"  {pattern}")
        raise SystemExit(1)

    print("\nEvery parameter the adapter covers changed somewhere: the merge applied the deltas.")
    print("Individual experts that did not change were never routed during training, and parameters")
    print("marked 'not targeted' have no adapter weights, so matching the base is correct for them.")


if __name__ == "__main__":
    main()
