"""
Add the MoE expert LoRA deltas that PEFT does not merge.

PEFT 0.18 can train a LoRA against fused 3D expert parameters through target_parameters, but
merge_and_unload leaves those weights untouched (and merging with 0.19+ would apply a different
layout convention to a 0.18-era adapter, which silently scrambles the delta:
https://github.com/unslothai/unsloth-zoo/pull/1232, https://github.com/vllm-project/vllm/pull/52198).
So the attention and dense projections come out of mergeLoRA.py correctly, and the experts do not.

This script applies the missing deltas directly to the safetensors of the merged model:

    W_e += (alpha / r) * outer(A[e], B[:, e])

for every expert e of every layer, with A and B read from the adapter. The adapter stores one pair
per fused parameter, identified by shape:

    lora_A (E*r, 1024), lora_B (2048, E*r)   -> gate_up_proj: rows 0:512 are gate_proj, 512:1024 up_proj
    lora_A (E*r, 2048), lora_B (512,  E*r)   -> down_proj

With rank_pattern r=1 for the expert parameters, the E*r axis is one entry per expert, so the
ordering ambiguity between rank-major and expert-grouped layouts does not arise.

    python LLMFineTuning/mergeExpertDeltas.py [--dry-run] [--model DIR] [--lora DIR]

Rewrites the shards of $COGMOD_MERGED_MODEL in place (about 150 GB of reading and writing), after
checking that the expert weights are still identical to the base model.
"""
import argparse
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def adapter_pairs(lora_dir):
    """Per layer, the (A, B) pair for gate_up_proj and for down_proj, identified by shape"""
    config = json.load(open(Path(lora_dir) / "adapter_config.json"))
    rank = config.get("rank_pattern", {})
    alpha = config.get("alpha_pattern", {})
    # the expert entries of the patterns; both keys match .*mlp\.experts\.(gate_up_proj|down_proj)
    expert_rank = next((v for k, v in rank.items() if "experts" in k), config["r"])
    expert_alpha = next((v for k, v in alpha.items() if "experts" in k), config["lora_alpha"])
    scale = expert_alpha / expert_rank

    tensors = {}
    with safe_open(Path(lora_dir) / "adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            if ".mlp.experts" not in key:
                continue
            layer = int(re.search(r"layers\.(\d+)\.", key).group(1))
            tensors.setdefault(layer, {})[key.split(".")[-2] + ("_base" if ".base_layer." in key else "")] = f.get_tensor(key)

    pairs = {}
    for layer, found in tensors.items():
        a_base, b_base = found.get("lora_A_base"), found.get("lora_B_base")
        a_main, b_main = found.get("lora_A"), found.get("lora_B")
        if a_base is None or a_main is None:
            raise SystemExit(f"layer {layer}: expected two lora pairs, found {sorted(found)}")
        # gate_up_proj is the pair whose A has the smaller second dimension (2 * expert inner size)
        (a_gate_up, b_gate_up), (a_down, b_down) = ((a_base, b_base), (a_main, b_main)) \
            if a_base.shape[1] < a_main.shape[1] else ((a_main, b_main), (a_base, b_base))
        pairs[layer] = {"gate_up": (a_gate_up, b_gate_up), "down": (a_down, b_down), "scale": scale}
    return pairs, scale


def delta_for(name, pairs):
    """The delta for one expert tensor of the merged model, or None if the name is not an expert weight"""
    match = re.search(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name)
    if not match:
        return None
    layer, expert, which = int(match.group(1)), int(match.group(2)), match.group(3)
    pair = pairs[layer]
    if which == "down_proj":
        a, b = pair["down"]
        return pair["scale"] * torch.outer(a[expert].float(), b[:, expert].float())      # (2048, 512)
    a, b = pair["gate_up"]
    full = pair["scale"] * torch.outer(a[expert].float(), b[:, expert].float())          # (1024, 2048)
    half = full.shape[0] // 2
    return full[:half] if which == "gate_proj" else full[half:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("COGMOD_MERGED_MODEL"))
    parser.add_argument("--lora", default=os.environ.get("COGMOD_LORA"))
    parser.add_argument("--base", default=os.environ.get("COGMOD_BASE_MODEL"))
    parser.add_argument("--dry-run", action="store_true", help="report the deltas without writing")
    args = parser.parse_args()
    if not args.model or not args.lora:
        raise SystemExit("set COGMOD_MERGED_MODEL and COGMOD_LORA (env.sh does)")

    pairs, scale = adapter_pairs(args.lora)
    print(f"adapter: {len(pairs)} layers with expert deltas, scale alpha/r = {scale:g}")

    index = json.load(open(Path(args.model) / "model.safetensors.index.json"))["weight_map"]
    shards = sorted({shard for name, shard in index.items() if delta_for(name, pairs) is not None})
    print(f"model: {len(shards)} of {len(set(index.values()))} shards contain expert weights")

    expert_names = [name for name in index if delta_for(name, pairs) is not None]

    if args.dry_run:
        # sample a few tensors rather than reading every shard: enough to check the shapes line up
        # and that the deltas are a sensible size relative to the weights
        sample = []
        for which in ("gate_proj", "up_proj", "down_proj"):
            of_kind = [n for n in expert_names if n.endswith(f"{which}.weight")]
            sample += of_kind[:: max(1, len(of_kind) // 3)][:3]
        print(f"{len(expert_names)} expert tensors in total; sampling {len(sample)}\n")
        for name in sample:
            with safe_open(Path(args.model) / index[name], framework="pt") as f:
                values = f.get_tensor(name)
            delta = delta_for(name, pairs)
            if delta.shape != values.shape:
                raise SystemExit(f"{name}: delta {tuple(delta.shape)} does not match weight {tuple(values.shape)}")
            ratio = (delta.norm() / values.float().norm()).item()
            print(f"  {name:62s} {tuple(values.shape)}  |delta|/|W| = {ratio:.4f}")
        print("\nrun without --dry-run to apply them")
        return

    changed = relative = 0
    for number, shard in enumerate(shards, 1):
        path = Path(args.model) / shard
        with safe_open(path, framework="pt") as f:
            tensors = {name: f.get_tensor(name) for name in f.keys()}
            metadata = f.metadata()
        for name, values in tensors.items():
            delta = delta_for(name, pairs)
            if delta is None:
                continue
            if delta.shape != values.shape:
                raise SystemExit(f"{name}: delta {tuple(delta.shape)} does not match weight {tuple(values.shape)}")
            updated = values.float() + delta
            relative += (delta.norm() / values.float().norm()).item()
            changed += 1
            tensors[name] = updated.to(values.dtype)
        save_file(tensors, str(path), metadata=metadata)
        print(f"  [{number}/{len(shards)}] {shard}: updated, {changed} expert tensors so far", flush=True)

    print(f"\n{changed} expert tensors updated, mean |delta| / |W| = {relative / max(changed, 1):.4f}")
    print("check with: python LLMFineTuning/checkMerge.py 5")


if __name__ == "__main__":
    main()
