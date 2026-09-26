"""
Summarise a fine-tuning run: whether it finished, whether it learned, and whether the adapter it
saved actually contains trained expert weights.

    python LLMFineTuning/summariseFineTune.py [run_dir]

run_dir defaults to $COGMOD_FINETUNE_OUT. Reads SFTTrainer_logs.csv and baseline_metrics.csv, and
inspects the saved adapter directly: the published adapter's expert lora_B tensors were all zero,
which no log would have revealed, so this reads the weights rather than trusting the run.
"""
import os
import sys
from pathlib import Path

import pandas as pd


def adapter_report(adapter_dir):
    """Per-target norms of the saved adapter, so all-zero expert weights are visible"""
    from safetensors import safe_open

    path = adapter_dir / "adapter_model.safetensors"
    if not path.is_file():
        print(f"  no adapter at {path}")
        return

    expert_zero = expert_live = other_zero = other_live = 0
    smallest = None
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if "lora_B" not in key:
                continue                                  # lora_A starts random; lora_B starts at 0
            norm = f.get_tensor(key).float().norm().item()
            if ".mlp.experts" in key:
                if norm > 0:
                    expert_live += 1
                    smallest = norm if smallest is None else min(smallest, norm)
                else:
                    expert_zero += 1
            else:
                other_live += 1 if norm > 0 else 0
                other_zero += 0 if norm > 0 else 1

    print(f"  expert lora_B: {expert_live} non-zero, {expert_zero} zero")
    print(f"  other  lora_B: {other_live} non-zero, {other_zero} zero")
    if smallest is not None:
        print(f"  smallest non-zero expert norm: {smallest:.3g}")
    if expert_zero and not expert_live:
        print("  the experts were never trained: this is the failure the published adapter had, and\n"
              "  merging it would change nothing outside attention and the shared MLPs.")
    elif expert_live:
        print("  the experts carry trained weights, so merge_and_unload will apply them.")


def loss_report(run_dir):
    logs = run_dir / "SFTTrainer_logs.csv"
    if not logs.is_file():
        print(f"  no {logs.name}: the run did not reach the end")
        return
    history = pd.read_csv(logs)

    train = history[history.get("loss").notna()] if "loss" in history else pd.DataFrame()
    if len(train):
        print(f"  training loss: {train['loss'].iloc[0]:.4f} at step {int(train['step'].iloc[0])} "
              f"-> {train['loss'].iloc[-1]:.4f} at step {int(train['step'].iloc[-1])}")
        print(f"  epochs reached: {history['epoch'].max():.2f}")

    if "eval_loss" in history:
        # the final evaluate() logs at the same step as the last periodic one, so drop the repeat
        evals = history[history["eval_loss"].notna()][["step", "eval_loss"]].drop_duplicates()
        baseline = run_dir / "baseline_metrics.csv"
        if baseline.is_file():
            before = pd.read_csv(baseline)
            if "eval_loss" in before:
                print(f"  validation loss before any training: {before['eval_loss'].iloc[0]:.4f}")
        for _, row in evals.iterrows():
            print(f"    step {int(row['step']):4d}   validation loss {row['eval_loss']:.4f}")
        if len(evals) and evals["eval_loss"].iloc[-1] > evals["eval_loss"].min():
            print("  the last evaluation is not the best one: the run was past its best when it "
                  "stopped, so an earlier checkpoint may be the one to merge.")


if __name__ == "__main__":
    # an unset COGMOD_FINETUNE_OUT must not quietly become the working directory, which then reports
    # that the run never finished
    given = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COGMOD_FINETUNE_OUT") or ""
    if not given.strip():
        project = os.environ.get("PROJECT")
        given = f"{project}/FineTuneNew" if project else ""
        if not given:
            raise SystemExit("pass the run directory, or set COGMOD_FINETUNE_OUT (env.sh does)")
        print(f"COGMOD_FINETUNE_OUT is not set; trying the default {given}")

    run_dir = Path(given).expanduser()
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir!s} (pass it, or set COGMOD_FINETUNE_OUT)")

    print(f"run: {run_dir}")
    contents = sorted(p.name + ("/" if p.is_dir() else "") for p in run_dir.iterdir())
    print(f"  contains: {', '.join(contents) if contents else '(nothing)'}")

    # a run writes its output where COGMOD_FINETUNE_OUT pointed at the time, which is easy to lose
    # track of between a smoke run and a full one
    project = os.environ.get("PROJECT")
    others = []
    if project:
        others += [p for p in Path(project).glob("FineTune*") if p.is_dir()]
    data = os.environ.get("COGMOD_DATA_PATH")
    if data:
        # where an earlier version of FineTuneQ3LoRA.py defaulted to, so a run's output can be here
        others += [p for p in Path(data).glob("FineTune*") if p.is_dir()]
    others = [p for p in others if p.resolve() != run_dir.resolve()]
    if others:
        print("  other fine-tuning directories:")
        for other in others:
            entries = sorted(p.name for p in other.iterdir())
            print(f"    {other}: {', '.join(entries) if entries else '(empty)'}")

    loss_report(run_dir)

    for name in ("Final_LoRA", "Smoke_LoRA"):
        if (run_dir / name).is_dir():
            print(f"\nadapter {name}:")
            adapter_report(run_dir / name)

    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if checkpoints:
        print(f"\ncheckpoints: {', '.join(p.name for p in checkpoints)}")
        print(f"\nadapter in the last checkpoint ({checkpoints[-1].name}):")
        adapter_report(checkpoints[-1])
