"""
Summarise an OpenEvolve run: what the LLM produced, what the evaluator did with it, and why.

    python OpenEvolve/summariseRun.py <output_dir> [run_log]

<output_dir> is the --output of the run (its checkpoints/ subdirectory is read), and
[run_log] the SLURM log, which adds LLM-side statistics (invalid diffs, iteration times).
"""
import json
import re
import sys
import collections
from pathlib import Path

# Rescorla-Wagner scored with this evaluator on the five included experiments: 0.635 with the
# defaults (validation participants, no penalty), 0.633 with the BIC penalty, 0.637 on training data
RESCORLA_WAGNER = 0.635


def latest_checkpoint(output_dir):
    checkpoints = sorted(Path(output_dir).glob("checkpoints/checkpoint_*"),
                         key=lambda p: int(p.name.split("_")[-1]))
    if not checkpoints:
        raise SystemExit(f"No checkpoints under {output_dir}")
    return checkpoints[-1]

def load_programs(checkpoint):
    return [json.load(open(f)) for f in (checkpoint / "programs").glob("*.json")]

def short_error(metrics):
    error = str(metrics.get("error", ""))
    error = re.sub(r"tmp\w+\.py", "<program>", error)
    error = re.sub(r"\(\d+,\s*\d+\)", "(n, m)", error)
    return error[:90]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    checkpoint = latest_checkpoint(sys.argv[1])
    programs = load_programs(checkpoint)
    print(f"checkpoint: {checkpoint}")
    print(f"programs stored: {len(programs)}")

    ok = [p for p in programs if p["metrics"].get("runs_successfully", 0) == 1]
    failed = [p for p in programs if p not in ok]
    print(f"  ran successfully: {len(ok)}   failed: {len(failed)}")

    if failed:
        print(f"\nwhy programs failed ({len(failed)/len(programs):.0%} of all programs):")
        for error, count in collections.Counter(short_error(p["metrics"]) for p in failed).most_common(8):
            print(f"  {count:4d}  {error}")

    if ok:
        scores = sorted(ok, key=lambda p: p["metrics"].get("combined_score", 0))
        scores = [(p["metrics"].get("combined_score", 0), p) for p in scores]
        print(f"\nscores of the {len(ok)} programs that ran:")
        print(f"  best {scores[-1][0]:.4f}   median {scores[len(scores)//2][0]:.4f}   worst {scores[0][0]:.4f}")
        initial = [p for p in programs if p.get("parent_id") is None]
        if initial:
            print(f"  initial program: {initial[0]['metrics'].get('combined_score', 0):.4f}")
        print(f"  reference points: {initial[0]['metrics'].get('combined_score', 0.45):.3f} = initial program, "
              f"{RESCORLA_WAGNER} = fitted Rescorla-Wagner" if initial else f"  Rescorla-Wagner scores {RESCORLA_WAGNER}")

        jax = [p["metrics"].get("jax_fit") for p in ok if "jax_fit" in p["metrics"]]
        if jax:
            print(f"\ngradient fitting: jax_fit averaged {sum(jax)/len(jax):.2f} "
                  f"({sum(1 for j in jax if j == 1)} of {len(jax)} programs fitted entirely with JAX)")

        initial_score = initial[0]["metrics"].get("combined_score", 0) if initial else 0.45
        better_than_initial = [p for p in ok if p["metrics"].get("combined_score", 0) > initial_score + 1e-6]
        beats_rw = [p for p in ok if p["metrics"].get("combined_score", 0) > RESCORLA_WAGNER]
        print(f"\nprograms better than the initial program: {len(better_than_initial)} of {len(ok)}")
        print(f"programs better than Rescorla-Wagner ({RESCORLA_WAGNER}): {len(beats_rw)}")

        print("\nbest programs (score, complexity, iteration):")
        for _, program in scores[:-6:-1]:
            metrics = program["metrics"]
            print(f"  {metrics.get('combined_score', 0):.4f}   {metrics.get('model_complexity', 0):7.0f}   "
                  f"iteration {program.get('iteration_found')}")

        # the smallest program within 0.01 of the best, as a Pareto-style reference
        near_best = [p for p in ok if p["metrics"].get("combined_score", 0) >= scores[-1][0] - 0.01]
        smallest = min(near_best, key=lambda p: p["metrics"].get("model_complexity", 1e9))
        print(f"smallest program within 0.01 of the best: score {smallest['metrics'].get('combined_score', 0):.4f}, "
              f"complexity {smallest['metrics'].get('model_complexity', 0):.0f}, iteration {smallest.get('iteration_found')}")

    if len(sys.argv) > 2:
        log_path = Path(sys.argv[2])
        log = log_path.read_text(errors="replace")
        # SLURM writes warnings and LLM errors to the matching errors_*.txt, not to the output log
        errors_path = log_path.with_name(log_path.name.replace("results_", "errors_")).with_suffix(".txt")
        if errors_path.exists():
            log += errors_path.read_text(errors="replace")
            print(f"\n(also read {errors_path.name})")
        iterations = re.findall(r"Iteration (\d+): Program", log)
        no_diff = len(re.findall(r"No valid diffs found", log))
        times = [float(t) for t in re.findall(r"completed in ([\d.]+)s", log)]
        print(f"\nfrom the log:")
        print(f"  iterations that produced a program: {len(iterations)}")
        print(f"  responses without a usable code edit: {no_diff}")
        if times:
            times.sort()
            print(f"  iteration time: median {times[len(times)//2]:.0f}s, max {times[-1]:.0f}s")
        for pattern, label in [(r"404|does not exist", "LLM requests refused (model name mismatch?)"),
                               (r"os\.fork\(\) was called", "fork warnings (JAX deadlock risk)"),
                               (r"Feature dimension", "programs dropped from the database"),
                               (r"Timeout|timed out", "timeouts"),
                               (r"context length|maximum context", "context length errors")]:
            count = len(re.findall(pattern, log))
            if count:
                print(f"  {label}: {count}")
