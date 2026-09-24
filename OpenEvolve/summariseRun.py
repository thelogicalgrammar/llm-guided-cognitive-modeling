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
        print("\nwhy programs failed:")
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
        print("  reference points: 0.450 = initial program, 0.636 = fitted Rescorla-Wagner")

        jax = [p["metrics"].get("jax_fit") for p in ok if "jax_fit" in p["metrics"]]
        if jax:
            print(f"\ngradient fitting: jax_fit averaged {sum(jax)/len(jax):.2f} "
                  f"({sum(1 for j in jax if j == 1)} of {len(jax)} programs fitted entirely with JAX)")

        best = scores[-1][1]
        print(f"\nbest program: iteration {best.get('iteration_found')}, "
              f"score {best['metrics'].get('combined_score', 0):.4f}, complexity {best['metrics'].get('model_complexity')}")
        improvements = sorted((p['metrics'].get('combined_score', 0), p.get('iteration_found', 0))
                              for p in ok if p['metrics'].get('combined_score', 0) > (initial[0]['metrics'].get('combined_score', 0) if initial else 0))
        print(f"programs better than the initial one: {len(improvements)}")
        if improvements:
            print("  first few:", [(round(s, 4), i) for s, i in improvements[:5]])

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
