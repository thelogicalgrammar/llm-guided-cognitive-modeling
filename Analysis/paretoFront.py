"""
Pareto fronts of complexity against accuracy, for one or more OpenEvolve runs.

    python Analysis/paretoFront.py <run_dir> [<run_dir> ...]
        [--checkpoint N] [--x effective_parameters] [--out Images/Pareto.png]

Reads <run_dir>/checkpoints/checkpoint_<N>/programs/*.json, the same files
OpenEvolve/summariseRun.py reads, so it works on a run that is still going.

It pools every checkpoint up to the one chosen, deduplicating by program id. config.yaml
sets population_size 800 against max_iterations 10000, so any one checkpoint holds only
the programs alive at that moment, and eviction goes by fitness: a snapshot thins out the
cheap, inaccurate programs that anchor the low-complexity end of a front, and would make
both arms look better and more similar than they were. --snapshot gives the single-
checkpoint view. Neither recovers programs that lived and died between two checkpoints.

Unlike the Pareto plot in Analysis/OpenEvolveAnalysis.ipynb, which reads the published
CSVs and plots the AST-based model_complexity against combined_score, this defaults to
effective_parameters against nll: effective_parameters is what a BIC penalty would
charge (parameter_count + CONSTANT_WEIGHT * charged_constants, OpenEvolve/evaluator.py),
so a front drawn against it says what the extra structure actually bought. --x accepts
model_complexity, parameters or hardcoded_constants to get the other views.

Both axes are minimised, so the front is the lower-left skyline: no program on it is
beaten on accuracy *and* complexity by another.

Comparing two runs, it defaults to the highest checkpoint they have in common rather than
each one's latest. Two arms compared at different iteration counts differ in how long they
searched as well as in the model, and the longer one will usually look better for that
reason alone.
"""
import argparse
import json
import sys
from pathlib import Path

X_CHOICES = ["effective_parameters", "model_complexity", "parameters", "hardcoded_constants"]


def checkpoints_of(run_dir):
    """{iteration: path} for every checkpoint of a run"""
    found = {}
    for path in Path(run_dir).glob("checkpoints/checkpoint_*"):
        try:
            found[int(path.name.split("_")[-1])] = path
        except ValueError:
            continue
    if not found:
        raise SystemExit(f"no checkpoints under {run_dir}/checkpoints/")
    return found


def load_points(checkpoints, x_key):
    """(x, nll, program_id) for every distinct program across the given checkpoints

    Pooling matters: config.yaml sets population_size 800 against max_iterations 10000, so a
    single checkpoint holds only the programs alive at that moment. Eviction is by fitness, so
    a snapshot is a *selected* sample - it thins out exactly the cheap, inaccurate programs
    that anchor the low-complexity end of the front. Programs carry stable ids, so the union
    over checkpoints recovers everything that survived to any checkpoint boundary.
    """
    by_id = {}
    for checkpoint in checkpoints:
        for path in (checkpoint / "programs").glob("*.json"):
            try:
                program = json.load(open(path))
            except (json.JSONDecodeError, OSError):
                continue  # a checkpoint being written while a run is live can hold a partial file
            metrics = program.get("metrics", {})
            if metrics.get("runs_successfully", 0) != 1:
                continue
            if x_key not in metrics or "nll" not in metrics:
                continue
            program_id = program.get("id", str(path))
            by_id[program_id] = (float(metrics[x_key]), float(metrics["nll"]), program_id)
    return list(by_id.values())


def pareto_front(points):
    """The lower-left skyline: minimise both coordinates

    Sorted by x then y, a point joins the front only if it beats every less complex point
    on nll. Ties in x are ordered by y, so the best program at a given complexity wins.
    """
    front, best_nll = [], float("inf")
    for point in sorted(points, key=lambda p: (p[0], p[1])):
        if point[1] < best_nll:
            front.append(point)
            best_nll = point[1]
    return front


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="OpenEvolve --output directories")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="iteration to read (default: the highest all runs share)")
    parser.add_argument("--x", default="effective_parameters", choices=X_CHOICES)
    parser.add_argument("--out", default=None, help="PNG to write (default: no plot, text only)")
    parser.add_argument("--snapshot", action="store_true",
                        help="read only the chosen checkpoint instead of pooling every checkpoint "
                             "up to it: what the search was holding, not what it found")
    args = parser.parse_args()

    available = {run: checkpoints_of(run) for run in args.runs}
    if args.checkpoint is not None:
        iteration = args.checkpoint
        missing = [run for run, found in available.items() if iteration not in found]
        if missing:
            raise SystemExit(f"checkpoint_{iteration} missing from: {', '.join(missing)}")
    else:
        shared = set.intersection(*(set(found) for found in available.values()))
        if not shared:
            raise SystemExit("the runs share no checkpoint; pass --checkpoint explicitly")
        iteration = max(shared)
        for run, found in available.items():
            if max(found) > iteration:
                print(f"note: {Path(run).name} reaches checkpoint_{max(found)}, but comparing at "
                      f"checkpoint_{iteration}, the highest shared with the other run(s)")

    scope = f"checkpoint_{iteration} only" if args.snapshot else f"checkpoints up to {iteration}"
    print(f"\n{scope}, x = {args.x}, y = nll (both minimised)")
    series = []
    for run in args.runs:
        label = Path(run).name
        chosen = ([available[run][iteration]] if args.snapshot
                  else [path for step, path in sorted(available[run].items()) if step <= iteration])
        points = load_points(chosen, args.x)
        if not points:
            print(f"\n{label}: no programs with both {args.x} and nll")
            continue
        front = pareto_front(points)
        series.append((label, points, front))

        best = min(points, key=lambda p: p[1])
        print(f"\n{label}: {len(points)} distinct programs that ran"
              f"{'' if args.snapshot else f' over {len(chosen)} checkpoints'}, "
              f"{len(front)} on the front")
        print(f"  best nll {best[1]:.4f} at {args.x} {best[0]:g}")
        print(f"  {'front: ' + args.x:>28} | nll")
        for x, nll, _ in front:
            print(f"  {x:>28g} | {nll:.4f}")

    if not args.out:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display on a login node
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is not available here; rerun without --out for the text front")

    plt.figure(figsize=(8, 6))
    for index, (label, points, front) in enumerate(series):
        colour = f"C{index}"
        plt.scatter([p[0] for p in points], [p[1] for p in points], alpha=0.15, color=colour)
        plt.plot([p[0] for p in front], [p[1] for p in front], color=colour, linewidth=2,
                 marker="o", markersize=4, label=f"{label} ({len(points)} programs)")
    plt.xlabel(args.x.replace("_", " "))
    plt.ylabel("validation nll (lower is better)")
    plt.title(f"Complexity against accuracy at checkpoint_{iteration}")
    plt.legend()
    plt.grid(alpha=0.3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
