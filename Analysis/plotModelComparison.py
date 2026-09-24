"""
Compare fitted models across experiments: nll on the participants used for fitting, and on the
held-out test participants.

    python Analysis/plotModelComparison.py RescorlaWagner Base4000 [--out figure.png]

Reads $COGMOD_RESULTS_PATH/<model>/Results_summary.csv, written by
EvolvedModelInference/evaluator.py. Lower nll is better; chance is log(num_options).
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots 1-4 of the reference palette, validated for colour-vision deficiency
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdcd8", "#fcfcfb"

LABELS = {"TwoBandit_exp1": "TwoBandit 1", "TwoBandit_exp2": "TwoBandit 2",
          "DriftingBandit_exp0": "Drifting Bandit", "HorizonSomer_exp0": "Horizon Somer",
          "HorizonWaltz_exp0": "Horizon Waltz", "HorizonSade_exp0": "Horizon Sade",
          "HorizonFeng_exp0": "Horizon Feng", "ChangingBandit_exp0": "Changing Bandit",
          "MaggiesFarm_exp0": "Maggie's Farm"}


def load(model, results_dir):
    path = Path(results_dir) / model / "Results_summary.csv"
    if not path.is_file():
        raise SystemExit(f"No summary for {model}: {path}")
    table = pd.read_csv(path)
    table["experiment"] = table["name"] + "_" + table["experiment"]
    return table.set_index("experiment")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--results", default=os.environ.get("COGMOD_RESULTS_PATH", "Results/EvolvedCogModels"))
    parser.add_argument("--out", default="Images/ModelComparison.png")
    args = parser.parse_args()

    tables = {model: load(model, args.results) for model in args.models}
    order = [e for e in LABELS if e in next(iter(tables.values())).index]
    held_out = [e for e in order if bool(next(iter(tables.values())).loc[e].get("held_out", False))]

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True, sharex=True)
    figure.patch.set_facecolor(SURFACE)
    positions = list(range(len(order)))
    # a small vertical dodge per model, so near-identical values stay visible
    offsets = [(i - (len(args.models) - 1) / 2) * 0.17 for i in range(len(args.models))]

    for axis, column, title in zip(axes, ["train_nll", "test_nll"],
                                   ["Participants used for fitting", "Held-out test participants"]):
        axis.set_facecolor(SURFACE)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)

        # shade the held-out tasks rather than drawing a line and labelling it twice
        if held_out:
            axis.axhspan(len(order) - len(held_out) - 0.5, len(order) - 0.4,
                         color=MUTED, alpha=0.05, zorder=0)

        for y, experiment in zip(positions, order):
            values = [tables[m].loc[experiment, column] for m in args.models]
            axis.plot([min(values), max(values)], [y, y], color=GRID, linewidth=1.5, zorder=1,
                      solid_capstyle="round")
        for colour, model, offset in zip(COLOURS, args.models, offsets):
            values = [tables[model].loc[e, column] for e in order]
            axis.scatter(values, [y + offset for y in positions], s=80, color=colour, zorder=3,
                         edgecolor=SURFACE, linewidth=1.5, label=model)

        axis.set_title(title, color=INK, fontsize=11, pad=8)
        axis.set_xlabel("negative log-likelihood per free choice", color=MUTED, fontsize=9)
        axis.tick_params(axis="x", colors=MUTED, labelsize=9)
        axis.tick_params(axis="y", length=0, colors=INK, labelsize=10)

    if held_out:
        axes[1].text(axes[1].get_xlim()[1], len(order) - len(held_out) - 0.35, "held-out tasks ",
                     fontsize=8.5, color=MUTED, va="top", ha="right")

    axes[0].set_yticks(positions, [LABELS.get(e, e) for e in order])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncol=len(args.models), frameon=False,
                  bbox_to_anchor=(0.99, 0.99), fontsize=10, labelcolor=INK)
    figure.text(0.012, 0.955, "Model fit across tasks", fontsize=13, color=INK, va="top")
    figure.text(0.012, 0.905, "negative log-likelihood per free choice, lower is better",
                fontsize=9, color=MUTED, va="top")
    means = "   ".join(f"{model}: {tables[model]['test_nll'].mean():.4f}" for model in args.models)
    figure.text(0.012, 0.022, f"mean test nll over {len(order)} tasks -  {means}", fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0.045, 1, 0.87))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"written: {out}")
    for model in args.models:
        print(f"  {model:20s} mean train {tables[model]['train_nll'].mean():.4f}   "
              f"mean test {tables[model]['test_nll'].mean():.4f}")


if __name__ == "__main__":
    main()
