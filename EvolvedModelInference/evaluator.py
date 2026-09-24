"""
Fit selected evolved programs on the training sets and score them on the test sets.

Uses the same fitting machinery as the search (cogmod_core), so the numbers are comparable.
Writes, per program and experiment, a CSV with one row per free-choice test trial
(logits, probability of the human choice, nll) plus a summary over experiments.

    python EvolvedModelInference/evaluator.py [Base/Best FineTuned/Best ...]

Data comes from $COGMOD_DATA_PATH, programs from $COGMOD_PROGRAMS_PATH
(default: EvolvedModelInference/BestModels), results go to $COGMOD_RESULTS_PATH
(default: <repo>/Results/EvolvedCogModels).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cogmod_core import (load_blocks, evalModel, fit_experiment, clear_jax_caches, load_program,
                         FIT_MAXFEV, FIT_STARTS, FIT_METHOD)

script_dir = Path(__file__).resolve().parent
data_path = os.environ.get("COGMOD_DATA_PATH", str(script_dir.parent / "Data"))
results_dir = Path(os.environ.get("COGMOD_RESULTS_PATH", str(script_dir.parent / "Results" / "EvolvedCogModels")))
programs_dir = Path(os.environ.get("COGMOD_PROGRAMS_PATH", str(script_dir / "BestModels")))

# the five experiments the search used, followed by the four held-out ones
experiments = [
    {'name': 'TwoBandit',      'dir': 'TB',  'experiment': 'exp1', 'num_options': 2, 'held_out': False},
    {'name': 'TwoBandit',      'dir': 'TB',  'experiment': 'exp2', 'num_options': 2, 'held_out': False},
    {'name': 'DriftingBandit', 'dir': 'DB',  'experiment': 'exp0', 'num_options': 4, 'held_out': False},
    {'name': 'HorizonSomer',   'dir': 'HSo', 'experiment': 'exp0', 'num_options': 2, 'held_out': False},
    {'name': 'HorizonWaltz',   'dir': 'HW',  'experiment': 'exp0', 'num_options': 2, 'held_out': False},
    {'name': 'HorizonSade',    'dir': 'HSa', 'experiment': 'exp0', 'num_options': 2, 'held_out': True},
    {'name': 'HorizonFeng',    'dir': 'HF',  'experiment': 'exp0', 'num_options': 2, 'held_out': True},
    {'name': 'ChangingBandit', 'dir': 'CB',  'experiment': 'exp0', 'num_options': 2, 'held_out': True},
    {'name': 'MaggiesFarm',    'dir': 'MF',  'experiment': 'exp0', 'num_options': 3, 'held_out': True},
]

DEFAULT_MODELS = ["Base/1000", "Base/1500", "Base/2000", "Base/Best",
                  "FineTuned/1000", "FineTuned/1500", "FineTuned/2000", "FineTuned/Best"]


def experiment_files(exp):
    """Training and test file of an experiment, or None if either is missing"""
    train = Path(data_path) / exp['dir'] / f"struc_Train_{exp['experiment']}.npy"
    test = Path(data_path) / exp['dir'] / f"struc_Test_{exp['experiment']}.npy"
    return (train, test) if train.exists() and test.exists() else None

def trial_table(blocks, logits, num_options):
    """One row per free-choice trial of the test set, in the order of the input data"""
    block, position = blocks['row_block'], blocks['row_position']
    free = blocks['free'][block, position]
    rows = {
        'participant': blocks['participant_of_block'][block][free],
        'game': blocks['game_of_block'][block][free],
        'horizon': blocks['horizon'][block, position][free],
        'trial': blocks['trial'][block, position][free],
        'reward': blocks['reward'][block, position][free],
        'human_choice': blocks['choice'][block, position][free],
    }
    row_logits = logits[block, position][free]                              # (n_free, num_options)
    chosen = row_logits[np.arange(len(row_logits)), rows['human_choice']]
    lse = scipy.special.logsumexp(row_logits, axis=1)
    rows['HCProb'] = np.exp(chosen - lse)
    for option in range(num_options):
        rows[f'EvCogModLog_{option}'] = row_logits[:, option]
    rows['nll'] = lse - chosen
    return pd.DataFrame(rows)

def evaluate_program(model_path):
    """Fit one program on every experiment's training set and score it on the test set"""
    program_path = programs_dir / model_path / "best_program.py"
    program = load_program(str(program_path))
    np.seterr(all='raise', under='ignore')

    out_dir = results_dir / model_path
    out_dir.mkdir(parents=True, exist_ok=True)

    param_bounds = program.define_parameters_and_bounds()
    param_keys = list(param_bounds.keys())

    summary = []
    for i, exp in enumerate(experiments):
        files = experiment_files(exp)
        if files is None:
            print(f"{model_path}: skipping {exp['name']}_{exp['experiment']}, data not found in {data_path}", flush=True)
            continue
        train_file, test_file = files

        start_time = time.time()
        train_blocks, test_blocks = load_blocks(np.load(train_file)), load_blocks(np.load(test_file))

        # the initial values of experiments 1-5 are the ones the search used; held-out experiments fall back to the defaults
        init_param_dict = program.get_init_param(i + 1)
        param_init = np.array([init_param_dict[k] for k in param_keys], dtype=np.float64)

        fit = fit_experiment(program, train_blocks, exp['num_options'], param_bounds, param_init, seed=i,
                             n_starts=FIT_STARTS, max_eval=FIT_MAXFEV, method=FIT_METHOD)

        # score the fitted parameters on the test set and keep the logits of every trial
        params = np.array([fit['params'][k] for k in param_keys], dtype=np.float64)
        test_nll, logits = evalModel(params, test_blocks, exp['num_options'], param_keys, program.Model(), return_logits=True)

        # participant and game of each block, for the per-trial table
        raw = np.load(test_file)
        block = test_blocks['row_block']
        for field, column in [('participant_of_block', 0), ('game_of_block', 1)]:
            test_blocks[field] = np.zeros(test_blocks['n_blocks'], dtype=np.int64)
            test_blocks[field][block] = raw[:, column]

        table = trial_table(test_blocks, logits, exp['num_options'])
        table.to_csv(out_dir / f"{exp['name']}_{exp['experiment']}.csv", index=False)

        summary.append({'name': exp['name'], 'experiment': exp['experiment'], 'held_out': exp['held_out'],
                        'train_nll': fit['nll'], 'test_nll': test_nll, 'jax_fit': float(fit['jax_used']),
                        'parameters': fit['params'], 'seconds': round(time.time() - start_time, 1)})
        print(f"{model_path}: {exp['name']}_{exp['experiment']}: train {fit['nll']:.4f}, test {test_nll:.4f}, "
              f"{'jax' if fit['jax_used'] else 'powell'}, {summary[-1]['seconds']}s", flush=True)

    clear_jax_caches()
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "Results_summary.csv", index=False)
    print(f"{model_path}: mean test nll {summary['test_nll'].mean():.4f} over {len(summary)} experiments", flush=True)
    return model_path, summary['test_nll'].mean()


if __name__ == "__main__":
    models = sys.argv[1:] or DEFAULT_MODELS
    print(f"fitting {len(models)} programs with method={FIT_METHOD}, {FIT_STARTS} starts, maxfev={FIT_MAXFEV}")
    print(f"data: {data_path}\nresults: {results_dir}", flush=True)

    workers = min(len(models), int(os.environ.get("COGMOD_WORKERS", 4)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(evaluate_program, models))

    print("\nmean test nll per program:")
    for model_path, mean_nll in sorted(results, key=lambda r: r[1]):
        print(f"  {model_path:20s} {mean_nll:.4f}")
