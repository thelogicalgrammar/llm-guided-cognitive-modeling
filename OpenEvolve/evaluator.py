import numpy as np
from openevolve.evaluation_result import EvaluationResult
import pandas as pd
from scipy.optimize import OptimizeWarning
import warnings
import traceback
import sys
import ast
import os

# shared fitting machinery, also used by EvolvedModelInference/evaluator.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cogmod_core import (load_blocks, evalModel, fit_experiment, clear_jax_caches, load_program,
                         FIT_MAXFEV, FIT_STARTS, FIT_METHOD, JAX_AVAILABLE)

# ------ Load Data Globally ------
experiments = [
    {'name': 'TB', 'experiment': 'exp1', 'num_options': 2},
    {'name': 'TB', 'experiment': 'exp2', 'num_options': 2},
    {'name': 'DB', 'experiment': 'exp0', 'num_options': 4},
    {'name': 'HSo', 'experiment': 'exp0', 'num_options': 2},
    {'name': 'HW', 'experiment': 'exp0', 'num_options': 2}
] # different names are used here to ensure the LLM cannot get the names of the experiments

data_path = os.environ.get("COGMOD_DATA_PATH", "Path/to/your/data/")  # Update this path (or set COGMOD_DATA_PATH) to your actual data directory
prog_path = os.environ.get("COGMOD_PROG_PATH", "Path/to/your/programs/")  # Update this path (or set COGMOD_PROG_PATH) to your actual programs directory

TRAIN_BLOCKS = [load_blocks(np.load(os.path.join(data_path, exp['name'], f"struc_Train_{exp['experiment']}.npy"))) for exp in experiments]

# Programs are fitted on the training participants and scored on held-out validation participants,
# so that models are selected for generalisation rather than for fitting the training set. Set
# COGMOD_SCORE_SPLIT=Train to score on the training participants instead, as the published runs did.
SCORE_SPLIT = os.environ.get("COGMOD_SCORE_SPLIT", "Val")

# Selection is by likelihood on held-out participants, which already prices extra parameters
# through the generalisation gap, so no penalty is applied by default. COGMOD_BIC_PENALTY=1 adds
# a per-trial BIC of log(N)/(2N) per effective parameter, for a run scored the classical way.
BIC_PENALTY = os.environ.get("COGMOD_BIC_PENALTY", "0") == "1"

# Hardcoded constants are tuned by the search across iterations, so they fit the data like
# parameters and count towards k, at this weight. They are counted more cautiously than they are
# reported: only non-integer literals, since integers are usually structural (indices, 2, 10).
CONSTANT_WEIGHT = float(os.environ.get("COGMOD_CONSTANT_WEIGHT", 0.5))

def score_blocks(exp):
    path = os.path.join(data_path, exp['name'], f"struc_{SCORE_SPLIT}_{exp['experiment']}.npy")
    return load_blocks(np.load(path)) if os.path.isfile(path) else None

SCORE_BLOCKS = [score_blocks(exp) for exp in experiments] if SCORE_SPLIT != "Train" else [None] * len(experiments)
if SCORE_SPLIT != "Train" and any(b is None for b in SCORE_BLOCKS):
    missing = [f"{e['name']}/{e['experiment']}" for e, b in zip(experiments, SCORE_BLOCKS) if b is None]
    print(f"evaluator: no struc_{SCORE_SPLIT} data for {missing}; those experiments are scored on the training participants")


def compute_source_complexity(program_path):
    """Compute a simple AST-based complexity score for the source file."""
    with open(program_path, "r", encoding="utf-8", errors="replace") as source_file:
        source = source_file.read()

    tree = ast.parse(source, filename=program_path)
    complexity = sum(1 for _ in ast.walk(tree))
    return complexity

def count_hardcoded_constants(program_path):
    """
    Numeric literals inside the Model class, as (reported, charged).

    Reported: everything except 0 and 1, which is what the program is told about.
    Charged: only non-integer values other than 0.5, since integers are usually structural
    (indices, 2, 10) while values like 0.18 or 0.0022 are tuned by the search and fit the data
    like parameters.
    """
    try:
        with open(program_path, "r", encoding="utf-8", errors="replace") as source_file:
            tree = ast.parse(source_file.read(), filename=program_path)
    except Exception:
        return 0.0, 0.0
    model_class = next((node for node in ast.walk(tree)
                        if isinstance(node, ast.ClassDef) and node.name == "Model"), None)
    if model_class is None:
        return 0.0, 0.0
    values = [node.value for node in ast.walk(model_class)
              if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
              and not isinstance(node.value, bool)]
    reported = [v for v in values if abs(v) not in (0, 1)]
    charged = [v for v in reported if not float(v).is_integer() and abs(v) != 0.5]
    return float(len(reported)), float(len(charged))

def safe_model_complexity(program_path):
    """Model complexity for failed programs, so OpenEvolve can still place them in the feature grid."""
    try:
        return float(compute_source_complexity(program_path))
    except Exception:
        return 0.0

def run_model(program_path, max_eval, n_starts=1, method="powell"):
    """
    This function is called once for each iteration by OpenEvolve, which includes:
    1. training the model by optimizing the adjustable parameters.
    2. returning feedback metrics.
    """

    # load program
    program = load_program(program_path)

    # get source_complexity based on model AST for combined_score penalization
    source_complexity = compute_source_complexity(program_path)

    # initialize numpy to consider all warnings as errors, except underflow, which is harmless
    # (e.g. inside logsumexp for confident predictions) and would otherwise reject well-fitted models
    np.seterr(all='raise', under='ignore')
    warnings.simplefilter("error", OptimizeWarning)

    # get parameter bounds, the keys, and number of parameters
    param_bounds = program.define_parameters_and_bounds()
    param_keys = list(param_bounds.keys())
    parameter_count = len(param_bounds)

    # compute model complexity
    model_complexity = float(source_complexity + (parameter_count * 150))

    # numeric literals the search has tuned by hand: parameters in all but name
    hardcoded_constants, charged_constants = count_hardcoded_constants(program_path)
    effective_parameters = parameter_count + CONSTANT_WEIGHT * charged_constants

    # use JAX for this program until tracing fails once
    use_jax = True
    jax_failure = None
    n_jax_fits = 0

    # loop over experiments and store results
    results = []
    for i, exp in enumerate(experiments):

        # catch SciPy and Numpy warnings
        try:

            # extract parameters from the helper function defined by the LLM
            init_param_dict = program.get_init_param(i+1)

            # raise an error if there is a mismatch between the keys in the bounds and the keys in the initial parameters
            if param_keys != list(init_param_dict.keys()):
                raise ValueError(f"Mismatch between parameter keys in bounds and initial values"
                                 f"Bounds keys: {list(param_bounds.keys())}, Initial values keys: {list(init_param_dict.keys())}")

            # get initial parameter values in the correct order for optimization
            param_init = np.array([init_param_dict[k] for k in param_keys], dtype=np.float64)

            # fit, with gradients from JAX where possible and Powell for the rest of this program otherwise
            fit = fit_experiment(program, TRAIN_BLOCKS[i], exp['num_options'], param_bounds, param_init, seed=i,
                                 n_starts=n_starts, max_eval=max_eval, method=method, use_jax=use_jax)
            if fit['jax_used']:
                n_jax_fits += 1
            elif fit['jax_failure']:
                use_jax, jax_failure = False, fit['jax_failure']

            # score the fitted parameters on participants the fit never saw
            train_nll = fit['nll']
            if SCORE_BLOCKS[i] is not None:
                parameters = np.array([fit['params'][k] for k in param_keys], dtype=np.float64)
                score_nll = evalModel(parameters, SCORE_BLOCKS[i], exp['num_options'], param_keys, program.Model())
            else:
                score_nll = train_nll

            # BIC per trial: each effective parameter costs log(N) / (2N), with N the number of
            # trials the parameters were fitted on. Kept separate, so the reported nll stays bare.
            n_trials = TRAIN_BLOCKS[i]['n_free']
            penalty = effective_parameters * np.log(n_trials) / (2 * n_trials) if BIC_PENALTY else 0.0

        # catch all errors (and warnings raised as errors)
        except (FloatingPointError, OptimizeWarning, Exception) as e:
            # extract traceback
            _, _, tb = sys.exc_info()

            # get the last call, for exact location of the error
            last_call = traceback.extract_tb(tb)[-1]

            # create format of the error, and return as tuple
            error_type = type(e).__name__
            error_message = e
            error_location =  f"line {last_call.lineno}, in {last_call.name}: {last_call.line}"
            return {'Experiment': i+1, 'error_type': error_type, 'error_message': str(error_message), 'error_location':error_location}

        # store all the data in a list of dictionaries
        results.append({"Experiment": i+1, "combined_score": np.exp(-(score_nll + penalty)), "nll": score_nll,
                        "train_nll": train_nll, "bic_penalty": penalty,
                        "model_complexity": model_complexity, "parameters": fit['params']})

    # clear compiled JAX functions, as every program compiles new ones
    clear_jax_caches()

    # return the results as a dataframe, with the fitting method as metadata
    results = pd.DataFrame(results)
    results.attrs['jax_fit'] = n_jax_fits / len(experiments)
    results.attrs['hardcoded_constants'] = hardcoded_constants
    results.attrs['parameter_count'] = float(parameter_count)
    results.attrs['effective_parameters'] = effective_parameters
    results.attrs['jax_failure'] = jax_failure
    return results

# Stage-based evaluation for cascade evaluation
def evaluate(program_path, max_eval=20, stage=2, n_starts=1, method="powell"):
    """Evaluate a program by fitting its parameters on each experiment from `n_starts` starting points, with JAX gradients or Powell (`max_eval` evaluations per start)"""

    # model complexity is also reported for failed programs, as it is a feature dimension of the program database
    failed_complexity = safe_model_complexity(program_path)

    try:
        # Run a single trial with timeout
        result = run_model(program_path, max_eval, n_starts, method)

        # catch caused by wrong implementations
        if isinstance(result, dict):

            # create a single error message
            error_summary = f"[Exp {result['Experiment']}] {result['error_type']}: {result['error_message']} @ {result['error_location']}"

            # advice for the specific failure, which reaches the LLM with the program that caused it
            message = str(result['error_message'])
            specific = ""
            if "broadcast" in message:
                specific = ("Shapes: a per-game quantity has shape (n_games,) and must be written q[:, None] before it is "
                            "combined with an array of shape (n_games, num_options). Build per-option values with "
                            "np.stack([...], axis=1). predict must return (n_games, num_options).")
            elif "invalid index to scalar" in message or "too many indices" in message:
                specific = ("Indexing: every input is an array with one entry per game, not a number. Read the chosen "
                            "option with values[np.arange(n_games), choice] or np.sum(values * np.eye(num_options)[choice], axis=1).")
            elif "does not declare" in message or "Mismatch between parameter keys" in message:
                specific = ("Parameters: define_parameters_and_bounds, get_init_param and the params_dict that set_params "
                            "reads must use exactly the same keys. Adding a parameter means adding it in all three.")
            elif "has no attribute" in message:
                specific = "State: create every array the model uses in reset(), before predict or update refer to it."

            error_artifacts = {
                'errors:': error_summary,
                "suggestion": ("Ensure all required methods of the class Model exist, namely: __init__, set_params, reset, predict, update."
                               "All inputs are NumPy arrays with one entry per block, state should be arrays with one row per block, and predict must return an array of shape (n_blocks, num_options)."
                               "Ensure define_parameters_and_bounds and get_init_param exists, and that all these functions return the correct format."
                               "Also ensure math is handled correctly, and that bounds and initial guesses are properly set for the trainable parameters."
                               )
            }
            if specific:
                error_artifacts["how to fix this error"] = specific
            
            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "model_complexity": failed_complexity,
                    "error": result['error_message']
                },
                artifacts=error_artifacts
            )

        # Handle different result formats
        if isinstance(result, pd.DataFrame):
            if result.shape[1] == 7:
                pass
            else:
                error_artifacts = {
                    "error_type": "InvalidReturnFormat",
                    "error_message": f"Stage {stage}: Invalid result format, expected got n-results: {result.shape[1]}",
                    "suggestion": "Ensure predict returns an array of shape (n_blocks, num_options)."
                }
                
                return EvaluationResult(
                    metrics={
                        "runs_successfully": 0.0, 
                        "combined_score": 0.0,
                        "model_complexity": failed_complexity,
                        "error": "Invalid result format"
                    },
                    artifacts=error_artifacts
                )
        else:
            # print(f"Stage {stage}: Invalid result format, expected a dictionary (experiment, combined_score, nll, model_complexity, parameters) but got: {type(result)}")
            
            error_artifacts = {
                "error_type": "InvalidReturnType",
                "error_message": f"Stage {stage}: Function returned {type(result)}, expected dictionary (experiment, combined_score, nll, model_complexity, parameters)",
                "suggestion": "run_model() must return a dictionary with the items (experiment, combined_score, nll, model_complexity, parameters)."
            }
            
            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "model_complexity": failed_complexity,
                    "error": "Invalid result format"
                },
                artifacts=error_artifacts
            )

        # extract data from the results            
        nll_df = result[['Experiment', 'nll']]

        # Check if the result is valid
        if (
            any(np.isnan(nll_df['nll']))
            or any(np.isinf(nll_df['nll']))
        ):
            # print(f"Stage {stage} validation: Invalid result: {nll_df.to_string()}")
            
            error_artifacts = {
                "error_type": "InvalidResultValues",
                "error_message": f"Stage {stage}: Got invalid values: {nll_df.to_string()}",
                "suggestion": "Function returned NaN or infinite values. Check for division by zero, invalid math operations, or uninitialized variables"
            }
            
            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.1, 
                    "combined_score": 0.0,
                    "model_complexity": failed_complexity,
                    "error": "Invalid result values"
                },
                artifacts=error_artifacts
            )
        
        # compute mean combined score
        best_stats = result[['combined_score', 'nll', 'train_nll', 'bic_penalty', 'model_complexity']].mean().to_dict()

        # exp of the mean nll, not the mean of exp(-nll): the latter weights an improvement on an
        # easy experiment more than the same improvement on a hard one, and the hard experiments
        # (DriftingBandit) are where the models differ most
        best_stats['combined_score'] = float(np.exp(-(best_stats['nll'] + best_stats['bic_penalty'])))
        jax_fit, jax_failure = result.attrs.get('jax_fit', 0.0), result.attrs.get('jax_failure')
        hardcoded_constants = result.attrs.get('hardcoded_constants', 0.0)
        parameter_count = result.attrs.get('parameter_count', 0.0)
        effective_parameters = result.attrs.get('effective_parameters', parameter_count)
        result.drop(columns=['model_complexity'], inplace=True)
        
        # Per-experiment feedback deliberately excludes the validation numbers the score is based
        # on: reporting them back would let the search tune against the participants it is scored on.
        reported = result[['Experiment', 'train_nll', 'bic_penalty', 'parameters']]
        evaluation_artifacts = {
            "Experiment History (nll on the participants used for fitting)\n": reported.to_dict(orient='records')
        }
        if hardcoded_constants >= 10:
            evaluation_artifacts["hardcoded constants"] = (
                f"The Model class contains {hardcoded_constants:.0f} numeric literals other than 0 and 1, of which "
                f"{effective_parameters - parameter_count:.1f} parameters' worth are charged in the score: values like "
                "0.18 or 0.0022 are tuned across iterations and fit the data like parameters, so they are counted as "
                "such. Replace the ones that matter with trainable parameters, and delete the rest.")
        if method == "jax" and jax_failure:
            evaluation_artifacts["fitting"] = (f"Parameters were fitted with the slower Powell method, because JAX could not trace the program ({jax_failure}). "
                                               "Write array code that also works with jax.numpy: no in-place array assignment, no Python if on array values, "
                                               "no float() or int() on parameters, and create all state arrays in reset().")

        return EvaluationResult(
            metrics={
                "runs_successfully": 1.0,
                "combined_score": best_stats['combined_score'],
                "nll": best_stats['nll'],                      # validation nll, without the penalty
                "train_nll": best_stats['train_nll'],
                "generalisation_gap": best_stats['nll'] - best_stats['train_nll'],
                "bic_penalty": best_stats['bic_penalty'],
                "model_complexity": best_stats['model_complexity'],
                "parameters": parameter_count,
                "hardcoded_constants": hardcoded_constants,
                "effective_parameters": effective_parameters,
                "jax_fit": jax_fit
            },
            artifacts=evaluation_artifacts
        )
    except TimeoutError as e:
        # print(f"Stage {stage} evaluation timed out: {e}")
        
        error_artifacts = {
            "error_type": "TimeoutError",
            "error_message": f"Stage {stage}: Function execution exceeded the evaluator timeout",
            "suggestion": "Function is likely stuck in infinite loop or doing too much computation. Try reducing iterations or adding early termination conditions"
        }
        
        return EvaluationResult(
            metrics={
                "runs_successfully": 0.0, 
                "combined_score": 0.0,
                "model_complexity": failed_complexity,
                "error": "Timeout"
            },
            artifacts=error_artifacts
        )
    except IndexError as e:
        # Specifically handle IndexError which often happens with early termination checks
        # print(f"Stage {stage} evaluation failed with IndexError: {e}")
        # print("This is likely due to a list index check before the list is fully populated.")
        
        error_artifacts = {
            "error_type": "IndexError",
            "error_message": f"Stage {stage}: {str(e)}",
            "suggestion": "List index out of range - likely accessing empty list or wrong index. Check list initialization and bounds"
        }
        
        return EvaluationResult(
            metrics={
                "runs_successfully": 0.0, 
                "combined_score": 0.0,
                "model_complexity": failed_complexity,
                "error": f"IndexError: {str(e)}"
            },
            artifacts=error_artifacts
        )
    except Exception as e:
        # print(f"Stage {stage} evaluation failed: {e}")
        # print(traceback.format_exc())
        
        error_artifacts = {
            "error_type": type(e).__name__,
            "error_message": f"Stage {stage}: {str(e)}",
            "full_traceback": traceback.format_exc(),
            "suggestion": "Unexpected error occurred. Check the traceback for specific issue"
        }
        
        return EvaluationResult(
            metrics={
                "runs_successfully": 0.0, 
                "combined_score": 0.0,
                "model_complexity": failed_complexity,
                "error": str(e)
            },
            artifacts=error_artifacts
        )

# define the cascade evaluation functions with differing amounts of times to evaluate the program for (due to non-determinism of minimize)
def evaluate_stage1(program_path):
    return evaluate(program_path, 3, 1)

def evaluate_stage2(program_path):
    return evaluate(program_path, FIT_MAXFEV, 2, FIT_STARTS, FIT_METHOD)

if __name__ == "__main__":
    print(evaluate_stage2(f"{prog_path}initial_program.py")) # just for testing, the other print statements are also purely for testing purposes