import importlib.util
import numpy as np
from openevolve.evaluation_result import EvaluationResult
import scipy
import pandas as pd
from scipy.optimize import minimize, OptimizeWarning
import warnings
import traceback
import sys
import ast
import os

# ------ JAX Setup ------
# one CPU thread per evaluation process, as OpenEvolve already runs many evaluations in parallel (must be set before importing jax)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as jax_logsumexp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

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

def load_blocks(data):
    """
    Reshape trial rows into arrays of shape (n_blocks, n_trials), with one block per participant and game.

    All experiments contain the data columns:
    participant:    int32   - The participant ID
    game:           int32   - The number of the game, starts at 1
    horizon:        int32   - The number of trials per game, -1 if not known by the participant
    trial:          int32   - The trial number, resets every game and starts at 1
    forced:         int32   - Either 0 or 1, with 1 representing a non-trial, which won't be taken into account during computation of nll
    human_choice:   int32   - The human choice that has to be modelled
    reward:         int32   - The reward as a consequence of the human_choice
    hazard_rate:    int32   - Indicates the degree of abrupt expected point change, it ranges from 0-10, with 1 representing a 10% change

    Blocks shorter than the longest block are padded at the end with forced trials (choice 0, reward 0).
    """
    data = np.asarray(data)

    # identify blocks and the position of each row within its block (rows keep their order within a block)
    _, block = np.unique(data[:, :2].astype(np.int64), axis=0, return_inverse=True)
    block = block.ravel()
    counts = np.bincount(block)
    order = np.argsort(block, kind='stable')
    position = np.empty(len(data), dtype=np.int64)
    position[order] = np.arange(len(data)) - np.repeat(np.cumsum(counts) - counts, counts)
    n_blocks, n_trials = len(counts), int(counts.max())

    def grid(column, fill, dtype):
        values = np.full((n_blocks, n_trials), fill, dtype=dtype)
        values[block, position] = data[:, column]
        return values

    # horizon and hazard rate are constant within a block, padding repeats the block's first value
    first_row = order[np.cumsum(counts) - counts]
    horizon = np.repeat(data[first_row, 2].astype(np.int64)[:, None], n_trials, axis=1)
    hazard_rate = np.repeat(data[first_row, 7].astype(np.float64)[:, None], n_trials, axis=1)
    trial = np.repeat(np.arange(1, n_trials + 1, dtype=np.int64)[None, :], n_blocks, axis=0)
    trial[block, position] = data[:, 3]

    blocks = {
        'horizon': horizon,
        'hazard_rate': hazard_rate,
        'trial': trial,
        'forced': grid(4, 1, np.int64),
        'choice': grid(5, 0, np.int64),
        'reward': grid(6, 0, np.float64),
    }

    # make inputs read-only, so programs cannot modify the data in place
    for values in blocks.values():
        values.flags.writeable = False

    blocks['free'] = blocks['forced'] == 0
    blocks['n_free'] = int(blocks['free'].sum())
    blocks['n_blocks'], blocks['n_trials'] = n_blocks, n_trials
    return blocks

TRAIN_BLOCKS = [load_blocks(np.load(os.path.join(data_path, exp['name'], f"struc_Train_{exp['experiment']}.npy"))) for exp in experiments]

# ------ Fitting Budget ------
# Powell evaluations per start and number of starts per experiment for the full (stage 2) evaluation.
# The first start uses the program's own initial values, the others are random points within the bounds.
FIT_MAXFEV = int(os.environ.get("COGMOD_FIT_MAXFEV", 1000))
FIT_STARTS = int(os.environ.get("COGMOD_FIT_STARTS", 5))

# Fitting method for the full evaluation: "jax" fits with gradients (L-BFGS-B, gradients from JAX) and falls back to Powell
# for programs that JAX cannot trace; "powell" always uses Powell.
FIT_METHOD = os.environ.get("COGMOD_FIT_METHOD", "jax")
JAX_MAXITER = int(os.environ.get("COGMOD_JAX_MAXITER", 500))


def compute_source_complexity(program_path):
    """Compute a simple AST-based complexity score for the source file."""
    with open(program_path, "r", encoding="utf-8", errors="replace") as source_file:
        source = source_file.read()

    tree = ast.parse(source, filename=program_path)
    complexity = sum(1 for _ in ast.walk(tree))
    return complexity

def safe_model_complexity(program_path):
    """Model complexity for failed programs, so OpenEvolve can still place them in the feature grid."""
    try:
        return float(compute_source_complexity(program_path))
    except Exception:
        return 0.0

def evalModel(x, blocks, num_op, param_keys, model):
    """
    Evaluate the model on all blocks of an experiment at once. First, set the parameters. Second, reset the latent state of all blocks.
    Third, step through the trial positions, predicting logits and updating the model for all blocks simultaneously.
    """
    # create the parameter dictionary from the static keys and dynamic values
    params_dict = dict(zip(param_keys, x))

    # create bare-bones model object
    model.set_params(num_op, params_dict)

    # initialize the latent state of all blocks
    model.reset(blocks['horizon'][:, 0], blocks['hazard_rate'][:, 0])

    nll_sum = 0.0
    for t in range(blocks['n_trials']):
        trial = blocks['trial'][:, t]
        horizon = blocks['horizon'][:, t]
        hazard_rate = blocks['hazard_rate'][:, t]
        forced = blocks['forced'][:, t]

        # Predict action logits for all blocks
        logits = np.asarray(model.predict(trial, horizon, hazard_rate, forced), dtype=np.float64)
        if logits.shape != (blocks['n_blocks'], num_op):
            raise ValueError(f"predict returned logits of shape {logits.shape}, expected (n_blocks, num_options) = {(blocks['n_blocks'], num_op)}")

        # accumulate nll over free-choice trials
        free = blocks['free'][:, t]
        if free.any():
            free_logits = logits[free]
            nll_sum += np.sum(scipy.special.logsumexp(free_logits, axis=1) - free_logits[np.arange(len(free_logits)), blocks['choice'][free, t]])

        # Update model based on human choice and reward
        model.update(trial, horizon, hazard_rate, forced, blocks['choice'][:, t], blocks['reward'][:, t])

    # compute nll
    nll = nll_sum / blocks['n_free']

    # set best_nll and best_params as global so they are consistent across functions
    global best_nll, best_params

    # store best nll and params
    if nll < best_nll:
        best_nll = nll
        best_params = params_dict

    return nll

BLOCK_FIELDS = ['trial', 'horizon', 'hazard_rate', 'forced', 'choice', 'reward']
JAX_BLOCKS = {}

def jax_blocks(i):
    """Training blocks of experiment i as JAX arrays with time first, shape (n_trials, n_blocks), for jax.lax.scan"""
    if i not in JAX_BLOCKS:
        blocks = TRAIN_BLOCKS[i]
        xs = {k: jnp.asarray(np.ascontiguousarray(blocks[k].T)) for k in BLOCK_FIELDS}
        xs['free'] = jnp.asarray(blocks['free'].T.astype(np.float64))
        JAX_BLOCKS[i] = xs
    return JAX_BLOCKS[i]

class numpy_as_jax:
    """Temporarily point the program's `np` to jax.numpy, so the same program code can be traced and differentiated"""
    def __init__(self, program):
        self.program = program
    def __enter__(self):
        self.saved = self.program.__dict__.get('np')
        self.program.np = jnp
    def __exit__(self, *exc):
        self.program.np = self.saved

def make_jax_value_and_grad(program, i, num_op, param_keys):
    """
    Build a compiled function returning the mean nll over free-choice trials and its gradient with respect to the parameters.
    The trial loop runs in jax.lax.scan, with every JAX array attribute of the model after reset() as the scan state.
    """
    blocks = TRAIN_BLOCKS[i]
    horizon, hazard_rate, n_blocks, n_free = jnp.asarray(blocks['horizon'][:, 0]), jnp.asarray(blocks['hazard_rate'][:, 0]), blocks['n_blocks'], blocks['n_free']

    def loss(x, xs):
        model = program.Model()
        model.set_params(num_op, {k: x[j] for j, k in enumerate(param_keys)})
        model.reset(horizon, hazard_rate)
        state = {k: v for k, v in vars(model).items() if isinstance(v, jax.Array)}
        static = {k: v for k, v in vars(model).items() if k not in state}

        def step(state, s):
            step_model = program.Model.__new__(program.Model)
            step_model.__dict__.update(static)
            step_model.__dict__.update(state)
            logits = jnp.asarray(step_model.predict(s['trial'], s['horizon'], s['hazard_rate'], s['forced']))
            if logits.shape != (n_blocks, num_op):
                raise ValueError(f"predict returned logits of shape {logits.shape}, expected (n_blocks, num_options) = {(n_blocks, num_op)}")
            chosen = jnp.take_along_axis(logits, s['choice'][:, None], axis=1)[:, 0]
            nll = jnp.sum((jax_logsumexp(logits, axis=1) - chosen) * s['free'])
            step_model.update(s['trial'], s['horizon'], s['hazard_rate'], s['forced'], s['choice'], s['reward'])
            return {k: jnp.asarray(getattr(step_model, k)) for k in state}, nll

        _, nlls = jax.lax.scan(step, state, xs)
        return jnp.sum(nlls) / n_free

    return jax.jit(jax.value_and_grad(loss))

def fit_with_jax(program, i, num_op, param_keys, param_bounds, starts):
    """
    Fit the parameters with L-BFGS-B from every start, using gradients from JAX. Returns the optimum of every start that converged to a finite nll.
    Raises an exception if the program cannot be traced by JAX, so the caller can fall back to Powell.
    """
    xs = jax_blocks(i)
    with numpy_as_jax(program):
        value_and_grad = make_jax_value_and_grad(program, i, num_op, param_keys)

        def fun(x):
            value, grad = value_and_grad(jnp.asarray(x, dtype=jnp.float64), xs)
            return float(value), np.asarray(grad, dtype=np.float64)

        # compile and check the first start; tracing errors propagate to the caller
        value, grad = fun(starts[0])

        optima = []
        for x0 in starts:
            try:
                result = minimize(fun, x0, jac=True, method='L-BFGS-B', bounds=list(param_bounds.values()), options={'maxiter': JAX_MAXITER})
            # a numerical failure from one start only discards that start
            except (FloatingPointError, ValueError, OverflowError, ZeroDivisionError):
                continue
            if np.isfinite(result.fun) and np.all(np.isfinite(result.x)):
                optima.append(result.x)

    if not optima:
        raise RuntimeError("JAX fitting found no finite optimum")
    return optima

def run_model(program_path, max_eval, n_starts=1, method="powell"):
    """
    This function is called once for each iteration by OpenEvolve, which includes:
    1. training the model by optimizing the adjustable parameters.
    2. returning feedback metrics.
    """

    # load program
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

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

    # construct the model
    model = program.Model()

    # use JAX for this program until tracing fails once
    use_jax = method == "jax" and JAX_AVAILABLE
    jax_failure = None if JAX_AVAILABLE or method != "jax" else "jax is not installed"
    n_jax_fits = 0

    # loop over experiments and store results
    results = []
    for i, exp in enumerate(experiments):

        # set best_nll and best_params as global so they are consistent across functions
        global best_nll, best_params
        best_nll = np.inf
        best_params = {}

        # get data for this experiment
        train_data = TRAIN_BLOCKS[i]

        # get number of options
        num_options = exp['num_options']

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

            # starting points: the program's own initial values, followed by random points within the bounds (deterministic per experiment)
            rng = np.random.default_rng(i)
            starts = [param_init] + [np.array([rng.uniform(lo, hi) if lo is not None and hi is not None and np.isfinite([lo, hi]).all() else init
                                               for (lo, hi), init in zip(param_bounds.values(), param_init)], dtype=np.float64)
                                     for _ in range(n_starts - 1)]

            # fit with gradients from JAX, falling back to Powell for the rest of this program if JAX cannot trace it
            jax_optima = None
            if use_jax:
                try:
                    jax_optima = fit_with_jax(program, i, num_options, param_keys, param_bounds, starts)
                except Exception as e:
                    use_jax = False
                    jax_failure = f"{type(e).__name__}: {str(e)[:500]}"

            if jax_optima is not None:
                # score the JAX optima with the NumPy evaluation, so nll values and errors match the Powell path
                n_jax_fits += 1
                numerical_errors = []
                for x in jax_optima:
                    try:
                        evalModel(x, train_data, num_options, param_keys, model)
                    # an optimum that is numerically unstable in NumPy is discarded, unless all optima are
                    except (FloatingPointError, OverflowError, ZeroDivisionError) as e:
                        numerical_errors.append(e)
                if len(numerical_errors) == len(jax_optima):
                    raise numerical_errors[0]
            else:
                # run optimization of trainable parameters from the program's own initial values
                minimize(
                    evalModel,
                    x0=param_init,
                    args=(train_data, num_options, param_keys, model),
                    options={'maxfev': max_eval},
                    method='Powell',
                    bounds=list(param_bounds.values())
                )

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

        # additional random starts for Powell, best result over all starts is kept
        for x0 in (starts[1:] if jax_optima is None else []):
            try:
                minimize(
                    evalModel,
                    x0=x0,
                    args=(train_data, num_options, param_keys, model),
                    options={'maxfev': max_eval},
                    method='Powell',
                    bounds=list(param_bounds.values())
                )
            # a numerical failure from a random start only discards that start
            except (FloatingPointError, OptimizeWarning, ValueError, OverflowError, ZeroDivisionError):
                continue

        # store all the data in a list of dictionaries
        results.append({"Experiment": i+1, "combined_score": np.exp(-best_nll), "nll": best_nll, "model_complexity": model_complexity, "parameters": best_params})
            
    # clear compiled JAX functions, as every program compiles new ones
    if JAX_AVAILABLE:
        jax.clear_caches()

    # return the results as a dataframe, with the fitting method as metadata
    results = pd.DataFrame(results)
    results.attrs['jax_fit'] = n_jax_fits / len(experiments)
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

            error_artifacts = {
                'errors:': error_summary,
                "suggestion": ("Ensure all required methods of the class Model exist, namely: __init__, set_params, reset, predict, update."
                               "All inputs are NumPy arrays with one entry per block, state should be arrays with one row per block, and predict must return an array of shape (n_blocks, num_options)."
                               "Ensure define_parameters_and_bounds and get_init_param exists, and that all these functions return the correct format."
                               "Also ensure math is handled correctly, and that bounds and initial guesses are properly set for the trainable parameters."
                               )
            }
            
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
            if result.shape[1] == 5:
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
        best_stats = result[['combined_score', 'nll', 'model_complexity']].mean().to_dict()
        jax_fit, jax_failure = result.attrs.get('jax_fit', 0.0), result.attrs.get('jax_failure')
        result.drop(columns=['model_complexity'], inplace=True)
        
        # Add artifacts for successful stage 1
        evaluation_artifacts = {
            "Experiment History\n": result.to_dict(orient='records')
        }
        if method == "jax" and jax_failure:
            evaluation_artifacts["fitting"] = (f"Parameters were fitted with the slower Powell method, because JAX could not trace the program ({jax_failure}). "
                                               "Write array code that also works with jax.numpy: no in-place array assignment, no Python if on array values, "
                                               "no float() or int() on parameters, and create all state arrays in reset().")

        return EvaluationResult(
            metrics={
                "runs_successfully": 1.0,
                "combined_score": best_stats['combined_score'],
                "nll": best_stats['nll'],
                "model_complexity": best_stats['model_complexity'],
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