"""
Shared machinery for evaluating and fitting the cognitive models.

Used by OpenEvolve/evaluator.py (scoring candidates during the search) and by
EvolvedModelInference/evaluator.py (fitting selected programs and scoring them on
the test sets), so both fit models in exactly the same way.

A model processes all blocks (participant x game) of an experiment at once:
every input is an array with one entry per block, and its latent state is arrays
with one row per block. Parameters are fitted with gradients from JAX where the
program can be traced, and with Powell otherwise.
"""
import os
import numpy as np
import scipy
from scipy.optimize import minimize

# one CPU thread per process, as many evaluations run in parallel (must be set before importing jax)
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as jax_logsumexp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# ------ Fitting Budget ------
# Powell evaluations per start, number of starts per experiment, and the fitting method:
# "jax" fits with gradients (L-BFGS-B) and falls back to Powell for programs JAX cannot trace.
FIT_MAXFEV = int(os.environ.get("COGMOD_FIT_MAXFEV", 1000))
FIT_STARTS = int(os.environ.get("COGMOD_FIT_STARTS", 5))
FIT_METHOD = os.environ.get("COGMOD_FIT_METHOD", "jax")
JAX_MAXITER = int(os.environ.get("COGMOD_JAX_MAXITER", 500))

BLOCK_FIELDS = ['trial', 'horizon', 'hazard_rate', 'forced', 'choice', 'reward']


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
    'row_block' and 'row_position' give the grid cell of every input row, so results can be written back per trial.
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
    blocks['row_block'], blocks['row_position'] = block, position
    return blocks

def evalModel(x, blocks, num_op, param_keys, model, return_logits=False):
    """
    Evaluate the model on all blocks of an experiment at once. First, set the parameters. Second, reset the latent state of all blocks.
    Third, step through the trial positions, predicting logits and updating the model for all blocks simultaneously.

    Returns the mean nll over free-choice trials, and with return_logits also the logits of shape (n_blocks, n_trials, num_op).
    """
    # create the parameter dictionary from the static keys and dynamic values
    params_dict = dict(zip(param_keys, x))

    # create bare-bones model object
    model.set_params(num_op, params_dict)

    # initialize the latent state of all blocks
    model.reset(blocks['horizon'][:, 0], blocks['hazard_rate'][:, 0])

    all_logits = np.empty((blocks['n_blocks'], blocks['n_trials'], num_op)) if return_logits else None

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
        if return_logits:
            all_logits[:, t, :] = logits

        # accumulate nll over free-choice trials
        free = blocks['free'][:, t]
        if free.any():
            free_logits = logits[free]
            nll_sum += np.sum(scipy.special.logsumexp(free_logits, axis=1) - free_logits[np.arange(len(free_logits)), blocks['choice'][free, t]])

        # Update model based on human choice and reward
        model.update(trial, horizon, hazard_rate, forced, blocks['choice'][:, t], blocks['reward'][:, t])

    nll = nll_sum / blocks['n_free']
    return (nll, all_logits) if return_logits else nll

# ------ JAX Fitting ------

class numpy_as_jax:
    """Temporarily point the program's `np` to jax.numpy, so the same program code can be traced and differentiated"""
    def __init__(self, program):
        self.program = program
    def __enter__(self):
        self.saved = self.program.__dict__.get('np')
        self.program.np = jnp
    def __exit__(self, *exc):
        self.program.np = self.saved

def blocks_to_jax(blocks):
    """Block arrays as JAX arrays with time first, shape (n_trials, n_blocks), for jax.lax.scan"""
    if 'jax_xs' not in blocks:
        xs = {k: jnp.asarray(np.ascontiguousarray(blocks[k].T)) for k in BLOCK_FIELDS}
        xs['free'] = jnp.asarray(blocks['free'].T.astype(np.float64))
        blocks['jax_xs'] = xs
    return blocks['jax_xs']

def make_jax_value_and_grad(program, blocks, num_op, param_keys):
    """
    Build a compiled function returning the mean nll over free-choice trials and its gradient with respect to the parameters.
    The trial loop runs in jax.lax.scan, with every JAX array attribute of the model after reset() as the scan state.
    """
    horizon, hazard_rate = jnp.asarray(blocks['horizon'][:, 0]), jnp.asarray(blocks['hazard_rate'][:, 0])
    n_blocks, n_free = blocks['n_blocks'], blocks['n_free']

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

def fit_with_jax(program, blocks, num_op, param_keys, param_bounds, starts):
    """
    Fit the parameters with L-BFGS-B from every start, using gradients from JAX. Returns the optimum of every start that converged to a finite nll.
    Raises an exception if the program cannot be traced by JAX, so the caller can fall back to Powell.
    """
    xs = blocks_to_jax(blocks)
    with numpy_as_jax(program):
        value_and_grad = make_jax_value_and_grad(program, blocks, num_op, param_keys)

        def fun(x):
            value, grad = value_and_grad(jnp.asarray(x, dtype=jnp.float64), xs)
            return float(value), np.asarray(grad, dtype=np.float64)

        # compile and check the first start; tracing errors propagate to the caller
        fun(starts[0])

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

# ------ Fitting ------

def make_starts(param_bounds, param_init, n_starts, seed):
    """The program's own initial values, followed by random points within the bounds (deterministic per experiment)"""
    rng = np.random.default_rng(seed)
    return [param_init] + [np.array([rng.uniform(lo, hi) if lo is not None and hi is not None and np.isfinite([lo, hi]).all() else init
                                     for (lo, hi), init in zip(param_bounds.values(), param_init)], dtype=np.float64)
                           for _ in range(n_starts - 1)]

def fit_experiment(program, blocks, num_op, param_bounds, param_init, seed,
                   n_starts=FIT_STARTS, max_eval=FIT_MAXFEV, method=FIT_METHOD, use_jax=True):
    """
    Fit one experiment and return {'nll', 'params', 'jax_used', 'jax_failure'}.

    Every candidate optimum is scored with the NumPy evaluation, so nll values and errors are
    the same whichever fitting method was used.
    """
    param_keys = list(param_bounds.keys())
    model = program.Model()
    starts = make_starts(param_bounds, param_init, n_starts, seed)

    best = {'nll': np.inf, 'params': {}}
    def score(x):
        nll = evalModel(x, blocks, num_op, param_keys, model)
        if nll < best['nll']:
            best['nll'], best['params'] = nll, dict(zip(param_keys, x))
        return nll

    jax_failure = None
    jax_optima = None
    if method == "jax" and use_jax and JAX_AVAILABLE:
        try:
            jax_optima = fit_with_jax(program, blocks, num_op, param_keys, param_bounds, starts)
        except Exception as e:
            jax_failure = f"{type(e).__name__}: {str(e)[:500]}"
    elif method == "jax" and not JAX_AVAILABLE:
        jax_failure = "jax is not installed"

    if jax_optima is not None:
        numerical_errors = []
        for x in jax_optima:
            try:
                score(x)
            # an optimum that is numerically unstable in NumPy is discarded, unless all optima are
            except (FloatingPointError, OverflowError, ZeroDivisionError) as e:
                numerical_errors.append(e)
        if len(numerical_errors) == len(jax_optima):
            raise numerical_errors[0]
    else:
        # the program's own initial values must work; failures there are reported to the caller
        minimize(score, x0=starts[0], options={'maxfev': max_eval}, method='Powell', bounds=list(param_bounds.values()))
        for x0 in starts[1:]:
            try:
                minimize(score, x0=x0, options={'maxfev': max_eval}, method='Powell', bounds=list(param_bounds.values()))
            # a numerical failure from one start only discards that start
            except (FloatingPointError, ValueError, OverflowError, ZeroDivisionError):
                continue

    return {'nll': best['nll'], 'params': best['params'], 'jax_used': jax_optima is not None, 'jax_failure': jax_failure}

def clear_jax_caches():
    """Every program compiles new functions, so the caches are cleared between programs"""
    if JAX_AVAILABLE:
        jax.clear_caches()

def load_program(program_path):
    """Import a program file as a module"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return program
