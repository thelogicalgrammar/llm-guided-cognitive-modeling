# EVOLVE-BLOCK START
import numpy as np


class Model:
    # Avoid unnecessary randomness inside predict/update.
    # Prefer deterministic mechanisms unless stochasticity is essential.
    # Expensive random sampling significantly slows optimization.

    # All blocks (games) of an experiment are processed at once: every input is a NumPy array with one entry per block,
    # and latent state is stored as arrays with one row per block. Do not loop over blocks in Python.
    # The same code is also run with jax.numpy to fit parameters with gradients: create new arrays instead of assigning in place,
    # use np.where instead of Python if on arrays, and create all state arrays in reset().

    def __init__(self):
        """
        This initialization function is called once when the model is created.
        It is only meant to avoid initialize the model object multiple times.
        """
        pass

    def set_params(self, num_options, params_dict):
        """
        This function is called during parameter optimization per experiment, and should update the model's parameters based on the input dictionary.
        """
        self.num_options = num_options

        # extract parameters
        self.alpha = params_dict["alpha"]      # learning rate

    def reset(self, horizon, hazard_rate):
        # Initialize the latent state of all blocks here, e.g. values of shape (n_blocks, num_options).
        # horizon and hazard_rate are arrays of shape (n_blocks,). This is called before the first trial of every evaluation.
        self.n_blocks = len(horizon)

    def predict(self, trial, horizon, hazard_rate, forced):
        # Returns raw action logits of shape (n_blocks, num_options), do NOT apply softmax or convert to probabilities.
        logits = np.full((self.n_blocks, self.num_options), self.alpha)
        return logits

    def update(self, trial, horizon, hazard_rate, forced, choice, reward):
        # update state based on trial feedback, all inputs are arrays of shape (n_blocks,)
        pass


def define_parameters_and_bounds():
    """
    Define trainable parameters and their optimization bounds.
    Parameter definitions are shared across experiments, but parameter values are optimized independently per experiment.

    Returns:
    param_bounds_dictionary: Dictionary of parameter names and their (min, max) bounds for optimization
    """

    # define prameters and their bounds
    return {
        'alpha': (0.1, 2.0)    # alpha: Some parameter
    }

def get_init_param(experiment):
    """
    This function is called at the start of optimization to set the initial parameters of the model.
    Note, these initial parameters are matched against the bounds dictionary.
    An error will be raised if any additional parameters are added or some are unused for a particular experiment.
    """
    default_params = {'alpha': 1}

    # set initial parameters values depending on the experiment
    match experiment:
        case 1:
            return {'alpha': 1}
        case 2:
            return {'alpha': 1}
        case 3:
            return {'alpha': 1}
        case 4:
            return {'alpha': 1}
        case 5:
            return {'alpha': 1}
        case _:
            return default_params

# EVOLVE-BLOCK END
