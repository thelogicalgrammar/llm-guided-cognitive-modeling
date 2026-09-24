# Centaur's Rescorla-Wagner model (RescorlaWagner/CogModel.py) in the vectorised interface:
# separate learning rates for better- and worse-than-expected outcomes, stickiness and an
# information bonus, with values reset per game. Written without in-place array updates, so
# it can be fitted with gradients like any evolved program.
#
# Scored through EvolvedModelInference/evaluator.py it reproduces the published test nll
# (HorizonSomer 0.4079, HorizonWaltz 0.2947, TwoBandit exp2 0.3531), which makes it the
# like-for-like baseline for programs from a search.
import numpy as np

class Model:
    def set_params(self, num_options, p):
        self.n = num_options
        self.ap, self.am, self.v0 = p["alpha_plus"], p["alpha_minus"], p["init_value"]
        self.beta, self.stick, self.info = p["beta"], p["stickiness"], p["info_bonus"]
    def reset(self, horizon, hazard_rate):
        b = horizon.shape[0]
        self.V = np.zeros((b, self.n)) + self.v0; self.prev = np.zeros((b, self.n)); self.counts = np.zeros((b, self.n))
    def predict(self, trial, horizon, hazard_rate, forced):
        return self.beta * self.V + self.stick * self.prev + self.info * self.counts
    def update(self, trial, horizon, hazard_rate, forced, choice, reward):
        onehot = np.eye(self.n)[choice]
        pe = reward - np.sum(self.V * onehot, axis=1)
        self.V = self.V + (np.where(pe >= 0, self.ap, self.am) * pe)[:, None] * onehot
        free = (forced == 0)[:, None]
        self.prev = onehot * free
        self.counts = self.counts + onehot * free

def define_parameters_and_bounds():
    return {"alpha_plus": (0.0, 1.0), "alpha_minus": (0.0, 1.0), "init_value": (-100.0, 100.0),
            "beta": (-10.0, 10.0), "stickiness": (-10.0, 10.0), "info_bonus": (-10.0, 10.0)}

def get_init_param(experiment):
    return {"alpha_plus": 0.5, "alpha_minus": 0.5, "init_value": 0.0, "beta": 0.01, "stickiness": 0.0, "info_bonus": 0.0}
