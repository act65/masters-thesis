import random
import jax.numpy as np

from jax import jit, grad, vmap

@jit
def sample(p, temperature=1.0):  # TODO need to check this...
    g = -np.log(-np.log(rnd.random(p.shape))) * temperature
    idx = np.argmax(np.log(p) + g, axis=-1)
    return idx

class MDP():
    def __init__(self, n_states, n_actions, discount=0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        P = rnd.random((n_states, n_states, n_actions))
        self.P = P/P.sum(axis=0, keepdims=True)
        self.r = rnd.standard_normal((n_states, n_actions))
        d0 = rnd.random((n_states, 1))
        self.d0 = d0/d0.sum(axis=0, keepdims=True)
        self.discount = discount

    def reset(self):
        self.current_state = sample(self.d0)

    def step(self, a):
        self.current_state = sample(self.P[:, self.current_state, a])
        return self.current_state, self.r[self.current_state, a]
