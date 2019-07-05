import numpy as np
import numpy.random as rnd
import collections

mdp = collections.namedtuple('mdp', ['S', 'A', 'P', 'r', 'discount', 'd0'])

def build_random_mdp(n_states, n_actions, discount):
    P = rnd.random((n_states, n_states, n_actions))
    r = rnd.standard_normal((n_states, n_actions))
    d0 = rnd.random((n_states, 1))
    return mdp(n_states, n_actions, P/P.sum(axis=0, keepdims=True), r, discount, d0/d0.sum(axis=0, keepdims=True))


mdp = build_random_mdp(2,2,0.9)
print(np.sum(mdp.P[:,1,0]))
