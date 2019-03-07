import jax.numpy as np
import numpy.random as rnd

from jax import jit

import logging
# logging.basicConfig(level=logging.DEBUG)

"""
TODO
- want to make a new version the multi-step simulation that chooses actions (like mcts)
- want to use freq to define options make them avaliable as actions
- want to extend to cts actions (will not be trivial...!)
- want to reuse the previously calculated plans!? (only possible if they actually arrived in the right state)
- use the mpc searches to backup and aid value estimates!? (but this assumes they are reliable...)
- use jit compilation to make things faster!?
"""

def onehot(idx, N):
    return np.eye(N)[idx]

def simulation(transition_fn, init_s, pis):
    """
    Args:
        transition_fn (callable):
        init_s (np.array):the init state. [dims]
        pis (np.array): a matrix of actions [time x batch x dims]

    Returns:
        (np.array): the final states [batch x dims]
    """
    B = pis.shape[1]
    s = np.vstack([init_s for _ in range(B)]).reshape((B, -1))
    # TODO collect all states. so we can use reward + discounts!?
    for a in pis:
        s = transition_fn(s, a.reshape((B, -1))).reshape((B, -1))
    return s

def whiten(x):
    return (x - np.mean(x))/np.sqrt(np.var(x) + 1e-8)

def mpc(s_init, transition_fn, n_actions, T, N, value_fn=None, reward_fn=None, gamma=0.9):
    pis = np.stack([onehot(rnd.randint(0, n_actions, (T,)), n_actions) for _ in range(N)], axis=1)
    # run pis in parallel
    # TODO wrap in jit
    states = simulation(transition_fn, s_init, pis)

    # learned value fn
    # if we are only using the last then we dont need to store the others!
    # this reminds me more of AlphaGO?? but mcts would avg and backup?!
    vs = whiten(value_fn(states))
    return sample_action(pis, vs[0])

def discounted_return(rewards, discount):
    # TODO faster way to do this? some sort of conv?
    return np.sum([discount**i * r for i, r in enumerate(rewards)])

def sample(logits):  # BUG not working with rnd
    g = - np.log(-np.log(rnd.random(logits.shape)))
    return np.argmax(logits + g)

def sample_action(pis, evals):
    idx = sample(evals)  # bigger is more likely. TODO what about adding memory / momentum here?
    return pis[idx][0]  # only take the first action.
