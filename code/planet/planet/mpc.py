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

def simulate(transition_fn, init_s, pis):
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
    states = [s]
    for a in pis:
        states.append(transition_fn(states[-1], a.reshape((B, -1))).reshape((B, -1)))
    return states

def whiten(x):
    return (x - np.mean(x))/np.sqrt(np.var(x) + 1e-8)

def rnd_policy_generator(n_actions, N, T):
    for _ in range(N):
        yield onehot(rnd.randint(0, n_actions, (T,)), n_actions)

def mpc(s_init, transition_fn, n_actions, T, N, value_fn=None, reward_fn=None, gamma=0.9):
    pis = np.stack(list(rnd_policy_generator(n_actions, N, T)), axis=1)
    # run pis in parallel
    # TODO wrap in jit
    states = simulate(transition_fn, s_init, pis)

    # learned value fn
    # if we are only using the last then we dont need to store the others!
    # this reminds me more of AlphaGO?? but mcts would avg and backup?!
    vs = whiten(value_fn(states[-1]))
    return sample_policy(pis, vs[0])[0]   # [0] only take the first action.

def discounted_return(rewards, discount):
    # TODO faster way to do this? some sort of conv?
    return np.sum([discount**i * r for i, r in enumerate(rewards)])

def sample(logits):
    g = - np.log(-np.log(rnd.random(logits.shape)))
    return np.argmax(logits + g)

def sample_policy(pis, evals):
    """
    Samples policies according to their value
    Args:
        pis(np.array): [time x batch x dims]
        evals (np.array): [batch]

    Returns:
        [time x dims]
    """
    idx = sample(evals)  # bigger is more likely. TODO what about adding memory / momentum here?
    return pis[:, idx, :]

def exploration_value(s, s_t):
    """
    Using planning. We could explore many possible future trajectories.
    And pick the one that leads us to the most different place!?
    max distance(current_state, future_state) --> but the transition fn might not be accurate there? (have seen the data)
    want the transition fn to genralise the change in actions. aka regular actions...

    but if one action tends to change the state more than another, then we will just keep picking the same action?
    is that such a bad thing?
    """
    return np.sum((s-s_t)**2, axis=-1)

def make_planner(transition_net, value_net):
    # !! this works nicely. way faster!
    @jit
    def mpc(s_init, transition_params, value_params, n_actions, T, N):
        pis = np.stack(list(rnd_policy_generator(n_actions, N, T)), axis=1)
        # run pis in parallel
        # TODO wrap in jit
        trans_fn = lambda s,a: transition_net.fn(transition_params, s, a)
        states = simulate(trans_fn, s_init, pis)

        # learned value fn
        # if we are only using the last then we dont need to store the others!
        # this reminds me more of AlphaGO?? but mcts would avg and backup?!
        value_fn = lambda s: value_net.fn(value_params, s)
        vs = whiten(value_fn(states[-1]))

        return sample_policy(pis, vs[0])[0]   # [0] only take the first action.

    return mpc
