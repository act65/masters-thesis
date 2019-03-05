import numpy as np
import numpy.random as rnd

import logging
logging.basicConfig(level=logging.DEBUG)

"""
TODO
- want to make a new version the multi-step simulation that chooses actions
- want to use some sort of byte pair / freq based encoding and add them to avaliable actions
- want to extend to cts actions (will not be trivial...!)
- want to reuse the previously calculated plans!?
"""

def onehot(idx, N):
    return np.eye(N)[idx]

def multi_step_transitions(transition_fn, s, pi):
    """
    Args:
        transition_fn (callable s x a -> s): takes a state and action and maps to a new state
        s (np.array): a vector of shape (n_states, )
        pi (list): a list of actions. actions are ints in range [0,n_actions-1]

    Returns:
        (list): a list of the intermediate states reached via pi
    """

    if len(pi) == 1:
        return [transition_fn(s, pi[0])]
    else:
        new_s = transition_fn(s, pi[0]).reshape((1, -1))
        return [new_s] + multi_step_transitions(transition_fn, new_s, pi[1:])

def discounted_return(rewards, discount):
    # faster way to do this? some sort of conv?
    return np.sum([discount**i * r for i, r in enumerate(rewards)])

def evaluate_policy_w_value(transition_fn, value_fn, gamma, s_init, policy):
    """
    Args:
        ...
    """
    states = multi_step_transitions(transition_fn, s_init, policy)

    # learned value fn
    # if we are only using the last then we dont need to store the others!
    return value_fn(states[-1])

def evaluate_policy_w_reward(transition_fn, reward_fn, gamma, s_init, policy):
    """
    Args:
        ...
    """
    states = multi_step_transitions(transition_fn, s_init, policy)
    ### setimate the value of an action seq as;
    # sum of rewards
    rewards = [reward_fn(s) for s in states] # TODO + value of final state
    return discounted_return(rewards, 0.9)

def rnd_policy_generator(n_actions, T, N):
    """
    Args:
        n_actions (int): the number of discrete actions
        T (int): the number of times steps to simulate
        N (int): the number of policies to generate

    Returns:
        (list): a list of actions. aka a policy
    """

    logging.info('{} steps total'.format(T*N))
    for i in range(N):
        yield rnd.randint(0, n_actions, (T,))

def simulate_policies(generator, evaluator):
    """
    Args:
        generator (generator): yields policies
        evaluator (callable): takes a policy and returns its value

    Returns:
        (tuple):
            (np.array) policies
            (np.array) their values
    """
    # TODO parallelise these calls. could use vmap!?
    # NOTE this is turning into something close to MCTS!?

    return tuple([np.array(v) for v in zip(*[(pi, evaluator(pi)) for pi in generator])])

def sample(logits):  # BUG not working with rnd
    g = - np.log(-np.log(rnd.random(logits.shape)))
    return np.argmax(logits + g)

def sample_action(pis, evals):
    idx = sample(evals)  # bigger is more likely. TODO what about adding memory / momentum here?
    return pis[idx][0]  # only take the first action.

def mpc(s_init, transition_fn, n_actions, T, N, value_fn=None, reward_fn=None, gamma=0.9):
    """
    A wrapper to handle planning.
    """
    if value_fn is not None:
        evaluator = lambda pi: evaluate_policy_w_value(transition_fn, value_fn, gamma, s_init, pi)
    elif reward_fn is not None:
        evaluator = lambda pi: evaluate_policy_w_reward(transition_fn, reward_fn, gamma, s_init, pi)
    else:
        raise ValueError('must provide a fn to evaluate. reward or value')

    generator = rnd_policy_generator(n_actions, T, N)
    pis, evals = simulate_policies(generator, evaluator)
    return sample_action(pis, evals)
