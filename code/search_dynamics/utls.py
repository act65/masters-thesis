"""
Explore how the different search spaces effect the GD dynamics.
"""
import functools
import collections

import jax.numpy as np
from jax import grad, jit, jacrev

import numpy.random as rnd

mdp = collections.namedtuple('mdp', ['S', 'A', 'P', 'r', 'discount', 'd0'])

def build_random_mdp(n_states, n_actions, discount):
    P = rnd.random((n_states, n_states, n_actions))
    r = rnd.standard_normal((n_states, n_actions))
    d0 = rnd.random((n_states, 1))
    return mdp(n_states, n_actions, P/P.sum(axis=0, keepdims=True), r, discount, d0/d0.sum(axis=0, keepdims=True))

######################

"""
Various ways to parameterise these fns.
Want to be able to try different topologies!!
"""

def random_parameterised_matrix(n, m, d_hidden, n_hidden):
    glorot_init = lambda shape: (1/np.sqrt(shape[0] + shape[1]))*rnd.standard_normal(shape)
    cores = [glorot_init((d_hidden, d_hidden)) for _ in range(n_hidden)]
    cores = [glorot_init((n, d_hidden))] + cores + [glorot_init((d_hidden, m))]
    return cores

def combine_svd(u, s, vT):
    return np.dot(u, np.dot(np.diag(s), vT))

def random_reparameterisation(cores, i):
    assert i > 0 and i < len(cores)-1
    n = cores[i].shape[-1]
    m = cores[i+1].shape[0]
    assert n == m  # else not invertible...

    # M = rnd.standard_normal((n,m))
    # Mm1 = np.linalg.inv(M)
    # assert np.isclose(np.dot(M, Mm1), np.eye(n), atol=1e-4).all()
    # return cores[:i-1] + [np.dot(cores[i], M)] + [np.dot(Mm1, cores[i+1])] + cores[i+2:]

    u_i, s_i, vT_i = np.linalg.svd(cores[i])
    u_ip1, s_ip1, vT_ip1 = np.linalg.svd(cores[i+1])

    # X = np.dot(cores[i], cores[i+1])
    # Y = np.dot(combine_svd(u_i, s_i, np.dot(vT_i, u_ip1)), combine_svd(np.eye(n), s_ip1, vT_ip1))
    # print(np.isclose(X, Y, atol=1e-6).all())

    return (
        cores[:i] +
        [combine_svd(u_i, s_i, np.dot(vT_i, u_ip1))] +
        [combine_svd(np.eye(n), s_ip1, vT_ip1)] +
        cores[i+2:]
        )

def value(cores):
    return functools.reduce(np.dot, cores)

def pi(cores):
    M = np.abs(functools.reduce(np.dot, cores))
    return M/M.sum(axis=1, keepdims=True)

"""
Tools for simulating dyanmical systems.
"""

def isclose(x, y, atol=1e-8):
    if isinstance(x, np.ndarray):
        return np.isclose(x, y, atol=atol).all()
    elif isinstance(x, list):
        # return all(np.isclose(x[0], y[0], atol=1e-03).all() for i in range(len(x)))
        return np.isclose(value(x), value(y), atol=atol).all()
    elif isinstance(x, tuple) and isinstance(x[0], np.ndarray):
        return np.isclose(x[0], y[0], atol=atol).all()
    elif isinstance(x, tuple) and isinstance(x[0], list):
        return np.isclose(value(x[0]), value(y[0]), atol=atol).all()
    else:
        raise ValueError('wrong format')

def converged(l):
    if len(l)>10:
        if len(l)>10000:
            return True
        elif isclose(l[-1], l[-2]):
            return True
        else:
            False
    else:
        False

def solve(update_fn, init):
    xs = [init]
    x = init
    while not converged(xs):
        print('\rStep: {}'.format(len(xs)), end='', flush=True)
        x = update_fn(x)
        xs.append(x)
    return xs

"""
Some useful functions that will be repeately used.
- `value_functional`: evaluates a policy within a mdp
- `bellman_optimality_operator`: calculates a step of the bellman operator
- `state_visitation_distribution`: calculates the expected distribution over states given a mdp + policy
"""

def value_functional(P, r, pi, discount):
    """
    V = r_{\pi} + \gamma P_{\pi} V
      = (I-\gamma P_{\pi})^{-1}r_{\pi}

    Args:
        P (np.ndarray): [n_states x n_states x n_actions]
        r (np.ndarray): [n_states x n_actions]
        pi (np.ndarray): [n_states x n_actions]
        discount (float): the temporal discount value
    """
    n = P.shape[-1]
    P_pi = np.einsum('ijk,jk->ij', P, pi)
    r_pi = np.einsum('jk,jk->jk', r, pi)

    assert np.isclose(pi/pi.sum(axis=1, keepdims=True), pi).all()
    assert np.isclose(P_pi/P_pi.sum(axis=1, keepdims=True), P_pi).all()

    return np.dot(np.linalg.inv(np.eye(n) - discount*P_pi), r_pi)

def bellman_optimality_operator(P, r, Q, discount):
    """
    Args:
        P (np.ndarray): [n_states x n_states x n_actions]
        r (np.ndarray): [n_states x n_actions]
        Q (np.ndarray): [n_states x n_actions]
        discount (float): the temporal discount value
    """
    return r + discount * np.einsum('ijk,i->jk', P, np.argmax(Q, axis=1))

def state_visitation_distribution(P, pi, discount, d0):
    """
    Ps + yPPs + yyPPPs + yyyPPPPs ...
    (P + yPP + yyPPP + yyyPPPP ... )s
    (I - yP)^-1 s
    """
    n = d0.size
    P_pi = np.einsum('ijk,jk->ij', P, pi)

    # check we have been given normalised distributions
    assert np.isclose(d0/d0.sum(), d0).all()
    if np.isclose(P_pi/P_pi.sum(axis=1, keepdims=True), P_pi, atol=1e-8).all():
        print(P_pi.sum(axis=1, keepdims=True))
        raise ValueError('P_pi is not normalised')

    return (1-discount)*np.dot(np.linalg.inv(np.eye(n) - discount * P_pi), d0)

"""
Value iteration;
- Q_t+1 = Q_t + lr . (TQ_t - Q)
- and a parameterised version. where Q is a fn of some params.
"""

def value_iteration(mdp, lr):
    T = lambda Q: bellman_optimality_operator(mdp.P, mdp.r, Q, mdp.discount)
    U = lambda Q: Q + lr * (T(Q) - Q)
    return jit(U)

def parameterised_value_iteration(mdp, lr):
    T = lambda Q: bellman_optimality_operator(mdp.P, mdp.r, Q, mdp.discount)
    TD = lambda cores: T(value(cores)) - value(cores)
    dVdw = jacrev(value)

    def update_fn(cores):
        delta = TD(cores)
        grads = [np.einsum('ij,ijkl->kl', delta, dc) for dc in dVdw(cores)]
        # TODO attempt to understand the properties of dc. and its relation to K
        return [c+lr*g for c, g in zip(cores, grads)]
    return jit(update_fn)

# NOTE this isnt really value iteration...!?
# def parameterised_expected_value_iteration(mdp, lr):
#     pi = lambda cores: softmax(value(cores), axis=1)
#     d = lambda cores: state_visitation_distribution(mdp.P, mpi(pi(cores)), mdp.discount, mdp.d0)
#     EV = lambda cores: np.sum(d(cores) * softmax(value(cores), axis=1))
#     def update_fn(cores):
#         dEVdw = grad(EV)
#         return [c+lr*g for c, g in zip(cores, dEVdw(cores))]
#     return update_fn

######################
# Neural tangent kernel and ...
######################

"""
Inspired by Towards Characterizing Divergence in Deep Q-Learning
https://arxiv.org/abs/1903.08894
"""

def adjusted_value_iteration(mdp, lr, D, K):
    T = lambda Q: bellman_optimality_operator(mdp.P, mdp.r, Q.reshape((-1, 1)), mdp.discount)
    U = lambda Q: Q + lr * np.dot(K, np.dot(D, T(Q) - Q))
    return jit(U)

# def corrected_value_iteration(mdp, lr):
#     T = lambda theta: mdp.r + mdp.discount * np.argmax(mdp.P * Q(w))
#     dQdw = lambda w: grad(Q)
#     Km1 = lambda w: np.linalg.inv(np.dot(dQdw(w).T, dQdw(w)))
#     U = lambda w: w + lr * np.dot(dQdw(w), np.dot(Km1, T(Q(w) - Q(w))))
#     return U

######################
# Policy iteration
######################

# def policy_iteration(mdp, lr):
#     V = lambda pi: value_functional(mdp.P, mdp.r, mpi(pi), mdp.discount)
#     delta = lambda pi: np.dot(dpidw, 1/(pi*np.log(mdp.A)))
#     U = lambda pi: pi + lr * np.dot(V, delta(pi))
#     return U

def softmax(x, axis=-1):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

def policy_gradient_iteration_logits(mdp, lr):
    # d/dlogits V = E_{\pi}[V] = E[V . d/dlogit log \pi]
    dpi_dlogit_ = jacrev(softmax)
    dlogpi_dlogit_ = jacrev(lambda logits: np.log(softmax(logits)))
    def update_fn(logits):
        V = value_functional(mdp.P, mdp.r.reshape((-1, 1)), mpi(softmax(logits)), mdp.discount)
        dpi_dlogit = np.sum(dpi_dlogit_(logits), axis=[0, 1])  # BUG or axis=[-1, -1]?
        dlogpi_dlogit = np.sum(dlogpi_dlogit_(logits), axis=[0, 1])
        return logits + lr * np.dot(dlogpi_dlogit.T, V)  # BUG?!?!?!?
    return update_fn

def clip_by_norm(x, axis=-1, norm=2):
    v = np.linalg.norm(x, axis=axis, ord=norm, keepdims=True)
    norm_vals = np.where(v>1, v, np.ones_like(v))
    return x/norm_vals

def policy_gradient_iteration_projected(mdp, lr):
    V = lambda pi: value_functional(mdp.P, mdp.r.reshape((-1, 1)), mpi(pi), mdp.discount)
    delta = lambda pi: 1/(pi)
    U = lambda pi: clip_by_norm(pi + lr * np.dot(delta(pi), V(pi)), axis=1, norm=1)
    # this feels weird. would prefer to use logits??
    # dynamics will depend on which projection is used!??
    return U

# def parameterised_policy_gradient_iteration(mdp, lr):
#     dlogpidw = grad(np.log(pi(w)))
#     dpidw = grad(pi(w))
#     delta = lambda w: np.dot(dpidw(w), dlogpdw(w))
#     U = lambda w: w + lr * delta
#     return U

######################

def momentum_bundler(update_fn, decay):
    """
    Wraps an update fn in with its exponentially averaged grad.
    Uses the exponentially averaged grad to make updates rather than the grad itself.

    Args:
        U (callable): The update fn. U: params-> new_params.
        decay (float): the amount of exponential decay

    Returns:
        (callable): The new update fn. U: params'-> new_params'. Where params' = [params, param_momentum].
    """
    def momentum_update_fn(x):
        W_t, M_t = x[0], x[1]

        # TODO want a nicer way to do thisself.
        # nested fn application
        if isinstance(W_t, np.ndarray):
            dW = update_fn(W_t) - W_t  # should divide by lr here?
            M_tp1 = decay * M_t + dW
            W_tp1 = W_t + (1 - decay) * M_tp1
        elif isinstance(W_t, list):
            dW = [w_tp1 - w_t for w_tp1, w_t in zip(update_fn(W_t), W_t)]
            M_tp1 = [decay * m_t + dw for m_t, dw in zip(M_t, dW)]
            W_tp1 = [w_t + (1 - decay) * m_tp1 for w_t, m_tp1 in zip(W_t, M_tp1)]
        else:
            raise ValueError('Unknown format: {}'.format(type(W_t)))

        return W_tp1, M_tp1
    return jit(momentum_update_fn)

if __name__ == '__main__':
    n_states, n_actions = 2, 2

    # # TEST parameterised
    # C = random_parameterised_matrix(n_states, n_actions, 8, 2)
    # print(value(C).shape)
    #
    # # but these wont be distributed unformly in policy space!?
    # C = random_parameterised_matrix(n_states, n_actions, 8, 2)
    # print(pi(C).shape)


    mdp = build_random_mdp(n_states, n_actions, 0.9)
    init = rnd.standard_normal((n_states, n_actions))
    init /= init.sum(axis=1, keepdims=True)
    # pis = solve(policy_gradient_iteration(mdp, 0.01), init)
    # print(pis)

    # V = lambda pi: value_functional(mdp.P, mdp.r.reshape((-1, 1)), mpi(init), mdp.discount)
    # delta = lambda pi: (1/(pi*np.log(mdp.A)))  # dpi . dlog pi
    # print(delta(init).shape, V(init).shape)

    # U = policy_gradient_iteration(mdp, 0.01)
    # print(U(init))


    # x = np.abs(rnd.standard_normal((5,5)))
    # x = clip_by_norm(x, axis=1, norm=1)
    # print(x.sum(axis=1))
