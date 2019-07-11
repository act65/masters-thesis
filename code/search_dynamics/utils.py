"""
Explore how the different search spaces effect the GD dynamics.
"""
import functools
import itertools
import collections

import numpy
import jax.numpy as np
from jax import grad, jit, jacrev, vmap

import numpy.random as rnd

MDP = collections.namedtuple('mdp', ['S', 'A', 'P', 'r', 'discount', 'd0'])

def build_random_mdp(n_states, n_actions, discount):
    P = rnd.random((n_states, n_states, n_actions))
    r = rnd.standard_normal((n_states, n_actions))
    d0 = rnd.random((n_states, 1))
    return MDP(n_states, n_actions, P/P.sum(axis=0, keepdims=True), r, discount, d0/d0.sum(axis=0, keepdims=True))

######################

def gen_grid_policies(N):
    # special case for 2 x 2
    p1s, p2s = np.linspace(0,1,N), np.linspace(0,1,N)
    p1s = p1s.ravel()
    p2s = p2s.ravel()
    return [np.array([[p1, 1-p1],[1-p2, p2]]) for p1 in p1s for p2 in p2s]

# @jit
def polytope(P, r, discount, pis):
    print('n pis:{}'.format(len(pis)))
    def V(pi):
        return np.sum(value_functional(P, r, pi, discount), axis=1)
    vs = np.vstack([V(pi) for pi in pis])
    return vs

#############################
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

def build(cores):
    return functools.reduce(np.dot, cores)

"""
Tools for simulating dyanmical systems.
"""

def isclose(x, y, atol=1e-8):
    if isinstance(x, np.ndarray):
        return np.isclose(x, y, atol=atol).all()
    elif isinstance(x, list):
        # return all(np.isclose(x[0], y[0], atol=1e-03).all() for i in range(len(x)))
        return np.isclose(build(x), build(y), atol=atol).all()
    elif isinstance(x, tuple) and isinstance(x[0], np.ndarray):
        return np.isclose(x[0], y[0], atol=atol).all()
    elif isinstance(x, tuple) and isinstance(x[0], list):
        return np.isclose(build(x[0]), build(y[0]), atol=atol).all()
    else:
        raise ValueError('wrong format')

def converged(l):
    if len(l)>10:
        if len(l)>50000:
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
"""

@jit
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
    # P_{\pi}(s_t+1 | s_t) = sum_{a_t} P(s_{t+1} | s_t, a_t)\pi(a_t | s_t)
    P_pi = np.einsum('ijk,jk->ij', P, pi)
    r_pi = np.expand_dims(np.einsum('ij,ij->i', pi, r), 1)

    # assert np.isclose(pi/pi.sum(axis=1, keepdims=True), pi).all()
    # assert np.isclose(P_pi/P_pi.sum(axis=0, keepdims=True), P_pi, atol=1e-4).all()

    # BUG why transpose here?!?!
    vs = np.dot(np.linalg.inv(np.eye(n) - discount*P_pi.T), r_pi)
    # print(vs.shape, P_pi.shape)
    return vs

def bellman_optimality_operator(P, r, Q, discount):
    """
    Args:
        P (np.ndarray): [n_states x n_states x n_actions]
        r (np.ndarray): [n_states x n_actions]
        Q (np.ndarray): [n_states x n_actions]
        discount (float): the temporal discount value

    Returns:
        (np.ndarray): [n_states, n_actions]
    """
    if Q.shape[1] == 1:  # Q == V
        # Q(s, a) =  r(s, a) + \gamma E_{s'~P(s' | s, a)} V(s')
        return r + discount*np.einsum('ijk,il->jk', P, Q)
    else:
        # Q(s, a) =  r(s, a) + \gamma max_a' E_{s'~P(s' | s, a)} Q(s', a')
        return r + discount*np.max(np.einsum('ijk,il->jkl', P, Q), axis=-1)

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
    TD = lambda cores: T(build(cores)) - build(cores)
    dVdw = jacrev(build)

    @jit
    def update_fn(cores):
        delta = TD(cores)
        grads = [np.einsum('ij,ijkl->kl', delta, dc) for dc in dVdw(cores)]
        # TODO attempt to understand the properties of dc. and its relation to K
        return [c+lr*g for c, g in zip(cores, grads)]
    return jit(update_fn)

######################
# Neural tangent kernel and ...
######################

"""
Inspired by Towards Characterizing Divergence in Deep Q-Learning
https://arxiv.org/abs/1903.08894
"""

def adjusted_value_iteration(mdp, lr, D, K):
    T = lambda Q: bellman_optimality_operator(mdp.P, mdp.r, Q, mdp.discount)
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

def entropy(p, axis=1):
    return -np.sum(np.log(p) * p)

def softmax(x, axis=-1):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

def policy_gradient_iteration_logits(mdp, lr):
    # d/dlogits V = E_{\pi}[V] = E[V . d/dlogit log \pi]
    dlogpi_dlogit = jacrev(lambda logits: np.log(softmax(logits)))
    dHdlogit = jacrev(lambda logits: entropy(softmax(logits)))

    @jit
    def update_fn(logits):
        V = value_functional(mdp.P, mdp.r, softmax(logits), mdp.discount)
        Q = bellman_optimality_operator(mdp.P, mdp.r, V, mdp.discount)
        A = Q-V
        g = np.einsum('ijkl,ij->kl', dlogpi_dlogit(logits), A)
        return logits + 1e-4*dHdlogit(logits) + lr * g
    return update_fn

def parameterised_policy_gradient_iteration(mdp, lr):
    dlogpi_dw = jacrev(lambda cores: np.log(softmax(build(cores), axis=1)))
    dHdw = jacrev(lambda cores: entropy(softmax(build(cores))))

    @jit
    def update_fn(cores):
        V = value_functional(mdp.P, mdp.r, softmax(build(cores), axis=1), mdp.discount)
        Q = bellman_optimality_operator(mdp.P, mdp.r, V, mdp.discount)
        A = Q-V
        grads = [np.einsum('ijkl,ij->kl', d, A) for d in dlogpi_dw(cores)]
        return [c+lr*g+1e-4*dH for c, g, dH in zip(cores, grads, dHdw(cores))]
    return update_fn

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

def approximate(v, cores):
    """
    cores = random_parameterised_matrix(2, 1, d_hidden=8, n_hidden=4)
    v = rnd.standard_normal((2,1))
    cores_ = approximate(v, cores)
    print(v, '\n',build(cores_))
    """
    loss = lambda cores: np.sum(np.square(v - build(cores)))
    dl2dc = grad(loss)
    l2_update_fn = lambda cores: [c - 0.01*g for g, c in zip(dl2dc(cores), cores)]
    init = (cores, [np.zeros_like(c) for c in cores])
    final_variables, momentum_var = solve(momentum_bundler(l2_update_fn, 0.9), init)[-1]
    return final_variables


if __name__ == '__main__':
    n_states, n_actions = 2, 2

    # # TEST parameterised
    # C = random_parameterised_matrix(n_states, n_actions, 8, 2)
    # print(build(C).shape)
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
