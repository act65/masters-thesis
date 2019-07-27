import numpy
from numpy import linalg
import numpy.random as rnd

import jax.numpy as np
from jax import grad, jit

def rnd_mdp(n_states, n_actions):
    P = rnd.random((n_states, n_states, n_actions))
    P = P/P.sum(0)

    r = rnd.standard_normal((n_states, n_actions))
    return P, r

def rnd_lmdp(n_states, n_actions):
    p = rnd.random((n_states, n_states))
    p = p/p.sum(0)
    q = rnd.random((n_states, 1))
    return p, q

def mdp_encoder(P, r):
    """
    Args:
        P (np.array): The transition function. shape = [n_states, n_states, n_actions]
        r (np.array): The reward function. shape = [n_states, n_actions]

    Returns:
        p (np.array): the uncontrolled transition matrix
        q (np.array): the pseudo reward

    Needs to be solved for every state.
    """
    # QUESTION Are there similarities between the action embeddings in each state?
    # QUESTION How does this embedding change the set of policies that can be represented? Does this transformation preserve the optima?
    def embed_state(idx_x):
        """
        For each state, we have a system of |A| linear equations.
        Each eqn requiring that there exists u(.|a), a in A, s.t.;
        - p(s' | s, a) = u(s'|a).p(s'|s)
        - r(s, a) = q(s) - KL(u(.|a)p(.|s), p(.| s))

        This ensures that p, q are able to `represent`#?!?# the original dynamics and
        reward.

        See supplementary material of todorov 2009 for more info
        https://www.pnas.org/content/106/28/11478
        """
        # b_a = r(x, a) - E_s'~p(s' | s, a) log( p(s' | s, a) / p(x' | x))
        b = r[idx_x, :] - np.sum(P[:, idx_x, :] * np.log(P[:, idx_x, :]+1e-8), axis=0) # swap +/- of 2nd term
        # D_ax' = p(x' | x, a)
        D = -P[:, idx_x, :]  # swap +/-

        # D(q.1 - m) = b, c = q.1 - m
        c = np.dot(b, linalg.pinv(D))
        # min c
        q = -np.log(np.sum(np.exp(-c)))

        m = q - c
        p = np.exp(m)

        return p, np.array([q])

    # TODO, can solve these in parallel.
    # maybe even share some knowledge???!
    pnqs = [embed_state(i) for i in range(P.shape[0])]
    p, q = tuple([np.stack(val, axis=1) for val in zip(*pnqs)])
    return p, np.squeeze(q)

def lmdp_solver(p, q, discount):
    """
    v = -log(z)
    v = max_u q(s) + log G - K(u || p . z / G)
    v* = q(s) + log G
    -log z = q + log G
    z = exp(-q - log G)
    z = exp(-q)exp(-log G)
    z = QG^-1  !!?!
    z = QPz

    Args:

    Returns:
        (): the optimal policy
        (): the value of the optimal policy
    """
    # BUG doesnt work for large values of discount 0.999.
    # BUG need to alter to handle maximisation...?!?

    # Evaluate
    # Solve z = QPz
    Q = np.diag(np.exp(q))
    z = solve(Q, p, a=discount)

    # no. this is the value of the unconstrained dynamics, kinda
    # z = np.dot((np.linalg.pinv(np.eye(p.shape[0]) - discount * p)), np.exp(q) + KL(?? || p))

    v = -np.log(z)

    # Calculate the optimal control
    # G(x) = sum_x' p(x' | x) z(x')
    G = np.einsum('ij,i->j', p, z)
    # u*(x' | x) = p(x' | x) z(x') / G[z](x)
    u = p * z[:, np.newaxis] / G[np.newaxis, :]

    return u, v

def solve(Q, p, a):
    # Solve z = QPz^a
    # huh, this is pretty consistent.
    # approx 1:100?
    A = np.dot(Q, p)
    fn = lambda x: np.dot(A, x**a)
    z = dynamical_system_solver(fn, np.ones((p.shape[-1], 1)))
    return z.squeeze()

def dynamical_system_solver(fn, init):
    xs = [init]
    while not converged(xs):
        xs.append(fn(xs[-1]))
    return xs[-1]

def converged(xs):
    if len(xs) <= 1:
        return False
    elif len(xs) > 1000 and np.isclose(xs[-1], xs[-2], atol=1e-6).all():
        print('Close enough...')
        return True
    elif len(xs) > 10000:
        raise ValueError('not converged')
    elif np.isnan(xs[-1]).any():
        raise ValueError('Nan')
    else:
        return np.isclose(xs[-1], xs[-2], atol=1e-8).all()

def softmax(x, axis=1):
    return np.exp(x)/np.sum(np.exp(x), axis=axis, keepdims=True)

def lmdp_decoder(u, P, lr=10):
    """
    Given optimal control dynamics.
    Find a policy that yields those same dynamics.
    Find pi(a | x) s.t. u(x' | x) ~= sum_a pi(a | x) P(x'| x, a)
    or relax to min_pi KL(u, P_pi)
    """
    # TODO need to convert back into discrete actions!
    # NOTE is there a way to solve this using linear equations?!
    # W = log(P_pi)
    # = sum_a log(P[a]) + log(pi[a])
    # M = log(u)
    # UW - UM = 0
    # U(W-M) = 0, W = M = sum_a log(P[a]) + log(pi[a])
    # 0 = sum_a log(P[a]) + log(pi[a]) - M

    def loss(pi_logits):
        pi = softmax(pi_logits)
        # P_pi(s'|s) = \sum_a pi(a|s)p(s'|s, a)
        P_pi = np.einsum('ijk,jk->ij', P, pi)
        return np.sum(np.multiply(u, np.log(u/P_pi)))  # KL

    dLdw = jit(grad(loss))
    def update_fn(w):
        return w - lr * dLdw(w)

    init = rnd.standard_normal((P.shape[0], P.shape[-1]))
    pi_star_logits = dynamical_system_solver(update_fn, init)

    return softmax(pi_star_logits)

def KL(P,Q):
    return -np.sum(P*np.log((Q+1e-8)/(P+1e-8)))

def test_embedded_dynamics():
    """
    Explore how the unconstrained dynamics in a simple setting.
    """
    # What about when p(s'| s) = 0, is not possible under the true dynamics?!
    r = np.array([
        [1, 0],
        [0, 0]
    ])

    # Indexed by [s' x s x a]
    # ensure we have a distribution over s'
    p000 = 1
    p100 = 1 - p000

    p001 = 0
    p101 = 1 - p001

    p010 = 0
    p110 = 1 - p010

    p011 = 1
    p111 = 1 - p011

    P = np.array([
        [[p000, p001],
         [p010, p011]],
        [[p100, p101],
         [p110, p111]],
    ])
    # BUG ??? only seems to work for deterministic transitions!?
    # oh, this is because deterministic transitions satisfy the row rank requirement??!
    # P = np.random.random((2, 2, 2))
    # P = P/np.sum(P, axis=0)

    # a distribution over future states
    assert np.isclose(np.sum(P, axis=0), np.ones((2,2))).all()

    pi = softmax(r, axis=1)  # exp Q vals w gamma = 0
    # a distribution over actions
    assert np.isclose(np.sum(pi, axis=1), np.ones((2,))).all()

    p, q = mdp_encoder(P, r)

    print('q', q)
    P_pi = np.einsum('ijk,jk->ij', P, pi)

    print('p', p)
    print('P_pi', P_pi)

    # the unconstrained dynamics with deterministic transitions,
    # are the same was using a gamma = 0 boltzman Q vals
    print("exp(r) is close to p? {}".format(np.isclose(p, P_pi, atol=1e-4).all()))

    # r(s, a) = q(s) - KL(P(. | s, a) || p(. | s))
    # TODO how to do with matrices!?
    # kl = - (np.einsum('ijk,ij->jk', P, np.log(p)) - np.einsum('ijk,ijk->jk', P, np.log(P)))
    kl = numpy.zeros((2, 2))
    for j in range(2):
        for k in range(2): # actions
            kl[j, k] = KL(P[:, j, k], p[:, j])

    r_approx = -q[:, np.newaxis] - kl

    print(np.around(r, 3))
    print(np.around(r_approx, 3))
    print('r ~= q - KL(P || p): {}'.format(np.isclose(r, r_approx, atol=1e-3).all()))

def test_lmdp_solver():
    """
    Want to set up a env that will test long term value over short term rewards.
    """
    p = np.array([
        [0.75, 0.25],
        [0.25, 0.75]
    ])
    q = np.array([1, 0])
    u, v = lmdp_solver(p, q, 0.9)
    print(u)
    print(v)

def test_decoder_simple():
    # Indexed by [s' x s x a]
    # ensure we have a distribution over s'
    p000 = 1
    p100 = 1 - p000

    p001 = 0
    p101 = 1 - p001

    p010 = 0
    p110 = 1 - p010

    p011 = 1
    p111 = 1 - p011

    P = np.array([
        [[p000, p001],
         [p010, p011]],
        [[p100, p101],
         [p110, p111]],
    ])

    u = np.array([
        [0.95, 0.25],
        [0.05, 0.75]
    ])

    pi = lmdp_decoder(u, P, lr=1)
    P_pi = np.einsum('ijk,jk->ij', P, pi)

    assert np.isclose(P_pi, u, atol=1e-4).all()
    print(P_pi)
    print(u)

def test_decoder_rnd():
    n_states = 6
    n_actions = 6

    P = rnd.random((n_states, n_states, n_actions))
    P /= P.sum(0, keepdims=True)

    u = rnd.random((n_states, n_states))
    u /= u.sum(0, keepdims=True)

    pi = lmdp_decoder(u, P, lr=1)
    P_pi = np.einsum('ijk,jk->ij', P, pi)

    print(P_pi)
    print(u)
    print(KL(P_pi,u))
    assert np.isclose(P_pi, u, atol=1e-2).all()

if __name__ == "__main__":
    # test_embedded_dynamics()
    # test_lmdp_solver()
    # test_decoder_simple()
    test_decoder_rnd()

    # p, q = rnd_lmdp(n_states, n_actions)

    # u, v = lmdp_solver(p, q, 0.9)
    # # print(v)
    # # print(u)
    #
    # pi = lmdp_decoder(u, P)
    # print(pi)
