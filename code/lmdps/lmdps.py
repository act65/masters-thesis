import jax.numpy as npj
from jax import grad, jit
import numpy.random as rnd
import numpy as np

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
        Solves
        r(s, a) = q(s) + E_s'~p(s' | s, a) log( p(s' | s, a) / p(x' | x))
        See supplementary material of todorov 2009 for more info
        """
        # b_a = r(x, a) - E_s'~p(s' | s, a) log( p(s' | s, a) / p(x' | x))
        b = r[idx_x] - np.sum(P[:, idx_x, :] * np.log(P[:, idx_x, :]), axis=0)
        # D_ax' = p(x' | x, a)
        D = P[:, idx_x, :]

        # D(q.1 - m) = b, c = q.1 - m
        c = np.dot(b, np.linalg.pinv(D))
        q = -np.log(np.sum(np.exp(-c)))

        m = np.sum(q) - c
        p = np.exp(m)

        return p, [q]

    # TODO, should be able to solve these in parallel.
    # maybe even share some knowledge???!
    pnqs = [embed_state(i) for i in range(P.shape[0])]
    return tuple([np.stack(val, axis=1) for val in zip(*pnqs)])

def softmax(x):
    return npj.exp(x)/npj.sum(npj.exp(x), axis=1, keepdims=True)

def lmdp_decoder(u, P, lr=10):
    """
    Given optimal control dynamics.
    Find a policy that yields those same dynamics.
    Find pi(a | x) s.t. u(x' | x) ~= sum_a pi(a | x) P(x'| x, a)
    or relax to min_pi KL(u, P_pi)
    """
    # TODO need to convert back into discrete actions!
    # QUESTION If we cannot accurately decode an action. What about using an option?
    # QUESTION how expensive is this?!

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
        P_pi = npj.einsum('ijk,jk->ij', P, pi)
        return npj.sum(npj.multiply(u, npj.log(u/P_pi)))  # KL

    dLdw = jit(grad(loss))
    def update_fn(w):
        return w - lr * dLdw(w)

    init = rnd.standard_normal((P.shape[0], P.shape[-1]))
    pi_star_logits = dynamical_system_solver(update_fn, init)

    return softmax(pi_star_logits)

def lmdp_solver(p, q, discount):
    """
    z = QPz

    Args:

    Returns:
        (): the optimal policy
        (): the value of the optimal policy
    """
    # Evaluate
    # Solve z = QPz
    Q = np.diag(np.exp(np.squeeze(-q)))
    z = solve(Q, p, a=discount)

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
    z = dynamical_system_solver(fn, rnd.random((p.shape[-1], 1)))
    return z.squeeze()

def dynamical_system_solver(fn, init):
    xs = [init]
    while not converged(xs):
        xs.append(fn(xs[-1]))
    return xs[-1]

def converged(xs):
    if len(xs) <= 1:
        return False
    elif len(xs) > 10000:
        raise ValueError('not converged')
    elif np.isnan(xs[-1]).any():
        raise ValueError('Nan')
    else:
        return np.isclose(xs[-1], xs[-2], atol=1e-8).all()


if __name__ == "__main__":
    n_states, n_actions = 3, 2
    P, r = rnd_mdp(n_states, n_actions)
    p, q = lmdp_embedding(P, r)

    # p, q = rnd_lmdp(n_states, n_actions)

    u, v = lmdp_solver(p, q, 0.9)
    # print(v)
    # print(u)

    pi = lmdp_control_decoder(u, P)
    print(pi)
