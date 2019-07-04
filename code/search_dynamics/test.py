import numpy as np
import functools
import trl_utils as trl
import collections

def random_parameterised_matrix(n, m, d_hidden, n_hidden):
    cores = [np.random.standard_normal((d_hidden, d_hidden)) for _ in range(n_hidden)]
    cores = [np.random.standard_normal((n, d_hidden))] + cores + [np.random.standard_normal((d_hidden, m))]
    return cores

def value(cores):
    return functools.reduce(np.dot, cores)

def mpi(cores):
    M = functools.reduce(np.dot, cores)
    return M/M.sum(axis=1, keepdims=True)

mdp = collections.namedtuple('mdp', ['S', 'A', 'P', 'r', 'discount', 'd0'])

def build_random_mdp(n_states, n_actions, discount):
    P = np.random.standard_normal((n_states, n_states*n_actions))
    r = np.random.standard_normal((n_states, n_actions))
    d0 = np.random.standard_normal((n_states, 1))
    return mdp(n_states, n_actions, P/P.sum(axis=1, keepdims=True), r, discount, d0)

def converged(l):
    if len(l)>10:
        if len(l)>1000:
            return True
        elif np.isclose(l[-1], l[-2]).all():
            return True
        else:
            False
    else:
        False

def solve(update_fn, init):
    xs = []
    x = init
    while not converged(xs):
        x = update_fn(x)
        xs.append(x)
    return xs

def value_iteration(mdp, lr):
    T = lambda Q: mdp.r + mdp.discount * np.argmax(mdp.P * Q.reshape((1, -1)))  # bellman optimality operator
    U = lambda Q: Q + lr * (T(Q) - Q)  # GD update operator
    return U

def parameterised_value_iteration(mdp, lr):
    T = lambda w: mdp.r + mdp.discount * np.argmax(mdp.P * Q(w))
    dQdw = lambda w: grad(Q)
    U = lambda w: w + lr * np.dot((T(Q(w)) - Q(w)), dQdw(w))
    return U

def corrected_value_iteration(mdp, lr):
    T = lambda theta: mdp.r + mdp.discount * np.argmax(mdp.P * Q(w))
    dQdw = lambda w: grad(Q)
    Km1 = lambda w: np.linalg.inv(np.dot(dQdw(w).T, dQdw(w)))
    U = lambda w: w + lr * np.dot(dQdw(w), np.dot(Km1, T(Q(w) - Q(w))))
    return U

def policy_gradient_iteration(mdp, lr):
    dlogpidw = grad(np.log(pi(w)))
    dpidw = grad(pi(w))
    delta = lambda w: np.dot(dpidw(w), dlogpdw(w))
    U = lambda w: w + lr * delta
    return U

def momentum_bundler(U, decay):
    """
    Wraps an update fn in
    """
    def momentum_update_fn(x):
        w_t, m_t = x[0], x[1]
        dw = U(w_t) - w_t
        m_tp1 = decay * m_t + dw
        w_tp1 = w_t + m_tp1
        return np.stack([w_tp1, m_tp1], axis=0)
    return momentum_update_fn


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_states = 2
    n_actions = 2

    # TEST parameterised
    # C = random_parameterised_matrix(n_states, n_actions, 8, 2)
    # print(value(C).shape)
    #
    # # but these wont be distributed unformly in policy space!?
    # C = random_parameterised_matrix(n_states, n_actions, 8, 2)
    # print(trl.generate_Mpi(n_states, n_actions, mpi(C)).shape)


    mdp = build_random_mdp(n_states, n_actions, 0.99)
    qs = solve(value_iteration(mdp, 0.1), np.random.standard_normal((mdp.S, mdp.A)))
    vs = np.vstack([np.max(q, axis=1) for q in qs])

    plt.scatter(vs[:, 0], vs[:, 1], c=range(vs.shape[0]))

    init = np.stack([np.zeros((mdp.S, mdp.A)), np.random.standard_normal((mdp.S, mdp.A))], axis=0)
    qs = solve(momentum_bundler(value_iteration(mdp, 0.001), 0.9), init)
    vs = np.vstack([np.max(q[0], axis=1) for q in qs])

    plt.scatter(vs[:, 0], vs[:, 1], c=range(vs.shape[0]))
    plt.show()
