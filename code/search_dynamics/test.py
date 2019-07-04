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
        if len(l)>10000:
            return True
        elif np.isclose(l[-1], l[-2], atol=1e-03).all():
            return True
        else:
            False
    else:
        False

def solve(update_fn, init):
    xs = [init]
    x = init
    while not converged(xs):
        x = update_fn(x)
        xs.append(x)
    return xs

def value_iteration(mdp, lr):
    # bellman optimality operator
    T = lambda Q: mdp.r + mdp.discount * np.argmax(mdp.P * Q.reshape((1, -1)))
    # GD update operator
    U = lambda Q: Q + lr * (T(Q) - Q)
    return U

def parameterised_value_iteration(mdp, lr):
    T = lambda w: mdp.r + mdp.discount * np.argmax(mdp.P * Q(w))
    dQdw = lambda w: grad(Q)  # might need to do some reshaping here!?
    U = lambda w: w + lr * np.dot((T(Q(w)) - Q(w)), dQdw(w))
    return U

def corrected_value_iteration(mdp, lr):
    T = lambda theta: mdp.r + mdp.discount * np.argmax(mdp.P * Q(w))
    dQdw = lambda w: grad(Q)
    Km1 = lambda w: np.linalg.inv(np.dot(dQdw(w).T, dQdw(w)))
    U = lambda w: w + lr * np.dot(dQdw(w), np.dot(Km1, T(Q(w) - Q(w))))
    return U

def policy_gradient_iteration(mdp, lr):
    delta = lambda pi: np.dot(dpidw, 1/pi)
    U = lambda pi: pi + lr * delta
    return U

def parameterised_policy_gradient_iteration(mdp, lr):
    dlogpidw = grad(np.log(pi(w)))
    dpidw = grad(pi(w))
    delta = lambda w: np.dot(dpidw(w), dlogpdw(w))
    U = lambda w: w + lr * delta
    return U

def momentum_bundler(U, decay):
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
        w_t, m_t = x[0], x[1]
        dw = U(w_t) - w_t  # should divide by lr here?
        m_tp1 = decay * m_t + dw
        w_tp1 = w_t + m_tp1
        return np.stack([w_tp1, m_tp1], axis=0)
    return momentum_update_fn

def generate_vi_sgd_vs_mom():
    n_states, n_actions = 2, 2

    mdp = build_random_mdp(n_states, n_actions, 0.9)

    # sgd
    init = np.random.standard_normal((mdp.S, mdp.A))
    qs = solve(value_iteration(mdp, 0.01), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])

    n = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(n), cmap='summer', label='sgd')

    # momentum
    init = np.stack([init, np.zeros((mdp.S, mdp.A))], axis=0)
    qs = solve(momentum_bundler(value_iteration(mdp, 0.01), 0.9), init)
    vs = np.vstack([np.max(q[0], axis=1) for q in qs])

    m = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', label='momentum')
    plt.title('SGD: {}, Mom {}'.format(n, m))
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    generate_vi_sgd_vs_mom()

    # TEST parameterised
    # C = random_parameterised_matrix(n_states, n_actions, 8, 2)
    # print(value(C).shape)
    #
    # # but these wont be distributed unformly in policy space!?
    # C = random_parameterised_matrix(n_states, n_actions, 8, 2)
    # print(trl.generate_Mpi(n_states, n_actions, mpi(C)).shape)
