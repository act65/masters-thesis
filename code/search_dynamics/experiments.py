from utls import *
import numpy as np
import matplotlib.pyplot as plt

def generate_avi_vs_vi():
    n_states, n_actions = 2, 2

    mdp = build_random_mdp(n_states, n_actions, 0.9)

    # vi
    init = np.random.standard_normal((mdp.S, mdp.A))
    qs = solve(value_iteration(mdp, 0.01), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])

    n = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(n), cmap='summer', label='vi')

    # avi
    # K = np.eye(n_states)
    K = np.random.standard_normal((n_states))
    K = (K + K.T)/2 + 1  # this can accelerate learning. but also lead to divergence. want to explore this more!!!
    d = np.random.random(n_states)
    D = np.diag(d/d.sum())  # this can change the optima!!
    # D = np.eye(n_states)
    qs = solve(adjusted_value_iteration(mdp, 0.01, D, K), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])

    m = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', label='avi')
    plt.title('VI: {}, Avi {}'.format(n, m))
    plt.show()

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
    init = np.stack([init, np.zeros_like(init)], axis=0)
    qs = solve(momentum_bundler(value_iteration(mdp, 0.01), 0.9), init)
    vs = np.vstack([np.max(q[0], axis=1) for q in qs])

    m = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', label='momentum')
    plt.title('SGD: {}, Mom {}'.format(n, m))
    plt.show()


def generate_PG_vs_MPG():
    n_states, n_actions = 2, 2
    lr = 0.1

    mdp = build_random_mdp(n_states, n_actions, 0.9)

    # PG
    init = rnd.standard_normal((n_states, n_actions))
    init /= init.sum(axis=1, keepdims=True)
    pis = solve(policy_gradient_iteration_logits(mdp, lr), init)
    vs = np.stack([value_functional(mdp.P, mdp.r.reshape((-1, 1)), mpi(pi), mdp.discount) for pi in pis], axis=0).squeeze()
    n = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(n), cmap='summer', label='PG')

    # Momentum + PG
    init = np.stack([init, np.zeros_like(init)], axis=0)
    pis = solve(momentum_bundler(policy_gradient_iteration_logits(mdp, lr), 0.9), init)
    vs = np.stack([value_functional(mdp.P, mdp.r.reshape((-1, 1)), mpi(pi[0]), mdp.discount) for pi in pis], axis=0).squeeze()
    n = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(n), cmap='autumn', label='MPG')
    plt.show()


# pg vs pi
# param vi vs vi vs param mom vs mom

if __name__ == '__main__':
    # generate_vi_sgd_vs_mom()
    # generate_avi_vs_vi()
    generate_PG_vs_MPG()
