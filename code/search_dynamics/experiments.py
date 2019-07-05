from utls import *
import numpy as np
import matplotlib.pyplot as plt
import copy

def generate_avi_vs_vi():
    print('Running AVI vs VI')
    n_states, n_actions = 2, 2
    lr = 0.1

    mdp = build_random_mdp(n_states, n_actions, 0.9)

    # vi
    init = np.random.standard_normal((mdp.S, mdp.A))
    qs = solve(value_iteration(mdp, lr), init)
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
    qs = solve(adjusted_value_iteration(mdp, lr, D, K), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])

    m = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', label='avi')
    plt.title('VI: {}, Avi {}'.format(n, m))

    vs = polytope(mdp.P, mdp.r, mdp.discount)
    plt.scatter(vs[:, 0], vs[:, 1], s=5, alpha=0.5)
    plt.savefig('figs/avi-vs-vi.png')
    plt.close()

def generate_vi_sgd_vs_mom():
    print('Running VI SGD vs Mom')
    n_states, n_actions = 2, 2

    mdp = build_random_mdp(n_states, n_actions, 0.9)

    # sgd
    init = np.random.standard_normal((mdp.S, mdp.A))
    qs = solve(value_iteration(mdp, 0.01), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])

    n = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(n), cmap='summer', label='sgd')

    # momentum
    init = (init, np.zeros_like(init))
    qs = solve(momentum_bundler(value_iteration(mdp, 0.01), 0.9), init)
    vs = np.vstack([np.max(q[0], axis=1) for q in qs])
    m = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', label='momentum')
    plt.title('SGD: {}, Mom {}'.format(n, m))

    vs = polytope(mdp.P, mdp.r, mdp.discount)
    plt.scatter(vs[:, 0], vs[:, 1], s=5, alpha=0.5)

    plt.savefig('figs/vi_sgd-vs-vi_mom.png')
    plt.close()

def generate_PG_vs_VI():
    print('Running PG vs VI')
    n_states, n_actions = 2, 2
    lr = 0.1

    mdp = build_random_mdp(n_states, n_actions, 0.9)

    # PG
    init = rnd.standard_normal((n_states, n_actions))
    init /= init.sum(axis=1, keepdims=True)
    logits = solve(policy_gradient_iteration_logits(mdp, lr), init)
    vs = np.vstack([np.max(value_functional(mdp.P, mdp.r, softmax(logit, axis=-1), mdp.discount), axis=1) for logit in logits])
    n = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(n), cmap='autumn', label='PG')

    # VI
    init = value_functional(mdp.P, mdp.r, softmax(init, axis=-1), mdp.discount)
    qs = solve(value_iteration(mdp, lr), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])
    m = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='summer', label='vi')
    plt.title('PG: {}, VI {}'.format(n, m))

    vs = polytope(mdp.P, mdp.r, mdp.discount)
    plt.scatter(vs[:, 0], vs[:, 1], s=5, alpha=0.5)
    plt.savefig('figs/pg-vs-vi.png')
    plt.close()

def generate_pvi_vs_vi():
    print('Running PVI vs VI')
    n_states, n_actions = 2, 2
    lr = 0.001

    mdp = build_random_mdp(n_states, n_actions, 0.9)

    # pvi
    core_init = random_parameterised_matrix(2, 2, 32, 2)
    params = solve(parameterised_value_iteration(mdp, lr), core_init)
    vs = np.hstack([value(c) for c in params]).T
    m = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', label='pvi')

    # vi
    init = value(core_init)  # use the same init
    qs = solve(value_iteration(mdp, lr), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])
    n = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(n), cmap='summer', label='vi')

    plt.title('VI: {}, PVI {}'.format(n, m))


    vs = polytope(mdp.P, mdp.r, mdp.discount)
    plt.scatter(vs[:, 0], vs[:, 1], s=5, alpha=0.5)

    plt.savefig('figs/vi-vs-pvi.png')
    plt.close()

def generate_mpvi_vs_mvi():
    print('Running MPVI vs MVI')
    n_states, n_actions = 2, 2

    mdp = build_random_mdp(n_states, n_actions, 0.9)

    # mpvi
    core_init = random_parameterised_matrix(2, 2, 32, 8)
    init = (core_init, [np.zeros_like(c) for c in core_init])
    params = solve(momentum_bundler(parameterised_value_iteration(mdp, 0.01), 0.9), init)
    vs = np.vstack([np.max(value(c[0]), axis=-1) for c in params])
    m = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', label='mpvi')

    # mvi
    init = copy.deepcopy(value(core_init))  # use the same init
    init = (init, np.zeros_like(init))
    qs = solve(momentum_bundler(value_iteration(mdp, 0.01), 0.9), init)
    vs = np.vstack([np.max(q[0], axis=-1) for q in qs])
    n = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(n), cmap='summer', label='mvi')

    plt.title('MVI: {}, MPVI {}'.format(n, m))

    vs = polytope(mdp.P, mdp.r, mdp.discount)
    plt.scatter(vs[:, 0], vs[:, 1], s=5, alpha=0.5)


    plt.savefig('figs/mpvi-vs-pvi.png')
    plt.close()

def generate_mpvi_inits():
    print('Running MPVI inits')
    n_states, n_actions = 2, 2

    mdp = build_random_mdp(n_states, n_actions, 0.9)

    core_init = random_parameterised_matrix(2, 2, 16, 6)
    init = (core_init, [np.zeros_like(c) for c in core_init])
    params = solve(momentum_bundler(parameterised_value_iteration(mdp, 0.01), 0.9), init)
    vs = np.vstack([np.max(value(c[0]), axis=-1) for c in params])
    m = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', label='mpvi', alpha=0.5)


    new_init = random_reparameterisation(core_init, 2)
    # sanity check
    assert np.isclose(value(core_init), value(new_init), atol=1e-4).all()
    init = (new_init, [np.zeros_like(c) for c in core_init])
    params = solve(momentum_bundler(parameterised_value_iteration(mdp, 0.01), 0.9), init)
    vs = np.vstack([np.max(value(c[0]), axis=-1) for c in params])
    m = vs.shape[0]
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='summer', label='mpvi', alpha=0.5)

    vs = polytope(mdp.P, mdp.r, mdp.discount)
    plt.scatter(vs[:, 0], vs[:, 1], s=5, alpha=0.5)

    plt.savefig('figs/mpvi-inits.png')
    plt.close()
    """
    Hmm. I thought this would change the dynamics.
    It is because the value is only a linear function of the parameters???
    """

# def generate_pvi_vs_apvi():
#     print('Running PVI vs APVI')
#     n_states, n_actions = 2, 2
#     lr = 0.01
#
#     mdp = build_random_mdp(n_states, n_actions, 0.9)
#
#     core_init = random_parameterised_matrix(2, 2, 32, 2)
#     # pvi
#     params = solve(parameterised_value_iteration(mdp, lr), core_init)
#     vs = np.hstack([value(c) for c in params]).T
#     m = vs.shape[0]
#     plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', label='pvi')
#
#     # TODO want to visualise.
#     # @jit
#     def K(dQ):
#         return np.tensordot(dQ, dQ, axes=([-1,-2],[-1,-2]))
#
#     dVdw = jit(jacrev(value))
#     dQs = [dVdw(cores) for cores in params][0:10]
#
#     Ks = [sum([K(dq) for dq in dQ]).reshape((mdp.S * mdp.A, mdp.S * mdp.A)) for dQ in dQs]
#
#     n = 100
#     x = np.stack(np.meshgrid(np.linspace(-1,1,n), np.linspace(-1,1,n)), axis=0).reshape((2, n**2))
#     print(x.shape)
#
#     # plt.show()


# pg vs pi

if __name__ == '__main__':
    generate_vi_sgd_vs_mom()
    generate_avi_vs_vi()
    generate_pvi_vs_vi()
    generate_mpvi_vs_mvi()
    generate_mpvi_inits()
    generate_PG_vs_VI()
    # generate_pvi_vs_apvi()
