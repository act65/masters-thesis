'traj-figsfrom utls import *
import numpy as np
import matplotlib.pyplot as plt
import copy

def generate_vi_sgd_vs_mom(mdp, init):
    print('\nRunning VI SGD vs Mom')
    lr = 0.01

    # sgd
    qs = solve(value_iteration(mdp, lr), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='sgd')
    plt.scatter(vs[:, 0], vs[:, 1], c=range(n), cmap='summer', s=5)

    # momentum
    init = (init, np.zeros_like(init))
    qs = solve(momentum_bundler(value_iteration(mdp, lr), 0.99), init)
    vs = np.vstack([np.max(q[0], axis=1) for q in qs])
    m = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='momentum')
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', s=5)

    plt.title('SGD: {}, Mom {}'.format(n, m))
    plt.legend()
    plt.colorbar()

    plt.savefig('traj-figs/vi_sgd-vs-vi_mom.png')
    plt.close()


def generate_pvi_vs_vi(mdp, init):
    print('\nRunning PVI vs VI')
    lr = 0.01

    # pvi
    core_init = random_parameterised_matrix(2, 2, 32, 4)
    core_init = approximate(init, core_init)
    params = solve(parameterised_value_iteration(mdp, lr), core_init)
    vs = np.vstack([np.max(build(c), axis=1) for c in params])
    m = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='pvi')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(m-2), cmap='autumn', s=5)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='r', marker='x')

    # vi
    qs = solve(value_iteration(mdp, lr), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='vi')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='summer', s=5)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')

    plt.title('VI: {}, PVI {}'.format(n, m))
    plt.legend()
    # plt.colorbar()

    plt.savefig('traj-figs/vi-vs-pvi.png', dpi=300)
    plt.close()


###################################################



def generate_avi_vs_vi(mdp):
    print('\nRunning AVI vs VI')
    lr = 0.1

    # vi
    init = np.random.standard_normal((mdp.S, mdp.A))
    qs = solve(value_iteration(mdp, lr), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])

    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='vi')
    plt.scatter(vs[:, 0], vs[:, 1], c=range(n), cmap='summer', s=5)

    # avi
    # K = np.eye(n_states)
    K = np.random.standard_normal((mdp.S))
    K = (K + K.T)/2 + 1  # this can accelerate learning. but also lead to divergence. want to explore this more!!!
    d = np.random.random(mdp.S)
    D = np.diag(d/d.sum())  # this can change the optima!!
    # D = np.eye(n_states)
    qs = solve(adjusted_value_iteration(mdp, lr, D, K), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])

    m = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='avi')
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', s=5)
    plt.title('VI: {}, Avi {}'.format(n, m))
    plt.legend()
    plt.colorbar()

    plt.savefig('traj-figs/avi-vs-vi.png')
    plt.close()

def generate_PG_vs_VI(mdp, init):
    print('\nRunning PG vs VI')
    lr = 0.1

    # PG
    logits = solve(policy_gradient_iteration_logits(mdp, lr), init)
    vs = np.vstack([value_functional(mdp.P, mdp.r, softmax(logit), mdp.discount).T for logit in logits])
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='PG')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='summer', s=5)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')

    # VI
    v = value_functional(mdp.P, mdp.r, softmax(init), mdp.discount)
    init = np.einsum('ijk,jl->jk', mdp.P, v)  # V->Q
    qs = solve(value_iteration(mdp, lr), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])
    m = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='VI')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(m-2), cmap='autumn', s=5)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='r', marker='x')
    plt.legend()
    plt.title('PG: {}, VI {}'.format(n, m))
    # plt.colorbar()

    plt.savefig('traj-figs/pg-vs-vi.png')
    plt.close()

def generate_PG_vs_PPG(mdp, init):
    print('\nRunning PG vs PPG')
    lr = 0.1

    # PPG
    core_init = random_parameterised_matrix(2, 2, 32, 8)
    core_init = approximate(init, core_init)
    all_params = solve(parameterised_policy_gradient_iteration(mdp, lr), core_init)
    vs = np.vstack([np.max(value_functional(mdp.P, mdp.r, softmax(build(params), axis=-1), mdp.discount), axis=1)
                    for params in all_params])
    m = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='PPG')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(m-2), cmap='summer', s=5)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')

    # PG
    logits = solve(policy_gradient_iteration_logits(mdp, lr), init)
    vs = np.vstack([np.max(value_functional(mdp.P, mdp.r, softmax(logit, axis=-1), mdp.discount), axis=1) for logit in logits])
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='PG')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='autumn', s=5)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='r', marker='x')

    plt.title('PG: {}, PPG {}'.format(n, m))
    plt.legend()
    # plt.colorbar()

    plt.savefig('traj-figs/pg-vs-ppg.png', dpi=300)
    plt.close()

def generate_mpvi_vs_mvi(mdp, init):
    print('\nRunning MPVI vs MVI')
    # mpvi
    core_init = random_parameterised_matrix(2, 2, 32, 8)
    core_init = approximate(init, core_init)
    c_init = (core_init, [np.zeros_like(c) for c in core_init])
    params = solve(momentum_bundler(parameterised_value_iteration(mdp, 0.01), 0.9), c_init)
    vs = np.vstack([np.max(build(c[0]), axis=-1) for c in params])
    m = vs.shape[0]

    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='mpvi')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(m-2), cmap='autumn', s=5)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='r', marker='x')

    # mvi
    init = (init, np.zeros_like(init))
    qs = solve(momentum_bundler(value_iteration(mdp, 0.01), 0.9), init)
    vs = np.vstack([np.max(q[0], axis=-1) for q in qs])
    n = vs.shape[0]

    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='mvi')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='summer', s=5)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')


    plt.title('MVI: {}, MPVI {}'.format(n, m))
    plt.legend()

    plt.savefig('traj-figs/mpvi-vs-pvi.png')
    plt.close()




# def generate_mpvi_inits(mdp):
#     print('Running MPVI inits')
#
#     core_init = random_parameterised_matrix(2, 2, 16, 6)
#     init = (core_init, [np.zeros_like(c) for c in core_init])
#     params = solve(momentum_bundler(parameterised_value_iteration(mdp, 0.01), 0.9), init)
#     vs = np.vstack([np.max(build(c[0]), axis=-1) for c in params])
#     m = vs.shape[0]
#     plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', label='mpvi')
#
#
#     new_init = random_reparameterisation(core_init, 2)
#     # sanity check
#     assert np.isclose(build(core_init), build(new_init), atol=1e-4).all()
#     init = (new_init, [np.zeros_like(c) for c in core_init])
#     params = solve(momentum_bundler(parameterised_value_iteration(mdp, 0.01), 0.9), init)
#     vs = np.vstack([np.max(build(c[0]), axis=-1) for c in params])
#     m = vs.shape[0]
#     plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='summer', label='mpvi')
#
#     plt.savefig('traj-figs/mpvi-inits.png')
#     plt.close()
#     """
#     Hmm. I thought this would change the dynamics.
#     It is because the value is only a linear function of the parameters???
#     """

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
#     vs = np.hstack([build(c) for c in params]).T
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

# def argumentparser():
#     parser = argparse.ArgumentParser(description='Visualise losses and returns')
#     parser.add_argument('--logdir', type=str, default='logs',
#                         help='location to save logs')
#     return parser.parse_args()

# def plot(vs1, vs2, save_path):


if __name__ == '__main__':
    n_states, n_actions = 2, 2
    mdp = build_random_mdp(n_states, n_actions, 0.5)
    pis = gen_grid_policies(41)
    vs = polytope(mdp.P, mdp.r, mdp.discount, pis)
    init = np.random.standard_normal((mdp.S, mdp.A))

    experiments = [
        # generate_vi_sgd_vs_mom,
        # generate_pvi_vs_vi,
        # generate_mpvi_vs_mvi,
        #
        # generate_avi_vs_vi,
        #
        generate_PG_vs_VI,
        # generate_PG_vs_PPG,
        #
        # generate_mpvi_inits,
    ]

    for exp in experiments:
        plt.figure(figsize=(16,16))
        plt.scatter(vs[:, 0], vs[:, 1], s=10, alpha=0.5)
        exp(mdp, init)
        # plt.show()
