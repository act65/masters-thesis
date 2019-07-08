from utls import *
import jax.numpy as np
from jax import jit, grad
import numpy.random as rnd
import matplotlib.pyplot as plt
import copy
import json

def vi_vector_field(mdp, qs, lr):
    update_fn = value_iteration(mdp, lr)
    delta = lambda q: np.max(update_fn(q) - q, axis=1)
    return np.vstack([delta(q) for q in qs])

def approximate(v, cores):
    """
    cores = random_parameterised_matrix(2, 1, d_hidden=8, n_hidden=4)
    v = rnd.standard_normal((2,1))
    cores_ = approximate(v, cores)
    print(v, '\n',value(cores_))
    """
    loss = lambda cores: np.sum(np.square(v - value(cores)))
    dl2dc = grad(loss)
    l2_update_fn = lambda cores: [c - 0.01*g for g, c in zip(dl2dc(cores), cores)]
    init = (cores, [np.zeros_like(c) for c in cores])
    traj = solve(momentum_bundler(l2_update_fn, 0.9), init)[-1]
    return traj[0]  # the parameters, not the momentum variables

def pvi_vector_field(mdp, many_cores, lr):
    update_fn = parameterised_value_iteration(mdp, lr)
    delta = lambda cores: np.max(value(cores),axis=1) - np.max(value(update_fn(cores)), axis=1)
    return np.vstack([delta(cores) for cores in many_cores])

def fitted_cores(mdp, qs):
    cores = random_parameterised_matrix(2, 2, d_hidden=8, n_hidden=4)
    approxs = [approximate(q, cores) for q in qs]
    return approxs

def cts_lr_fields(mdp):
    lrs = np.linspace(0, 1, 10)

    pis = gen_grid_policies(N=11)
    vs = polytope(mdp.P, mdp.r, mdp.discount, pis)
    qs = [np.einsum('ijk,i->jk', mdp.P, v) for v in vs]
    many_cores = fitted_cores(mdp, qs)

    for lr in lrs:
        dpvis = pvi_vector_field(mdp, many_cores, lr)
        dvis = vi_vector_field(mdp, qs, lr)

def generate_pvi_vs_vi(mdp):
    print('\nRunning PVI vs VI')
    lr = 0.1

    pis = gen_grid_policies(N=11)
    vs = polytope(mdp.P, mdp.r, mdp.discount, pis)
    qs = [np.einsum('ijk,i->jk', mdp.P, v) for v in vs]

    plt.figure(figsize=(16,16))

    # pvi
    many_cores = fitted_cores(mdp, qs)
    dpvis = pvi_vector_field(mdp, many_cores, lr)
    mags = np.linalg.norm(dpvis, axis=1, keepdims=True)
    normed_dpvis = dpvis/mags

    plt.subplot(2,1,1)
    plt.title('Pamameterised VI')
    plt.quiver(vs[:, 0], vs[:, 1], normed_dpvis[:, 0], normed_dpvis[:,1], mags)
    plt.colorbar()

    # vi
    dvis = vi_vector_field(mdp, qs, lr)
    mags = np.linalg.norm(dvis, axis=1, keepdims=True)
    normed_dvis = dvis/mags

    plt.subplot(2,1,2)
    plt.title('VI')
    plt.quiver(vs[:, 0], vs[:, 1], normed_dvis[:, 0], normed_dvis[:,1], mags)
    plt.colorbar()

    plt.show()



################################################
#
# # @jit
# def pg_vector_field(mdp, pis, lr):
#     V = lambda logit: value_functional(mdp.P, mdp.r, softmax(logit, axis=1), mdp.discount)
#     logits = [np.log(pi+1e-8) for pi in pis]
#     pg_update_fn = policy_gradient_iteration_logits(mdp, lr)
#     return np.hstack([V(pg_update_fn(logit)) - V(logit) for logit in logits]).T
#
#
# def generate_pg_vs_vi(mdp):
#     print('\nRunning PG vs VI')
#     lr = 0.1
#
#     pis = gen_grid_policies(N=31)
#     vs = polytope(mdp.P, mdp.r, mdp.discount, pis)
#
#     plt.figure(figsize=(16,16))
#
#     # pg
#     dpgs = pg_vector_field(mdp, pis, lr)
#     mags = np.linalg.norm(dpgs, axis=1, keepdims=True)
#     normed_dpg = dpgs/mags
#
#     plt.subplot(2,1,1)
#     plt.quiver(vs[:, 0], vs[:, 1], normed_dpg[:, 0], normed_dpg[:,1], mags)
#     plt.colorbar()
#
#     # vi
#     dvis = vi_vector_field(mdp, pis, lr)
#     mags = np.linalg.norm(dvis, axis=1, keepdims=True)
#     normed_dvis = dvis/mags
#
#     plt.subplot(2,1,2)
#     plt.quiver(vs[:, 0], vs[:, 1], normed_dvis[:, 0], normed_dvis[:,1], mags)
#
#
#     plt.show()


if __name__ == '__main__':
    n_states, n_actions = 2, 2
    mdp = build_random_mdp(n_states, n_actions, 0.5)
    generate_pvi_vs_vi(mdp)
    # cts_lr_fields(mdp)
