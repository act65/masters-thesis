import numpy as np
import matplotlib.pyplot as plt

import trl_utils as trl

def softmax(x, axis=1, temp=10.0):
    x *= temp
    return np.exp(x)/np.sum(np.exp(x), axis=1)

def policy_iteration_partitions(n_states, n_actions, P, r, N=31):
    """
    For each policy on a uniform grid.
    Use that policy as an init and solve the MDP.
    Lens = how many steps are required to solve the MDP.
    """
    lens, Vs, pis = [], [], []

    M_pis = [trl.generate_Mpi(n_states, n_actions, pi)
             for pi in trl.gen_grid_policies(n_states,n_actions,N)]
    for M_pi in M_pis:
        pi, vs = trl.solve(trl.policy_iteration_update, P, r, M_pi, 0.9)

        Vs.append(vs[0])
        pis.append(pi[0][:, ::2].sum(axis=1, keepdims=True))
        lens.append(len(vs))

    return lens, Vs, pis

def plot_partitions(Vs, lens):
    fig = plt.scatter(*[x for x in np.hstack(Vs)], c=lens, s=1)
    plt.colorbar()
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

def generate_partition_figures():
    """
    Print how many gpi steps are required to converge to the optima wrt pi/v.
    """
    n_states = 2
    n_actions = 2  # TODO want to generalise to nD.
    nx = 5
    ny = 2
    plt.figure(figsize=(16, 16))
    count = 0
    for i in range(nx*ny):
        print(i)
        P, r = trl.generate_rnd_problem(n_states, n_actions)
        lens, Vs, pis = policy_iteration_partitions(n_states, n_actions, P, r, 41)

        count += 1
        plt.subplot(nx,ny*2,count)
        plot_partitions(pis, lens)

        count += 1
        plt.subplot(nx,ny*2,count)
        plot_partitions(Vs, lens)


    plt.tight_layout()
    plt.savefig('../pictures/figures/gpi-partitions.png')


if __name__ =='__main__':
    generate_partition_figures()
