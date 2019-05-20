"""
Want to generate vector fields

NOTE. Want a measure of how easy a space is to optimise.
The homogeneity of the gradients. !!!
For each neighborhood, take the laplacian!? / the det!?
"""
import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

import trl_utils as trl

def generate_field():
    """
    """
    n_states = 2
    n_actions = 2
    discount = 0.8

    n = 3
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        P, r = trl.generate_rnd_problem(n_states, n_actions)
        M_pis = [trl.generate_Mpi(n_states, n_actions, pi) for pi in trl.gen_grid_policies(2,2,21)]
        Vs = np.hstack([trl.value_functional(P, r, M_pi, discount) for M_pi in M_pis])
        dVs = np.sum(np.stack([trl.value_jacobian(np.dot(M_pi, r), np.dot(M_pi, P), discount) for M_pi in M_pis]), axis=1)  # QUESTION 1 or 2?
        n_dVs = np.stack([v/np.linalg.norm(v) for v in dVs], axis=1)

        fig = plt.quiver(Vs[0, :], Vs[1, :], dVs[0, :], dVs[1, :], np.sum(dVs, axis=1))
        #
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.colorbar()

    plt.tight_layout()
    # plt.savefig('../pictures/figures/polytope_vector_fields.png')
    plt.show()


def distribution_of_grads():
    """
    """
    n_states = 2
    n_actions = 2
    discount = 0.8

    n = 10000
    vals = []
    M_pis = [trl.generate_Mpi(n_states, n_actions, pi) for pi in trl.gen_grid_policies(2,2,4)]
    for i in range(n):
        P, r = trl.generate_rnd_problem(n_states, n_actions)
        dVs = np.sum(np.stack([trl.value_jacobian(np.dot(M_pi, r), np.dot(M_pi, P), discount) for M_pi in M_pis]), axis=1)  # 1 or 2?
        dVs = list(sorted(dVs, key=np.linalg.norm))
        vals.append(np.linalg.norm(dVs[0] - dVs[-1]))

    # print(vals)
    plt.hist(vals, bins=100)
    plt.show()


if __name__ =='__main__':
    generate_field()
    # distribution_of_grads()
