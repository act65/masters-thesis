"""
Want to generate entropy polytope plots.

TODO
- compare with different abstractions
- ???
"""
import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

from polytope_tools import *

def generate_polytope_densities():
    """

    """
    n_states, n_actions = 2, 2
    M_pis = [generate_Mpi(n_states, n_actions, pi) for pi in gen_grid_policies(2,2,31)]

    nx = 4
    ny = 5
    plt.figure(figsize=(16, 16))

    for i in range(nx*ny):
        print(i)
        P, r = generate_rnd_problem(n_states, n_actions)
        Vs = np.hstack([value_functional(P, r, M_pi, 0.9) for M_pi in M_pis])

        # just set all to be the same probability
        # does that make sense?
        px = 0.1

        pVs = [density_value_functional(px, P, r, M_pi, 0.9) for M_pi in M_pis]

        plt.subplot(nx,ny,i+1)
        fig = plt.scatter(Vs[0, :], Vs[1, :], c=pVs)
        # plt.colorbar()
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig('figs/polytope_densities.png'.format(i))


    # plt.show()

if __name__ =='__main__':
    # generate_rnd_polytope_densities()
    generate_polytope_densities()
