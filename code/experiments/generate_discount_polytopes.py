import numpy as np
import matplotlib.pyplot as plt

import trl_utils as trl

def generate_discounted_polytopes_forvideo():
    """
    ffmpeg -framerate 10 -start_number 0 -i disc%d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
    """
    n_states = 2
    n_actions = 2

    N = 100
    n = 4
    M_pis = [trl.generate_Mpi(n_states, n_actions, pi) for pi in trl.gen_grid_policies(2,2,31)]
    Prs = [trl.generate_rnd_problem(n_states,n_actions)for _ in range(n*n)]

    for i, discount in enumerate(np.linspace(0, 1-1e-4,N)):
        print(i)
        plt.figure()
        for j in range(n*n):
            ax = plt.subplot(n,n,j+1)
            P, r = Prs[j]
            Vs = np.hstack([trl.value_functional(P, r, M_pi, discount) for M_pi in M_pis])
            fig = plt.plot(Vs[0, :], Vs[1, :], 'b.')[0]
            ax.set_xlim(np.min(Vs[0, :]),np.max(Vs[0, :]))
            ax.set_ylim(np.min(Vs[1, :]),np.max(Vs[1, :]))
            plt.title('{:.4f}'.format(discount))

            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.savefig('../pictures/figures/discounts/disc{}.png'.format(i))

def generate_discounted_polytopes():
    n_states = 2
    n_actions = 2

    ny = 6
    nx = 12
    M_pis = [trl.generate_Mpi(n_states, n_actions, pi) for pi in trl.gen_grid_policies(2,2,31)]
    Prs = [trl.generate_rnd_problem(n_states,n_actions)for _ in range(ny)]
    count = 0
    plt.figure(figsize=(16,16))
    for j in range(ny):
        print(j)
        for i, discount in enumerate(np.linspace(0, 1-1e-4,nx)):
            count += 1
            ax = plt.subplot(ny,nx,count)

            P, r = Prs[j]
            Vs = np.hstack([trl.value_functional(P, r, M_pi, discount) for M_pi in M_pis])
            pVs = [trl.density_value_functional(0.1, P, r, M_pi, 0.9) for M_pi in M_pis]

            fig = plt.scatter(Vs[0, :], Vs[1, :], c=pVs)
            ax.set_xlim(np.min(Vs[0, :]),np.max(Vs[0, :]))
            ax.set_ylim(np.min(Vs[1, :]),np.max(Vs[1, :]))

            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig('../pictures/figures/discounts.png')

if __name__ == '__main__':
    generate_discounted_polytopes()
