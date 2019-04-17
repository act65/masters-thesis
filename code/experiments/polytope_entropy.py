"""
Want to generate entropy polytope plots.

TODO
- compare with different abstractions
- ???
"""
import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

import trl_utils as trl

def normalise(x, axis):
    return x/np.sum(x, axis=axis, keepdims=True)

def exp_dist(x, lambda_=3.5):  # why 3.5!?
    return lambda_*np.exp(-lambda_*x)

def uniform_simplex(shape):
    # god damn. it's not as simple as I thought to generate uniform distributions on simplexs
    # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    # http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
    return normalise(exp_dist(rnd.random(shape)),axis=1)

def generate_rnd_polytope_densities():
    """
    Randomly sample from `n_action` simplexes.
    And then use the value functional to solve for V.
    """
    generate_rnd_policy = lambda n_states, n_actions: trl.generate_Mpi(n_states,
                                                                   n_actions,
                                                                   uniform_simplex((n_states, n_actions)))
    n_states = 2


    n = 4
    N = 100000

    n_actions = [2]*8 + [3]*8
    plt.figure(figsize=(16, 16))
    for i in range(n*n):
        print(i)
        P, r = trl.generate_rnd_problem(n_states, n_actions[i])

        # TODO can we calculate this analytically? p(V) = inv(abs(1/det(dVdx))) p(x)
        Vs = np.hstack([trl.value_functional(P, r, generate_rnd_policy(n_states, n_actions[i]), 0.9) for _ in range(N)])
        im, _, _ = np.histogram2d(Vs[0, :], Vs[1, :], (100, 100))

        plt.subplot(n,n,i+1)
        # BUG not sure why we need to flip the ims
        fig = plt.imshow(np.flip(im, axis=1), cmap=plt.cm.jet)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig('../pictures/figures/polytope_rnd_densities.png'.format(i))

if __name__ =='__main__':
    generate_rnd_polytope_densities()
