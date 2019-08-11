import jax.numpy as np
import numpy
import numpy.random as rnd
from jax import grad, jit
from numpy import linalg
import src.utils as utils

"""
Related??? https://arxiv.org/pdf/1901.11530.pdf
"""

def construct_mdp_basis(det_pis, mdp):
    V_det_pis = [utils.value_functional(mdp.P, mdp.r, pi, mdp.discount) for pi in det_pis]
    return np.hstack(V_det_pis)  # [n_states x n_dep_pis]

def mdp_topology(det_pis):
    n = len(det_pis)
    A = numpy.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = np.sum(np.abs(det_pis[i]- det_pis[j]))
                A[i, j] = 1 if diff == 2 else 0
    return A

def estimate_coeffs(basis, x):
    """
    \sum \alpha_i . V^d_i = V_pi
    V_ds . a = V_pi
    Ax = b
    """
    # TODO could instead do some sort of sparse solver?
    # or min of l2 distances?
    # and they should be all positive?!
    alphas = np.dot(x, linalg.pinv(basis))
    return alphas

def mse(x, y):
    return np.sum(np.square(x-y))

def sparse_coeffs(basis, b, gamma=1e-2, lr=1e-1):
    """
    Want x s.t. b ~= basis . x
    min || basis . x - b ||_2^2 + gamma * ||x||_1
    """
    assert basis.shape[0] == b.shape[0]

    def sparse_loss(x):
        a = utils.softmax(x)  # convex combination
        return mse(np.dot(basis, a), b) + gamma * np.sum(np.abs(a))

    dLdx = grad(sparse_loss)
    def update_fn(x):
        g = dLdx(x)
        # print(x)
        return x - lr * g

    init = 1e-3*rnd.standard_normal((basis.shape[1], ))
    return utils.solve(update_fn, init)[-1]
