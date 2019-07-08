import numpy.random as rnd
import jax.numpy as np
from jax import vmap

W = rnd.random((5,5))
x = rnd.random((5,1))
X = rnd.random((100,5,1))

def f(x, W):
    print(x.shape)
    return np.dot(W, x)

vf = vmap(lambda x: f(x, W))
print(vf(X).shape)

def value_functional(P, r, pi, discount):
    """
    V = r_{\pi} + \gamma P_{\pi} V
      = (I-\gamma P_{\pi})^{-1}r_{\pi}

    Args:
        P (np.ndarray): [n_states x n_states x n_actions]
        r (np.ndarray): [n_states x n_actions]
        pi (np.ndarray): [n_states x n_actions]
        discount (float): the temporal discount value
    """
    print(pi.shape)
    n = P.shape[-1]
    # P_{\pi}(s_t+1 | s_t) = sum_{a_t} P(s_{t+1} | s_t, a_t)\pi(a_t | s_t)
    P_pi = np.einsum('ijk,jk->ij', P, pi)
    r_pi = np.expand_dims(np.einsum('ij,ij->i', pi, r), 1)

    # assert np.isclose(pi/pi.sum(axis=1, keepdims=True), pi).all()
    # assert np.isclose(P_pi/P_pi.sum(axis=0, keepdims=True), P_pi, atol=1e-4).all()

    # BUG why transpose here?!?!
    vs = np.dot(np.linalg.inv(np.eye(n) - discount*P_pi.T), r_pi)
    # print(vs.shape, P_pi.shape)
    return vs

P = rnd.random((2,2,2))
r = rnd.random((2,2))
gamma = 0.9
pis = rnd.random((100, 2,2))

vvf = vmap(lambda pi: value_functional(P, r, pi, gamma))
vvf(pis)
