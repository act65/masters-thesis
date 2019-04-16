import numpy as np

def generate_rnd_problem(n_states, n_actions):
    P = rnd.random((n_states * n_actions, n_states))**2
    P = P/P.sum(axis=1, keepdims=True)
    r = rnd.random((n_states * n_actions, 1))
    return P, r

def value_functional(P, r, M_pi, gamma):
    """
    V = f(pi)
      = (T-\gamma P_{\pi})^{-1}r_{\pi}
    """
    n = P.shape[-1]
    P_pi = np.dot(M_pi, P)
    r_pi = np.dot(M_pi, r)
    return np.dot(np.linalg.inv(np.eye(n) - gamma*P_pi), r_pi)
