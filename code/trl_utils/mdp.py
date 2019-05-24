import numpy as np
import numpy.random as rnd

def generate_rnd_problem(n_states, n_actions):
    P = rnd.random((n_states * n_actions, n_states))**2
    P = P/P.sum(axis=1, keepdims=True)
    r = rnd.random((n_states * n_actions, 1))
    return P, r

def value_functional(P, r, M_pi, discount):
    """
    V = r_{\pi} + \gamma P_{\pi} V
      = (I-\gamma P_{\pi})^{-1}r_{\pi}

    Args:
        P ():
        r ():
        M_pi ():
        discount (float): the temporal discount value
    """
    n = P.shape[-1]
    P_pi = np.dot(M_pi, P)
    r_pi = np.dot(M_pi, r)
    return np.dot(np.linalg.inv(np.eye(n) - discount*P_pi), r_pi)

def density_value_functional(px, P, r, M_pi, discount):
    P_pi = np.dot(M_pi, P)
    r_pi = np.dot(M_pi, r)

    J = value_jacobian(r_pi, P_pi, discount)
    return probability_chain_rule(px, J)

def value_jacobian(r_pi, P_pi, discount):
    """
    Returns:
        [inputs x outputs] ???
    """
    return r_pi * (np.eye(P_pi.shape[0]) - discount * P_pi)**(-2)

def entropy_jacobian(pi):
    """
    H(pi) = - sum p log p
    dHdpi(j) = 1 + log p
    """
    return -1 - np.log(pi)

def probability_chain_rule(px, J):
    """
    p(f(x)) = abs(|J|)^-1 . p(x)
    """
    return (np.abs(np.linalg.det(J))**(-1)) * px

def get_pi(M_pi):
    n, m = M_pi.shape
    n_actions = m // n
    pi = np.zeros((n, n_actions))
    for i in range(n):
        for j in range(n_actions):
            pi[i, j] = M_pi[i, i*n_actions + j]
    return pi
