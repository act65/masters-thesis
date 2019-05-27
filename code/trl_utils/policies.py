import numpy as np
import numpy.random as rnd
import itertools

def generate_Mpi(n_states, n_actions, ps):
    """
    A policy is represented by a block diagonal |S| x |S||A| matrix M.

    Args:
        n_states (int): the number of states
        n-actions (int): the number of actions
        ps (array[n_states, n_actions]): the probabilities of taking action a in state s.

    Returns:
        (array[n_states, n_states x n_actions])
    """
    A = np.ones((1, n_actions))
    S = np.eye(n_states)

    M_pi = np.zeros((n_states, n_states * n_actions))
    M_pi[np.where(np.equal(1, np.kron(S, A)))] = ps.reshape(-1)
    return M_pi

def pi(M_pi, s, a):
    """
    Let state s be indexed by i and the action a be indexed by j.
    Then we have that M[i, i x |A| + j] = pi(a|s)
    """
    return M_pi[s, s*n_actions+a]

def normalise(x, axis):
    return x/np.sum(x, axis=axis, keepdims=True)

def exp_dist(x, lambda_=3.5):  # why 3.5!?
    return lambda_*np.exp(-lambda_*x)

def uniform_simplex(shape):
    # god damn. it's not as simple as I thought to generate uniform distributions on simplexs
    # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    # http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
    return normalise(exp_dist(rnd.random(shape)),axis=1)

def generate_rnd_policy(n_states, n_actions):
    return generate_Mpi(n_states, n_actions, uniform_simplex((n_states, n_actions)))

# def gen_grid_policies(n_states, n_actions, N=11):
#     # TODO need to generalise to nD
#     # only works for 2 states, 2 actions
#     x = np.linspace(0, 1, N)
#     return [np.array([x[i],1-x[i],x[j],1-x[j]]) # will not generalise to n states
#                   for i in range(N)
#                   for j in range(N)]

def gen_grid_policies(n_states, n_actions, N=11):
    """
    This scales badly!!!
    """
    simplicies = list(itertools.product(np.linspace(0, 1, N), repeat=n_actions))
    simplicies = [s/np.sum(s) for s in simplicies]
    return [np.vstack(x) for x in itertools.product(*[simplicies for _ in range(n_states)])]

def get_deterministic_policies(n_states, n_actions):
    simplicies = list([np.eye(n_actions)[i] for i in range(n_actions)])
    pis = list(itertools.product(*[simplicies for _ in range(n_states)]))
    return [np.stack(p) for p in pis]
