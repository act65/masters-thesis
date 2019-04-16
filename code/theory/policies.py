

def normalise(x, axis):
    return x/np.sum(x, axis=axis, keepdims=True)

def exp_dist(x, lambda_=3.5):  # why 3.5!?
    return lambda_*np.exp(-lambda_*x)

def uniform_simplex(shape):
    # god damn. it's not as simple as I thought to generate uniform distributions on simplexs
    # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    # http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
    return normalise(exp_dist(rnd.random(shape)),axis=1)


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
    # ps = len n_states. ps[0] = len n_actions
    M_pi = np.zeros((n_states, n_states * n_actions))
    M_pi[tuple(zip(*[(i, i*n_actions+j) for i in range(n_states) for j in range(n_actions) ]))] = ps.reshape(-1)
    return M_pi

def pi(M_pi, s, a):
    """
    Let state s be indexed by i and the action a be indexed by j.
    Then we have that M[i, i x |A| + j] = pi(a|s)
    """
    return M_pi[s, s*n_actions+a]
