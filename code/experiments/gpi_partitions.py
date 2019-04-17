import numpy as np

def softmax(x, axis=1, temp=10.0):
    x *= temp
    return np.exp(x)/np.sum(np.exp(x), axis=1)

def greedy_solution(V, P):
    n_states = P.shape[-1]
    n_actions = P.shape[0]//P.shape[-1]
    EV = np.dot(P, V).reshape((n_states, n_actions))  # expected value of each action in each state
    return generate_Mpi(n_states, n_actions, np.clip(np.round(softmax(EV)),0, 1))

def policy_iteration_update(P, r, M_pi, gamma):
    V = value_functional(P, r, M_pi, gamma)
    return greedy_solution(V, P)


def generate_rnd_partition_figures():
    pass

if __name__ =='__main__':
    generate_rnd_partition_figures()
