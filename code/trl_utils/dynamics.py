import numpy as np
import trl_utils as trl

def softmax(x):
    assert len(x.shape) == 2
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def onehot(x, n):
    return np.eye(n)[x]

def solve(update_fn, P, r, Mpi, discount, atol=1e-4, maxiters=10):
    """
    Generates the dynamics of update_fn .
    Initial condition = Mpi

    An ODE solver!?
    """
    converged = False
    pis = [Mpi]
    vs = [trl.value_functional(P, r, Mpi, 0.9)]
    count = 0
    while not converged:
        Mpi = update_fn(P, r, Mpi, discount)
        V = trl.value_functional(P, r, Mpi, 0.9)

        if np.isclose(vs[-1] - V, np.zeros(V.shape)).all():
            converged = True
        else:
            vs.append(V)
            pis.append(Mpi)


        count += 1
        if count > maxiters-2:
            break

    return pis, vs

def greedy_solution(V, P):
    # TODO this isnt correct?! should be: argmax_a r + gamma P V?!
    n_states = P.shape[-1]
    n_actions = P.shape[0]//P.shape[-1]
    EV = np.dot(P, V).reshape((n_states, n_actions))  # expected value of each action in each state

    return trl.generate_Mpi(n_states, n_actions, onehot(np.argmax(EV,axis=1), n_actions))

def policy_iteration_update(P, r, M_pi, gamma):
    V = trl.value_functional(P, r, M_pi, gamma)
    return greedy_solution(V, P)

def interior_iteration_update(P, r, M_pi, gamma):
    """
    How can we get more information about the shape of the polytope by
    tranversing the interior, rather than jumping to the edges.

    Like a compressed sensing problem?! Take a set of random actions,
    and mix together the directions.
    """
    pass
