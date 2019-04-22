import numpy as np
import trl_utils as trl

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
