"""
Want to generate pics of value under approximate transition fns

Question is. How likely is this error to push GPI to an erroneous solution?
Also. How would this scale in many dimension!?
More likely to be a partition that is 'close'.
"""
import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

import trl_utils as trl

def generate_lmdp_solutions():
    n_states = 2
    n_actions = 2
    discount = 0.9

    P = rnd.random((n_states, n_states, n_actions))
    P = P/np.sum(P, axis=(1,2))
    r = np.random.random((n_states, n_actions))

    M_pis = [trl.generate_Mpi(n_states, n_actions, pi) for pi in trl.gen_grid_policies(2,2,31)]

    p, q = trl.lmdp_embedding(P, r)
    u_star, v_star = trl.lmdp_solver(p, q, discount)
    print(u_star, v_star)

    # Vs = np.hstack([trl.value_functional(P, r, M_pi, discount) for M_pi in M_pis])




if __name__ =='__main__':
    generate_lmdp_solutions()
