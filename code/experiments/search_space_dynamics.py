"""
How can we measure the difference / bias in trajectories?
Why do we care?
How can we think of inductive bias in RL?
A bias towards certain policies?

When we have perfect evaluations there is only one optimal policy...
A bias doesnt do anything!?

When we have imperfect observations / evaluations.
We might not know which MDP we are in. Or the value of our policy.
A bias might prefer certain MDPs, or certain policies!? Or certain types of exploration?!
"""

import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

import trl_utils as trl


def plot_trajectory(traj):
    plt.scatter()

def dynamics():
    n_states, n_actions = 3, 3

    P, r = trl.generate_rnd_problem(n_states, n_actions)


    for algol in [value_iteration, policy_iteration, parameter_iteration]:
        vs, pis = trl.solve(P, r, discount, algol)
        plot_trajectory(vs)



if __name__ =='__main__':
    dynamics()
