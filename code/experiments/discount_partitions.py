import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import trl_utils as trl

import collections

def most_common(lst):
    data = collections.Counter(lst)
    return max(lst, key=data.get)

def check_differences(policies):
    m = len(policies)
    count = 0
    for i in range(m):
        for j in range(m):
            if i > j:
                if not np.equal(policies[i], policies[j]).all():
                    count += 1
    return count

def solve(P, r, discount, n=5):
    # n the number of retries
    # generate optimal pi
    n_states = P.shape[-1]
    n_actions = P.shape[0]//n_states
    stars = []

    done = False
    while not done:
        Mpi = trl.generate_rnd_policy(n_states, n_actions)
        pis, vs = trl.solve(trl.policy_iteration_update, P, r, Mpi, discount, atol=1e-5, maxiters=20)
        pi_star = pis[-1]
        stars.append(pi_star)
        diffs = check_differences(stars[-n:])
        if diffs == 0 and len(stars) > n:
            done = True
    return stars[-1]

def discount_partitions():
    """
    Plot the partitions of discounts wrt optimal policies.

    Ok, so a MDP that has many changes of the optimal policy wrt to discounts is
    a 'harder' one than one that has few changes.
    Can solve once / few times and transfer the policy between different discounts.
    When / why can this be done?
    """
    n_states, n_actions = 3, 3
    n = 100
    discounts = np.linspace(0.1, 0.999, n)


    for _ in range(10):
        P, r = trl.generate_rnd_problem(n_states, n_actions)
        stars = []
        for i in range(n):
            stars.append(solve(P, r, discounts[i]))

        diffs = [np.sum(np.abs(stars[i]-stars[i+1])) for i in range(n-1)]
        plt.plot(discounts[:-1], np.cumsum(diffs))

    plt.title('Discount partitions')
    plt.xlabel('Discount')
    plt.ylabel('Cumulative changes to the optimal policy')

    # plt.show()
    plt.savefig('../pictures/figures/discount-partitions.png')

if __name__ == '__main__':
    discount_partitions()
