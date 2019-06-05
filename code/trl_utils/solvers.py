import numpy as np
import trl_utils as trl

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

def solve_optimal(P, r, discount, n=5):
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
