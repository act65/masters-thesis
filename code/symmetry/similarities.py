import numpy as np

def whiten(x):
    return (x - np.mean(x)) / (np.sqrt(np.var(x)) + 1e-8)

def onehot(x, N):
    return np.eye(N)[x]

def discount_rewards(rs, discount):
    Rs = [rs[-1]]
    for r in reversed(list(rs[:-1])):
        Rs.append(r + discount*Rs[-1])
    return (1-discount)*np.stack(list(reversed(Rs)))

def cosine_similarity(x, y):
    return np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y))

def sigmoid(x, temp=1.0):
    return 1 / (1 + np.exp(-temp*x))

def value_pairs(traj, n_states, n_actions):
    """
    Returns training pairs.
    """
    # PROBLEM this doesnt seem like a good represnetation!?
    # it wont contain information about which action to take?!
    s_s, a_s, r_s = zip(*traj)
    r_s = whiten(r_s)
    v_s = discount_rewards(r_s, 0.99)

    idx = list(range(len(s_s)))
    for i, s_i, a_i, v_i in zip(idx, s_s, a_s, v_s):
        for j, s_j, a_j, v_j in zip(idx, s_s, a_s, v_s):
            if j != i:
                x_i = np.concatenate([s_i, onehot(a_i, n_actions)])
                x_j = np.concatenate([s_j, onehot(a_j, n_actions)])
                # two examples, and their similarity
                yield x_i, x_j, sigmoid(cosine_similarity(v_i , v_j), 10)

def transition_pairs(traj, n_states, n_actions):
    """
    Returns training triples.
    x1, x2, similarity(x1, x2)
    Two state-actions are 'similar' if they change the
    state/reward in a similar way.
    """
    # PROBLEM this assumes it makes sense to take the difference between states
    # aka that the state space has some sort of ordering / locality

    # QUESTION does this even make sense?
    # one state might change reward 1->2, and another 100->101.
    # we are saying that these should have similar representation!?!?
    # that doesnt seem right.
    s, a, r = zip(*traj)
    r = whiten(r)

    for i in range(1, len(s)-1):
        for j in range(1, len(s)-1):
            if i != j:
                x_i = np.concatenate([s[i], onehot(a[i], n_actions)])
                x_j = np.concatenate([s[j], onehot(a[j], n_actions)])

                y_i_t = np.concatenate([s[i], [r[i-1]]])
                y_i_tp1 = np.concatenate([s[i+1],  [r[i]]])
                y_j_t = np.concatenate([s[j], [r[j-1]]])
                y_j_tp1 = np.concatenate([s[j+1], [r[j]]])

                dy_i = y_i_tp1 - y_i_t
                dy_j = y_j_tp1 - y_j_t

                yield x_i, x_j, sigmoid(cosine_similarity(dy_i , dy_j), 10)

if __name__ == '__main__':
    traj = [
        (np.random.random((4,)),
        np.random.randint(0, 4),
        np.random.random((1,)))
        for _ in range(1000)]

    gen = transition_pairs(traj, 6, 4)
    print(next(gen))

    # gen = value_pairs(traj, 6, 4)
    # print(next(gen))
