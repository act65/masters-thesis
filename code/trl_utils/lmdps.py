import numpy as np

def lmdp_embedding(P, r):
    """

    """
    M_pi = np.array([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]])
    # use a the transition and reward of a random policy as q(x) and p(x' | x)
    q = np.dot(M_pi, r)
    p = np.dot(M_pi, P)

    return p, q

def lmdp_embedding(P, r):
    """
    Args:
        P (np.array): The transition function. shape = [n_states, n_states, n_actions]
        r (np.array): The reward function. shape = [n_states, n_actions]

    Returns:
        p (np.array): the uncontrolled transition matrix
        q (np.array): the pseudo reward

    Needs to be solved for every state.
    """
    # QUESTION Are there similarities between the embedding in each state?
    # QUESTION If we cannot accurately embed an action. What about using an option?
    # QUESTION How does this embedding change the set of policies that can be represented? Does this transformation preserve the optima?
    # QUESTION What if we have a prior on policies. The we can construct p(x' | x) via p(x' | x) = sum_a p(a | x) p(x' | x, a) and infer q(x).
    def embed_state(idx_x):
        """
        Solves
        r(s, a) = q(s) + E_s'~p(s' | s, a) log( p(s' | s, a) / p(x' | x))
        m_x' = log p (x' | x)
        b_a = r(x, a) - E_s'~p(s' | s, a) log( p(s' | s, a) / p(x' | x))
        D_ax' = p(x' | x, a)

        q1 - Dm = b
        Dc = b
        c = q1 - m
        """
        D = P[:, idx_x, :]
        b = r[idx_x] - np.sum(P[:, idx_x, :] * np.log(P[:, idx_x, :]), axis=0)

        c = np.dot(np.linalg.inv(D), b)
        q = -np.log(np.sum(np.exp(-c)))

        m = np.sum(q) - c
        p = np.exp(m)

        return p, q

    pqs = [embed_state(i) for i in range(P.shape[0])]
    return tuple([np.stack(val) for val in zip(*pqs)])

def lmdp_solver(p, q, discount):
    """
    z = QPz

    Args:

    Returns:
        (): the optimal policy
        (): the value of the optimal policy
    """
    Q = np.diag(np.exp(np.squeeze(-q)))

    z = solve(Q, p, a=discount)
    v = -np.log(z)  # BUG am getting negative values while r is positive.
    # u*(x' | x) = p(x' | x) z(x') / G[z](x)
    # G(x) = sum_x' p(x' | x) z(x')
    u = np.einsum('ij,ik->ij', p, z) / np.einsum('ij,ik->j', p, z)
    return u.T, v  # not sure why i need to transpose this.

def lmdp_control_decoder(u, P):
    """
    Give optimal control dynamics. We want to infer the optimal control policy.
    Find pi(a | x) s.t. u(x' | x) = sum_a pi(a | x) P(x'| x, a)
    """
    # TODO need to convert back into discrete actions!
    pass

def solve(Q, p, a=None):
    # Solve z = QPz
    if a is None:
        A = np.eye(Q.shape[0]) - np.dot(Q, p)
        # Ax = b. But b = 0. Therefore det(A) = 0
        val, vec = np.linalg.eig(np.dot(Q, p))
        # BUG no this doesnt solve z = QPz, it sovles l.z = QPz.
        # this doesnt make sense to me. would expect this system to diverge!? value is infinite...?

        x = vec[:, 0] * val[0] # np.dot(np.diag(val), vec)
        # QUESTION which eigen vector should I pick?
        return x

    # Solve z = QPz^a
    # HACK want a better way to solve this. HOW?
    elif a is not None:
        A = np.dot(Q, p)
        fn = lambda x: np.dot(A, x**a)
        solved = False
        while not solved:
            try:  # try for a few different random inits.
                z = matrix_power(fn, np.random.standard_normal((p.shape[-1], 1)))
                solved = True
            except ValueError as E:
                print(E)
        return z

def matrix_power(fn, init):
    # huh, this solver is pretty consistent.
    xs = [init]
    while not converged(xs):
        xs.append(fn(xs[-1]))
    return xs[-1]

def converged(xs):
    if len(xs) <= 1:
        return False
    elif len(xs) > 1000:
        raise ValueError('not converged')
    elif np.isnan(xs[-1]).any():
        raise ValueError('Nan')
    else:
        return np.isclose(xs[-1], xs[-2]).all()
