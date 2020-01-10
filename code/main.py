"""
Want to compare
- vanillia
- sample according to similarity
- sample w preference for symmetry

Compare on distributions of MDPs sampled from uniform and symmetric priors.
"""

import numpy as np

env = MDP()

def play_episode():
    s = env.reset()

    while not done:

        s, r = env.step(a)


def parse_params(params):
    n = params.shape[0]
    return params[:n//2, ...], params[n//2:, ...]

def reparam_sample(mu, var):
    e = rnd.standard_normal(var.shape)
    return mu + e * var

def rejection_sample_include_prior(conditional_dist, unnormalised_prior, k):
    """
    Want to sample from P(x, y) = P(x|y)P(y).
    But we dont know how to sample from P(y).
    Rather sample from P(x|y)U(y) and reject samples using P(y).

    # p(x|y)U(y)/(p(x|y) . p(y)) -> 1/ p(y)

    Args:
        unnormalised_distribution (callable): our
    """
    while not accepted:
        x = conditional_dist.sample()
        if k/unnormalised_prior(x) > z:
            break
    return x

def estimate_k(conditional_dist, unnormalised_prior, n):
    samples = [conditional_dist.sample() for _ in range(n)]
    return max([1/unnormalised_prior(s) for s in samples])

def build_abstraction(x):
    """
    Args:
        x(np.ndarray): a n x n matrix in {0, 1}.
    Returns:
        f (callable)
    """


    f = lambda x: np.dot(F, x)
    g = lambda x: np.dot(G, x)
    return f, g

class EnvModel():
    """
    Could be something more.
    - GP?
    - DNN?
    """
    def __init__(self):
        self.dLdp = grad(lambda params, obs: MAP(parse_params(params), obs))

        # init with high uncertainty.

    def update(s, a, r, s_tp1, a_tp1,):
        self.P_params -= self.lr * dLdp(self.P_params, (s, a, s_tp1))
        self.r_params -= self.lr * dLdp(self.r_params, (s, a, r))

    def sample(self):
        P = softmax(reparam_sample(*parse_params(self.P_params)), axis=0)
        r = reparam_sample(*parse_params(self.r_params))
        return P, r


class SimilarityModel():
    def __init__(self):
        pass

    def sample(self):
        return reparam_sample(parse_params(self.params))

    def p(self, x):
        return gaussian_dist(self.mu_similarity, self.var_similarity, x)

    def update(self, obs):
        # observations are Q values!?
        # incremental avg and var
        sim_t = np.max(obs[:, None, :] - obs[None, :, :], axis=-1))
        self.mu_similarity = None
        self.var_similarity = None

def symmetry_measure(x):
    """
    THIS IS THE KEY!
    A soft measure of the symmetry in x.
    Args:
        x (np.ndarray): a n x n matrix.
    Returns:
        float: the amount of symmetry in x. higher output means more symmetry
    """
    # what is this fns computational cost. how can we efficiently approx?

    # for all P in permutations.
    # how close similar is x ~ P.x?
    # if similar to all permutations, then max symmetry
    # if similar to only one (the identity) then min symmetry

    permutations = np.stack([])
    x_ps = np.einsum('ijl,lk->ijk', P, x)  # not sure this is what I want? we are permuting the rows.
    err = np.mean(np.linalg.norm(x[None, :, :] - x_ps, axis=(1,2)))
    return 1-sigmoid(err)

class TS_Player():
    def __init__(self, n_states, n_actions):
        self.Q_t = Q
        self.s_tm1, self.a_tm1, self.r_tm1 =(None, None, None)

    def __call__(self, s, r_tm1):
        # sample a MDP
        P, R = self.env_model.sample()

        # act optimally wrt the sampled MDP
        self.Q_t = utils.bellman_optimality_operator(P, R, discount, self.Q_t)  # q learning.
        a = utils.sample(np.exp(self.Q_t[s, :]))

        # update parameters and loop variables
        self.env_model.update(self.s_tm1, self.a_tm1, self.r_tm1, s, a)
        self.s_tm1, self.a_tm1 = s, a

        return a

class Similar_TS_Player(TS_Player):
    def __init__(self, n_states, n_actions):
        super(TS_Player).__init__(self, n_states, n_actions)

    def sample(self):
        P, r = self.env_model.sample()
        f, g = build_abstraction(self.similarity_model.sample())
        return P, r, (f, g)

    def update(self, s, a, r, s_tp1, a_tp1, Q_t):
        self.env_model.update(s, a, r, s_tp1, a_tp1,)
        self.similarity_model.update(Q_t)

    def __call__(self, s):
        P, r, (f, g) = self.sample()
        self.Q_t = g(utils.bellman_optimality_operator(f(P), f(r), f(self.Q_t), self.discount))  # q learning. in the abstracted space
        return utils.sample(np.exp(self.Q_t[s, :]))

class Symmetric_TS_Player(Similar_TS_Player):
    def __init__(self, n_states, n_actions):
        super(Similar_TS_Player).__init__(self, n_states, n_actions)
        self.decay = 0.9

    def sample(self):
        P, r = self.env_model.sample()
        self.k = self.decay*self.k + estimate_k(self.similarity_model, symmetry_measure, 10)  # need to do some reading on this. how important is it?
        f, g = build_abstraction(rejection_sample_include_prior(self.similarity_model, symmetry_measure, (1-self.decay)*self.k))
        return P, r, (f, g)
