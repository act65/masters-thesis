import numpy as np

def onehot(idx, N):
    return np.eye(N)[idx]

class DiscreteBipartideWrapper():
    def __init__(self, n_actions, env, mapping=None):
        self.n_actions = n_actions
        self.env = env

        if mapping is None:
            self.reset_mapping()
        else:
            self._mapping = mapping

    def reset_mapping(self):
        self._mapping = np.random.randint(0, 2, (self.env.action_space.n, self.n_actions))

    def _project_action_space(self, a):
        """
        Args:
            a (int): a number to be mapped to another.

        Returns:
            (int): the result.
        """
        return np.argmax(np.dot(self._mapping, onehot(a, self.n_actions)))

    def step(self, a, *args,**kwargs):
        """
        Project the action chosen into dims=env.action_space.n

        Args:
            a (int): an action in n_actions.

        Returns:
            obs, r, done, info (the return from gym.Env.step)
        """
        a_ = self._project_action_space(a)
        return self.env.step(a_, *args, **kwargs)

    def reset(self, *args,**kwargs):
        return self.env.reset(*args, **kwargs)

    def render(self, *args,**kwargs):
        self.env.render(*args, **kwargs)

    def close(self, *args,**kwargs):
        self.env.close(*args, **kwargs)

    def seed(self, *args,**kwargs):
        self.env.seed(*args, **kwargs)

    def sample_action(self):
        return np.random.randint(0, self.n_actions)

def shared_mappings(n_input_actions, n_output_actions, N, depth):
    """
    Is this a valid way to have abstract similarities between action spaces?

    A1 = X1 x Y1
    A2 = X1 x Y2
    A3 = X2 x Y1
    A4 = X2 x Y2
    (binary combinations of different permutations!?!)
    Could also do with layers of a NN - for cts actions??

    Args:
        n_input_actions (int): the number of actions seen by the agent
        n_output_actions (int): the number of true actions according to the env
        N (int): the number of mappings to generate
        depth (int): the depth of the heirarchy of sharing

    Returns:
        (list): a list of np.array's
    """
    def matmul_recurse(mapping, depth):
        if depth == 1:
            return mapping
        else:
            return np.dot(mapping, )

if __name__ == "__main__":
    import gym
    env = gym.make('CartPole-v1')
    env = DiscreteBipartideWrapper(12, env)

    done = False
    env.reset()
    while not done:
        a = env.sample_action()
        obs, r, done, info = env.step(a)

    env.reset_mapping()
    env.reset()
    while not done:
        a = env.sample_action()
        obs, r, done, info = env.step(a)
