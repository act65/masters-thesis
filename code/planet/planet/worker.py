import gym

import copy
import numpy as np
import random

def onehot(idx, N):
    return np.eye(N)[idx]

class ReplayBuffer(object):
    def __init__(self, max_size, overflow_cache_size=100):
        self.max_size = max_size
        self.cur_size = 0
        self.buffer = {}
        self.overflow_cache_size = overflow_cache_size

    @property
    def size(self):
        return self.cur_size

    def add(self, episode):
        """Add episodes to buffer."""

        self.buffer[self.cur_size] = episode
        self.cur_size += 1

        # batch removals together for speed!?
        if len(self.buffer) > self.max_size+self.overflow_cache_size:
            self.remove_n(self.overflow_cache_size)

    def remove_n(self, n):
        """Get n items for removal."""
        # random removal
        idxs = list(random.sample(range(self.cur_size), self.cur_size-n))
        # idxs = list(range(n))  # removes the oldest
        new_buffer = np.array(list(self.buffer.values()))[idxs]
        # overwrites the old dict. (possibly expensive, but not sure of a better way...)
        n = len(new_buffer)
        self.buffer = dict(zip(range(n), new_buffer))
        self.cur_size = n

    def get_batch(self, n):
        """Get batch of episodes to train on."""
        # random batch
        idxs = random.sample(range(self.cur_size), n)
        batch = [self.buffer[idx] for idx in idxs]  # a list of [[obs_s, a_s, r_s], ...]
        batch = [np.stack(x, axis=0) for x in zip(*batch)]
        return batch

class Worker():
    """
    Worker is in charge of;
    - evaluating the current policy
    - adding experience to the buffer
    - (could also be used to compute gradients)
    """
    def __init__(self, env_name, player, maxsteps=100):
        self.buffer = ReplayBuffer(max_size=10000)

        self.env = gym.make(env_name)
        obs = self.env.reset()

        self.player = player(n_inputs=obs.shape[0], n_actions=self.env.action_space.n)

        self.maxsteps = maxsteps

    def play_episode(self, render=False):
        obs = self.env.reset()
        done = False
        R = 0
        old_obs = copy.deepcopy(obs)
        while not done:
            # TODO use old_a in step so we are not blocking on prediction

            ### choose action and simulate
            a = self.player.choose_action(obs.reshape((1, -1)))
            obs, r, done, info = self.env.step(a)
            R += r

            if render:
                self.env.render()

            ### add experience to buffer
            # HACK although this is an episodic task
            # because it is almost full info we can break it up into pairs of examples
            self.buffer.add([old_obs, np.array([a]), np.array([r]), obs])

            old_obs = copy.deepcopy(old_obs)

        return R

    def work(self, n):
        returns = []
        for _ in range(n):
            returns.append(self.play_episode())
        return returns

    def get_batch(self, n):
        if self.buffer.size > n:
            return self.buffer.get_batch(n)
        else:
            return None
