import gym

import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.cur_size = 0
        self.buffer = {}

    @property
    def size(self):
        return self.cur_size

    def add(self, episode):
        """Add episodes to buffer."""

        self.buffer[self.cur_size] = episode
        self.cur_size += 1

        # batch removals together for speed!?
        if len(self.buffer) > self.max_size+100:
            self.remove_n(100)

    def remove_n(self, n):
        """Get n items for removal."""
        # random removal
        # idxs = random.sample(range(self.cur_size), self.cur_size-n)
        idxs = list(range(n))  # removes the oldest
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
    def __init__(self, env_name, player, is_remote, maxsteps=100):
        self.buffer = rp.ReplayBuffer(max_size=2000)

        self.env = gym.make(env_name)
        obs = self.env.reset()

        self.player = player(input_shape=obs.shape, n_actions=self.env.action_space.n)

        self.maxsteps = maxsteps

    def play_episode(self, weights=None, render=False):
        obs = self.env.reset()

        done = False
        R = 0
        count = 0

        old_obs = copy.deepcopy(obs)
        old_a = 0
        old_r = 0
        old_q = [0]*self.n_actions
        older_r = 0
        older_a = 0

        trajectory = []

        while not done:
            if count >= self.maxsteps:
                break
            count += 1

            # TODO use old_a in step so we are not blocking on prediction

            ### choose action and simulate
            a = self.player.choose_action(obs, weights=weights)
            obs, r, done, info = self.env.step(a)
            R += r

            if render:
                self.env.render()

            ### add experience to buffer
            trajectory.append([old_obs, np.array([old_a]), np.array([old_r]), obs])

            old_obs = copy.deepcopy(old_obs)
            old_a = copy.deepcopy(a)
            old_r = copy.deepcopy(r)

        # HACK although this is an episodic task
        # because it is almost full info we can break it up into pairs of examples
        self.buffer.add([np.stack(x, axis=0) for x in zip(*trajectory)])

        return R

    def work(self, n):
        returns = []
        for _ in range(n):
            returns.append(self.run_episode())
        return returns

    def get_batch(self, n):
        if self.buffer.size > n:
            return self.buffer.get_batch(n)
        else:
            return None
