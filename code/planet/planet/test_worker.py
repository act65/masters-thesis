import unittest

import gym
import numpy as np
import numpy.random as rnd

from worker import *

def gen_step():
    return [np.random.random([4]),
         np.random.random([1]),
         np.random.random([1])]

def gen_trajectory(n):
    return [gen_step() for _ in range(n)]

class TestBuffer(unittest.TestCase):
    def test_buffer_contents(self):

        mem = ReplayBuffer(max_size=2000)

        # generate trajectories of various lengths
        # and add to buffer
        for _ in range(10):
            trajectory = gen_step()
            mem.add(trajectory)

            # trajectory = [np.stack(x) for x in zip(*trajectory)]
            #
            # self.assertEqual(trajectory[0].shape, (T, 4))
            # self.assertEqual(trajectory[1].shape, (T, 1))
            # self.assertEqual(trajectory[2].shape, (T, 1))

        B = 5
        batch = mem.get_batch(B)

        self.assertEqual(batch[0].shape, (B, 4))
        self.assertEqual(batch[1].shape, (B, 1))
        self.assertEqual(batch[2].shape, (B, 1))

    def test_buffer_overflow(self):
        size = 100
        mem = ReplayBuffer(max_size=size)

        # generate trajectories of various lengths
        # and add to buffer
        for _ in range(500):
            trajectory = gen_step()
            mem.add(trajectory)

        self.assertTrue(mem.size < mem.overflow_cache_size + size + 1)

    def test_buffer_overwrite(self):
        size = 50
        mem = ReplayBuffer(max_size=size, overflow_cache_size=5)

        # generate trajectories of various lengths
        # and add to buffer
        for i in range(200):
            trajectory = gen_step()
            mem.add([i])

        # print(mem.buffer.values())

class RndPlayer():
    def __init__(self, *args, **kwargs):
        pass

    def choose_action(self, x):
        return rnd.randint(0, 2)

    def update(self, *args):
        pass

class TestWorker(unittest.TestCase):
    def test_worker(self):
        worker = Worker('CartPole-v1', RndPlayer)
        returns = worker.work(100)
        batch = worker.get_batch(5)
        self.assertEqual(batch[0].shape, (5, 2*4))

    def test_sample(self):
        """
        check action sampling follows the correct distribution
        """
        n_actions= 10
        logits = rnd.standard_normal((n_actions,))
        a_s = np.vstack([sample(logits, return_onehot=True) for _ in range(2000)])
        sample_dist = np.mean(a_s, axis=0)

        def norm(x):
            return x/np.max(x)

        # plt.figure()
        # plt.bar(range(10), norm(np.exp(logits)), alpha=0.75, label='logits')
        # plt.bar(range(10), norm(sample_dist), alpha=0.75, label='samples')
        # plt.legend()
        # plt.show()

        diff = np.mean((norm(np.exp(logits)) - norm(sample_dist))**2)
        self.assertTrue(diff < 1e-3)
