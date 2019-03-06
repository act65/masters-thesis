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
        self.assertEqual(batch[0].shape, (5, 4))
