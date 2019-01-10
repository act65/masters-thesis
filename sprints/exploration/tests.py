import os
import gym
import shutil
import unittest

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from utils import *
from reachability import *

class TestExplorerMethods(unittest.TestCase):
    def test_reach_memory_size(self):
        """make sure memory isnon zero"""
        exp = Explorer(2)

        for _ in range(200):
            x = tf.random_normal([32, 32, 4])
            exp(x)

        l = len(exp.memory)
        self.assertTrue(l > 0)
        self.assertTrue(l < 200)

    def test_action_set(self):
        """check that actions are within the right bounds/format"""
        n_actions = 4
        action_set = list(range(n_actions))
        exp = Explorer(n_actions)

        actions = [exp(tf.random_normal([32, 32, 4])) for _ in range(200)]
        for a, b in actions:
            self.assertTrue(a.numpy() in action_set)

    def test_bonus(self):
        """check the norvelty bonus"""
        exp = Explorer(5)

        for _ in range(200):
            x = tf.random_normal([32, 32, 4])
            a, b = exp(x)

            print(b)


class RndExplorer():  # used for TestTraining.test_run
    def __call__(self, *args, **kwargs):
        return env.action_space.sample(), 0

    def reset(self, *args, **kwargs):
        pass

    def train_step(self, *args, **kwargs):
        pass

class TestTraining(unittest.TestCase):
    def test_run(self):
        env = gym.make('MontezumaRevenge-v0')

        player = utl.Worker(RndExplorer(), batch_size=5)
        train(env, player, 10)


class RndPolicy():  # used for TestEpisodicMemory.test_accuracy
    def __init__(self, n_actions, *args, **kwargs):
        self.n_actions = n_actions

    @property
    def variables(self):
        return []

    def __call__(self, *args, **kwargs):
        return np.random.randint(0, self.n_actions)

    def get_loss(self, *args, **kwargs):
        return 0

class TestEpisodicMemory(unittest.TestCase):
    def test_pairs_shapes(self):
        """check the output format"""
        T, B, D = 200, 50, 3
        x = tf.random_normal([T, B, D])

        k, n = 4, 3
        reachable, not_reachable = reachable_training_pairs(x, k, n)

        self.assertTrue(reachable.shape, (B*n, D*2))
        self.assertTrue(not_reachable.shape, (B*n, D*2))

    def test_diff(self):
        """check that the outputs are differentiable wrt inputs"""
        with tf.GradientTape() as tape:
            x = tf.random_normal([5, 4, 3])
            tape.watch(x)
            reachable, not_reachable = reachable_training_pairs(x, 3, 3)
            y = reachable**2
        grad = tape.gradient(y, x)
        self.assertFalse(np.isclose(grad, np.zeros_like(grad)).all())

    def test_accuracy(self):
        """check that the similarty metric can learn to accurately predict reachability"""
        # use rnd policy, no memory, no extra training losses,
        # does it make sense to do this with a rnd policy?
        # reachabilty should be inv prop to diffusion.

        logdir = '/tmp/exp-tests/mem-acc'
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
        env = gym.make('MontezumaRevenge-v0')
        player = Worker(Explorer(RndPolicy,
                                 env.action_space.n,
                                 memory_max_size=2,
                                 encoder_beta=0.0,
                                 logdir=logdir),
                        batch_size=50)
        train(env, player, 10, 20000)

        # TODO read logs and fetch acc. ensure > 80%!?

    def test_bonus_range(self):
        """check that the bonus is the right scale"""
        pass


class TestIntegration():
    def test_performance(self):
        """the test we really care about.
        how does this extension improve the performance?
        can it be used as a simple drop in addition?
        """
        env = gym.make('MontezumaRevenge-v0')

        player = Worker(Explorer(Policy,
                                 env.action_space.n,
                                 memory_max_size=2,
                                 encoder_beta=0.0,
                                 logdir='/tmp/exp-tests/policy_mem'),
                        batch_size=50)
        train(env, player, 10)
        rewards = fetch()

        player = Worker(SimpleLearner(Policy,
                               env.action_space.n,
                               memory_max_size=2,
                               encoder_beta=0.0,
                               logdir='/tmp/exp-tests/policy'),
                        batch_size=50)
        train(env, player, 10)
        rewards = fetch()

if __name__ == '__main__':
    unittest.main()
