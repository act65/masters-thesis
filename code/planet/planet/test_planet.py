import unittest

import gym
import numpy as np
import numpy.random as rnd

from planet import *
from nets import *

class TestPlanet(unittest.TestCase):
    def test_planet(self):
        n_actions = 2
        learner = Planet(4, n_actions)

        x = rnd.standard_normal((1, 4))
        a = learner.choose_action(x)
        self.assertTrue(np.argmax(a) in range(n_actions))

    def test_cartpole_runs(self):
        env = gym.make('CartPole-v1')
        obs = env.reset()

        learner = Planet(obs.size, env.action_space.n)

        done = False
        R = 0
        while not done:
            a = learner.choose_action(obs.reshape((-1, 4)))
            obs, r, done, info = env.step(a)
            self.assertTrue(a in range(env.action_space.n))
            R += r
            print('\r Reward: {}'.format(R), end='', flush=True)

    def test_cartpole_update(self):
        env = gym.make('CartPole-v1')
        obs = env.reset()

        learner = Planet(obs.size, env.action_space.n)

        done = False
        R = 0
        while not done:
            a = learner.choose_action(obs.reshape((-1, 4)))
            obs, r, done, info = env.step(a)
            learner.update(obs, a, np.array([[r]]), obs)
            self.assertTrue(a in range(env.action_space.n))
            R += r
            print('\r Reward: {}'.format(R), end='', flush=True)
