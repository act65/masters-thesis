import unittest
import gym
import nets
from mpc import *

import jax.numpy as np
import jax.random as random

key = random.PRNGKey(0)

class TestPlanet(unittest.TestCase):
    def test_multistep(self):
        """
        Create a transition network and check that we can sumilate together multiple steps.
        """
        params, fn, out_shape, loss_fn, dldp = nets.make_transition_net((-1, 4+2), 32, 4)

        N = 5
        pi = [onehot(a, 2).reshape((-1, 2)) for a in random.randint(key, minval=0, maxval=2, shape=(N,))]
        states = multi_step_transitions(lambda s, a: fn(params, s, a), obs.reshape((1, -1)), pi)

        self.assertEqual(len(states), 5)

    def test_pi_generator(self):
        """
        Test that we can
        """
        possible_actions = range(10)
        for pi in policy_generator(n_actions=10, T=12, N=10):
            assert len(pi) == 12
            for a in pi:
                assert a in possible_actions
