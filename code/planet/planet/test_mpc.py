import unittest
import gym
import nets
from mpc import *

import numpy as np
import numpy.random as rnd

def create_discrete_toy(n_states, n_actions):
    transition_tensor = np.stack([np.eye(n_states)[rnd.permutation(n_states)]
                                  for _ in range(n_actions)], axis=0)
    goal = rnd.randint(0, n_states)

    def transition_fn(s, a):
        T = transition_tensor[a]
        return np.argmax(T[s])

    def reward_fn(s):
        if s == goal:
            return 1
        else:
            return 0

    return transition_fn, reward_fn


class TestMPC(unittest.TestCase):
    def test_multistep(self):
        """
        Create a transition network and check that we can sumilate together multiple steps.
        """
        params, fn, out_shape, loss_fn, dldp = nets.make_transition_net((-1, 4+2), 32, 4)

        N = 5
        pi = [onehot(a, 2).reshape((-1, 2)) for a in rnd.randint(0, 2, (N,))]
        states = multi_step_transitions(lambda s, a: fn(params, s, a), rnd.standard_normal((1, 4)), pi)

        self.assertEqual(len(states), 5)

    def test_grads(self):
        """
        check that the grads are correct
        """
        pass


    def test_overfit(self):
        """
        chekc that the network can (over) fit a simple problem like f(x)=sin(x)
        """
        pass

    def test_pi_generator(self):
        """
        Test that we can generate random policies of the right length and
        with the right actions.
        """
        possible_actions = range(10)
        for pi in rnd_policy_generator(n_actions=10, T=12, N=10):
            assert len(pi) == 12
            for a in pi:
                assert a in possible_actions


    def test_mpc(self):
        """
        integration test.
        """
        transition_fn, reward_fn = create_discrete_toy(32, 4)
        a = mpc(0, transition_fn, n_actions=4, T=5, N=10, reward_fn=reward_fn)
        self.assertTrue(a in range(4))

    def test_search(self):
        """
        check we can actually search a space and find the solution
        """

        transition_fn, reward_fn = create_discrete_toy(32, 4)

        s = 0
        done = False
        while not done:
            a = mpc(0, transition_fn, n_actions=4, T=5, N=10, reward_fn=reward_fn)
            s = transition_fn(s, a)
            if bool(reward_fn(s)):
                done = True
        self.assertTrue(done)
