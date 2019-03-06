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
        T = transition_tensor[np.argmax(a)]
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
        net = nets.make_transition_net(4, 2, 32, 4)

        N = 5
        pi = [onehot(a, 2).reshape((-1, 2)) for a in rnd.randint(0, 2, (N,))]

        transition_fn =  lambda s, a: net.fn(net.params, s, a)
        states = multi_step_transitions(transition_fn, rnd.standard_normal((1, 4)), pi)

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
            self.assertEqual(len(pi), 12)
            for a in pi:
                self.assertTrue(np.argmax(a) in possible_actions)


    def test_mpc(self):
        """
        integration test.
        """
        transition_fn, reward_fn = create_discrete_toy(32, 4)
        a = mpc(0, transition_fn, n_actions=4, T=5, N=10, reward_fn=reward_fn)
        self.assertTrue(np.argmax(a) in range(4))

    def test_search(self):
        """
        check we can actually search a space and find the solution
        """
        # TODO possible BUG need to acutally check if this does better than rnd search...
        n_actions = 6
        transition_fn, reward_fn = create_discrete_toy(64, n_actions)

        s = 0
        done = False
        counter = 0
        while not done:
            a = mpc(0, transition_fn, n_actions=n_actions, T=50, N=100, reward_fn=reward_fn)
            s = transition_fn(s, a)
            if bool(reward_fn(s)):
                done = True
            counter += 1
        self.assertTrue(done)
        logging.info('{} steps total'.format(counter))


    def test_sample(self):
        """
        check action sampling follows the correct distribution
        """
        n_actions = 12
        N = 30
        T= 4
        evals = rnd.standard_normal((N, 1))
        generator = rnd_policy_generator(n_actions=n_actions, T=T, N=N)
        pis = [next(generator) for i in range(N)]
        a = sample_action(pis, evals)
        a_s = [sample_action(pis, evals) for _ in range(200)]

        # max_idx = np.argmax(evals)
        # expected_val =

        # self.assertEqual(np.argmax(pis[max_idx]), np.argmax(counts))
        # the most commonly chosen action should match the best eval.
        # actually, that might not be true...
