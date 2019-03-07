import unittest
import gym
import nets
from mpc import *

import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

def create_discrete_toy(n_states, n_actions):
    transition_tensor = np.stack([np.eye(n_states)[rnd.permutation(n_states)]
                                  for _ in range(n_actions)], axis=0)
    goal = rnd.randint(0, n_states)

    def transition_fn(s, a):  # TODO could tidy this with vmap
        T = transition_tensor[np.argmax(a, axis=1)]
        s = onehot(s, n_states).reshape((n_states,-1))
        return np.argmax(np.einsum('ijk,ji->ik', T, s), axis=1)

    def reward_fn(s):
        return np.equal(s, np.ones_like(s)*goal).astype(np.int32)

    return transition_fn, reward_fn

class TestMPC(unittest.TestCase):
    def test_simulate(self):
        """
        Create a transition network and check that we can sumilate together multiple steps.
        """
        N = 5
        T = 10
        n_actions = 2
        net = nets.make_transition_net(4, n_actions, 32, 4)
        pis = np.stack(rnd_policy_generator(n_actions=n_actions, T=T, N=N), axis=1)

        transition_fn =  lambda s, a: net.fn(net.params, s, a)
        states = simulate(transition_fn, rnd.standard_normal((1, 4)), pis)

        self.assertEqual(len(states), T+1)
        for s in states:
            self.assertEqual(s.shape, (N, 4))

    def test_overfit(self):
        """
        chekc that the network can (over) fit a simple problem like f(x)=sin(x)
        """
        pass


    def test_mpc(self):
        """
        integration test.
        """
        transition_fn, reward_fn = create_discrete_toy(32, 4)
        a = mpc(np.array([[0]]), transition_fn, n_actions=4, T=5, N=10, value_fn=reward_fn)
        self.assertEqual(a.shape, (4,))

    def test_search(self):
        """
        check we can actually search a space and find the solution
        """
        # TODO possible BUG need to acutally check if this does better than rnd search...
        n_actions = 6
        transition_fn, reward_fn = create_discrete_toy(64, n_actions)

        s = np.array([[0]])
        done = False
        counter = 0
        while not done:
            a = mpc(s, transition_fn, n_actions=n_actions, T=50, N=100, value_fn=reward_fn)
            a = sample(a, return_onehot=True)
            s = transition_fn(s, a.reshape((1, -1)))
            if bool(reward_fn(s)[0]):
                done = True
            counter += 1
        self.assertTrue(done)
        logging.info('{} steps total'.format(counter))


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
