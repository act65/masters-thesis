import unittest
import gym
from nets import *

import jax.numpy as np
import numpy.random as rnd

import time
import matplotlib.pyplot as plt

def onehot(idx, N):
    return np.eye(N)[idx]

class TestNets(unittest.TestCase):
    def test_transition(self):

        net = make_transition_net(n_inputs=4, n_actions=2, width=32, n_outputs=4)

        x = rnd.standard_normal((1, 4))
        a = onehot(rnd.randint(0, 2, (1,)), 2)
        t = rnd.standard_normal((1, 4))

        y = net.fn(net.params, x, a)
        g = net.grad_fn(net.params, x, a, t)
        loss = net.loss_fn(net.params, x, a, t)
        self.assertEqual(y.shape, (1, 4))

    def test_value(self):

        net = make_value_net(4, 32)

        x = rnd.standard_normal((1, 4))
        a = rnd.standard_normal((1, 1))
        a_logits = rnd.standard_normal((1, 12))
        v_tp1 = rnd.standard_normal((1, 1))
        r = rnd.standard_normal((1, 1))

        v_t = net.fn(net.params, x)
        g = net.grad_fn(net.params, x, r, v_tp1, gamma=0.9, a=a, a_logits=a_logits)
        loss = net.loss_fn(net.params, x, r, v_tp1, gamma=0.9)
        self.assertEqual(v_t.shape, (1, 1))

    def test_transition_toy_train(self):
        """
        check these nets can learn something simple
        """
        # BUG QUESTION ok. this doesnt really work as wellas I would have expected
        # Relu and Tanh both seem to fit the lhs but not the rhs (negaives)
        net = make_transition_net(n_inputs=1, n_actions=2, width=256, n_outputs=1)

        batch_size = 20

        def toy_fn(x, a):
            return np.sin(3*x) #+ np.sin(20*x+2)

        def data_generator(N):
            for _ in range(N):
                x = rnd.random((batch_size, 1))*2 - 1
                a = np.zeros((batch_size, 2))
                t = toy_fn(x, a) + rnd.random((batch_size, 1)) * 0.1
                yield x, a, t

        losses = []
        for i, batch in enumerate(data_generator(2000)):
            net = opt_update(i, net, batch)
            L = net.loss_fn(net.params, *batch)
            losses.append(L)
            print("\r{}.".format(L), end='', flush=True)


        plt.figure()
        plt.subplot(2,1,1)
        plt.title('Loss')
        plt.plot(np.log(losses))

        N = 1000
        x = np.linspace(-1, 1, N)
        y = net.fn(net.params, x.reshape((N, 1)), np.zeros((N, 2)))

        plt.subplot(2,1,2)
        plt.title('Predictions and truth')
        plt.plot(x, y, label='prediction')
        plt.plot(x, toy_fn(x, None), label='truth')
        plt.legend()

        plt.show()


    # TODO !!!
    def test_value_train(self):
        """
        check these nets can learn something simple
        """
        net = make_value_td_net(n_inputs=1, width=64)

        def reward_fn(x):
            """
            a simple step fn.
            """
            t = 1e-1
            return 0.1*np.greater(x, -t).astype(np.float32) * np.greater(t, x).astype(np.float32)

        def transition_fn(s, a):
            return s + a

        def policy(s):
            return -np.sign(s)/100  # actions take us toward x=0
            # return -0.01  # always go left

        def data_generator(N):
            for _ in range(N):
                s_t = rnd.random((50, 1))*-1
                a_t = policy(s_t)
                s_tp1 = transition_fn(s_t, a_t)
                r_t = reward_fn(s_tp1)
                yield s_t, a_t, r_t, s_tp1

        opt_state = net.opt_state
        losses = []
        gamma = 0.9
        for i, batch in enumerate(data_generator(2000)):
            x_t, a_t, r_t, s_tp1 = tuple(batch)
            v_tp1 = net.fn(net.params, s_tp1)
            L = net.loss_fn(net.params, x_t, r_t, v_tp1, gamma, a_t)
            losses.append(L)
            print("\r{}.".format(L), end='', flush=True)
            opt_state = net.step(i, opt_state, (x_t, r_t, v_tp1, gamma, a_t))

        def play_episode(x, N):
            xs = [x]
            for _ in range(N-1):
                xs.append(transition_fn(xs[-1], policy(xs[-1])))
            return xs

        N = 100
        x = np.linspace(-1, 1, N)

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title('Loss')
        plt.plot(losses)

        plt.subplot(3, 1, 2)
        plt.title('Reward fn')
        plt.plot(x, reward_fn(x))

        plt.subplot(3, 1, 3)
        plt.title('Value')

        y = net.fn(optimizers.get_params(opt_state), x.reshape((N, 1)))
        plt.plot(x, y, label='estimate')

        rs = reward_fn(np.vstack(play_episode(x, 1000)))
        vs = discount(rs, gamma)
        plt.plot(x, vs, label='truth')
        plt.legend()

        plt.show()

def discount(rs, discount):
    return np.sum(np.vstack([r*discount**(i) for i, r in enumerate(rs)]), axis=0)

class TestActorCritic(unittest.TestCase):
    def test_actor_critic(self):
        """
        """
        n_actions = 8
        batch_size = 5
        net = make_actor_critic(n_inputs=10, width=128, n_actions=n_actions)

        x = rnd.standard_normal((batch_size, 10))
        a_logits = rnd.standard_normal((batch_size, n_actions))
        v_tp1 = rnd.standard_normal((batch_size, 1))
        r = rnd.standard_normal((batch_size, 1))

        a, v = net.fn(net.params, x)
        self.assertEqual(a.shape, (batch_size, n_actions))
        self.assertEqual(v.shape, (batch_size, 1))

        loss = net.loss_fn(net.params, x, r, v_tp1, 0.9, a, a_logits)
        g = net.grad_fn(net.params, x, r, v_tp1, 0.9, a, a_logits)


    def test_a2c_value(self):
        """
        check that the value fn is well behaved
        """

        n_actions = 8
        batch_size = 5
        net = make_actor_critic(n_inputs=1, width=32, n_actions=n_actions)

        def toy_fn(x):
            t = 1e-1
            return np.greater(x, -t).astype(np.float32) * np.greater(t, x).astype(np.float32)

        def data_generator(N):
            gamma = 0.99
            for _ in range(N):
                x_t = rnd.random((50, 1))*-1
                a_logits = -x_t/2  # actions take us toward x=0
                r_t = toy_fn(x_t+a_logits)
                yield x_t, r_t, gamma, np.zeros_like(a_logits), a_logits

        opt_state = net.opt_state
        losses = []
        for i, batch in enumerate(data_generator(2000)):
            x_t, r_t, gamma, a_t, a_logits = tuple(batch)
            a, v_tp1 = net.fn(net.params, x_t+a_logits)
            L = net.loss_fn(net.params, x_t, r_t, v_tp1, gamma, a_t, a_logits)
            losses.append(L)
            print("\r{}.".format(L), end='', flush=True)
            opt_state = net.step(i, opt_state, (x_t, r_t, v_tp1, gamma, a_t, a_logits))

        plt.plot(losses)
        plt.show()
