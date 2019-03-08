import unittest
import gym
from nets import *

import numpy as np
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
        net = make_value_net(n_inputs=1, width=128)

        def toy_fn(x):
            return np.sin(2*x)

        def data_generator(N):
            for _ in range(N):
                x = rnd.random((50, 1))*-1
                r_t = toy_fn(x)
                yield x, r_t, None, None

        opt_state = net.opt_state
        losses = []
        for i, batch in enumerate(data_generator(2000)):
            # net = opt_update(i, net, batch)
            L = net.loss_fn(net.params, *batch)
            losses.append(L)
            print("\r{}.".format(L), end='', flush=True)
            opt_state = net.step(i, opt_state, batch)

        plt.figure()
        plt.plot(losses)

        x = np.linspace(-1, 1, 100)
        y = net.fn(optimizers.get_params(opt_state), x.reshape((100, 1)))

        plt.figure()
        plt.plot(x, y)
        plt.plot(x, toy_fn(x))

        plt.show()


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
