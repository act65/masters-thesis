import unittest
import gym
from nets import *

import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

def onehot(idx, N):
    return np.eye(N)[idx]

class TestNet(unittest.TestCase):
    def test_transition(self):

        net = make_transition_net(n_inputs=4, n_actions=2, width=32, n_outputs=4)

        x = rnd.standard_normal((1, 4))
        a = onehot(rnd.randint(0, 2, (1,)), 2)
        print(a)
        t = rnd.standard_normal((1, 4))

        y = net.fn(net.params, x, a)
        g = net.grad_fn(net.params, x, a, t)
        loss = net.loss_fn(net.params, x, a, t)
        self.assertEqual(y.shape, (1, 4))

    def test_value(self):

        net = make_value_net(4, 32)

        x = rnd.standard_normal((1, 4))
        v_tp1 = rnd.standard_normal((1, 1))
        r = rnd.standard_normal((1, 1))

        v_t = net.fn(net.params, x)
        g = net.grad_fn(net.params, x, r, v_tp1, gamma=0.9)
        loss = net.loss_fn(net.params, x, r, v_tp1, gamma=0.9)
        self.assertEqual(v_t.shape, (1, 1))

    def test_transition_toy_train(self):
        """
        check these nets can learn something simple
        """
        net = make_transition_net(n_inputs=1, n_actions=2, width=32, n_outputs=1)

        def toy_fn(x, a):
            return np.sin(3*x)

        def data_generator(N):
            for _ in range(N):
                x = rnd.random((20, 1))
                a = np.zeros((20, 2))
                t = toy_fn(x, a)
                yield x, a, t

        losses = []
        for i, batch in enumerate(data_generator(100)):
            net = opt_update(i, net, batch)
            L = net.loss_fn(net.params, *batch)
            losses.append(L)
            print("\r{}.".format(L), end='', flush=True)

        plt.figure()
        plt.plot(losses)

        x = np.linspace(-1, 1, 100)
        y = net.fn(net.params, x.reshape((100, 1)), np.zeros((100, 2)))

        plt.figure()
        plt.plot(x, y)
        plt.plot(x, toy_fn(x, None))

        plt.show()

        # BUG well the loss goes down... but the fit...
        # are NNs really this bad, or have I done something wrong?

    # def test_value_train(self):
    #     """
    #     check these nets can learn something simple
    #     """
    #     net = make_value_net(n_inputs=1, width=128)
    #
    #     def toy_fn(x):
    #         return np.sin(2*x)
    #
    #     def data_generator(N):
    #         for _ in range(N):
    #             x = rnd.random((50, 1))*-1
    #             r_t = toy_fn(x)
    #             yield x, r_t, None, None
    #
    #     opt_state = net.opt_state
    #     losses = []
    #     for i, batch in enumerate(data_generator(2000)):
    #         # net = opt_update(i, net, batch)
    #         L = net.loss_fn(net.params, *batch)
    #         losses.append(L)
    #         print("\r{}.".format(L), end='', flush=True)
    #         opt_state = net.step(i, opt_state, batch)
    #
    #     plt.figure()
    #     plt.plot(losses)
    #
    #     x = np.linspace(-1, 1, 100)
    #     y = net.fn(optimizers.get_params(opt_state), x.reshape((100, 1)))
    #
    #     plt.figure()
    #     plt.plot(x, y)
    #     plt.plot(x, toy_fn(x))
    #
    #     plt.show()
