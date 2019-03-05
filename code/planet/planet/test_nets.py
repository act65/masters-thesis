import unittest
import gym
import nets
from nets import *

import numpy as np
import numpy.random as rnd

class TestNet(unittest.TestCase):
    def test_transition(self):

        net = make_transition_net((-1,4+2), 32, 4)

        x = rnd.standard_normal((1, 4))
        a = rnd.standard_normal((1, 2))
        t = rnd.standard_normal((1, 4))

        y = net.fn(net.params, x, a)
        g = net.grad_fn(net.params, x, a, t)
        loss = net.loss_fn(net.params, x, a, t)
        self.assertEqual(y.shape, (1, 4))

    def test_value(self):

        net = make_value_net((-1,4), 32)

        x = rnd.standard_normal((1, 4))
        v_tp1 = rnd.standard_normal((1, 1))
        r = rnd.standard_normal((1, 1))

        v_t = net.fn(net.params, x)
        g = net.grad_fn(net.params, x, r, v_tp1, gamma=0.9)
        loss = net.loss_fn(net.params, x, r, v_tp1, gamma=0.9)
        self.assertEqual(v_t.shape, (1, 1))
