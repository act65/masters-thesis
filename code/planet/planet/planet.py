import unittest
import gym
import nets
from mpc import *

import jax.numpy as np
import jax.random as random

class Planet():
    def __inti__(self):

        self.transition_fn = make_transition_net()

    def choose_action(self, s):
        # freeze the current nets and use them to plan
        transition = lambda s, a: self.transition.fn(self.transition.params, s, a)
        value = lambda s: self.value.fn(self.value.params, s)

        return mpc(s, transition, value, gamma)

    def step(self, s):
        

    def update(self, ):
        pass
