import nets
import mpc

import numpy as np

"""
Put together the fns in nets.py and mpc.py to build an agent for online RL.
"""

class Planet():
    def __init__(self, n_inputs, n_actions):
        self.transition = make_transition_net((-1, n_inputs+n_actions), width=32, n_outputs=n_inputs)
        self.value = make_value_net((-1, n_inputs), width=32)

    def choose_action(self, s):
        # freeze the current nets and use them to plan
        transition = lambda s, a: self.transition.fn(self.transition.params, s, a)
        value = lambda s: self.value.fn(self.value.params, s)

        return mpc(s, transition, value, gamma)

    def step(self, s):
        pass

    def update(self, ):
        pass
