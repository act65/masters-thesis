import nets
import mpc

import numpy as np

"""
Put together the fns in nets.py and mpc.py to build an agent for online RL.
"""

def onehot(idx, N):
    return np.eye(N)[idx]

class Planet():
    def __init__(self, n_inputs, n_actions, planning_window=5, n_plans=20, width=64):
        self.n_actions = n_actions
        self.planning_window = planning_window
        self.n_plans = n_plans

        # TODO state abstraction!?
        # s = self.encoder(obs)
        # TODO action abstraction
        # a = self.decoder(self.choose_action(a))
        # how to train this? what loss? an AE?
        # if mpc was differentiable we could train this end to end!?
        self.transition = nets.make_transition_net(n_inputs, n_actions, width=width, n_outputs=n_inputs)
        self.value = nets.make_value_net(n_inputs, width=width)

        self.step_counter = 0

    def choose_action(self, s):
        # freeze the current nets and use them to plan
        transition = lambda s, a: self.transition.fn(self.transition.params, s, a)
        value = lambda s: self.value.fn(self.value.params, s)

        a = mpc.mpc(s, transition, n_actions=self.n_actions,
                    T=self.planning_window, N=self.n_plans, value_fn=value)

        return np.argmax(a)  # convert from onehot to int

    def update(self, s_t, s_tp1, a_t, r_t):
        a_t = onehot(a_t, self.n_actions)
        v_tp1 = self.value.fn(self.value.params, s_tp1)
        self.transition = nets.opt_update(self.step_counter, self.transition, (s_t, a_t, s_tp1))
        self.value = nets.opt_update(self.step_counter, self.value, (s_t, r_t, a_t, v_tp1))
        self.step_counter += 1
