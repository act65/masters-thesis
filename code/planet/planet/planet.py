import nets
import mpc

import numpy as np

"""
Put together the fns in nets.py and mpc.py to build an agent for online RL.
"""

def onehot(idx, N):
    return np.eye(N)[idx]

class Planet():
    def __init__(self, n_inputs, n_actions, planning_window=5, n_plans=200, width=128):
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

        self.planner = mpc.make_planner(self.transition, self.value)

    def choose_action(self, s, random=False):
        if len(s.shape) != 2:
            raise ValueError('expected s as shape (Batch, Dim)')

        if random:
            a = np.random.randint(0, self.n_actions)
        else:
            # freeze the current nets and use them to plan
            a = self.planner(s, self.transition.params, self.value.params,
                             n_actions=self.n_actions, T=self.planning_window, N=self.n_plans)
            a = np.argmax(a)  # convert from onehot to int

        return a

    def update(self, s_t, a_t, r_t, s_tp1):
        a_t = onehot(a_t.reshape(-1), self.n_actions)
        v_tp1 = self.value.fn(self.value.params, s_tp1)
        self.transition = nets.opt_update(self.step_counter, self.transition, (s_t, a_t, s_tp1))
        self.value = nets.opt_update(self.step_counter, self.value, (s_t, r_t, a_t, v_tp1))
        self.step_counter += 1

        return self.loss(s_t, a_t, r_t, s_tp1, v_tp1)

    def loss(self, s_t, a_t, r_t, s_tp1, v_tp1):
        transition_loss = self.transition.loss_fn(self.transition.params, s_t, a_t, s_tp1)
        value_loss = self.value.loss_fn(self.value.params, s_t, r_t, v_tp1, 1.0)
        return transition_loss, value_loss
