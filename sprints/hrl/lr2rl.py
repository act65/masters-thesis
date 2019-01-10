import numpy as np
import trfl

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

"""
Learning to reinforcement learn.
https://arxiv.org/abs/1611.05763
"""

class Env():
    """
    A super simple environment to play with.
    A multi-armed bandit. It has n arms. Each are has a mean/stddev, and yeilds a payoff.
    """
    def __init__(self, n_bandits, maxsteps=100):
        self.n_bandits = n_bandits
        self.means = np.random.standard_normal(n_bandits)
        self.stddev = np.abs(np.random.standard_normal(n_bandits))

        self.reset()
        self.maxsteps = maxsteps

    def step(self, action):
        reward = self.means[action] + self.stddev[action] * 0.1*np.random.standard_normal()

        if self.timestep >= self.maxsteps-1:
            done = True
            self.reset()
        else:
            done = False
            self.timestep += 1

        # NOTE is it necessary to provide x=t? or can the RNN learn to count?
        return self.timestep+1, reward, done

    def reset(self):
        self.timestep = 0
        return self.timestep, 0, False

    def show(self):
        plt.figure(figsize=(8, 4))
        plt.title('Bandit arm values')
        plt.bar(range(self.n_bandits), self.means, 0.5, yerr=self.stddev)
        plt.xlabel('Arms')
        plt.ylabel('Payoff')

def play_episode(player, env, is_training):
    obs, r, done = env.reset()
    rewards = []
    while not done:
        a = player(obs, r, done, is_training)
        obs, r, done = env.step(a)
        rewards.append(r)
    a = player(obs, r, done)
    return rewards

def eval_player(env, player, n_episodes, is_training=True):
    """
    Args:
        env: must have a callable env.step fn that takes [state x action)s and returns (new_state, reward)
        player: a callable fn that returns actions given the current state
    """
    R = []
    for i in range(n_episodes):

        rs = play_episode(player, env, is_training)
        r = np.mean(rs)
        R.append(r)

        print('\rStep: {}, r: {}'.format(i, r), end='', flush=True)

    return R

def eval_meta_player(n_arms, player, n_episodes):
    """
    Meta learning. We give the learner a new problem to solve every time.
    We hope to see that as training progresses, the learner can solve them 'better'.
    """

    regrets = []
    for i in range(n_episodes):
#         if i % 100 == 0:
#             # "At the beginning of each episode, a new bandit task is sampled and held constant for 100 trials."
#             # should that matter!?
#             # why not a new bandit every time?!
        bandit = Env(n_arms)
        r = eval_player(env=bandit, player=player, n_episodes=1)
        r = np.mean(r)
        regrets.append(np.max(bandit.means)-r)
        print('\rStep: {}, R: {}'.format(i, regrets[-1]), end='', flush=True)

    return regrets

def discrete_reparam(x):
    return tfd.OneHotCategorical(logits=x).sample()

class Memory():
    def __init__(self, max_size=2000):
        self.mem = {}
        self.counter = 0
        self.max_size = max_size

    def append(self, x):
        self.mem[self.counter] = x
        self.counter += 1

        if self.counter > self.max_size + 200:
            self.remove(200)

    def remove(self, n):
        # this seems expensive?!
        latest = [self.mem[i] for i in range(n, self.counter)]
        self.mem = {i: x for i, x in zip(range(self.max_size), latest)}
        self.counter = len(self.mem)

    def get_batch(self, batch_size):
        idxs = np.random.randint(0,self.size, batch_size)
        batch = [self.mem[idx] for idx in idxs]  # a list of [[obs_s, a_s, r_s], ...]
        return [np.stack(arr, axis=0) for arr in zip(*batch)]

    @property
    def size(self):
        return len(self.mem)

class RNN():
    """
    Handles sharing parameters between worker and learner RNNs.
    Need to be able to run the RNN for;
    - next step prediction,
    - end to end training.
    """
    def __init__(self, n_hidden, batch_size):
        self.cell = tf.nn.rnn_cell.MultiRNNCell([
#             tf.nn.rnn_cell.LSTMCell(n_hidden),
            tf.nn.rnn_cell.LSTMCell(n_hidden)
        ])
        self.reset_state(batch_size)
        self.variables = self.cell.variables

    def __call__(self, x):
        # x = many time steps [B, T, D]
        ys = []
        for t in range(x.shape[1]):
            y, self.state = self.cell(x[:, t, :], state=self.state)
            ys.append(y)
        return ys

    def step(self, x, state):
        # next step prediciton only. x = one time step [B, D]
        return self.cell(x, state=state)

    def reset_state(self, batch_size):
        self.state = self.cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        return self.state

class Worker():
    def __init__(self, learner, batch_size):
        self.buffer = Memory()
        self.trajectory = []

        self.learner = learner
        self.batch_size = batch_size

        self.old_x = 0.0
        self.old_a = 0.0

    def __call__(self, x, r, done, is_training=True):
        if x is None:
            x = 0
        # call policy and take action
        a, _ = self.learner(x, self.old_a, r)

        if is_training:
            # add experience to buffer
            if done:
                self.buffer.append([np.stack(arr, axis=0) for arr in zip(*self.trajectory)])
                self.trajectory = []
            else:
                # obs_t, action_t -> reward_t
                self.trajectory.append([self.old_x, self.old_a, r])

            self.old_x = x
            self.old_a = a.numpy()

            # train
            if self.buffer.size > self.batch_size and done:
                # runs a training step every episode
                self.learner.train_step(*self.buffer.get_batch(self.batch_size))

        return a

class A2C():
    """
    Advantage actor critic
    """
    def __init__(self, n_actions, time_steps, batch_size, lr=1e-3):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.time_steps = time_steps

        self.rnn = RNN(128, batch_size=batch_size)
        self.nn = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.selu),
            tf.keras.layers.Dense(n_actions+1)
        ])

        # state and action for online calls - maybe could be manage by the worker?
        self.state = self.rnn.reset_state(1)
        self.a_old = tf.constant(0, shape=[1, 1], dtype=tf.float32)

        self.opt = tf.train.AdamOptimizer(lr)
        self.global_step = tf.train.get_or_create_global_step()

    def __call__(self, x, a, r):
        """
        Choose actions online. Use the current parameters to choose an action.
        Keep the state from the last choice.
        """
        # current observation/state (x), last action taken and the r from t-1.
        x = tf.constant(x, shape=[1, 1], dtype=tf.float32)
        a = tf.constant(a, shape=[1, 1], dtype=tf.float32)
        r = tf.constant(r, shape=[1, 1], dtype=tf.float32)

        inputs = tf.concat([x, a, r], axis=-1)
        h, self.state = self.rnn.step(inputs, self.state)

        z = self.nn(h)
        a = tf.squeeze(tf.argmax(discrete_reparam(z[..., :-1]), axis=1))
        v = z[..., -1:]
        return a, v

    def forward(self, inputs):
        """
        A forward function for BPTT.
        The key is that we need to be able to differentiate wrt all inputs/params.
        """
        # QUESTION how can the net explore? cannot take random actions
        # as it is a deterministic fn of its inputs.
        # it could enumerate the different actions and try each!?
        self.rnn.reset_state(self.batch_size)
        hs = self.rnn(inputs)
        # BUG gradients are not working!!! why!?
        zs = tf.stack(list(map(self.nn, hs)), axis=0)

        # NOTE tbh it feels kinds weird returning a, v.
        # the v is supposed to be the estimated value of the action taken?
        # sharing parameters seems unusual. need to explore further
        return zs[:, :,:self.n_actions], zs[:, :,self.n_actions:]

    def train_step(self, x, a, r):
        """
        Train on a batch of data.

        x: (B, T)
        a: (B, T)
        r: (B, T)
        """
        x = tf.constant(x, shape=[self.batch_size, self.time_steps, 1], dtype=tf.float32)
        a = tf.constant(a, shape=[self.batch_size, self.time_steps, 1], dtype=tf.float32)
        r = tf.constant(r, shape=[self.batch_size, self.time_steps, 1], dtype=tf.float32)

        # current obs, old action, old reward
        inputs = tf.concat([x[:, 1:, :], a[:, :-1:, ], r[:, :-1, :]], axis=-1)

        actions_taken = tf.transpose(tf.squeeze(tf.cast(a[:, 1:, :], tf.int32)))
        rewards_received = tf.transpose(r[:, 1:, 0])
        returns = tf.reduce_sum(rewards_received, axis=0)

        with tf.GradientTape() as tape:

            logits, v = self.forward(inputs)
            # NOTE does this loss fn correct for its off policy nature?
            policy_loss, extra = trfl.sequence_advantage_actor_critic_loss(
                policy_logits=logits,
                baseline_values=v[..., 0], # Q how can A2C be extended with distributional estimates of value?
                actions=actions_taken,
                rewards=rewards_received,
                pcontinues=0.99*tf.ones_like(rewards_received),
                bootstrap_value=returns,
                entropy_cost=1.0,
                lambda_=0.5
            )
            beta = tf.constant(1.0 * 0.5 ** (self.global_step/2000))
            beta = tf.cast(beta, tf.float32)

            loss = tf.reduce_mean(0.05*extra.baseline_loss + extra.policy_gradient_loss + beta*extra.entropy_loss)

        variables = self.nn.variables+self.rnn.variables
        grads = tape.gradient(loss, variables)
        self.opt.apply_gradients(zip(grads, variables), global_step=self.global_step)

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.scalar('loss', loss)
            tf.contrib.summary.scalar('extra/baseline_loss', tf.reduce_mean(extra.baseline_loss))
            tf.contrib.summary.scalar('extra/policy_gradient_loss', tf.reduce_mean(extra.policy_gradient_loss))
            tf.contrib.summary.scalar('extra/entropy_loss', tf.reduce_mean(extra.entropy_loss))
            tf.contrib.summary.scalar('total_R', tf.reduce_sum(r))
            tf.contrib.summary.histogram('actions', a)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_arms", type=int, default=2, help="the number of arms")
    parser.add_argument("--trial_num", type=int, default=0, help="the id of the trial")
    parser.add_argument("--logdir", type=str, default="/tmp/lr2rl", help="the location to save logs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n_train_steps", type=int, default=20000, help="n of train steps")
    parser.add_argument("--name", type=str, default='', help="name of trial")


    def experiment1(args, sanity=True):
        """
        Attempt to reproduce the results of experiment 1 in
        https://arxiv.org/abs/1611.05763.
        """
        time_steps = 100

        if not sanity:
            # a sanity check to verify the network learns
            writer = tf.contrib.summary.create_file_writer('{}/single/{}-{}'.format(
                        args.logdir, args.trial_num, args.name))
            writer.set_as_default()
            player = Worker(
                A2C(
                    args.n_arms,
                    batch_size=50,
                    time_steps=time_steps),
                50)
            rs = eval_player(Env(args.n_arms), player, 500)

        # now the real experiment
        writer = tf.contrib.summary.create_file_writer('{}/meta/{}-{}'.format(
                    args.logdir, args.trial_num, args.name))
        writer.set_as_default()

        player = Worker(
            A2C(
                args.n_arms,
                batch_size=256,
                time_steps=time_steps,
                lr=args.lr),
            256)
        regrets = eval_meta_player(args.n_arms, player, args.n_train_steps)  # that's a lot of steps...

    tf.enable_eager_execution()
    experiment1(parser.parse_args())
