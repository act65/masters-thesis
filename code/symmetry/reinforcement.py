import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

class Explorer():
    # TODO Count based exploration
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

        # self.counts = np.ones([n_states, n_actions])

    def __call__(self, s):
        # TODO should use UCB instead!?
        # explore the state-actions that have small count.
        # p = softmax(normalise(-self.counts[s, ...]))
        # a = sample(np.arange(self.n_actions), p)
        # self.counts[s, a] += 1
        return np.random.randint(0, self.n_actions)

class ReplayBuffer():
    def __init__(self, size=1000):
        self.buffer = Deque()

    def add():
        pass

    def sample(self, N):
        pass

def estimate_qvals(value_estimator, s_t, n_actions):
    """
    Takes a value predictor and applies to to all the possible discrete actions.
    Q(s, a) for all a.

    Args:
        value_estimator (callable): A neural network that predicts the value of
            state-action pairs
        s_t (tf.tensor): the current state. [batch_size, n_dims]
        n_actions (int): the number of discrete actions

    Returns:
        (tf.tensor): shape = [batch_size, n_actions]
    """
    B = tf.shape(s_t)[0]
    # the fn being mapped over actions
    def fn(a):
        # repeat the action for all states in the batch
        a_s = tf.stack([a for i in range(B)], axis=0)
        return value_estimator(tf.concat([s_t, a_s], axis=-1))
    # the NN expects one_hot representations for the actions.
    actions = tf.cast(tf.one_hot(tf.range(n_actions), n_actions), tf.float32)
    return tf.transpose(tf.map_fn(fn, actions), [1,0])

class QLearner():
    def __init__(self, model, ):
        self.model = model
        self.value_net = make_NN(n_inputs, n_outputs)
        self.buffer = ReplayBuffer()

    def __call__(self, s_t, a_tm1, r_tm1):
        self.buffer.append((self.s_tm1, a_tm1, r_tm1))

        q_vals = estimate_qvals(self.value_net, self.model(s_t), self.n_actions)
        a = sample(q_vals)

        if self.step % 100 == 0:
            batch = self.buffer.sample(self.batch_size)
            self.update(batch)

        return a

    def loss_fn(self, s_t, a_t, r_t, s_tp1):
        s_t = self.model(tf.concat([s_t, a_t]))
        a_t, r_t, s_tp1

    def update(self, batch):
        pass

if __name__ == '__main__':
    W = tf.get_variable(shape=(6, 1), name='W')
    def net(x):
        y = tf.matmul(x, W)
        return tf.reduce_sum(y**2, axis=1)

    v = estimate_qvals(net, tf.random.normal((5, 4)), 2)
    print(v)
