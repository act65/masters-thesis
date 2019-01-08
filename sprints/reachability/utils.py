import random
import numpy as np
import tensorflow as tf

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.cur_size = 0
        self.buffer = {}

    @property
    def size(self):
        return self.cur_size

    def add(self, episode):
        """Add episodes to buffer."""

        self.buffer[self.cur_size] = episode
        self.cur_size += 1

        # batch removals together for speed!?
        if len(self.buffer) > self.max_size+100:
            self.remove_n(100)

    def remove_n(self, n):
        """Get n items for removal."""
        # random removal
        # idxs = random.sample(range(self.cur_size), self.cur_size-n)
        idxs = list(range(n))  # removes the oldest
        new_buffer = np.array(list(self.buffer.values()))[idxs]
        # overwrites the old dict. (possibly expensive, but not sure of a better way...)
        n = len(new_buffer)
        self.buffer = dict(zip(range(n), new_buffer))
        self.cur_size = n

    def get_batch(self, n):
        """Get batch of episodes to train on."""
        # random batch
        idxs = random.sample(range(self.cur_size), n)
        batch = [self.buffer[idx] for idx in idxs]  # a list of [[obs_s, a_s, r_s], ...]
        return [np.stack(x, axis=1) for x in zip(*batch)]

def preprocess(x, old_x, stride=5):
    shape = x.shape

    # down sample
    x = x[::stride, ::stride, ...]
    old_x = np.zeros_like(x) if old_x is None else old_x[::stride, ::stride, ...]

    # preprocessing. add differences for velocity
    diff = np.sum(x, axis=-1, keepdims=True)-np.sum(old_x, axis=-1, keepdims=True)

    x = np.concatenate([x, diff], axis=-1)
    return x

class Worker():
    """
    A worker to handle offline learning.
    It collects experience in a replay buffer.
    """
    def __init__(self, learner, batch_size=50):
        self.old_a = None
        self.old_x = None
        self.old_obs = None

        self.buffer = ReplayBuffer(2000)
        self.learner = learner
        self.episode = []
        self.batch_size = batch_size

    def __call__(self, obs, r, done):
        x = preprocess(obs, self.old_obs)
        a, b = self.learner(tf.constant(x, dtype=tf.float32))

        if self.old_a is not None:
            self.episode.append([self.old_x, self.old_a, r, b])

        self.old_x = x
        self.old_a = a

        if done:
            self.buffer.add([np.stack(x, axis=0) for x in zip(*self.episode)])
            self.reset()

            # only train when done an episode.
            # otherwise will train hundreds of times for every episode we recieve
            if self.buffer.size > self.batch_size:
                L = self.learner.train_step(*self.buffer.get_batch(self.batch_size))

        return a

    def reset(self):
        self.episode = []
        self.old_a = None
        self.old_x = None
        # self.learner.reset()


class SetEmbedding():
    def __init__(self, n_hidden, n_outputs):
        self.encoding_fn = tf.keras.Sequential([
            tf.keras.layers.Dense(n_hidden, activation=tf.nn.selu),
            tf.keras.layers.Dense(n_hidden)
        ])

        self.decoding_fn = tf.keras.Sequential([
            tf.keras.layers.Dense(n_hidden, activation=tf.nn.selu),
            tf.keras.layers.Dense(n_hidden)
        ])

        # can memoize simply based on size of inputs.
        # if the size the the memory doesnt change, the memory hasnt changed
        self.memoize_state_size = 0
        self.memoized_result = None
        # QUESTION ^^^ but is this still differentiable!?!?

    def __call__(self, xs):
        n = len(xs)
        if n == self.memoize_state_size:
            return self.memoized_result
        else:
            result = self.decoding_fn(tf.add_n([self.encoding_fn(x) for x in xs])/len(xs))
            self.memoized_result = result
            return result

class Residual(tf.keras.layers.Layer):
    def __init__(self, n_hidden, n_outputs, **kwargs):
        self.n_hidden = n_hidden
        self.output_dim = n_outputs
        super(Residual, self).__init__(**kwargs)

    def build(self, input_shape):
        self.fn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.n_hidden, activation=tf.nn.selu),
            tf.keras.layers.Dense(self.output_dim),
        ])
        super(Residual, self).build(input_shape)

    def call(self, inputs):
        # n_outputs = n_inputs
        return inputs + self.fn(inputs)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)
