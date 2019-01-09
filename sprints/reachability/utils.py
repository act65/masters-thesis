import random
import numpy as np
import tensorflow as tf
import trfl

class ReplayBuffer(object):
    def __init__(self, max_size, buffer_buffer=100):
        self.max_size = max_size
        self.buffer_buffer = buffer_buffer  # buffer the deletions. they are expensive
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
        if len(self.buffer) > self.max_size+self.buffer_buffer:
            self.remove_n(self.buffer_buffer)

    def remove_n(self, n):
        """Get n items for removal."""
        # random removal
        idxs = random.sample(range(self.cur_size), self.cur_size-n)
        # idxs = list(range(n))  # removes the oldest
        new_buffer = [val for i, val in enumerate(list(self.buffer.values())) if i in idxs]
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


class Policy():
    # Currently A2C TODO change to PPO or something smarter!
    def __init__(self, n_actions, n_hidden):
        self.policy_fn = tf.keras.Sequential([
            tf.keras.layers.Dense(n_hidden, activation=tf.nn.selu),
            Residual(n_hidden//4, n_hidden),
            Residual(n_hidden//4, n_hidden),
            Residual(n_hidden//4, n_hidden),
            tf.keras.layers.Dense(n_actions)
        ])

        self.value_fn = tf.keras.Sequential([
            tf.keras.layers.Dense(n_hidden, activation=tf.nn.selu),
            Residual(n_hidden//4, n_hidden),
            Residual(n_hidden//4, n_hidden),
            Residual(n_hidden//4, n_hidden),
            tf.keras.layers.Dense(1)
        ])

    @property
    def variables(self):
        return self.value_fn.variables + self.policy_fn.variables

    def __call__(self, x, return_logits=False):
        # TODO the policy needs a fn of the memory as well!!
        # else how does it know where it should explore?

        # but how!? the memory changes in size. k-means? <- but order might change!?
        # could use a graph NN?! or f(sum(g(m)))

        # this breaks everything... now we need to add the memory contents to the buffer!?
        # m = self.memory_sketch(self.memory)
        # logits = self.policy_fn(tf.concat([x, m], axis=-1))

        logits = self.policy_fn(x)

        if return_logits:
            return logits

        else:
            # gumbel-softmax trick
            g = -tf.log(-tf.log(tf.random_uniform(shape=tf.shape(logits), minval=0, maxval=1, dtype=tf.float32)))
            return tf.argmax(logits + g, axis=-1)

    def get_loss(self, x, at, r):
        # TODO change to PPO?
        T, B = tf.shape(r)

        policy_logits = tf.map_fn(lambda x: self.__call__(x, return_logits=True), x)
        v = tf.squeeze(tf.map_fn(self.value_fn, tf.concat([x, policy_logits], axis=-1)))  # action-value estimates

        pg_loss, extra = trfl.sequence_advantage_actor_critic_loss(
            policy_logits=policy_logits,
            baseline_values=v,
            actions=at,
            rewards=r,
            pcontinues=tf.ones_like(r),
            bootstrap_value=tf.ones(tf.shape(r)[1])
        )

        loss = tf.reduce_mean(pg_loss)
        tf.contrib.summary.scalar('loss/policy', loss)
        return loss


def ortho_loss(x):
    B, D = tf.shape(x)
    return tf.norm(tf.matmul(x, x, transpose_b=True) - tf.eye(B), ord='fro', axis=[0, 1])

class Encoder():
    def __init__(self, n_hidden, beta=1.0):
        self.fn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(n_hidden, 4, 2, 'same', activation=tf.nn.selu),
            tf.keras.layers.Conv2D(n_hidden, 4, 2, 'same', activation=tf.nn.selu),
            tf.keras.layers.Conv2D(n_hidden, 4, 2, 'same', activation=tf.nn.selu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(n_hidden),
        ])
        self.beta = beta

    @property
    def variables(self):
        return self.fn.variables

    def __call__(self, x):
        return self.fn(x)

    def get_loss(self, x):
        T, B, D = tf.shape(x)
        x = tf.reshape(x, [T*B, D])
        loss = self.beta*ortho_loss(x)
        tf.contrib.summary.scalar('loss/embed', loss)
        return loss
