import numpy as np
import tensorflow as tf
import trfl
import random

def cosine_dist(u, v):
    return tf.reduce_sum(tf.matmul(u, v, transpose_b=True))/(tf.norm(u)*tf.norm(v))

class Explorer():
    """
    An RL learner that uses reachability estimates (estimated via memory) to guide exploration.
    https://arxiv.org/abs/1810.02274
    """
    def __init__(self, n_actions, n_hidden=32):
        self.writer = tf.contrib.summary.create_file_writer("/tmp/test_explore/0")
        self.writer.set_as_default()

        self.policy_fn = tf.keras.Sequential([
            tf.keras.layers.Dense(n_hidden, activation=tf.nn.selu),
            tf.keras.layers.Dense(n_actions)
        ])

        self.value_fn = tf.keras.Sequential([
            tf.keras.layers.Dense(n_hidden, activation=tf.nn.selu),
            tf.keras.layers.Dense(1)
        ])

        self.embed = tf.keras.Sequential([
            tf.keras.layers.Conv2D(n_hidden//2, 4, 2, 'same', activation=tf.nn.selu),
            tf.keras.layers.Conv2D(n_hidden//2, 4, 2, 'same', activation=tf.nn.selu),
            tf.keras.layers.Conv2D(n_hidden//2, 4, 2, 'same', activation=tf.nn.selu),
            tf.keras.layers.Flatten(),
        ])

        # 1 if inputs are similar, else 0
        self.sim = tf.keras.Sequential([
            tf.keras.layers.Dense(n_hidden, activation=tf.nn.selu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        # what does this learn!? what makes two states similar

        self.memory = []
        self.max_size = 100

        self.alpha = 1
        self.beta = 0.5

        self.opt = tf.train.AdamOptimizer(1e-4)
        self.global_step = tf.train.get_or_create_global_step()

    def reset(self):
        self.memory = []

    def __call__(self, x):
        """
        Args:
            x: the current state, in a fully observed setting, this can simply be the observations
        Returns:
            a: the action chosen
            b: the reward bonus
        """
        x = tf.expand_dims(x, 0)
        e = self.embed(x)
        s = self.compare(self.memory, e)

        # reward bonus (for discovering something dissimilar to items in memory)
        # tuning this could be a pain
        b = self.alpha * (self.beta - s)

        if s < 0.2 and len(self.memory) < self.max_size:
            self.memory.append(e)

        a = self.choose_action(e)

        return a, b

    def compare(self, memory, e):
        if len(memory) == 0:
            return 0
        else:
            # this is O(n). as memory gets larger this costs more.
            cs = [self.sim(tf.concat([e, m],axis=-1)) for m in memory]

            # or use a unlearned version!?
#             cs = [cosine_dist(e, m) for m in memory]

            i = int(len(memory)/10)
            return tf.reduce_mean(tf.contrib.framework.sort(tf.stack(cs))[i])

    def choose_action(self, x):
        # TODO the policy should be a fn of the memory as well!?
        logits = self.policy_fn(x)
        g = -tf.log(-tf.log(tf.random_normal(tf.shape(logits))))
        return tf.argmax(logits + g, axis=-1)

    def train_step(self, x, at, r):
        """
        Args:
            batch: (x, a, r) wth shape [Timesteps, Batch, ...]
        """
        x = tf.constant(x, dtype=tf.float32)
        at = tf.squeeze(tf.constant(at, dtype=tf.int32))
        r = tf.constant(r, dtype=tf.float32)

        e = tf.map_fn(self.embed, x)

        with tf.GradientTape() as tape:
            pg_loss = self.get_pg_loss(e, at, r)
            reach_loss = get_reach_loss(self.sim, e)
            loss = pg_loss + reach_loss  # HACK should be ok to do this!?

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.scalar('loss/pg', pg_loss)
            tf.contrib.summary.scalar('loss/reach', reach_loss)
            tf.contrib.summary.scalar('R', tf.reduce_mean(r))

        variables = self.policy_fn.variables + self.embed.variables + self.sim.variables
        grads = tape.gradient(loss, variables)
        self.opt.apply_gradients(zip(grads, variables), global_step=self.global_step)
        return loss

    def get_pg_loss(self, x, at, r):
        T, B = tf.shape(r)

        policy_logits = tf.map_fn(self.policy_fn, x)
        v = tf.squeeze(tf.map_fn(self.value_fn, tf.concat([x, policy_logits], axis=-1)))  # action-value estimates

        pg_loss, extra = trfl.sequence_advantage_actor_critic_loss(
            policy_logits=policy_logits,
            baseline_values=v,
            actions=at,
            rewards=r,
            pcontinues=tf.ones_like(r),
            bootstrap_value=tf.ones(tf.shape(r)[1])
        )

        return tf.reduce_mean(pg_loss)

def get_reach_loss(reachability_metric, embeddings):
    # how necessary is it to train this!?
    # could do with a random similarity measure?!
    reachable, not_reachable = reachable_training_pairs(embeddings, 5, 3)

    sim_reach = reachability_metric(reachable)  # should be = 1
    sim_not_reach = reachability_metric(not_reachable)  # should be = 0

    return tf.reduce_mean(-tf.log(sim_reach+1e-6)) + tf.reduce_mean(-tf.log(1-sim_not_reach+1e-6))

def reachable_training_pairs(x, k, n):
    """
    Two states are considered 'reachable' if they occur within k timesteps.

    Args:
        x ([T, B, D]): A sequence of elements
        k: the number of steps that makes two states mutually reachable
        n: the number of elements to sample

    Returns:
        tuple ([n*B x D], [n*B x D]):
            reachable:
            not_reachable:
    """
    T, B, D = tf.shape(x)

    reachable_pairs = [(i, i+m) for i in range(T) for m in range(k) if i+m<T]
    not_reachable_pairs = [(i, i+m) for i in range(T) for m in range(k, T) if i+m<T]
    reachable = [tf.concat([x[i, ...], x[j, ...]], axis=-1) for i, j in random.sample(reachable_pairs, n)]
    not_reachable = [tf.concat([x[i, ...], x[j, ...]], axis=-1) for i, j in random.sample(not_reachable_pairs, n)]

    return tf.concat(reachable, axis=0), tf.concat(not_reachable, axis=0)
