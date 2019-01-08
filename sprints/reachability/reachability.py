import numpy as np
import tensorflow as tf
import trfl
import random
import math
import os
import utils as utl

def cosine_dist(u, v):
    return tf.reduce_sum(tf.matmul(u, v, transpose_b=True))/(tf.norm(u)*tf.norm(v))


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


class EpisodicMemory():
    """
    """
    # QUESTION Why is this called episodic memory!?
    # QUESTION how is this the same as the transition function!?
    def __init__(self, max_size, n_hidden):
        self.max_size = max_size
        self.memory = set()

        self.alpha = 1
        self.beta = 0.5
        self.gamma = 0.1  # the threshold for adding a new elem to memory

        # 1 if inputs are similar, else 0
        self.sim = tf.keras.Sequential([
            tf.keras.layers.Dense(n_hidden, activation=tf.nn.selu),
            utl.Residual(n_hidden//4, n_hidden),
            utl.Residual(n_hidden//4, n_hidden),
            utl.Residual(n_hidden//4, n_hidden),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        # what does this learn!? what makes two states similar
        # reachability is dependent on the current policy!!

        # self.memory_sketch = SetEmbedding(n_hidden, n_hidden)
        # could do something like. take top K reachable memories. use them to predict likely observation!?
        # how does that help? allows use to learn a meaningful summary of the memory

    @property
    def variables(self):
        return self.sim.variables

    def __call__(self, x):
        s = self.compare(x)

        if s < self.gamma:
            self.add(x)

        # reward bonus (for discovering something dissimilar to items in memory)
        # tuning this could be a pain
        # TODO want to use online calc of moments here!?
        return self.alpha * (self.beta - s)

    def add(self, x):
        if len(self.memory) > self.max_size:
            # NOTE would prefer a least recently used mechanism!?
            self.memory.remove(*random.sample(self.memory, 1))
        else:
            self.memory.add(x)

    def reset(self):
        # NOTE problem(?). because we reset the episodic memory. there will always be
        # exploration rewards at the start. good or bad!?
        # OHH! this is actually quite important. it means we still have a
        # chance to revisit some places, and get reward, again.
        # (despite its lack of novelty, we just forgot it...)
        # if this is not tuned right, it can get endless entertainment from little exploration
        # not quite. reachability will punish nearby states!?
        # this avoids the problem of detachment - see https://eng.uber.com/go-explore/
        self.memory = set()

    def compare(self, e):
        """
        Compare the current state with memory and estimate their similarity.
        0 means dissimilar/novel
        1 means similar/not new

        Args:
            list: a list of tensors [1 x n_hidden]
            tensor: a tensor [1 x n_hidden]

        Returns
            float: the similarity between memory and e
        """
        # BUG as we learn the similarity metric, the elements in our memory become weird...
        # can we assume that the metric doesnt really change wrt memory!?
        # only if seq_len ~= episode_len...
        # distributed offline training kinda solves this!?

        n = self.memory
        if len(n) == 0:
            # not similar to anything in memory
            return 0
        else:
            # this is O(n). as memory gets larger this costs more.
            # TODO is this parallelised?
            cs = tf.squeeze(tf.stack([self.sim(tf.concat([e, m],axis=-1)) for m in self.memory], axis=0))

            # or use a unlearned version!?
            # cs = [cosine_dist(e, m) for m in memory]

            i = math.ceil(len(n)/10)

            if len(n) == 1:
                return cs
            else:
                return tf.reduce_mean(tf.contrib.framework.sort(cs)[:i])

    def get_loss(self, x):
        # how necessary is it to train this!?
        # could do with a random similarity measure?!
        reachable, not_reachable = reachable_training_pairs(x, 5, 3)

        sim_reach = self.sim(reachable)  # should be = 1
        sim_not_reach = self.sim(not_reachable)  # should be = 0

        return tf.reduce_mean(-tf.log(sim_reach+1e-6)) + tf.reduce_mean(-tf.log(1-sim_not_reach+1e-6))

class Policy():
    # Currently A2C TODO change to PPO or something smarter!
    def __init__(self, n_actions, n_hidden):
        self.policy_fn = tf.keras.Sequential([
            tf.keras.layers.Dense(n_hidden, activation=tf.nn.selu),
            utl.Residual(n_hidden//4, n_hidden),
            utl.Residual(n_hidden//4, n_hidden),
            utl.Residual(n_hidden//4, n_hidden),
            tf.keras.layers.Dense(n_actions)
        ])

        self.value_fn = tf.keras.Sequential([
            tf.keras.layers.Dense(n_hidden, activation=tf.nn.selu),
            utl.Residual(n_hidden//4, n_hidden),
            utl.Residual(n_hidden//4, n_hidden),
            utl.Residual(n_hidden//4, n_hidden),
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

        return tf.reduce_mean(pg_loss)

class Explorer():
    """
    An RL learner that uses reachability estimates (estimated via memory) to guide exploration.
    https://arxiv.org/abs/1810.02274

    Puts everything together.
    Manages gradient computation, summaries, resets, ...
    """
    def __init__(self, n_actions, n_hidden=64, logdir="/tmp/exp/0"):
        self.writer = tf.contrib.summary.create_file_writer(logdir)
        self.writer.set_as_default()

        self.embed = tf.keras.Sequential([
            tf.keras.layers.Conv2D(n_hidden, 4, 2, 'same', activation=tf.nn.selu),
            tf.keras.layers.Conv2D(n_hidden, 4, 2, 'same', activation=tf.nn.selu),
            tf.keras.layers.Conv2D(n_hidden, 4, 2, 'same', activation=tf.nn.selu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(n_hidden),
        ])

        self.memory = EpisodicMemory(200, n_hidden)
        self.policy = Policy(n_actions, n_hidden)

        self.opt = tf.train.AdamOptimizer(1e-4)
        self.global_step = tf.train.get_or_create_global_step()

        # checkpoint_dir = '/tmp/exp-ckpts/0'
        # os.makedirs(checkpoint_dir, exist_ok=True)
        # self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        # self.root = tf.train.Checkpoint(
        #     sim=self.memory.sim,
        #     embed=self.embed,
        #     policy=self.policy_fn,
        #     value=self.value_fn
        # )

    def reset(self):
        self.memory.reset()

    def __call__(self, x):
        """
        Online call for the explorer. It picks actions and gives rewards.

        Args:
            x: the current state, in a fully observed setting, this can simply be the observations
        Returns:
            a: the action chosen
            b: the reward bonus
        """
        x = tf.expand_dims(x, 0)
        e = self.embed(x)
        b = self.memory(e)
        # TODO want the policy to be a fn of the memory!?!?
        a = self.policy(e)

        return a, b

    def train_step(self, x, at, r, b):
        """
        Args:
            batched inputs:
                (x, a, r, b) with shapes [Timesteps, Batch, ...]
        """
        x = tf.constant(x, dtype=tf.float32)
        at = tf.squeeze(tf.constant(at, dtype=tf.int32))
        r = tf.constant(r, dtype=tf.float32)
        b = tf.constant(b, dtype=tf.float32)

        e = tf.map_fn(self.embed, x)

        with tf.GradientTape() as tape:
            pg_loss = self.policy.get_loss(e, at, r+b)
            reach_loss = self.memory.get_loss(e)
            loss = pg_loss + reach_loss  # HACK should be ok to do this!?

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            tf.contrib.summary.scalar('loss/pg', pg_loss)
            tf.contrib.summary.scalar('loss/reach', reach_loss)
            tf.contrib.summary.scalar('rewards/R', tf.reduce_mean(r))
            tf.contrib.summary.scalar('rewards/B', tf.reduce_mean(b))

        variables = self.policy.variables + self.embed.variables + self.memory.variables
        grads = tape.gradient(loss, variables)
        self.opt.apply_gradients(zip(grads, variables), global_step=self.global_step)
        return loss

    def save(self):
        self.save_loc = self.root.save(self.checkpoint_prefix)

    def load(self):
        self.root.restore(self.save_loc)
