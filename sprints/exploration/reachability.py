import os
import math
import random

import numpy as np
import tensorflow as tf

import utils as utl

def reachable_training_pairs(x, k, n):
    """
    Two states are considered 'reachable' if they occur within k timesteps.

    Args:
        x ([T, B, D]): A sequence of elements
        k: the number of steps that makes two states mutually reachable
        n: the number of elements to sample

    Returns:
        tuple ([n*B x D*2], [n*B x D*2]):
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
            return 0  # not similar to anything in memory
        else:
            # this is O(n). as memory gets larger this costs more.
            # TODO is this parallelised well?
            cs = tf.squeeze(tf.stack([self.sim(tf.concat([e, m],axis=-1)) for m in self.memory], axis=0))

            i = math.ceil(len(n)/10)

            if len(n) == 1:
                return cs
            else:
                return tf.reduce_mean(tf.contrib.framework.sort(cs)[:i])

    def get_loss(self, x):
        """
        Args:
            x: tensor of shape [T, B, ...]

        Returns:
            loss (tf.tensor): shape = []
        """
        reachable, not_reachable = reachable_training_pairs(x, 5, 3)

        sim_reach = self.sim(reachable)  # should be = 1
        sim_not_reach = self.sim(not_reachable)  # should be = 0

        acc_reach = utl.accuracy(tf.round(sim_reach), tf.ones_like(sim_reach))
        acc_not_reach = utl.accuracy(tf.round(sim_not_reach), tf.zeros_like(sim_not_reach))
        loss = tf.reduce_mean(-tf.log(sim_reach+1e-6)) + tf.reduce_mean(-tf.log(1-sim_not_reach+1e-6))

        tf.contrib.summary.scalar('loss/reach', loss)
        tf.contrib.summary.scalar('acc/reach', acc_reach)
        tf.contrib.summary.scalar('acc/not_reach', acc_not_reach)

        return loss


class Explorer():
    """
    An RL learner that uses reachability estimates (estimated via memory) to guide exploration.
    https://arxiv.org/abs/1810.02274

    Puts everything together.
    Manages gradient computation, summaries, resets, ...
    """
    def __init__(self, policy_spec, n_actions, n_hidden=64, memory_max_size=200, encoder_beta=1e-3, lr=1e-3, logdir="/tmp/exp/0"):
        self.writer = tf.contrib.summary.create_file_writer(logdir)
        self.writer.set_as_default()

        self.memory = EpisodicMemory(memory_max_size, n_hidden)
        self.policy = policy_spec(n_actions, n_hidden)
        self.embed = utl.get_conv_net(n_hidden)

        self.opt = tf.train.AdamOptimizer(lr)
        self.global_step = tf.train.get_or_create_global_step()

        # checkpoint_dir = '/tmp/exp-ckpts/0'
        # os.makedirs(checkpoint_dir, exist_ok=True)
        # self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        # self.root = tf.train.Checkpoint(
        #     sim=self.memory.sim,
        #     embed=self.embed,
        #     policy=self.policy.policy_fn,
        #     value=self.policy.value_fn
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
        # get a representation of the current state
        e = self.embed(x)
        # look up in memory
        b = self.memory(e)
        # choose action
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
            with tf.contrib.summary.record_summaries_every_n_global_steps(10):

                policy_loss = self.policy.get_loss(e, at, r+b)
                reach_loss = self.memory.get_loss(e)
                loss = policy_loss + reach_loss  # HACK should be ok to do this!?

                tf.contrib.summary.scalar('rewards/R', tf.reduce_mean(r))
                tf.contrib.summary.scalar('rewards/B', tf.reduce_mean(b))

        variables = self.policy.variables + self.memory.variables + self.embed.variables
        grads = tape.gradient(loss, variables)
        self.opt.apply_gradients(zip(grads, variables), global_step=self.global_step)
        return loss

    def save(self):
        self.save_loc = self.root.save(self.checkpoint_prefix)

    def load(self):
        self.root.restore(self.save_loc)
