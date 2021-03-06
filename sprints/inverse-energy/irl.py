import gym
import replaybuffers as rp
import rl
import tensorflow as tf
import numpy as np
import copy

class InverseReinforcementLearner():
    """
    Implementation of matching feature expectation.
    https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf

    and IOC
    https://homes.cs.washington.edu/~todorov/papers/DvijothamICML10.pdf
    """
    def __init__(self, writer=None):
        self.writer = tf.summary.FileWriter('/tmp/irl/0') if writer is None else writer

        self.buffer = rp.ReplayBufferv2(max_size=2000)

        self.batch_size = 50
        self.gamma = 0.9

        self.build_graph(2, 64)

        self.old_obs = None
        self.old_a = None

    def build_graph(self, n_actions, n_hidden):
        with tf.Graph().as_default():
            self.sess = tf.Session()
            self.opt = tf.train.AdamOptimizer()
            self.global_step = tf.train.get_or_create_global_step()

            # batch x time x ...
            self.obs = tf.placeholder(name='obs', shape=[None, 4], dtype=tf.float32)
            self.a = tf.placeholder(name='a', shape=[None, 1], dtype=tf.int32)
            self.obs_t = tf.placeholder(name='obs_t', shape=[None, 4], dtype=tf.float32)
            self.a_t = tf.placeholder(name='a_t', shape=[None, 1], dtype=tf.int32)

            # trained as an AE
            # or maybe not trained. just rnd projection!?
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation=tf.nn.selu),
                tf.keras.layers.Dense(256, activation=tf.nn.selu),
                tf.keras.layers.Dense(n_hidden)
            ])
            self.decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation=tf.nn.selu),
                tf.keras.layers.Dense(256, activation=tf.nn.selu),
                tf.keras.layers.Dense(4+1)
            ])

            # trained with td error
            self.discounted_features = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation=tf.nn.selu),
                tf.keras.layers.Dense(256, activation=tf.nn.selu),
                tf.keras.layers.Dense(n_hidden)
            ])

            # trained with cross entropy on the optimal policy
            self.linear = tf.keras.layers.Dense(1)

            x = tf.concat([self.obs, tf.cast(self.a, tf.float32)], axis=-1)
            x_t = tf.concat([self.obs_t, tf.cast(self.a_t, tf.float32)], axis=-1)

            features = self.encoder(x)
            loss_enc = tf.losses.mean_squared_error(x, self.decoder(features))

            mu = self.discounted_features(x) # use obs or the features?
            mu_t = self.discounted_features(x_t)
            loss_mu = tf.losses.mean_squared_error(
                        mu, tf.stop_gradient(features)
                        + self.gamma*tf.stop_gradient(mu_t))

            xs = [tf.concat([self.obs, i*tf.ones_like(self.a, dtype=tf.float32)], axis=-1) for i in range(n_actions)]
            mus = tf.map_fn(self.discounted_features, xs, dtype=tf.float32)
            logits = tf.squeeze(tf.concat(tf.map_fn(self.linear, mus), axis=-1))
            loss_w = tf.losses.sparse_softmax_cross_entropy(labels=self.a, logits=logits)

            gnvs = (
                self.opt.compute_gradients(loss_mu, self.discounted_features.variables)
                + self.opt.compute_gradients(loss_w, self.linear.variables)
                + self.opt.compute_gradients(loss_enc, self.encoder.variables+self.decoder.variables)
            )

            self.loss = loss_w + loss_mu + loss_enc
            tf.summary.scalar('loss_w', loss_w)
            tf.summary.scalar('loss_mu', loss_mu)
            tf.summary.scalar('loss_enc', loss_enc)
            self.summaries = tf.summary.merge_all()

            self.train_op = self.opt.apply_gradients(gnvs, global_step=self.global_step)
            self.sess.run(tf.global_variables_initializer())

    def train(self, batch):
        feed = dict(zip([self.obs, self.a, self.obs_t, self.a_t], batch))
        _, loss, step, summ  = self.sess.run([self.train_op, self.loss, self.global_step, self.summaries], feed_dict=feed)
        self.writer.add_summary(summ, step)
        return loss

    def __call__(self, obs, a):
        if self.old_obs is not None:
            self.buffer.add([self.old_obs, np.array([self.old_a]), obs, np.array([a])])

        self.old_obs = copy.deepcopy(obs)
        self.old_a = copy.deepcopy(a)

        if self.buffer.size > self.batch_size:
            batch = self.buffer.get_batch(self.batch_size)
            loss = self.train(batch)

        return None

    def evaluate(self):
        # evaluate the accuracy of the estimated reward
        # could just draw pictures? rotate around 360 degrees and plot rewards
        pass
