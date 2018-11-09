import gym
import replaybuffers as rp
import rl
import tensorflow as tf
import numpy as np
import copy

class InverseReinforcementLearner():
    """
    Implementation of https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf.
    Max entropy IRL.

    No. Implementation of matching feature expectation.
    """
    def __init__(self):
        self.reward_fn = None
        self.buffer = rp.ReplayBufferv2(max_size=2000)
        self.batch_size = 50
        self.gamma = 0.9

        self.build_graph(2, 16)

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
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation=tf.nn.selu),
                tf.keras.layers.Dense(64, activation=tf.nn.selu),
                tf.keras.layers.Dense(n_hidden*2)
            ])
            self.decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation=tf.nn.selu),
                tf.keras.layers.Dense(64, activation=tf.nn.selu),
                tf.keras.layers.Dense(4+1)
            ])

            # trained with td error
            self.discounted_features = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation=tf.nn.selu),
                tf.keras.layers.Dense(64, activation=tf.nn.selu),
                tf.keras.layers.Dense(n_hidden*2)
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
                self.opt.compute_gradients(loss_mu, self.discounted_features.variables) +
                self.opt.compute_gradients(loss_w, self.linear.variables) +
                self.opt.compute_gradients(loss_enc, self.encoder.variables+self.decoder.variables)
            )

            self.train_op = self.opt.apply_gradients(gnvs, global_step=self.global_step)

            self.sess.run(tf.global_variables_initializer())

    def train(self, batch):
        feed = dict(zip([self.obs, self.a, self.obs_t, self.a_t], batch))
        loss = self.sess.run([self.train_op], feed_dict=feed)
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
        # could just draw pictures?
        pass

def main():
    actor = rl.PG()
    observer = InverseReinforcementLearner()

    env = gym.make('CartPole-v1')

    def run_episode():
        obs = env.reset()
        done = False
        R = 0
        reward = 0

        while not done:
            action = actor(obs, reward)
            reward_fn = observer(obs, action)
            obs, reward, done, info = env.step(action)

            R += reward

        return R

    for i in range(1000):
        R = run_episode()

        summary = tf.Summary()
        summary.value.add(tag='return', simple_value=R)
        actor.writer.add_summary(summary, i)
        actor.writer.flush()

        print('\r{}'.format(R), end='', flush=True)

if __name__ == '__main__':
    main()
