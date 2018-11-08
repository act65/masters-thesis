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
        self.buffer = rp.ReplayBuffer(max_size=2000,max_len=100)
        self.done = False
        self.trajectory = []
        self.batch_size = 50

    def build_graph(self):
        with tf.Graph().as_default():
            self.sess = tf.Session()
            self.opt = tf.train.AdamOptimizer()
            self.global_step = tf.train.get_or_create_global_step()

            # batch x time x ...
            self.obs = tf.placeholder(name='obs', shape=[None, None, 4], dtype=tf.float32)
            self.a = tf.placeholder(name='a', shape=[None, None], dtype=tf.int32)

            obs = tf.reshape(self.obs, [-1, 4])
            a = tf.reshape(self.a)

            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation=tf.nn.selu),
                tf.keras.layers.Dense(64, activation=tf.nn.selu),
                tf.keras.layers.Dense(8)
            ])

            self.linear = tf.keras.layers.Dense(1)

            features = tf.map_fn(self.encoder, self.obs)

            # funny business with the discounting
            discounted_features = features #???

            values = 

            # values = None
            #
            # prob = tf.nn.softmax()
            #
            # action_dist = tfd.RelaxedOneHotCategorical(1.0, prob=prob)
            # action_dist.log_prob(a)
            #
            #
            # train_op = self.opt.minimize(loss, global_step=self.global_step)


            self.sess.run(tf.global_variabiles_initializer())

    def train(self, batch):
        # self.sess.run([self.train_op], feed=dict(zip([self.obs, self.a], batch)))
        pass

    def __call__(self, obs, a, done):
        if done:
            self.buffer.add(copy.deepcopy([np.stack(x, axis=0) for x in zip(*self.trajectory)]))
            self.trajectory = []
        else:
            self.trajectory.append([obs, np.array([a])])

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
        count = 0
        old_obs = obs

        while not done:
            if count >= 100:
                break
            count += 1

            action = actor(obs, reward)
            obs, reward, done, info = env.step(action)
            reward_fn = observer(old_obs, action, done)

            R += reward
            old_obs = obs

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
