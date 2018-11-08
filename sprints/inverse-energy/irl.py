import gym
import replaybuffers as rp
import rl
import tensorflow as tf
import numpy as np
import copy

class InverseReinforcementLearner():
    """
    Implementation of https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf
    Max entropy IRL.
    """
    def __init__(self):
        self.reward_fn = None
        self.buffer = rp.ReplayBuffer(max_size=2000)
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

            ### use policy to pick action
            self.reward_fn = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation=tf.nn.selu),
                tf.keras.layers.Dense(64, activation=tf.nn.selu),
                tf.keras.layers.Dense(1)
            ])

            x = tf.concat([self.obs, tf.cast(self.a, tf.float32)], axis=-1)
            cost = tf.reduce_sum(tf.map_fn(self.reward_fn, x), axis=1)  # sum over times

            x_ = x + tf.random_normal(tf.shape(x), stddev=0.5)
            cost_ = tf.reduce_sum(tf.map_fn(self.reward_fn, x_), axis=1)  # sum over times

            # is it possible to view only per step? or do we need entire trajectories?
            # do I need a global view!?
            diff = tf.clip_by_value(cost_ - cost, -2.0, 2.0)
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=-1))
            loss += 0.0001*tf.reduce_mean(tf.abs(cost))  # regularise for sparsity
            train_op = self.opt.minimize(loss, global_step=self.global_step)

            # not sure the above training makes sense. both x and x_ might achieve the true goal!?
            # a reward has been gven because a specific state has been reached. this required many actions to be taken in the past.

            # NOTE but this isnt going to recover the reward fn. probably something closer to the value fn!???
            # which priors should be included?
            # - sparsity
            # - reward provided at end of episode?



            self.sess.run(tf.global_variabiles_initializer())

    def train(self, batch):
        print([x.shape for x in batch])
        raise SystemExit

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
