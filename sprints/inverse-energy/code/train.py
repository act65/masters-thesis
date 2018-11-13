import gym
import replaybuffers as rp
import rl
import tensorflow as tf
import numpy as np
import copy
import argparse

def argumentparser():
    parser = argparse.ArgumentParser(description='IRL')
    parser.add_argument('--env_name', type=str, default='LunarLander-v2',
                        choices=['LunarLander-v2', 'BipedalWalkerHardcore-v2'],
                        help='Which Gym env')
    parser.add_argument('--logdir', type=str, default='/tmp/irl/0',
                        help='location to save logs')
    return parser.parse_args()

def main(args):
    writer = tf.summary.FileWriter(args.logdir)

    env = gym.make(args.env_name)
    obs = env.reset()

    actor = rl.DIFF(
        writer=writer,
        batch_size=512,
        n_inputs=obs.shape[0],
        n_hidden=128,
        n_actions=env.action_space.n,
        cts_actions='float' in str(env.action_space.dtype)
    )
    # observer = InverseReinforcementLearner(writer)

    def run_episode():
        obs = env.reset()
        done = False
        R = 0
        reward = 0

        while not done:
            action = actor.call(np.array(obs).astype(np.float32), reward)
            # reward_fn = observer(obs, action)
            env.render()
            obs, reward, done, info = env.step(action)

            R += reward

        return R

    for i in range(10000):
        R = np.mean([run_episode() for _ in range(100)])

        summary = tf.Summary()
        summary.value.add(tag='return', simple_value=R)
        writer.add_summary(summary, i)
        writer.flush()

        print('\r{}'.format(R), end='', flush=True)

if __name__ == '__main__':
    main(argumentparser())
