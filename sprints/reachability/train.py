import argparse

import gym
import tensorflow as tf

import reachability as rch
import utils as utl


def argumentparser():
    parser = argparse.ArgumentParser(description='Explore')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--logdir', type=str, default='/tmp/exp/0',
                        help='location to save logs')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='length of sequences to train on')
    parser.add_argument('--memory_max_size', type=int, default=200,
                        help='size of memory')
    parser.add_argument('--encoder_beta', type=float, default=1e-3,
                        help='strength of training loss for encoder')
    return parser.parse_args()


def run(env, player, seq_len):
    obs = env.reset()

    R = 0
    r = 0.0
    count = 1
    episodes_played = 0

    while True:
        # HACK soln to padding sequences. wrap them and continue
        # rather than using the done flag

        seq_break = True if (count % (seq_len+1) == 0) and (count != 1) else False

        a = player(obs, r, seq_break)
        obs, r, done, info = env.step(a)
        R += r

        if done:
            player.learner.reset()  # reset the explorers memory
            obs = env.reset()
            R = 0
            episodes_played += 1

        count += 1

        if count % 20 == 0:
            M = len(player.learner.memory.memory)
            B = len(player.buffer.buffer)

            print('\ri: {} R: {} M: {}, B: {}'.format(episodes_played, R, M, B), end='', flush=True)

        # if count % 10000 == 0:
        #     player.learner.save()

    return R, M


if __name__ == "__main__":
    tf.enable_eager_execution()
    args = argumentparser()
    env = gym.make('MontezumaRevenge-v0')
    player = utl.Worker(rch.Explorer(utl.RndPolicy,
                                     env.action_space.n,
                                     memory_max_size=args.memory_max_size,
                                     encoder_beta=args.encoder_beta,
                                     logdir=args.logdir),
                        batch_size=args.batch_size)
    run(env, player, args.seq_len)
