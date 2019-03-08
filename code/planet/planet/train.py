import numpy as np

import argparse

import learners as ln
import worker as wk

import matplotlib.pyplot as plt

"""
Main training script.
Offline learning.
"""

def argumentparser():
    parser = argparse.ArgumentParser(description='ModelBased learner')
    parser.add_argument('--trials', type=int, default=200,
                        help='number of trials')
    parser.add_argument('--logdir', type=str, default='/tmp/mpc/0',
                        help='location to save logs')
    parser.add_argument('--env_name', type=str, default='CartPole-v1',
                        choices=['LunarLander-v2', 'BipedalWalkerHardcore-v2', 'CartPole-v1', 'Acrobot-v-1'],
                        help='Which Gym env')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch_size')
    return parser.parse_args()

def main(args):
    worker = wk.Worker(env_name=args.env_name, player=ln.Planet)
    # worker = wk.Worker(env_name=args.env_name, player=ln.ActorCritic)

    losses = []
    returns = []

    # generalised policy iteration.
    for i in range(args.trials):
        ### Evaluate the current policy
        # use workers to collect data
        Rs = worker.work(1)
        returns.append(np.mean(Rs))

        ### Update the policy
        batch = worker.get_batch(args.batch_size)
        if batch is not None:
            losses.append(worker.player.update(*batch))
            print('\rStep: {} Loss: {}, Return: {}'.format(i, losses[-1], returns[-1]), end='', flush=True)

        # if i % 50 == 0:
        #     Rs = worker.play_episode(render=True)

    transition_losses, value_losses = tuple(zip(*losses))
    plt.subplot(3,1,1)
    plt.plot(np.log(transition_losses))
    plt.subplot(3,1,2)
    plt.plot(np.log(value_losses))
    plt.subplot(3,1,3)
    plt.plot(returns)

    # plt.subplot(2,1,1)
    # plt.plot(np.log(losses))
    # plt.subplot(2,1,2)
    # plt.plot(returns)

    plt.show()

if __name__ == '__main__':
    main(argumentparser())
