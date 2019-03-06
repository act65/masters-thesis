import numpy as np

import argparse

import planet as pl
import worker as wk
"""

"""

def argumentparser():
    parser = argparse.ArgumentParser(description='ModelBased learner')
    parser.add_argument('--trials', type=int, default=50,
                        help='number of trials')
    parser.add_argument('--logdir', type=str, default='/tmp/offline_offpolicy/0',
                        help='location to save logs')
    parser.add_argument('--env_name', type=str, default='LunarLander-v2',
                        choices=['LunarLander-v2', 'BipedalWalkerHardcore-v2'],
                        help='Which Gym env')
    return parser.parse_args()

def main(args):
    worker = wk.Worker(env_name=args.env_name, player=pl.Planet)

    for i in range(args.trials):
        ### Evaluate the current policy


        # use workers to collect data
        returns = worker.work(1)

        ### Update the policy
        batch = worker.get_batch(5)
        if batch is not None and i>10:
            loss = worker.player.update(*batch)
            print('\rLoss: {}, Return: {}'.format(loss, np.mean(returns)), end='', flush=True)
            # local_worker.run_episode(weights, render=True)

if __name__ == '__main__':
    main(argumentparser())
