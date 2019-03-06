import ray
import numpy as np

import argparse

"""
An offline learner using off policy methods. Aka a DQN with many workers.
- Offline learning. Store data in a buffer.
- Off policy. Learn the optimal policy while following another policy.

Problems with this learner;
- long episodes do not fit in memory...
- at the start many episodes can be played without ever receiving a +1
- ?
"""

def argumentparser():
    parser = argparse.ArgumentParser(description='Offline offpolicy learner')
    parser.add_argument('--trials', type=int, default=50,
                        help='number of trials')
    parser.add_argument('--logdir', type=str, default='/tmp/offline_offpolicy/0',
                        help='location to save logs')
    parser.add_argument('--env_name', type=str, default='LunarLander-v2',
                        choices=['LunarLander-v2', 'BipedalWalkerHardcore-v2', 'LunarLanderContinuous-v2'],
                        help='Which Gym env')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of workers')
    return parser.parse_args()

def main(args):
    local_worker = rl.Worker(env_name=args.env_name, player=rl.Player, is_remote=False)

    remote_worker = ray.remote(rl.Worker)
    workers = [remote_worker.remote(env_name=args.env_name, player=rl.Player, is_remote=True) for _ in range(args.n_workers)]

    for i in range(10000):
        ### Evaluate the current policy
        # get current weights
        weights = local_worker.player.get_weights()
        weights = {k: v + 0.1*np.random.standard_normal(v.shape) for k, v in weights.items()}
        weights_id = ray.put(weights)

        # use workers to collect data
        returns = [worker.run_episode.remote(weights_id) for worker in workers]

        ### Update the policy
        # gather experience from each worker's buffer
        batches = ray.get([worker.get_batch.remote(10) for worker in workers])

        # train
        if batches[0] is not None and i>10:
            batches = [np.vstack(x) for x in zip(*batches)]

            loss = local_worker.player.train(batches)
            print('\rLoss: {}, Return: {}'.format(loss, np.mean(ray.get(returns))), end='', flush=True)
            local_worker.run_episode(weights, render=True)

if __name__ == '__main__':
    ray.init()
    main(argumentparser())
