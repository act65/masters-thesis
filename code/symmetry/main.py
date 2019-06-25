import numpy as np
import gym

import reinforcement as rl
import similarities as sim
import abstractions as abs

import argparse

def argumentparser():
    parser = argparse.ArgumentParser(description='ModelBased learner')
    parser.add_argument('--logdir', type=str, default='/tmp/',
                        help='location to save logs')
    return parser.parse_args()

def play_episode(env, player, trajectory=[]):
    obs = env.reset()
    done = False
    V = 0
    s, a, r, = (0,0,0)

    # play an episode
    while not done:
        a = player(s)
        s, r, done, _ = env.step(a)
        V += r

        trajectory.append((s, a, r))

    return V, trajectory

def main(args):
    n_states = 8
    n_actions = 4

    env = gym.make('LunarLander-v2')
    player = rl.Explorer(n_states, n_actions)
    traj = []
    for _ in range(10):
        Rs, traj = play_episode(env, player, traj)
        print(Rs)

    # pairs = sim.value_pairs(traj, n_states, n_actions)
    pairs = sim.transition_pairs(traj, n_states, n_actions)
    model = abs.Abstraction(8+4, 2, logdir=args.logdir)
    model.train(pairs)

    learner = QLearner(model, d_states, n_actions)

    for _ in range(10):
        Rs, traj = play_episode(env, player)

if __name__ == '__main__':
    main(argumentparser())

    # representations = [
    #     StateAbstraction,
    #     ActionAbstraction,
    #     StateActionAbstraction, ]
    #
    # similarities = [
    #     value_based,
    #     transtion_reward_based,
    # ]
    #
    # for rep_class in representations:
    #     for sim in similarities:
    #         # construct training pairs based on ...?
    #         data_set = make_dataset(sim)
    #         rep = rep_class()
    #
    #         # pretrain a representation using the exploration data
    #         train(rep, dataset)
    #
    #         # evaluate on ???
    #         rs = evaluate(rep)
