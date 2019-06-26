import numpy as np

import argparse

def argumentparser():
    parser = argparse.ArgumentParser(description='Search spce dynamics')
    parser.add_argument('--logdir', type=str, default='/tmp/',
                        help='location to save logs')
    return parser.parse_args()

def main(args):
    """
    Want to explore the inductive bias of different search spaces.
    """
    mdp = random_mdp()

    """
    Properties of trajectories???
    - curvature
    - length
    - ?
    """

    solvers = [
        value_iteration,
        policy_iteration,
        parameter_iteration
    ]

    optimisers = [
        sgd,
        momentum
    ]


if __name__ == '__main__':
    main(argumentparser())
