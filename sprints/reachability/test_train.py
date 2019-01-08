import gym
import tensorflow as tf
import reachability as rch
import utils as utl
from train import *

import unittest

class RndExplorer():
    def __call__(self, *args, **kwargs):
        return env.action_space.sample(), 0

    def reset(self, *args, **kwargs):
        pass

    def train_step(self, *args, **kwargs):
        pass

class TestTrainer(unittest.TestCase):
    def test_run(self):
        env = gym.make('MontezumaRevenge-v0')

        player = utl.Worker(RndExplorer(), batch_size=5)
        run(env, player, 10)

if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
