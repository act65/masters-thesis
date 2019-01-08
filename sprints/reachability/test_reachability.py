from reachability import *
from utils import *
import tensorflow as tf
tf.enable_eager_execution()

import unittest

class TestExplorerMethods(unittest.TestCase):
    def test_reach_memory_size(self):
        """make sure memory isnon zero"""
        exp = Explorer(2)

        for _ in range(200):
            x = tf.random_normal([32, 32, 4])
            exp(x)

        l = len(exp.memory)
        self.assertTrue(l > 0)
        self.assertTrue(l < 200)

    def test_action_set(self):
        """check that actions are within the right bounds/format"""
        n_actions = 4
        action_set = list(range(n_actions))
        exp = Explorer(n_actions)

        actions = [exp(tf.random_normal([32, 32, 4])) for _ in range(200)]
        for a, b in actions:
            self.assertTrue(a.numpy() in action_set)

class TestReachPairs(unittest.TestCase):
    x = tf.random_normal([200, 50, 3])
    reachable, not_reachable = reachable_training_pairs(x, 4, 3)
    print(reachable.shape, not_reachable.shape)

if __name__ == '__main__':
    unittest.main()
