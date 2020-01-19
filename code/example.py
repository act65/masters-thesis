"""
Potential examples!?
- Race track ('mirror' symmetry)
- ??? (cyclic)
- ??? (dihedral)
- ??? (permutation)
- ??? (quarternoin)
- ??? (alternating)
"""

import numpy as np
from mdp import utils

# TODO would be nice to be able to generate these, for arbitrary grid sizes?
# yeah. i know how to do that!
# transition fn. 6 x 6 x 4
P = np.zeros((6, 6, 4))
# up
P[:, :, 0] = np.array([
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1,0,0,0],
    [0,0,0,1,1,1],
])

# right
P[:, :, 1] = np.array([
    [0,0,0,0,0,0],
    [1,1,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,1,1,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,1,1],
])

# down
P[:, :, 2] = np.array([
    [1,0,1,0,0,0],
    [0,1,0,1,0,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
])

# left
P[:, :, 3] = np.array([
    [1,1,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,1,1,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,1,1],
    [0,0,0,0,0,0],
])


# rewards. 6 x 4
r = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 0],  # rewarded for going up at the finish.
    [1, 0, 0, 0],
])


# initial distribution
d0 = np.array([
    [0.5, 0.5, 0, 0, 0, 0]
])

pi = np.array(utils.random_policy(6, 4))
pi[[0, 2, 4]] = pi[[1, 3, 5]]
V = utils.value_functional(P, r, pi, 0.5)
Q_t = utils.bellman_operator(P, r, V, 0.5)
# print(np.sum(P, axis=-1))
print(Q_t)
