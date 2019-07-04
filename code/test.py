import jax.numpy as np
from jax import grad

import numpy.random as rnd
import functools

x = [rnd.random((2,2)) for _ in range(3)]

def fn(W):
    return np.sum(functools.reduce(np.dot, W))

dfn = grad(fn)
print(dfn(x))
